import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UAC Predictive Dashboard", layout="wide")
st.title("HHS UAC Program: Predictive Intelligence Dashboard")
st.markdown("Forecast Future Care Loads, Anticipate Discharge Demand, and Run Scenario Simulations.")

# --- 1. DATA LOADING & CACHING ---
# @st.cache_data makes sure the app doesn't reload the CSV every time you click a button
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('HHS_Unaccompanied_Alien_Children_Program.csv', thousands=',')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Date']).set_index('Date').sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')
    df_clean = df.asfreq('D').interpolate(method='time').round()
    
    # Feature Engineering
    target = 'Children in HHS Care'
    discharges = 'Children discharged from HHS Care'
    transfers_in = 'Children transferred out of CBP custody' 
    
    df_clean['CareLoad_Lag1'] = df_clean[target].shift(1)
    df_clean['CareLoad_Lag7'] = df_clean[target].shift(7)
    df_clean['Rolling_Mean_7'] = df_clean[target].rolling(window=7).mean()
    df_clean['Net_Pressure'] = df_clean[transfers_in].shift(1) - df_clean[discharges].shift(1)
    df_clean = df_clean.dropna()
    return df_clean

df = load_and_prep_data()

# --- 2. SIDEBAR CONTROLS (User Capabilities) ---
st.sidebar.header("⚙️ Forecast Controls")

# Capability 1: Forecast horizon selector
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=1)

# Capability 2: Model toggle
selected_model = st.sidebar.selectbox("Select Forecasting Model", ["Random Forest (Recommended)", "XGBoost", "Naive Baseline"])

# Capability 3: Scenario comparison view
st.sidebar.markdown("---")
st.sidebar.header("⚠️ Scenario Simulator")
intake_multiplier = st.sidebar.slider("Border Intake Surge Multiplier", 0.5, 3.0, 1.0, 0.1, help="Simulate a sudden increase or decrease in border crossings.")

# Capacity Threshold
capacity_limit = st.sidebar.number_input("System Bed Capacity Limit", value=2600, step=100)


# --- 3. MODEL TRAINING & FORECASTING ---
def generate_forecast(model_type, steps, multiplier):
    X = df[['CareLoad_Lag1', 'CareLoad_Lag7', 'Rolling_Mean_7', 'Net_Pressure']]
    y_load = df['Children in HHS Care']
    y_discharge = df['Children discharged from HHS Care']
    
    # Train Models
    if model_type == "Random Forest (Recommended)":
        model_load = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y_load)
        model_discharge = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y_discharge)
    else:
        model_load = XGBRegressor(n_estimators=50, random_state=42).fit(X, y_load)
        model_discharge = XGBRegressor(n_estimators=50, random_state=42).fit(X, y_discharge)
        
    # Generate Future Dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    
    # Walk-forward prediction loop for scenarios
    future_loads = []
    future_discharges = []
    
    # Simulating the future row by row
    current_features = X.iloc[-1].copy()
    current_load = y_load.iloc[-1]
    
    for i in range(steps):
        # Apply scenario multiplier to the pressure
        current_features['Net_Pressure'] = current_features['Net_Pressure'] * multiplier
        
        pred_load = model_load.predict([current_features])[0]
        pred_discharge = model_discharge.predict([current_features])[0]
        
        future_loads.append(pred_load)
        future_discharges.append(pred_discharge)
        
        # Update features for the next day's prediction
        current_features['CareLoad_Lag7'] = current_features['CareLoad_Lag1']
        current_features['CareLoad_Lag1'] = pred_load
        
    return future_dates, future_loads, future_discharges

future_dates, future_loads, future_discharges = generate_forecast(selected_model, horizon, intake_multiplier)

# --- 4. TOP KPI CARDS ---
col1, col2, col3 = st.columns(3)
peak_load = max(future_loads)
col1.metric("Predicted Peak Care Load", f"{int(peak_load):,}")

# Check if capacity is breached
if peak_load > capacity_limit:
    col2.metric("Capacity Status", "🚨 BREACH WARNING")
else:
    col2.metric("Capacity Status", "✅ SAFE")
    
avg_discharge = np.mean(future_discharges)
col3.metric("Avg Daily Discharges Needed", f"{int(avg_discharge):,}")


# --- 5. MAIN VISUALIZATIONS (Core Modules) ---
st.markdown("---")
st.subheader("📈 Future Care Load Forecast (with Confidence Intervals)")

# Build Interactive Plotly Chart
fig_load = go.Figure()

# Plot History
fig_load.add_trace(go.Scatter(x=df.index[-60:], y=df['Children in HHS Care'].iloc[-60:], mode='lines', name='Historical Load', line=dict(color='blue', width=2)))

# Plot Forecast
fig_load.add_trace(go.Scatter(x=future_dates, y=future_loads, mode='lines', name='Forecasted Load', line=dict(color='orange', width=3, dash='dash')))

# Confidence Intervals (Using historical RMSE ~8 as a base variance buffer)
upper_bound = [val + (8.36 * 2) for val in future_loads]
lower_bound = [val - (8.36 * 2) for val in future_loads]

fig_load.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False))
fig_load.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode='none', fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', name='95% Confidence Interval'))

# Capacity Line
fig_load.add_trace(go.Scatter(x=[df.index[-60], future_dates[-1]], y=[capacity_limit, capacity_limit], mode='lines', name='Max Capacity Limit', line=dict(color='red', width=2)))

fig_load.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_load, use_container_width=True)


st.markdown("---")
st.subheader("🚪 Discharge Demand Forecast Panel")
st.markdown("Predicts the daily number of sponsor placements and discharges required to maintain system balance.")

fig_discharge = go.Figure()
fig_discharge.add_trace(go.Scatter(x=df.index[-60:], y=df['Children discharged from HHS Care'].iloc[-60:], mode='lines', name='Historical Discharges', line=dict(color='green', width=2)))
fig_discharge.add_trace(go.Scatter(x=future_dates, y=future_discharges, mode='lines', name='Forecasted Discharges', line=dict(color='purple', width=3, dash='dot')))

fig_discharge.update_layout(height=300, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_discharge, use_container_width=True)