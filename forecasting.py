import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# --- 1. DATA PREP & FEATURE ENGINEERING ---
df = pd.read_csv('HHS_Unaccompanied_Alien_Children_Program.csv', thousands=',')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Date']).set_index('Date').sort_index()
df = df.apply(pd.to_numeric, errors='coerce')
df_clean = df.asfreq('D').interpolate(method='time').round()

ml_df = df_clean.copy()
target = 'Children in HHS Care'
transfers_in = 'Children transferred out of CBP custody' 
discharges = 'Children discharged from HHS Care'

ml_df['CareLoad_Lag1'] = ml_df[target].shift(1)
ml_df['CareLoad_Lag7'] = ml_df[target].shift(7)
ml_df['Rolling_Mean_7'] = ml_df[target].rolling(window=7).mean().round(1)
ml_df['Net_Pressure'] = ml_df[transfers_in].shift(1) - ml_df[discharges].shift(1)
ml_df = ml_df.dropna()

X = ml_df.drop(columns=[target, transfers_in, discharges, 'Children apprehended and placed in CBP custody*', 'Children in CBP custody'])
y = ml_df[target]

X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

# --- 2. TRAIN THE WINNING MODEL (Random Forest) ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- 3. CALCULATE THE 4 EVALUATION METRICS ---
# 1. MAE
mae = mean_absolute_error(y_test, predictions)

# 2. RMSE (We calculate Mean Squared Error, then take the square root)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# 3. MAPE (Converted to a percentage)
mape = mean_absolute_percentage_error(y_test, predictions) * 100

# 4. Horizon Error (Breaking the 30 days into chunks)
horizon_1_7 = mean_absolute_percentage_error(y_test.iloc[0:7], predictions[0:7]) * 100
horizon_8_14 = mean_absolute_percentage_error(y_test.iloc[7:14], predictions[7:14]) * 100
horizon_15_30 = mean_absolute_percentage_error(y_test.iloc[14:30], predictions[14:30]) * 100


# --- 4. PRINT THE FINAL EVALUATION REPORT ---
print("\n==============================================")
print("📊 FINAL MODEL EVALUATION (Random Forest)")
print("==============================================")
print(f"1. MAE (Absolute Accuracy):   {mae:.2f} children")
print(f"2. RMSE (Large Error Check):  {rmse:.2f} children")
print(f"3. MAPE (Relative Error):     {mape:.2f} %")
print("\n4. HORIZON ERROR (Reliability over time):")
print(f"   -> Short-Term (Days 1-7):  {horizon_1_7:.2f} % error")
print(f"   -> Mid-Term (Days 8-14):   {horizon_8_14:.2f} % error")
print(f"   -> Long-Term (Days 15-30): {horizon_15_30:.2f} % error")
print("==============================================\n")