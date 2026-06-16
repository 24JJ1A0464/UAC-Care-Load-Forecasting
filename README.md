UAC Care Load Predictive Forecasting Dashboard
Overview
This project was developed during my Data Analyst internship at Unified Mentor. It transitions the federal Unaccompanied Alien Children (UAC) Program from reactive historical reporting to proactive predictive intelligence. The deployed Streamlit web application allows stakeholders to forecast 30-day care loads, anticipate discharge demand, and run "What-If" crisis scenarios to prevent system capacity breaches.

Key Features

Time-Series Interpolation: Cleaned and imputed missing reporting days using time-weighted linear interpolation to ensure a continuous federal reporting calendar.

Feature Engineering: Engineered sophisticated predictive signals, including 7-day rolling momentum, temporal lags, and a flow-based 'Net Pressure' indicator (Intake minus Discharges).

Machine Learning Pipeline: Evaluated multiple models (SARIMA, Gradient Boosting, XGBoost), ultimately deploying a Random Forest Regressor.

Interactive Dashboard: Built a live Streamlit application with dynamic Plotly visuals, interactive KPI scoring, and a border surge scenario simulator.

Project Results & KPIs

Forecast Accuracy: 99.73% overall accuracy on a 30-day blind holdout test.

Absolute Error: Mean Absolute Error (MAE) of only 6.56 children per day.

Long-Term Stability: Demonstrated an expanding horizon reliability (Short-Term Error: 0.38% -> Long-Term Error: 0.22%).

Tech Stack

Languages: Python

Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Statsmodels

Web Framework: Streamlit, Plotly

How to Run Locally

Clone the repository.

Install the requirements: pip install -r requirements.txt

Launch the dashboard: streamlit run app.py
