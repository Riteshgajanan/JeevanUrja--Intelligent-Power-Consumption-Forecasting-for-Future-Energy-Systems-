# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ---- Configuration ----
st.set_page_config(page_title="JeevanUrja — Energy Forecast", layout="wide")

# ---- Helper: load artifacts ----
@st.cache_resource(ttl=3600)
def load_artifacts(model_path='energy_consumption_model.pkl', scaler_path='scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# The features expected (order must match training)
FEATURE_ORDER = ['Temperature','Humidity','WindSpeed',
                 'GeneralDiffuseFlows','DiffuseFlows',
                 'Hour','DayOfWeek','IsWeekend','Month']

# ---- UI: Title ----
st.title("JeevanUrja — Intelligent Power Consumption Forecast")
st.markdown("""
Enter weather/time inputs below and click **Predict** to get the forecasted Total Power Consumption.
""")

# ---- Layout: Input panel and Output panel ----
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input features")
    # Weather inputs
    temp = st.number_input("Temperature (°C)", value=25.0, step=0.1, format="%.1f")
    humidity = st.number_input("Humidity (%)", value=60.0, step=0.1, format="%.1f")
    windspeed = st.number_input("Wind Speed (m/s)", value=3.0, step=0.1, format="%.1f")
    gdf = st.number_input("General Diffuse Flows (W/m²)", value=120.0, step=1.0)
    dfv = st.number_input("Diffuse Flows (W/m²)", value=80.0, step=1.0)

    # Datetime input
    st.subheader("Time input")
    dt = st.date_input("Date", value=datetime.today().date())
    t = st.time_input("Time (for Hour)", value=datetime.now().time())
    # derive hour/dayofweek/month/isweekend
    dt_full = datetime.combine(dt, t)
    hour = dt_full.hour
    dayofweek = dt_full.weekday()   # Monday=0
    month = dt_full.month
    isweekend = 1 if dayofweek >= 5 else 0

    st.write(f"Derived: hour = {hour}, day_of_week = {dayofweek}, month = {month}, is_weekend = {isweekend}")

    # Predict button
    if st.button("Predict Consumption"):
        # Prepare input vector in the same order as training
        input_df = pd.DataFrame([[temp, humidity, windspeed, gdf, dfv, hour, dayofweek, isweekend, month]],
                                columns=FEATURE_ORDER)

        # Scale numeric features (scaler expects shape (n_samples, n_features))
        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            st.error(f"Error scaling input: {e}")
            st.stop()

        # Predict
        pred = model.predict(input_scaled)[0]

        # Show result
        st.success(f"Predicted Total Consumption: {pred:,.2f} (same units as training data)")
        st.info("Note: prediction uses the same preprocessing as model training (StandardScaler)")

with col2:
    st.header("Prediction Explanation & Visuals")

    # Show feature importance if available
    try:
        importances = model.feature_importances_
        fi = pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6,4))
        fi.plot(kind='barh', ax=ax)
        ax.set_title("Feature Importance (Random Forest)")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    except Exception:
        st.write("Feature importance not available for this model.")

    # Optionally: show small historical sample (if CSV present)
    try:
        sample_df = pd.read_csv('powerconsumption.csv', parse_dates=['Datetime'])
        # create Total_Consumption column if not present
        if 'Total_Consumption' not in sample_df.columns:
            sample_df['Total_Consumption'] = (sample_df['PowerConsumption_Zone1'] +
                                             sample_df['PowerConsumption_Zone2'] +
                                             sample_df['PowerConsumption_Zone3'])
        # plot recent day
        recent = sample_df.set_index('Datetime').resample('H').mean().tail(48)
        st.subheader("Recent consumption (hourly) — dataset preview")
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(recent.index, recent['Total_Consumption'])
        ax2.set_xlabel("Datetime")
        ax2.set_ylabel("Total Consumption")
        st.pyplot(fig2)
    except FileNotFoundError:
        st.info("Dataset sample not found (powerconsumption.csv). Place it in app folder to enable preview.")
    except Exception as e:
        st.warning(f"Could not load sample dataset: {e}")

# ---- Footer / tips ----
st.markdown("---")
st.write("Tips:")
st.write("- Make sure the `best_model_random_forest.pkl` and `scaler.pkl` files are in the same directory as this app.")
st.write("- Use realistic weather/time values for meaningful predictions.")
st.write("- To deploy: push this folder to GitHub and connect to Streamlit Cloud (instructions included below).")
