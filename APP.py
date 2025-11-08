# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ---- Safe imports ----
try:
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    st.error("❌ Required libraries missing. Please make sure your `requirements.txt` "
             "contains joblib, matplotlib, and seaborn.")
    st.stop()

# ---- Page Config ----
st.set_page_config(page_title="JeevanUrja — Energy Forecast", layout="wide")

# ---- Load Model and Scaler ----
@st.cache_resource(ttl=3600)
def load_artifacts(model_path="energy_consumption_model.pkl", scaler_path="scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Failed to load model or scaler: {e}")
    st.stop()

# ---- Feature order ----
FEATURE_ORDER = [
    "Temperature", "Humidity", "WindSpeed",
    "GeneralDiffuseFlows", "DiffuseFlows",
    "Hour", "DayOfWeek", "IsWeekend", "Month"
]

# ---- Title ----
st.title("🔋 JeevanUrja — Intelligent Power Consumption Forecasting")
st.markdown("Predict **Total Power Consumption** based on weather and time inputs.")

col1, col2 = st.columns([1, 1])

# ---- Input Section ----
with col1:
    st.header("Input Parameters")

    temp = st.number_input("🌡️ Temperature (°C)", value=25.0, step=0.1)
    humidity = st.number_input("💧 Humidity (%)", value=60.0, step=0.1)
    windspeed = st.number_input("🌬️ Wind Speed (m/s)", value=3.0, step=0.1)
    gdf = st.number_input("☀️ General Diffuse Flows (W/m²)", value=120.0, step=1.0)
    dfv = st.number_input("🌤️ Diffuse Flows (W/m²)", value=80.0, step=1.0)

    st.subheader("⏰ Date and Time")
    dt = st.date_input("Select Date", value=datetime.today().date())
    t = st.time_input("Select Time", value=datetime.now().time())
    dt_full = datetime.combine(dt, t)

    hour = dt_full.hour
    dayofweek = dt_full.weekday()
    month = dt_full.month
    isweekend = 1 if dayofweek >= 5 else 0

    st.caption(f"Derived → Hour: {hour}, DayOfWeek: {dayofweek}, Month: {month}, Weekend: {isweekend}")

    if st.button("🔮 Predict Consumption"):
        input_df = pd.DataFrame([[temp, humidity, windspeed, gdf, dfv, hour, dayofweek, isweekend, month]],
                                columns=FEATURE_ORDER)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"⚡ Predicted Total Consumption: **{prediction:,.2f} units**")

# ---- Right Section ----
with col2:
    st.header("📊 Insights & Visuals")

    # Feature importance
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fi.plot(kind="barh", color="skyblue", ax=ax)
            ax.set_title("Feature Importance (Random Forest)")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")

    # Show recent trends
    try:
        df = pd.read_csv("powerconsumption.csv", parse_dates=["Datetime"])
        if "Total_Consumption" not in df.columns:
            df["Total_Consumption"] = (
                df["PowerConsumption_Zone1"] +
                df["PowerConsumption_Zone2"] +
                df["PowerConsumption_Zone3"]
            )
        recent = df.set_index("Datetime").resample("H").mean().tail(48)
        st.subheader("Recent Hourly Consumption Trend")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(recent.index, recent["Total_Consumption"], color="orange")
        ax2.set_xlabel("Datetime")
        ax2.set_ylabel("Total Consumption")
        st.pyplot(fig2)
    except FileNotFoundError:
        st.info("📁 'powerconsumption.csv' not found.")
    except Exception as e:
        st.warning(f"Error loading dataset preview: {e}")

# ---- Footer ----
st.markdown("---")
st.caption("Developed by Ritesh — JeevanUrja: Intelligent Power Consumption Forecasting")
