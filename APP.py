# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Try importing joblib safely
try:
    import joblib
except ModuleNotFoundError:
    st.error("⚠️ The 'joblib' library is missing. Please install it using 'pip install joblib'.")
    st.stop()

# ---- Configuration ----
st.set_page_config(page_title="JeevanUrja — Energy Forecast", layout="wide")

# ---- Helper: Load model and scaler ----
@st.cache_resource(ttl=3600)
def load_artifacts(model_path='energy_consumption_model.pkl', scaler_path='scaler.pkl'):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model or Scaler file missing: {e}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading artifacts: {e}")
        st.stop()

model, scaler = load_artifacts()

# ---- Define the features (must match training order) ----
FEATURE_ORDER = [
    'Temperature', 'Humidity', 'WindSpeed',
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'Hour', 'DayOfWeek', 'IsWeekend', 'Month'
]

# ---- UI: Title ----
st.title("🔋 JeevanUrja — Intelligent Power Consumption Forecast")
st.markdown("""
Enter weather and time details below to predict **Total Power Consumption**.
""")

# ---- Input Layout ----
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Features")

    # Weather-related inputs
    temp = st.number_input("🌡️ Temperature (°C)", value=25.0, step=0.1, format="%.1f")
    humidity = st.number_input("💧 Humidity (%)", value=60.0, step=0.1, format="%.1f")
    windspeed = st.number_input("🌬️ Wind Speed (m/s)", value=3.0, step=0.1, format="%.1f")
    gdf = st.number_input("☀️ General Diffuse Flows (W/m²)", value=120.0, step=1.0)
    dfv = st.number_input("🌤️ Diffuse Flows (W/m²)", value=80.0, step=1.0)

    # Time inputs
    st.subheader("⏰ Time Inputs")
    dt = st.date_input("Select Date", value=datetime.today().date())
    t = st.time_input("Select Time", value=datetime.now().time())

    # Derive datetime-based features
    dt_full = datetime.combine(dt, t)
    hour = dt_full.hour
    dayofweek = dt_full.weekday()
    month = dt_full.month
    isweekend = 1 if dayofweek >= 5 else 0

    st.caption(f"Derived Features → Hour: {hour}, DayOfWeek: {dayofweek}, Month: {month}, Weekend: {isweekend}")

    # ---- Prediction Button ----
    if st.button("🔮 Predict Consumption"):
        # Prepare input data
        input_df = pd.DataFrame([[temp, humidity, windspeed, gdf, dfv, hour, dayofweek, isweekend, month]],
                                columns=FEATURE_ORDER)

        # Apply scaling
        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()

        # Make prediction
        try:
            prediction = model.predict(input_scaled)[0]
            st.success(f"⚡ Predicted Total Consumption: **{prediction:,.2f} units**")
            st.info("Note: Units are the same as in training data.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ---- Right panel (Visuals & Info) ----
with col2:
    st.header("📊 Insights & Visuals")

    # Feature Importance (only for tree models)
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fi.plot(kind='barh', color='skyblue', ax=ax)
            ax.set_title("Feature Importance (Random Forest)")
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        else:
            st.write("Feature importance not available for this model.")
    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")

    # Show recent dataset preview
    try:
        sample_df = pd.read_csv("powerconsumption.csv", parse_dates=["Datetime"])
        if "Total_Consumption" not in sample_df.columns:
            sample_df["Total_Consumption"] = (
                sample_df["PowerConsumption_Zone1"] +
                sample_df["PowerConsumption_Zone2"] +
                sample_df["PowerConsumption_Zone3"]
            )
        recent = sample_df.set_index("Datetime").resample("H").mean().tail(48)
        st.subheader("Recent Hourly Consumption — Dataset Preview")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(recent.index, recent["Total_Consumption"], color="orange")
        ax2.set_xlabel("Datetime")
        ax2.set_ylabel("Total Consumption")
        st.pyplot(fig2)
    except FileNotFoundError:
        st.info("📁 Dataset file 'powerconsumption.csv' not found. Add it for preview charts.")
    except Exception as e:
        st.warning(f"Could not load dataset preview: {e}")

# ---- Footer ----
st.markdown("---")
st.caption("🚀 Developed by Ritesh — JeevanUrja: Intelligent Power Consumption Forecasting")
