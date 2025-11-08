# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ---- Safe imports for optional plotting ----
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    sns = None
    st.warning("⚠️ Optional plotting libraries (matplotlib/seaborn) not found. Visuals will be disabled.")

# ---- Safe import for joblib ----
try:
    import joblib
except ModuleNotFoundError:
    st.error("❌ Missing dependency: joblib. Please run 'pip install joblib'.")
    st.stop()

# ---- Page Config ----
st.set_page_config(page_title="JeevanUrja — Energy Forecast", layout="wide")

# ---- Load model and scaler ----
@st.cache_resource(ttl=3600)
def load_artifacts(model_path='energy_consumption_model.pkl', scaler_path='scaler.pkl'):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model or scaler file missing: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        st.stop()

model, scaler = load_artifacts()

# ---- Feature order ----
FEATURE_ORDER = [
    'Temperature', 'Humidity', 'WindSpeed',
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'Hour', 'DayOfWeek', 'IsWeekend', 'Month'
]

# ---- UI Header ----
st.title("🔋 JeevanUrja — Intelligent Power Consumption Forecasting")
st.markdown("""
Provide the weather and time details to predict **Total Power Consumption**.
""")

col1, col2 = st.columns([1, 1])

# ---- Left: Inputs ----
with col1:
    st.header("Input Parameters")

    # Weather inputs
    temp = st.number_input("🌡️ Temperature (°C)", value=25.0, step=0.1)
    humidity = st.number_input("💧 Humidity (%)", value=60.0, step=0.1)
    windspeed = st.number_input("🌬️ Wind Speed (m/s)", value=3.0, step=0.1)
    gdf = st.number_input("☀️ General Diffuse Flows (W/m²)", value=120.0, step=1.0)
    dfv = st.number_input("🌤️ Diffuse Flows (W/m²)", value=80.0, step=1.0)

    # Time-based inputs
    st.subheader("⏰ Date and Time")
    dt = st.date_input("Select Date", value=datetime.today().date())
    t = st.time_input("Select Time", value=datetime.now().time())
    dt_full = datetime.combine(dt, t)

    hour = dt_full.hour
    dayofweek = dt_full.weekday()
    month = dt_full.month
    isweekend = 1 if dayofweek >= 5 else 0

    st.caption(f"Derived features → Hour: {hour}, DayOfWeek: {dayofweek}, Month: {month}, Weekend: {isweekend}")

    if st.button("🔮 Predict Consumption"):
        try:
            input_df = pd.DataFrame([[temp, humidity, windspeed, gdf, dfv, hour, dayofweek, isweekend, month]],
                                    columns=FEATURE_ORDER)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            st.success(f"⚡ Predicted Total Consumption: **{prediction:,.2f} units**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---- Right: Visuals ----
with col2:
    st.header("📊 Insights & Visuals")

    if plt is not None:
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fi = pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                fi.plot(kind='barh', color='lightgreen', ax=ax)
                ax.set_title("Feature Importance (Random Forest)")
                ax.set_xlabel("Importance Score")
                st.pyplot(fig)
            else:
                st.write("Feature importance not available for this model.")
        except Exception as e:
            st.warning(f"Could not display feature importance: {e}")

        # Try showing recent data trends
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
            st.info("📁 Dataset file 'powerconsumption.csv' not found.")
        except Exception as e:
            st.warning(f"Could not load data preview: {e}")
    else:
        st.warning("⚠️ Matplotlib not available. Charts disabled.")

# ---- Footer ----
st.markdown("---")
st.caption("Developed by Ritesh — JeevanUrja: Intelligent Power Consumption Forecasting")
