# ...existing code...
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional, Tuple

print("APP.py starting (startup log)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JeevanUrja")

# Optional imports (graceful fallback)
_joblib_available = True
_matplotlib_available = True
try:
    import joblib  # type: ignore
except Exception:
    _joblib_available = False

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    _matplotlib_available = False

# Page config
st.set_page_config(page_title="JeevanUrja — Energy Forecast", layout="wide")

if not _joblib_available or not _matplotlib_available:
    missing = []
    if not _joblib_available:
        missing.append("joblib")
    if not _matplotlib_available:
        missing.append("matplotlib")
    st.sidebar.warning(
        "Optional libraries missing: "
        + ", ".join(missing)
        + ". App will run in demo/fallback mode. Add them to requirements.txt to enable full features."
    )

# Feature order expected by model
FEATURE_ORDER = [
    "Temperature", "Humidity", "WindSpeed",
    "GeneralDiffuseFlows", "DiffuseFlows",
    "Hour", "DayOfWeek", "IsWeekend", "Month"
]

# Lazy cached loader (only runs when called)
@st.cache_resource(ttl=3600)
def load_artifacts_cached(model_path: str = "energy_consumption_model.pkl",
                          scaler_path: str = "scaler.pkl") -> Tuple[Optional[object], Optional[object]]:
    if not _joblib_available:
        raise RuntimeError("joblib not available")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_artifacts_safe() -> Tuple[Optional[object], Optional[object], Optional[str]]:
    try:
        model, scaler = load_artifacts_cached()
        return model, scaler, None
    except FileNotFoundError as e:
        return None, None, f"Artifact file not found: {e}"
    except Exception as e:
        return None, None, str(e)

# Demo fallback predictor so UI always works
def demo_predict(input_df: pd.DataFrame) -> np.ndarray:
    w = {
        "Temperature": 0.5, "Humidity": 0.2, "WindSpeed": -0.3,
        "GeneralDiffuseFlows": 0.04, "DiffuseFlows": 0.03,
        "Hour": 0.6, "DayOfWeek": 1.0, "IsWeekend": -5.0, "Month": 0.2
    }
    base = 80.0
    # compute weighted sum vectorized
    pred = base + sum(input_df[col].astype(float) * w[col] for col in FEATURE_ORDER)
    return np.maximum(pred.values.reshape(-1), 0.0)

# UI
st.title("🔋 JeevanUrja — Intelligent Power Consumption Forecasting")
st.markdown("Demo-friendly UI. If model or files are missing, a demo predictor and charts are shown.")

col1, col2 = st.columns([1, 1])

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
        model, scaler, err = load_artifacts_safe()
        if model is not None and scaler is not None:
            try:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                st.success(f"⚡ Predicted Total Consumption: **{prediction:,.2f} units**")
            except Exception as e:
                st.warning(f"Model present but prediction failed: {e}. Using demo predictor.")
                pred = demo_predict(input_df)[0]
                st.success(f"⚡ Demo Predicted Total Consumption: **{pred:,.2f} units**")
        else:
            st.info("⚠️ Model or scaler not available — using demo predictor.")
            if err:
                st.caption(f"Load info: {err}")
            pred = demo_predict(input_df)[0]
            st.success(f"⚡ Demo Predicted Total Consumption: **{pred:,.2f} units**")

with col2:
    st.header("📊 Sample Chart & Data")

    # Always show a sample consumption chart so the app renders something
    def sample_df():
        times = pd.date_range(start="2025-01-01", periods=24, freq="H")
        base = 100 + 20 * np.sin(np.linspace(0, 2 * np.pi, 24))
        values = base + np.random.randn(24) * 5
        return pd.DataFrame({"Datetime": times, "Consumption": np.round(values, 2)}).set_index("Datetime")

    st.subheader("Sample Consumption (demo)")
    st.line_chart(sample_df())

    st.subheader("Upload / Preview CSV")
    uploaded = st.file_uploader("Upload powerconsumption.csv (optional)", type=["csv"])
    df_source = None
    if uploaded is not None:
        try:
            df_source = pd.read_csv(uploaded, parse_dates=["Datetime"])
        except Exception as e:
            st.warning(f"Uploaded file read error: {e}")

    if df_source is None:
        try:
            df_source = pd.read_csv("powerconsumption.csv", parse_dates=["Datetime"])
        except FileNotFoundError:
            df_source = None
        except Exception as e:
            st.warning(f"Error reading powerconsumption.csv: {e}")

    if df_source is not None:
        try:
            if "Total_Consumption" not in df_source.columns:
                zone_cols = [c for c in df_source.columns if "PowerConsumption" in c]
                if len(zone_cols) >= 1:
                    df_source["Total_Consumption"] = df_source[zone_cols].sum(axis=1)
                else:
                    numeric_cols = df_source.select_dtypes("number").columns
                    if len(numeric_cols) > 0:
                        df_source["Total_Consumption"] = df_source[numeric_cols[0]]
                    else:
                        raise ValueError("No numeric consumption column found")
            recent = df_source.set_index("Datetime").resample("H").mean().tail(48)
            st.subheader("Recent Hourly Consumption Trend")
            if _matplotlib_available:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(recent.index, recent["Total_Consumption"], color="orange")
                ax.set_xlabel("Datetime")
                ax.set_ylabel("Total Consumption")
                st.pyplot(fig)
            else:
                st.line_chart(recent["Total_Consumption"])
        except Exception as e:
            st.warning(f"Error processing CSV preview: {e}")
    else:
        st.info("No CSV available. Upload one to preview trends.")

st.markdown("---")
st.caption("Developed by Ritesh — JeevanUrja: Intelligent Power Consumption Forecasting")
# ...existing code...
