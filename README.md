# JeevanUrja--Intelligent-Power-Consumption-Forecasting-for-Future-Energy-Systems-
The JeevanUrja project is an intelligent system designed to forecast total energy consumption based on weather and time-related factors. It leverages machine learning to help households, industries, or power utilities predict electricity demand more accurately, enabling better planning and efficient energy usage.


⚡ Jeevan_Urja — Intelligent Power Consumption Forecasting for Future Energy Systems
Computational Intelligence Laboratory Project (2024–25)

MIT-ADT University, School of SEE Engineering
Division C | Batch C2

👥 Team Members
Name	PRN	Role
Ritesh Gajanan Asole	202301060014	Data preprocessing, modeling, deployment,	EDA, model evaluation, documentation

🌟 Project Overview

Jeevan_Urja is a Machine Learning–based system designed to predict short-term power consumption using environmental and zonal energy data. The project uses classical ML regressors and deploys an interactive dashboard for real-time prediction.

Power forecasting plays an essential role in smart grids, load balancing, demand response, and renewable integration. This project demonstrates how predictive analytics can improve energy efficiency and operational planning.

🎯 Objectives

Forecast short-term Total Power Consumption (kW) using environmental & time-based features.

Compare multiple ML models: Linear Regression, KNN, Random Forest.

Analyze feature correlations, consumption patterns, and behavioral trends.

Build a clean preprocessing pipeline for time-series energy data.

Deploy the best model using a Streamlit-based UI / Supabase API.

Provide explainability through feature importance and visual analysis.

🗂 Dataset Description

Source: Smart energy meter dataset (publicly available).
Duration: Jan 1 – Dec 30, 2017
Original Size: ~52,416 rows (10-minute interval readings)
Used Subset: 2200 rows (~15 days) for faster training & experimentation.

Features Used
Feature	Description
Temperature (°C)	Ambient temperature influencing cooling load
Humidity (%)	Affects AC usage & energy demand
WindSpeed (m/s)	Impacts thermal comfort
GeneralDiffuseFlows	Solar radiation inputs
DiffuseFlows	Weather-related radiation
Zone1, Zone2, Zone3 Power	Consumption from 3 monitored zones
Total_Consumption	Target variable = sum of all zones
Hour, DayOfWeek, Month	Extracted time features
IsWeekend	Binary indicator
🔧 Methodology
✔ 1. Data Preprocessing

Handled missing values (forward fill).

Removed duplicates.

Parsed datetime column correctly.

Created engineered features (Hour, DayOfWeek, Month, etc.).

Scaled inputs using StandardScaler.

Train-test split: 80/20 (time-ordered, no shuffle).

✔ 2. Exploratory Data Analysis

Plotted and analyzed:

Time-series trend (daily/weekly cycles)

Hourly average consumption

Weekly patterns (weekday vs weekend)

Correlation heatmap

Weather impact on consumption

Key Insight:
⚡ Temperature + Humidity + Hour are major predictors of energy usage.

✔ 3. Models Implemented
Model	MAE	RMSE	R² Score
Linear Regression	High error	High error	~0.69
KNN Regressor	Moderate	Better	~0.88
Random Forest Regressor	Lowest	Best	~0.97

Winner → Random Forest

Random Forest shows strong generalization and captures non-linear patterns effectively.

✔ 4. Model Evaluation

Visuals generated:

Actual vs Predicted curves

Scatter plots (prediction fit)

Feature importance rankings

Model comparison bar charts

Confusion matrix (consumption levels)

✔ 5. Deployment

Two deployment options were explored:

🔹 Streamlit Front-End

User inputs: Temperature, Humidity, WindSpeed, Weather conditions, etc.

Model predicts Total Power Consumption in real-time.

Clean UI with charts and explanation.

🔹 Supabase / API-Based Model Hosting

Model + scaler stored in cloud

Front-end communicates via API

Faster & more reliable than Streamlit Cloud (dependency issues fixed)

📈 Key Results

Random Forest achieved 97% accuracy (R² ≈ 0.97) on test data.

Strong relationship between weather conditions and power usage.

Prediction curve closely follows actual consumption pattern.

Supports data-driven decision-making for smart energy systems.

🔍 Conclusion

The project successfully demonstrates machine learning–based forecasting of short-term power consumption.

Random Forest emerged as the best model due to its ability to capture non-linear dependencies.

Weather variables + time features are essential for energy prediction.

The deployed model enables real-time forecasting, useful for smart-grid applications.

🚧 Limitations

Subset of 2200 rows used → limited seasonal representation.

Data not specific to a local region.

No hyperparameter tuning for advanced optimization.

🚀 Future Scope

Use full-year dataset for stronger seasonality modeling.

Implement LSTM / GRU for deep learning–based time-series forecasting.

Integrate IoT sensors for live data ingestion.

Build automated pipelines for continuous deployment (MLOps).

📁 Project Structure (Suggested for GitHub)
Jeevan_Urja/
│── data/
│   └── powerconsumption_2200.csv
│── notebooks/
│   └── modeling_notebook.ipynb
│── models/
│   ├── rf_model.pkl
│   └── scaler.pkl
│── app/
│   └── APP.py
│── utils/
│   └── preprocessing.py
│── README.md
│── requirements.txt
│── poster/
│   └── Jeevan_Urja_Poster.pdf
│── presentation/
│   └── Jeevan_Urja_PPT.pptx

🛠 Technologies Used

Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn

Streamlit

Supabase

Joblib (Model saving)

GitHub / Git
