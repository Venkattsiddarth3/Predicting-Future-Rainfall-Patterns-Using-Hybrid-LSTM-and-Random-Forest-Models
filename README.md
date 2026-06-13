# 🌧️ Predicting Future Rainfall Patterns Using Hybrid LSTM and Random Forest Models

## 📌 Overview

Predicting Future Rainfall Patterns Using Hybrid LSTM and Random Forest Models is an AI-powered weather forecasting system designed to provide accurate rainfall predictions using historical meteorological data. The project combines the strengths of Long Short-Term Memory (LSTM) neural networks and Random Forest classification models to improve forecasting accuracy and reliability. The system analyzes temperature, humidity, wind speed, atmospheric pressure, and historical rainfall records to generate future rainfall forecasts for major Indian cities. The solution is deployed through an interactive Streamlit dashboard with real-time visualization and downloadable forecast reports. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## 🎯 Objectives

- Predict future rainfall occurrence and rainfall intensity.
- Improve forecasting accuracy using a hybrid machine learning approach.
- Analyze historical weather patterns and seasonal variations.
- Provide interactive weather visualization and forecast reports.
- Support agriculture, disaster management, and water resource planning.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning & Deep Learning
- Random Forest Classifier
- Long Short-Term Memory (LSTM)

### Libraries & Frameworks
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Plotly
- Matplotlib
- Pickle
- JSON

---

## 📂 Dataset

The project uses historical Indian weather data from 2000–2024 containing:

- Date
- Maximum Temperature
- Minimum Temperature
- Humidity
- Wind Speed
- Atmospheric Pressure
- Rainfall Records

The dataset undergoes preprocessing, missing value handling, normalization, and feature engineering before training. :contentReference[oaicite:2]{index=2}

---

## ⚙️ Feature Engineering

The system generates advanced weather features including:

- Temperature Range
- Low Temperature Indicator
- Low Wind Indicator
- Fog Risk Index
- Seasonal Encoding
- Cyclical Date Features
- Lag Features (1-Day, 3-Day, 7-Day)

These engineered features help the model learn weather trends and seasonal rainfall patterns effectively. :contentReference[oaicite:3]{index=3}

---

## 🤖 Hybrid Model Architecture

### Random Forest Model
Used for:
- Rainfall Occurrence Prediction
- Rain Probability Estimation
- Classification of Rain/No Rain Events

### LSTM Model
Used for:
- Rainfall Amount Prediction
- Time-Series Forecasting
- Learning Long-Term Weather Dependencies

### Hybrid Forecasting Pipeline

1. Historical weather data preprocessing.
2. Feature engineering and scaling.
3. Random Forest predicts rainfall probability.
4. LSTM predicts rainfall intensity.
5. Combined output generates final rainfall forecast.

:contentReference[oaicite:4]{index=4}

---

## 🌍 Dashboard Features

The Streamlit dashboard provides:

### Forecast Controls
- City Selection
- Forecast Duration Selection
- Theme Customization
- Date Selection

### Weather Analytics
- Temperature Forecast
- Rainfall Forecast
- Rain Probability
- Humidity Prediction
- Wind Speed Prediction

### Visualization
- Interactive Rainfall Charts
- Daily Forecast Graphs
- Hourly Weather Breakdown
- Seasonal Trend Analysis

### Export Options
- CSV Download
- PNG Forecast Reports

:contentReference[oaicite:5]{index=5}
:contentReference[oaicite:6]{index=6}

---

## 📊 Model Evaluation Metrics

The models are evaluated using:

- Accuracy Score
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Rainfall Probability Analysis

The training pipeline automatically stores evaluation metrics for future analysis and comparison. :contentReference[oaicite:7]{index=7}

---

## 🚀 Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Random Forest Training
5. LSTM Training
6. Model Evaluation
7. Model Saving
8. Streamlit Dashboard Deployment
9. Forecast Generation
10. Visualization & Report Export

---

## 📁 Generated Files

After training, the following files are created:

- `lstm_model.h5`
- `rf_model.pkl`
- `scaler_lstm.pkl`
- `scaler_rf.pkl`
- `metrics.json`
- `sample_predictions.csv`

:contentReference[oaicite:8]{index=8}

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow streamlit plotly matplotlib
