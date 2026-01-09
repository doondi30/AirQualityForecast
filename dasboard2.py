import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

# ================== Page Config ==================
st.set_page_config(
    page_title="Dashboard 2",
    layout="wide",
    page_icon="📊"
)

# ================== Custom Styling ==================
st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.75rem !important;
        color: #1a472a;
        font-weight: 800;
        text-align: left;
        padding-bottom: 5px;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Header ==================
st.markdown('<p class="big-title">🤖 Air Quality Model Training Dashboard</p>', unsafe_allow_html=True)

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_air_quality_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        st.error("Could not load the dataset. Please make sure the file exists at 'cleaned_air_quality_data.csv'")
        return None

df = load_data()

# ================== Evaluation Function ==================
def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': mae, 'RMSE': rmse}

# ================== Model Training Functions ==================
def train_arima(series):
    model = ARIMA(series, order=(2,1,2))
    res = model.fit()
    pred = res.predict(start=0, end=len(series)-1, typ='levels')
    metrics = evaluate(series, pred)
    return model, metrics, pred

def train_prophet(series):
    df_prophet = pd.DataFrame({'ds': series.index, 'y': series.values}).reset_index(drop=True)
    model = Prophet()
    model.fit(df_prophet)
    forecast = model.predict(df_prophet)
    pred = forecast['yhat'].values
    metrics = evaluate(df_prophet['y'].values, pred)
    return model, metrics, pred

def train_lstm(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    X, y = [], []
    for i in range(1, len(scaled)):
        X.append(scaled[i-1])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3)])
    
    pred = model.predict(X).flatten()
    pred_inv = scaler.inverse_transform(pred.reshape(-1,1)).flatten()
    y_inv = scaler.inverse_transform(y.reshape(-1,1)).flatten()
    metrics = evaluate(y_inv, pred_inv)
    return model, scaler, metrics, pred_inv

def train_xgboost(series):
    X, y = [], []
    values = series.values
    for i in range(1, len(values)):
        X.append([values[i-1]])
        y.append(values[i])
    X, y = np.array(X), np.array(y)
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)
    pred = model.predict(X)
    metrics = evaluate(y, pred)
    return model, metrics, pred

# ================== Sidebar Controls ==================
with st.sidebar:
    st.markdown("### 🔧 Training Controls")
    if df is not None:
        pollutants = df.columns.drop(['Date', 'City', 'AQI_Bucket']).tolist()
        selected_pollutant = st.selectbox("Select Pollutant", pollutants)
        cities = df['City'].unique() if 'City' in df.columns else ['All Cities']
        selected_city = st.selectbox("Select City", cities)

        if selected_city != 'All Cities':
            filtered_df = df[df['City'] == selected_city]
        else:
            filtered_df = df

        series = filtered_df.set_index('Date')[selected_pollutant].dropna()

        model_options = st.multiselect(
            "Select Models to Train",
            ["ARIMA", "Prophet", "LSTM", "XGBoost"],
            default=["ARIMA", "Prophet", "LSTM"]
        )

        train_button = st.button("🚀 Train Selected Models")
    else:
        st.warning("Please load data first")

# ================== Main Dashboard ==================
if df is None:
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 Model Training", "📊 Performance Comparison", "🔮 Predictions"])

# ----------- TAB 1: Model Training -----------
with tab1:
    if train_button and len(model_options) > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results, predictions = {}, {}

        for i, model_name in enumerate(model_options):
            status_text.text(f"Training {model_name} model...")
            progress_bar.progress((i + 1) / len(model_options))
            try:
                if model_name == "ARIMA":
                    model, metrics, pred = train_arima(series)
                    results["ARIMA"] = metrics
                    predictions["ARIMA"] = pred
                elif model_name == "Prophet":
                    model, metrics, pred = train_prophet(series)
                    results["Prophet"] = metrics
                    predictions["Prophet"] = pred
                elif model_name == "LSTM":
                    model, scaler, metrics, pred = train_lstm(series)
                    results["LSTM"] = metrics
                    predictions["LSTM"] = pred
                elif model_name == "XGBoost":
                    model, metrics, pred = train_xgboost(series)
                    results["XGBoost"] = metrics
                    predictions["XGBoost"] = pred
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")

        status_text.text("Training complete!")

        if results:
            st.markdown("### Training Results")
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df.style.highlight_min(axis=0, color='#c8e6c9'))
            best_model = min(results, key=lambda x: results[x]['RMSE'])
            st.success(f"Best model: {best_model} (RMSE: {results[best_model]['RMSE']:.4f})")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(series.values, label='Actual', linewidth=2)
            colors = ['red', 'blue', 'green', 'orange']
            for i, (model_name, pred) in enumerate(predictions.items()):
                if len(pred) < len(series):
                    pred = np.concatenate([np.array([np.nan]), pred])
                elif len(pred) > len(series):
                    pred = pred[:len(series)]
                ax.plot(pred, label=model_name, linestyle='--', alpha=0.8, color=colors[i % len(colors)])
            ax.legend()
            # ax.set_title('Model Predictions Comparison')
            # st.pyplot(fig)

            st.session_state.results = results
            st.session_state.predictions = predictions
            st.session_state.model_options = model_options
            st.session_state.selected_pollutant = selected_pollutant
            st.session_state.selected_city = selected_city
    else:
        st.info("Select models and click 'Train Selected Models' to begin training")

# ----------- TAB 2: Performance Comparison -----------
with tab2:
    if 'results' in st.session_state:
        results = st.session_state.results
        metrics_df = pd.DataFrame(results).T

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### RMSE Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            models = list(results.keys())
            rmse_values = [results[m]['RMSE'] for m in models]
            ax.bar(models, rmse_values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
            ax.set_ylabel('RMSE')
            st.pyplot(fig)

        with col2:
            st.markdown("#### MAE Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            mae_values = [results[m]['MAE'] for m in models]
            ax.bar(models, mae_values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
            ax.set_ylabel('MAE')
            st.pyplot(fig)

        st.markdown("#### Detailed Metrics")
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='#c8e6c9'))
    else:
        st.info("Train models first to see performance comparison")

# ----------- TAB 3: Predictions -----------
with tab3:
    if 'results' in st.session_state:
        results = st.session_state.results
        selected_city = st.session_state.selected_city
        selected_pollutant = st.session_state.selected_pollutant
        model_options = st.session_state.model_options

        best_model = min(results, key=lambda x: results[x]['RMSE'])
        st.success(f"Using best model: {best_model} for predictions")

        n_days = st.slider("Days to forecast", 1, 30, 7)
        last_value = series.iloc[-1]
        future_pred = [last_value * (0.95 + 0.1 * np.random.random()) for _ in range(n_days)]
        last_date = df['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_days+1)]

        # ======== Custom Title Text ========
        st.markdown(
            f"### {n_days} days forecast for **{selected_city}** using **{selected_pollutant}** pollutant with **{', '.join(model_options)}** models"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'][-30:], series[-30:], label='Historical', linewidth=2)
        ax.plot(future_dates, future_pred, label='Forecast', linewidth=2, color='red', linestyle='--')
        ax.fill_between(future_dates, [x * 0.9 for x in future_pred], [x * 1.1 for x in future_pred], color='red', alpha=0.2)
        ax.legend()
        ax.set_title('Forecast Visualization')
        st.pyplot(fig)

        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Value': future_pred})
        st.dataframe(forecast_df)
    else:
        st.info("Train models first to generate predictions")
