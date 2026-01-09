import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import xgboost as xgb
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION + STYLE
# ---------------------------------------------------------
st.set_page_config(page_title="Dashboard 4", layout="wide")
st.markdown("""
<style>
.main {background-color: #f9f9fb;}
.banner {background: linear-gradient(90deg, #e9ecff, #dce2ff); padding: 25px 0; border-radius: 8px; text-align: center; margin-bottom: 25px;}
.banner h1 {color: #2e3b8e; font-weight: 800; font-size: 36px; margin-bottom: 5px;}
.section-title {color: #2e3b8e; font-weight: 700; font-size: 20px; margin-bottom: 12px;}
.alert-box {padding: 12px 18px; border-radius: 8px; margin-bottom: 10px; font-size: 15px; color: #2c3e50; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
.moderate {background-color:#fff9e6;}
.good {background-color:#eafbea;}
.unhealthy {background-color:#ffe6e6;}
.sidebar .sidebar-content {background-color: #eef0ff;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AUTO-CREATE DATA IF NOT EXIST
# ---------------------------------------------------------
if not os.path.exists("Fake_AQI_Data.csv"):
    cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bengaluru', 'Hyderabad']
    start_date = datetime(2024, 1, 1)
    days = 365
    data = []

    for city in cities:
        base_aqi = 80 + (cities.index(city) * 20)
        for i in range(days):
            date = start_date + timedelta(days=i)
            AQI = max(0, base_aqi + int(15 * np.sin(i / 30)))
            pollutants = {
                'PM2.5': AQI * 0.6,
                'PM10': AQI * 0.8,
                'NO': AQI * 0.2,
                'NO2': AQI * 0.4,
                'NOx': AQI * 0.5,
                'NH3': AQI * 0.1,
                'CO': AQI * 0.15,
                'SO2': AQI * 0.25,
                'O3': AQI * 0.45,
                'Benzene': AQI * 0.05,
                'Toluene': AQI * 0.06,
                'Xylene': AQI * 0.04
            }
            if AQI <= 50:
                bucket = "Good"
            elif AQI <= 100:
                bucket = "Satisfactory"
            elif AQI <= 200:
                bucket = "Moderate"
            elif AQI <= 300:
                bucket = "Poor"
            elif AQI <= 400:
                bucket = "Very Poor"
            else:
                bucket = "Severe"

            data.append({
                'City': city,
                'Date': date.strftime('%Y-%m-%d'),
                **pollutants,
                'AQI': AQI,
                'AQI_Bucket': bucket
            })

    pd.DataFrame(data).to_csv("Fake_AQI_Data.csv", index=False)
    print("✅ Fake_AQI_Data.csv generated successfully")

# ---------------------------------------------------------
# LOAD DATA FUNCTION
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Fake_AQI_Data.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    pollutant_cols = [c for c in df.columns if c not in ['City', 'Date', 'AQI_Bucket']]
    for col in pollutant_cols:
        if col != 'AQI' and df[col].mean() < 10:
            df[col] = df[col] * 1000
    return df

df = load_data()

# ---------------------------------------------------------
# AQI BUCKETS
# ---------------------------------------------------------
AQI_BUCKETS = pd.DataFrame({
    "AQI_Bucket": ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"],
    "min": [0, 51, 101, 201, 301, 401],
    "max": [50, 100, 200, 300, 400, float('inf')],
    "color": ["#2ecc71","#27ae60","#f1c40f","#e67e22","#d35400","#e74c3c"]
})

def get_aqi_category(val):
    val = max(val, 0)
    for _, row in AQI_BUCKETS.iterrows():
        if row['min'] <= val <= row['max']:
            return row['AQI_Bucket'], row['color']
    return "Unknown", "#95a5a6"

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Dashboard Controls")
admin_mode = st.sidebar.toggle("🛠️ Admin Mode")

cities = sorted(df['City'].unique())
pollutant_cols = [c for c in df.columns if c not in ['City', 'Date', 'AQI_Bucket']]

if "selected_city" not in st.session_state:
    st.session_state.selected_city = cities[0]
if "selected_pollutant" not in st.session_state:
    st.session_state.selected_pollutant = "AQI"
if "dashboard_ready" not in st.session_state:
    st.session_state.dashboard_ready = True

# ---------------------------------------------------------
# ADMIN CSV UPLOAD & CONCAT
# ---------------------------------------------------------
if admin_mode:
    st.sidebar.subheader("📤 Upload New Data (Admin Only)")
    uploaded_file = st.sidebar.file_uploader("Upload new air quality CSV", type=["csv"])
    
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        if 'Date' in new_data.columns:
            new_data['Date'] = pd.to_datetime(new_data['Date'], errors='coerce')

        if not {'City', 'Date'}.issubset(new_data.columns):
            st.sidebar.error("Uploaded CSV must contain 'City' and 'Date' columns.")
        else:
            st.sidebar.info("🔄 Integrating uploaded data ...")
            df = pd.concat([df, new_data], ignore_index=True)
            df.drop_duplicates(subset=['City','Date'], keep='last', inplace=True)
            df.to_csv("Fake_AQI_Data.csv", index=False)
            st.sidebar.success("✅ Data successfully integrated and saved!")
            st.cache_data.clear()
            df = load_data()

selected_city = st.sidebar.selectbox("Select City", cities, index=cities.index(st.session_state.selected_city))
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_cols, index=pollutant_cols.index(st.session_state.selected_pollutant))

if admin_mode and st.sidebar.button("🔄 Update Dashboard"):
    st.session_state.selected_city = selected_city
    st.session_state.selected_pollutant = selected_pollutant
    st.session_state.dashboard_ready = True
    st.success("✅ Dashboard updated successfully")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="banner">
    <h1>Air Quality Dashboard</h1>
    <h3>Live Pollutant & AQI Forecasting (Scaled Correctly)</h3>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FORECAST FUNCTION
# ---------------------------------------------------------
def forecast_pollutant(df, pollutant, days=7):
    df = df[['Date', pollutant]].dropna().sort_values('Date').copy()
    df[pollutant] = df[pollutant].apply(lambda x: max(x, 0))
    df['day_num'] = np.arange(len(df))
    X, y = df[['day_num']], df[pollutant]

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4
    )
    model.fit(X, y)

    future_days = pd.DataFrame({'day_num': np.arange(df['day_num'].iloc[-1] + 1,
                                                     df['day_num'].iloc[-1] + days + 1)})
    forecast = model.predict(future_days)
    future_dates = [df['Date'].iloc[-1] + timedelta(days=i + 1) for i in range(days)]
    forecast_df = pd.DataFrame({'Date': future_dates, pollutant: forecast})
    return df, forecast_df

# ---------------------------------------------------------
# CSV DOWNLOAD
# ---------------------------------------------------------
def generate_csv_download(forecast_df, pollutant):
    forecast_df_copy = forecast_df.copy()
    forecast_df_copy['AQI_Category'] = forecast_df_copy[pollutant].apply(lambda x: get_aqi_category(x)[0] if pollutant == 'AQI' else "")
    return forecast_df_copy.to_csv(index=False).encode('utf-8')

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
if st.session_state.dashboard_ready:
    filtered_df = df[df["City"] == st.session_state.selected_city].sort_values("Date")
    pollutant = st.session_state.selected_pollutant
    recent_df, forecast_df = forecast_pollutant(filtered_df, pollutant, days=7)
    latest_value = recent_df[pollutant].iloc[-1] if not recent_df.empty else np.nan

    if pollutant == "AQI":
        status, color = get_aqi_category(latest_value)
    else:
        if latest_value <= 50: status, color = "Good", "#2ecc71"
        elif latest_value <= 100: status, color = "Satisfactory", "#27ae60"
        elif latest_value <= 200: status, color = "Moderate", "#f1c40f"
        elif latest_value <= 300: status, color = "Poor", "#e67e22"
        elif latest_value <= 400: status, color = "Very Poor", "#d35400"
        else: status, color = "Severe", "#e74c3c"

    stats = recent_df[pollutant].describe()

    # CURRENT VALUE + STATS
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"<div class='section-title'>Current {pollutant}</div>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(values=[latest_value, stats['max']*1.2 - latest_value], hole=0.7,
                                     marker_colors=[color, "#f2f2f2"], textinfo="none")])
        fig.update_layout(showlegend=False, annotations=[dict(text=f"<b>{latest_value:.0f}</b><br>{status}",
                                                              x=0.5, y=0.5, font_size=18, showarrow=False)],
                          height=240, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"<div class='section-title'>{pollutant} Statistics</div>", unsafe_allow_html=True)
        st.metric("Latest Value", f"{latest_value:.0f}")
        st.metric("Mean", f"{stats['mean']:.0f}")
        st.metric("Max", f"{stats['max']:.0f}")
        st.metric("Min", f"{stats['min']:.0f}")

    # TRENDS + ALERTS SIDE-BY-SIDE
    col3, col4 = st.columns([2, 1])
    with col3:
        st.markdown(f"<div class='section-title'>{pollutant} Trends (Historical + Forecast)</div>", unsafe_allow_html=True)
        hist = recent_df[['Date', pollutant]].copy(); hist['Type'] = 'Historical'
        fut = forecast_df.copy(); fut['Type'] = 'Forecast'
        trend = pd.concat([hist, fut])
        trend['Date_str'] = trend['Date'].dt.strftime("%d-%b")
        fig_trend = px.line(trend, x='Date_str', y=pollutant, color='Type',
                            line_dash_sequence=['solid', 'dash'],
                            labels={pollutant: f"{pollutant} Level"})
        fig_trend.update_layout(height=300, colorway=['#27ae60', '#c0392b'])
        st.plotly_chart(fig_trend, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Alert Notifications</div>", unsafe_allow_html=True)
        
        forecast_dates = [datetime.now() + timedelta(days=i) for i in range(3)]
        for date in forecast_dates:
            date_str = date.strftime("%A, %d %B %Y")
            if date.date() == datetime.now().date():
                val = latest_value
            else:
                forecast_row = forecast_df[forecast_df['Date'].dt.date == date.date()]
                val = forecast_row[pollutant].values[0] if not forecast_row.empty else np.nan
            
            if pollutant == "AQI":
                status, _ = get_aqi_category(val)
                if status in ["Good", "Satisfactory"]:
                    level = "good"; msg = f"✅ Air quality is {status}"
                elif status in ["Moderate", "Poor"]:
                    level = "moderate"; msg = f"⚠️ Air quality is {status}"
                else:
                    level = "unhealthy"; msg = f"🚨 Air quality is {status}"
            else:
                if val <= 50:
                    level = "good"; msg = f"✅ {pollutant} level is Good"
                elif val <= 100:
                    level = "moderate"; msg = f"⚠️ {pollutant} level is Moderate"
                else:
                    level = "unhealthy"; msg = f"🚨 {pollutant} level is High"
            
            st.markdown(f"<div class='alert-box {level}'><b>{msg}</b><br><span>{date_str}</span></div>", unsafe_allow_html=True)

    # CSV download
    csv_bytes = generate_csv_download(forecast_df, selected_pollutant)
    st.download_button("💾 Download 7-Day Forecast Alerts (.csv)", data=csv_bytes, file_name=f"{st.session_state.selected_city}_{selected_pollutant}_forecast.csv", mime="text/csv")

else:
    st.warning("Dashboard is in default view. Enable Admin Mode and click 'Update Dashboard' to apply changes.")
