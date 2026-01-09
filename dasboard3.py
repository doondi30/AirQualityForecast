import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Dashboard 3", layout="wide")

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
.main { background-color: #fefcf9; }
.banner { background: linear-gradient(90deg, #fff3e0, #ffe0b2); padding: 25px 0; border-radius: 8px; text-align: center; margin-bottom: 25px; }
.banner h1 { color: #d35400; font-weight: 800; font-size: 36px; margin-bottom: 5px; }
.banner h3 { color: #e67e22; font-weight: 500; font-size: 18px; margin-top: 0; }
.section-title { color: #c0392b; font-weight: 700; font-size: 20px; margin-bottom: 12px; }
.forecast-card { display: inline-block; padding: 15px 22px; margin: 5px; border-radius: 10px; background-color: white; text-align: center; width: 90px; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }
.forecast-day { font-weight: 700; font-size: 15px; margin-bottom: 3px; }
.forecast-value { font-weight: 600; font-size: 13px; }
.alert-box { padding: 12px 18px; border-radius: 8px; margin-bottom: 10px; font-size: 15px; color: #2c3e50; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.good {background-color:#eafbea;}
.moderate {background-color:#fff9e6;}
.unhealthy {background-color:#fdecea;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AQI BUCKETS
# ---------------------------------------------------------

AQI_BUCKETS = pd.DataFrame({
    "AQI_Bucket": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
    "min": [0, 51, 101, 201, 301, 401],
    "max": [50, 100, 200, 300, 400, np.inf]  # Use np.inf for anything above 10
})


def get_aqi_category(val):
    """Determine AQI bucket based on AQI_Bucket min/max"""
    val = max(val, 0)  # Prevent negative AQI
    bucket_row = AQI_BUCKETS[(AQI_BUCKETS['min'] <= val) & (AQI_BUCKETS['max'] >= val)]
    if not bucket_row.empty:
        return bucket_row.iloc[0]['AQI_Bucket'], round(val, 1)
    return "Unknown", round(val, 1)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_air_quality_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Dashboard Controls")
cities = sorted(df['City'].unique())
selected_city = st.sidebar.selectbox("Select City", cities)
pollutant_cols = [c for c in df.columns if c not in ['City', 'Date', 'AQI']]
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_cols)
st.sidebar.info("Use this dashboard to visualize AQI and pollutant trends across cities.")

# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="banner">
    <h1>Air Quality Dashboard</h1>
    <h3>City-Level AQI and Pollutant Insights</h3>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FILTER DATA
# ---------------------------------------------------------
city_data = df[df['City'] == selected_city].sort_values('Date')

# ---------------------------------------------------------
# RECALCULATE AQI BASED ON SELECTED POLLUTANT
# ---------------------------------------------------------
if not city_data.empty:
    city_data['Recalculated_AQI'] = city_data[selected_pollutant]  # raw values
else:
    city_data['Recalculated_AQI'] = np.nan

# ---------------------------------------------------------
# CURRENT AQI GAUGE
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1.1])
with col1:
    st.markdown("<div class='section-title'>Current Air Quality</div>", unsafe_allow_html=True)
    if not city_data.empty:
        latest_val = city_data.iloc[-1]['Recalculated_AQI']
        category, aqi_val = get_aqi_category(latest_val)
        color_map = {
            "Good":"#2ecc71","Satisfactory":"#2ecc71",
            "Moderate":"#f1c40f","Poor":"#e67e22",
            "Very Poor":"#e67e22","Severe":"#e74c3c"
        }
        color = color_map.get(category,"#95a5a6")
        fig = go.Figure(data=[go.Pie(values=[aqi_val, 500-aqi_val], hole=0.7, marker_colors=[color,"#f2f2f2"], textinfo="none")])
        fig.update_layout(showlegend=False, annotations=[dict(text=f"<b>{aqi_val}</b><br>{category}", x=0.5,y=0.5,font_size=22,showarrow=False)], height=280, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f"{selected_city} — based on {selected_pollutant}")
    else:
        st.warning("No AQI data available for this city.")

# ---------------------------------------------------------
# 7-DAY FORECAST
# ---------------------------------------------------------
with col2:
    st.markdown("<div class='section-title'>7-Day AQI Forecast</div>", unsafe_allow_html=True)
    if not city_data.empty:
        recent = city_data['Recalculated_AQI'].tail(7)
        avg_change = recent.diff().mean()
        forecast_values = [recent.iloc[-1] + avg_change*i for i in range(1,8)]
    else:
        forecast_values = [np.nan]*7
    days = pd.date_range(datetime.date.today(), periods=7)
    forecast_html = ""
    for i,d in enumerate(days):
        val = forecast_values[i] if not pd.isna(forecast_values[i]) else 0
        category, aqi_val = get_aqi_category(val)
        color_map = {"Good":"#eafbea","Satisfactory":"#eafbea","Moderate":"#fff9e6","Poor":"#fdecea","Very Poor":"#fdecea","Severe":"#f5b7b1"}
        bg = color_map.get(category,"#ecf0f1")
        forecast_html += f"""
            <div class='forecast-card' style='background-color:{bg};'>
                <div class='forecast-day'>{d.strftime('%a')}</div>
                <div class='forecast-value'>AQI {aqi_val}</div>
                <div style='font-size:13px;color:#7f8c8d;'>{category}</div>
            </div>
        """
    st.markdown(forecast_html, unsafe_allow_html=True)


    # POLLUTANT TREND CHART
# ---------------------------------------------------------
col3, col4 = st.columns([1.2, 1])

with col3:
    st.markdown(f"<div class='section-title'>{selected_pollutant} Concentration Trend</div>", unsafe_allow_html=True)

    if not city_data.empty:
        fig_trend = px.line(city_data, x="Date", y=selected_pollutant,
                            labels={selected_pollutant: f"{selected_pollutant} (µg/m³)", "Date": "Date"},
                            color_discrete_sequence=["#e67e22"])
        fig_trend.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No pollutant data available for this city.")

# ---------------------------------------------------------
# ACTIVE ALERTS SECTION
# ---------------------------------------------------------
with col4:
    st.markdown("<div class='section-title'>Active Alerts</div>", unsafe_allow_html=True)

    today = datetime.date.today()
    forecast_days = [
        (today, "Today"),
        (today + datetime.timedelta(days=1), "Tomorrow"),
        (today + datetime.timedelta(days=2), "Day After Tomorrow")
    ]

    for i, (date, label) in enumerate(forecast_days):
        weekday = date.strftime("%A")
        if i < len(forecast_values):
            aqi_val = forecast_values[i]
        else:
            aqi_val = np.nan

        if pd.isna(aqi_val):
            msg = f"No data available for {weekday}."
            level = "good"
        else:
            category, _ = get_aqi_category(aqi_val)  # Fixed: unpack tuple
            msg = f"{weekday}: Expected {category} air quality (AQI {int(aqi_val)})."
            if category in ["Good", "Satisfactory"]:
                level = "good"
            elif category in ["Moderate", "Poor"]:
                level = "moderate"
            else:
                level = "unhealthy"

        st.markdown(
            f"<div class='alert-box {level}'><b>{label}</b><br>"
            f"<span style='font-size:14px;color:#2c3e50;'>{msg}</span></div>",
            unsafe_allow_html=True
        )
