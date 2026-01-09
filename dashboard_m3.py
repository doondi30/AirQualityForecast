import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Air Quality Alert System", layout="wide")

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #fefcf9;
}
.banner {
    background: linear-gradient(90deg, #fff3e0, #ffe0b2);
    padding: 25px 0;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 25px;
}
.banner h1 {
    color: #d35400;
    font-weight: 800;
    font-size: 36px;
    margin-bottom: 5px;
}
.banner h3 {
    color: #e67e22;
    font-weight: 500;
    font-size: 18px;
    margin-top: 0;
}
.section-title {
    color: #c0392b;
    font-weight: 700;
    font-size: 20px;
    margin-bottom: 12px;
}
.forecast-card {
    display: inline-block;
    padding: 15px 22px;
    margin: 5px;
    border-radius: 10px;
    background-color: white;
    text-align: center;
    width: 90px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.08);
}
.forecast-day {
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 3px;
}
.forecast-value {
    font-weight: 600;
    font-size: 13px;
}
.alert-box {
    padding: 12px 18px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 15px;
    color: #2c3e50;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.good {background-color:#eafbea;}
.moderate {background-color:#fff9e6;}
.unhealthy {background-color:#fdecea;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Dashboard Controls")
station = st.sidebar.selectbox("Monitoring Station", ["Downtown", "Suburban", "Industrial"])
pollutant = st.sidebar.selectbox("Select Pollutant", ["PM2.5", "PM10", "O3"])
st.sidebar.info("Use this dashboard to visualize AQI trends and alerts.")

# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="banner">
    <h1>Air Quality Alert System</h1>
    <h3>Milestone 3: Working Application (Weeks 5–6)</h3>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TOP ROW — AQI GAUGE + FORECAST
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1.1])

# ---- Current AQI Gauge ----
with col1:
    st.markdown("<div class='section-title'>Current Air Quality</div>", unsafe_allow_html=True)
    current_aqi = np.random.randint(45, 150)
    if current_aqi <= 50:
        color = "#2ecc71"; status = "Good"
    elif current_aqi <= 100:
        color = "#f1c40f"; status = "Moderate"
    elif current_aqi <= 150:
        color = "#e67e22"; status = "Unhealthy for Sensitive Groups"
    else:
        color = "#e74c3c"; status = "Unhealthy"

    fig = go.Figure(data=[go.Pie(
        values=[current_aqi, 200-current_aqi],
        hole=0.7,
        marker_colors=[color, "#f2f2f2"],
        textinfo="none"
    )])
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(text=f"<b>{current_aqi}</b><br>AQI<br>{status}", 
                 x=0.5, y=0.5, font_size=22, showarrow=False)
        ],
        height=280,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{station} Station")

# ---- 7-Day Forecast ----
with col2:
    st.markdown("<div class='section-title'>7-Day Forecast</div>", unsafe_allow_html=True)
    days = pd.date_range(datetime.date.today(), periods=7)
    aqi_values = np.random.randint(45, 150, 7)

    forecast_html = ""
    for i, d in enumerate(days):
        aqi = aqi_values[i]
        if aqi <= 50:
            label, bg = "Good", "#eafbea"
        elif aqi <= 100:
            label, bg = "Moderate", "#fff9e6"
        else:
            label, bg = "Unhealthy for Sensitive", "#fdecea"
        forecast_html += f"""
            <div class='forecast-card' style='background-color:{bg};'>
                <div class='forecast-day'>{d.strftime('%a')}</div>
                <div class='forecast-value'>AQI {aqi}</div>
                <div style='font-size:13px;color:#7f8c8d;'>{label}</div>
            </div>
        """
    st.markdown(forecast_html, unsafe_allow_html=True)
    st.markdown("""
        <div style="margin-top:10px;">
            <span style="background-color:#2ecc71;width:12px;height:12px;display:inline-block;"></span> Good &nbsp;&nbsp;
            <span style="background-color:#f1c40f;width:12px;height:12px;display:inline-block;"></span> Moderate &nbsp;&nbsp;
            <span style="background-color:#e67e22;width:12px;height:12px;display:inline-block;"></span> Unhealthy for Sensitive
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# BOTTOM ROW — POLLUTANT GRAPH + ALERTS
# ---------------------------------------------------------
col3, col4 = st.columns([1.1, 1])

# ---- Pollutant Concentration Graph ----
with col3:
    st.markdown("<div class='section-title'>Pollutant Concentrations</div>", unsafe_allow_html=True)
    hours = pd.date_range("00:00", "23:00", freq="3H").strftime("%H:%M")
    df = pd.DataFrame({
        "Time": hours,
        "PM2.5": np.random.randint(25, 70, len(hours)),
        "PM10": np.random.randint(30, 90, len(hours)),
        "O3": np.random.randint(15, 65, len(hours))
    })
    df_long = df.melt(id_vars="Time", var_name="Pollutant", value_name="Concentration")

    fig_trend = px.line(df_long, x="Time", y="Concentration", color="Pollutant",
                        color_discrete_map={"PM2.5": "#e67e22", "PM10": "#2980b9", "O3": "#27ae60"},
                        labels={"Concentration": "Concentration (µg/m³)", "Time": "Hour"})
    fig_trend.add_hline(y=50, line_dash="dot", annotation_text="WHO Limit", annotation_position="bottom right")
    fig_trend.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_trend, use_container_width=True)

# ---- Active Alerts ----
with col4:
    st.markdown("<div class='section-title'>Active Alerts</div>", unsafe_allow_html=True)
    alerts = [
        ("Unhealthy for Sensitive Groups", "Tomorrow, 10:00 AM", "unhealthy"),
        ("High Ozone Levels Expected", "Friday, 2:00 PM", "moderate"),
        ("Moderate Air Quality", "Today, 8:00 AM", "good")
    ]
    for msg, time, level in alerts:
        st.markdown(
            f"<div class='alert-box {level}'><b>{msg}</b><br>"
            f"<span style='font-size:13px;color:#7f8c8d;'>{time}</span></div>",
            unsafe_allow_html=True
        )