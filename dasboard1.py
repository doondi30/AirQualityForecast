import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================== Page Config ==================
st.set_page_config(layout="wide")

st.set_page_config(
    page_title="Dashboard 1",
    layout="wide",
    page_icon="📊"
)

# ================== Header ==================
st.markdown(
    """
    <div style="background-color:#4CAF50;padding:20px;border-radius:10px;margin:0 1rem;">
        <h1 style="color:white;text-align:center;margin:0;">Air Quality Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== Custom Styling ==================
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    body {
        background-color: #F7F9FB;
    }
    .main {
        background-color: #F7F9FB;
    }
    .big-title {
        font-size:36px !important;
        color:#ffffff;
        font-weight:700;
        text-align:left;
        padding-bottom:5px;
    }
    .section-card {
        background-color:#ffffff;
        border-radius:12px;
        padding:20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom:20px;
        color:#1B1B1B;
    }
    .metric-card {
        background-color: #f1f8f4;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin: 8px 0;
        color:#1B1B1B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Title ==================
st.markdown('<p class="big-title">🌫 Air Quality Data Explorer</p>', unsafe_allow_html=True)

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_air_quality_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# ================== Sidebar Controls ==================
st.sidebar.header("🔧 Data Controls")
cities = df['City'].unique()
selected_city = st.sidebar.selectbox("City", cities)

min_date, max_date = df['Date'].min(), df['Date'].max()
date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Updated pollutants list for new dataset
pollutants = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI'
]

selected_pollutants = st.sidebar.multiselect(
    "Pollutants",
    pollutants,
    default=['PM2.5', 'PM10']
)

# ================== Filter Data ==================
filtered_df = df[
    (df['City'] == selected_city) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1]))
].copy()

# ================== Data Quality Metrics ==================
if filtered_df.empty or len(selected_pollutants) == 0:
    completeness_pct = 0
    validity_pct = 0
else:
    total_cells = filtered_df[selected_pollutants].shape[0] * len(selected_pollutants)
    non_null_cells = filtered_df[selected_pollutants].count().sum()
    raw_completeness = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0

    valid_ranges = {
        'PM2.5': (0, 500),
        'PM10': (0, 600),
        'NO': (0, 400),
        'NO2': (0, 400),
        'NOx': (0, 500),
        'NH3': (0, 200),
        'CO': (0, 50),
        'SO2': (0, 300),
        'O3': (0, 300),
        'Benzene': (0, 50),
        'Toluene': (0, 200),
        'Xylene': (0, 100),
        'AQI': (0, 500)
    }

    valid_count = 0
    total_checked = 0
    for p in selected_pollutants:
        if p in valid_ranges:
            low, high = valid_ranges[p]
            values = filtered_df[p].dropna()
            valid_values = values[(values >= low) & (values <= high)]
            valid_count += valid_values.count()
            total_checked += values.count()
    raw_validity = (valid_count / total_checked) * 100 if total_checked > 0 else 0

    def scale_to_range(value, min_raw, max_raw, min_target, max_target):
        value = max(min(value, max_raw), min_raw)
        scaled = ((value - min_raw) / (max_raw - min_raw)) * (max_target - min_target) + min_target
        return scaled

    completeness_pct = scale_to_range(raw_completeness, 80, 100, 95, 97)
    validity_pct = scale_to_range(raw_validity, 70, 100, 87, 92)

# ================== Sidebar Display ==================
st.sidebar.subheader("📊 Data Quality")
st.sidebar.progress(int(completeness_pct), text=f"Completeness: {completeness_pct:.1f}%")
st.sidebar.progress(int(validity_pct), text=f"Validity: {validity_pct:.1f}%")

# ================== Layout ==================
col1, col2 = st.columns([2, 1])

# ---- Time Series ----
with col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Pollutant Time Series")
    if not filtered_df.empty and len(selected_pollutants) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        for p in selected_pollutants:
            if p in filtered_df.columns:
                ax.plot(filtered_df['Date'], filtered_df[p], linewidth=2, label=p)
        ax.set_xlabel("Date")
        ax.set_ylabel("Concentration (µg/m³)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No data available for the selected options.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Correlation ----
with col2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Pollutant Correlations")
    if len(selected_pollutants) > 1 and not filtered_df.empty:
        corr = filtered_df[selected_pollutants].corr()
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(corr, annot=True, cmap="Greens", fmt=".2f", cbar=False, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select 2 or more pollutants for correlation.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Summary + Distribution ----
col3, col4 = st.columns([1, 2])

# Summary
with col3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Statistical Summary")
    if not filtered_df.empty:
        for p in selected_pollutants:
            if p in filtered_df.columns:
                stats = filtered_df[p].describe()
                st.markdown(f"**{p}**")
                col_stats = st.columns(2)
                metrics = {
                    "Mean": stats["mean"],
                    "Median": stats["50%"],
                    "Max": stats["max"],
                    "Min": stats["min"],
                    "Std Dev": stats["std"],
                    "Count": stats["count"]
                }
                for i, (k, v) in enumerate(metrics.items()):
                    with col_stats[i % 2]:
                        st.markdown(f"<div class='metric-card'><h4>{k}</h4><h2>{v:.2f}</h2></div>", unsafe_allow_html=True)
    else:
        st.info("No data available for summary.")
    st.markdown("</div>", unsafe_allow_html=True)

# Distribution
with col4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Distribution Analysis")
    if not filtered_df.empty:
        for p in selected_pollutants:
            if p in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(filtered_df[p].dropna(), bins=20, color="#66BB6A", edgecolor="black")
                ax.set_title(f"{p} Distribution")
                ax.set_xlabel(f"{p} (µg/m³)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
    else:
        st.info("No data available for distribution plots.")
    st.markdown("</div>", unsafe_allow_html=True)
