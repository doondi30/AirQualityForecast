import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# 1️⃣ App Header
# ------------------------
st.title("Air Quality Monitoring & Forecasting Dashboard")
st.markdown("""
Upload your CSV, select city/pollutants, visualize historical data, forecast AQI & pollutants, and get alerts for next 5 days.
""")

# ------------------------
# 2️⃣ Upload CSV
# ------------------------
uploaded_file = st.file_uploader("Upload AQI dataset CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)

    # Detect date column
    date_col_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if not date_col_candidates:
        st.error("No date/time column found.")
    else:
        date_col = date_col_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(by=date_col)
        df.set_index(date_col, inplace=True)
        st.success(f"Data loaded: {df.shape[0]} rows")
        st.dataframe(df.head())

        # ------------------------
        # 3️⃣ Select city and pollutants
        # ------------------------
        city = st.selectbox("Select City", df['City'].unique())
        df_city = df[df['City'] == city].copy()
        pollutants = ['PM2.5','PM10','NO2','SO2','CO','O3']
        selected_pollutants = st.multiselect("Select Pollutants", pollutants, default=pollutants)

        # ------------------------
        # 4️⃣ Date range filter
        # ------------------------
        min_date = df_city.index.min()
        max_date = df_city.index.max()
        date_range = st.date_input("Select Date Range", [min_date, max_date])
        df_filtered = df_city.loc[date_range[0]:date_range[1], selected_pollutants + ['AQI']]

        st.subheader("Historical Data")
        st.line_chart(df_filtered[selected_pollutants + ['AQI']])

        # ------------------------
        # 5️⃣ Pollutant Visualizations
        # ------------------------
        st.subheader("Pollutant Histogram / Trend")
        fig, axes = plt.subplots(len(selected_pollutants), 2, figsize=(12,4*len(selected_pollutants)))
        for i, p in enumerate(selected_pollutants):
            # Histogram
            axes[i,0].hist(df_filtered[p].dropna(), bins=20, color='skyblue', edgecolor='black')
            axes[i,0].set_title(f"{p} Distribution")
            axes[i,0].set_ylabel("Frequency")
            # Trend
            axes[i,1].plot(df_filtered.index, df_filtered[p], marker='o', color='orange')
            axes[i,1].set_title(f"{p} Trend")
            axes[i,1].set_ylabel("Concentration")
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Pollutant Correlation")
        fig_corr, ax_corr = plt.subplots(figsize=(8,6))
        sns.heatmap(df_filtered[selected_pollutants].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        # ------------------------
        # 6️⃣ Forecast next 30 days
        # ------------------------
        forecast_days = 30
        future_dates = pd.date_range(start=df_city.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_dict = {}
        for col in selected_pollutants:
            ts = df_city[col].astype(float).fillna(method='ffill').fillna(method='bfill')
            try:
                model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,7),
                                enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                forecast_values = model_fit.forecast(steps=forecast_days)
                forecast_values[forecast_values < 0] = 0
                forecast_dict[col] = pd.Series(forecast_values.values, index=future_dates)
            except:
                st.warning(f"Could not forecast {col}")
                continue

        forecast_df = pd.DataFrame(forecast_dict)
        forecast_df.index.name = 'Date'

        # ------------------------
        # 7️⃣ Compute AQI for forecast
        # ------------------------
        aqi_breakpoints = {
            'PM2.5': [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),
                      (150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)],
            'PM10': [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),
                     (355,424,201,300),(425,504,301,400),(505,604,401,500)],
            'NO2': [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),
                    (650,1249,301,400),(1250,1649,301,400),(1650,2049,401,500)],
            'SO2': [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),
                    (305,604,201,300),(605,804,301,400),(805,1004,401,500)],
            'CO': [(0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),
                   (15.5,30.4,201,300),(30.5,40.4,301,400),(40.5,50.4,401,500)],
            'O3': [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),
                   (106,200,201,300),(201,300,301,400),(301,400,401,500)]
        }

        def compute_individual_aqi(pollutant, conc):
            breakpoints = aqi_breakpoints[pollutant]
            if conc < breakpoints[0][0]:
                return breakpoints[0][2]
            for (C_low, C_high, I_low, I_high) in breakpoints:
                if C_low <= conc <= C_high:
                    return ((I_high - I_low)/(C_high - C_low))*(conc - C_low) + I_low
            return breakpoints[-1][3]

        def compute_aqi_row(row):
            aqi_values = []
            for p in selected_pollutants:
                if p in aqi_breakpoints:
                    aqi_values.append(compute_individual_aqi(p, row[p]))
            return max(aqi_values) if aqi_values else np.nan

        forecast_df['AQI'] = forecast_df.apply(compute_aqi_row, axis=1)

        # ------------------------
        # 8️⃣ Alert System for next 5 days
        # ------------------------
        def aqi_alert_label(aqi):
            if aqi <= 50:
                return "Good", "✅", "green"
            elif aqi <= 100:
                return "Moderate", "⚠️", "yellow"
            elif aqi <= 200:
                return "Unhealthy", "⚠️", "orange"
            else:
                return "Hazardous", "☠️", "red"

        forecast_df['Alert'], forecast_df['Emoji'], forecast_df['Color'] = zip(*forecast_df['AQI'].apply(aqi_alert_label))

        st.subheader("Next 5 Days AQI Alerts")
        for idx, row in forecast_df.head(5).iterrows():
            st.markdown(f"""
            <div style='padding:10px; background-color:{row['Color']}; color:white; border-radius:8px;'>
                <b>{idx.date()} - AQI: {row['AQI']:.1f} {row['Emoji']} - {row['Alert']}</b>
            </div>
            """, unsafe_allow_html=True)

        # ------------------------
        # 9️⃣ Forecast Visualizations (30 days)
        # ------------------------
        st.subheader("Forecasted AQI Trend (30 Days)")
        plt.figure(figsize=(12,5))
        plt.plot(forecast_df.index, forecast_df['AQI'], color='red', marker='o')
        plt.xticks(rotation=45)
        plt.ylabel('AQI')
        plt.title(f"{city} Forecasted AQI")
        plt.grid(True)
        st.pyplot(plt)

        st.subheader("Forecasted Pollutant Trends (30 Days)")
        fig_forecast, ax_forecast = plt.subplots(figsize=(12,5))
        for p in selected_pollutants:
            ax_forecast.plot(forecast_df.index, forecast_df[p], label=p, marker='o')
        ax_forecast.set_ylabel("Concentration")
        ax_forecast.set_title(f"{city} Forecasted Pollutants")
        ax_forecast.legend()
        ax_forecast.grid(True)
        st.pyplot(fig_forecast)

        # ------------------------
        #  🔟 Download Forecast CSV
        # ------------------------
        csv = forecast_df.to_csv().encode('utf-8')
        st.download_button("Download Forecast CSV", csv, "forecasted_AQI.csv", "text/csv")