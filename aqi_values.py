import pandas as pd

# 1️⃣ Load the CSV
df = pd.read_csv("cleaned_air_quality_data.csv")

# 2️⃣ Ensure AQI column is numeric
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

# 3️⃣ Drop rows where AQI or AQI_Bucket is missing
df = df.dropna(subset=['AQI', 'AQI_Bucket'])

# 4️⃣ Group by AQI_Bucket and calculate min, max, mean
aqi_summary = df.groupby('AQI_Bucket')['AQI'].agg(['min', 'max', 'mean']).reset_index()

# 5️⃣ Round values for readability
aqi_summary = aqi_summary.round(2)

# 6️⃣ Sort by mean AQI (from low to high)
aqi_summary = aqi_summary.sort_values(by='mean')

print(aqi_summary)

# 7️⃣ Optionally save
aqi_summary.to_csv("aqi_bucket_summary.csv", index=False)
print("\nSaved as 'aqi_bucket_summary.csv'")
