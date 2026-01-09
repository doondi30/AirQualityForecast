

import numpy as np

# Breakpoints (India)
BP_TABLE = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300),
              (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300),
             (351, 430, 301, 400), (431, 600, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300),
            (281, 400, 301, 400), (401, 1000, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300),
            (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300),
           (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300),
           (209, 748, 301, 400), (749, 1000, 401, 500)],
}

# AQI categories
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe")
]


def get_bp_table():
    """Return AQI breakpoint table"""
    return BP_TABLE

def compute_sub_index(pollutant, concentration):
    """Compute AQI sub-index for a given pollutant concentration"""
    if pollutant not in BP_TABLE:
        return None
    for (low, high, index_low, index_high) in BP_TABLE[pollutant]:
        if low <= concentration <= high:
            return ((index_high - index_low)/(high - low))*(concentration - low) + index_low
    return None

def get_aqi_category(aqi_value):
    """Return AQI category"""
    for low, high, category in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return category
    return "Unknown"

def compute_overall_aqi(data_dict):
    """
    Compute overall AQI
    data_dict = {"PM2.5": value, "PM10": value, ...}
    Returns: overall_aqi, sub_indices, category
    """
    sub_indices = {}
    for pollutant, value in data_dict.items():
        if value is not None:
            sub_index = compute_sub_index(pollutant, value)
            if sub_index is not None:
                sub_indices[pollutant] = round(sub_index)
    if not sub_indices:
        return None, {}, "Unknown"
    
    overall_aqi = max(sub_indices.values())
    category = get_aqi_category(overall_aqi)
    return overall_aqi, sub_indices, category

def generate_alert(overall_aqi, threshold=200):
    """
    Generate alert if AQI exceeds threshold
    Returns True/False
    """
    return overall_aqi > threshold