import streamlit as st
import subprocess

st.title("Forecasting Air Quality Using Historical Pollution Data")

st.write("Click a button to run a Dashboards:")

if st.button("Dashboard 1"):
    subprocess.Popen(["streamlit", "run", "dasboard1.py"])
    st.success("Dashboard 1 is running in a new window/tab")

if st.button("Dashboard 2"):
    subprocess.Popen(["streamlit", "run", "dasboard2.py"])
    st.success("Dashboard 2 is running in a new window/tab")

if st.button("Dashboard 3"):
    subprocess.Popen(["streamlit", "run", "dasboard3..py"])
    st.success("Dashboard 3 is running in a new window/tab")

if st.button("Dashboard 4"):
    subprocess.Popen(["streamlit", "run", "dasboard4.py"])
    st.success("Dashboard 4 is running in a new window/tab")

