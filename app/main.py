import streamlit as st
import pandas as pd
import os
# from app.utils import load_data, plot_ghi_comparison
from utils import load_data, plot_ghi_comparison

st.set_page_config(page_title="Solar Potential Dashboard", layout="wide")

st.title("☀️ Solar Energy Potential Dashboard")

country = st.selectbox("Choose a Country", ["Benin", "Togo", "Sierra Leone", "All"])

df = load_data()

if country != "All":
    df = df[df["country"] == country]

plot_ghi_comparison(df)
