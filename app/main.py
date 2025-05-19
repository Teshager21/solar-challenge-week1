import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(page_title="🌞 Solar Potential Dashboard", layout="wide")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from utils import load_data, plot_ghi_comparison

# st.write("Current working directory:", os.getcwd())

# Setup page
# st.set_page_config(page_title="🌞 Solar Potential Dashboard", layout="wide")
st.title("🌍 Cross-Country Solar Potential Dashboard")
st.markdown("""
Analyze and compare solar irradiance metrics across Benin 🇧🇯, Togo 🇹🇬, and Sierra Leone 🇸🇱.
""")

# Load data (assuming data is in '../data/')
@st.cache_data
def load_data():
    benin = pd.read_csv("data/benin_clean.csv")
    togo = pd.read_csv("data/togo_clean.csv")
    sl = pd.read_csv("data/sierraleone_clean.csv")

    benin['country'] = 'Benin'
    togo['country'] = 'Togo'
    sl['country'] = 'Sierra Leone'

    return pd.concat([benin, togo, sl], ignore_index=True)

# Plot function
def plot_irradiance_distribution(df, metric):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='country', y=metric, data=df, palette='viridis', ax=ax)
    ax.set_title(f"{metric} Distribution by Country")
    ax.set_ylabel(f"{metric} (kWh/m²/day)")
    ax.set_xlabel("")
    st.pyplot(fig)

# Load and cache data
df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filters")
metric = st.sidebar.selectbox("Select Irradiance Metric", ['GHI', 'DNI', 'DHI'])

# Display summary table
st.subheader("📊 Summary Statistics")
st.dataframe(df.groupby('country')[[metric]].agg(['mean', 'median', 'std']).round(2))

# Display boxplot
st.subheader(f"📈 {metric} Comparison")
plot_irradiance_distribution(df, metric)

# Country-specific trends
st.subheader("📌 Country Trends")
country_option = st.selectbox("Select Country", df['country'].unique())

with st.expander("📉 View Daily Irradiance Trends"):
    country_df = df[df['country'] == country_option]
    if 'datetime' in country_df.columns:
        country_df['datetime'] = pd.to_datetime(country_df['datetime'])
        country_df.set_index('datetime', inplace=True)
        daily_avg = country_df[[metric]].resample('D').mean()

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        daily_avg.plot(ax=ax2, legend=False, color='orange')
        ax2.set_title(f"Daily Average {metric} in {country_option}")
        ax2.set_ylabel(f"{metric} (kWh/m²/day)")
        st.pyplot(fig2)
    else:
        st.info("No datetime column found for trend visualization.")

st.markdown("---")
st.markdown("Made with ❤️ for 10 Academy Week 1 Challenge — MoonLight Energy Solutions 🌞")
