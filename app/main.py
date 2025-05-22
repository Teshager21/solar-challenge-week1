import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Must be the first Streamlit command
st.set_page_config(page_title="🌞 Solar Potential Dashboard", layout="wide")


class SolarDashboard:
    def __init__(self):
        self.df = self.load_data()
        self.metric = 'GHI'  # default fallback
        self.country_option = None

    @staticmethod
    @st.cache_data
    def load_data():
        try:
            df_benin = pd.read_csv("data/benin_clean.csv")
            df_togo = pd.read_csv("data/togo_clean.csv")
            df_sl = pd.read_csv("data/sierraleone_clean.csv")

            df_benin["country"] = "Benin"
            df_togo["country"] = "Togo"
            df_sl["country"] = "Sierra Leone"

            df = pd.concat([df_benin, df_togo, df_sl], ignore_index=True)
        except FileNotFoundError:
            st.warning("⚠️ Data files not found. Using fallback simulated data.")
            countries = ["Benin", "Togo", "Sierra Leone"]
            rows_per_country = 30
            df = pd.DataFrame({
                "country": sum([[c]*rows_per_country for c in countries], []),
                "GHI": sum([list(np.random.normal(5.0 + i*0.1, 0.2, rows_per_country)) for i in range(3)], []),
                "DNI": sum([list(np.random.normal(6.0 + i*0.1, 0.3, rows_per_country)) for i in range(3)], []),
                "DHI": sum([list(np.random.normal(1.1, 0.1, rows_per_country)) for _ in range(3)], []),
                "temperature": np.tile(np.random.normal(28, 1, rows_per_country), 3),
                "month": np.tile(["Jan", "Feb", "Mar"], int(rows_per_country * 3 / 3))
            })
        return df

    def sidebar_filters(self):
        st.sidebar.header("🔎 Filters")
        available_metrics = [col for col in ['GHI', 'DNI', 'DHI'] if col in self.df.columns]
        self.metric = st.sidebar.selectbox("Select Irradiance Metric", available_metrics)
        self.country_option = st.sidebar.selectbox("Select Country", sorted(self.df['country'].unique()))

    def plot_irradiance_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x='country', y=self.metric, data=self.df, palette='viridis', ax=ax)
        ax.set_title(f"{self.metric} Distribution by Country")
        ax.set_ylabel(f"{self.metric} (kWh/m²/day)")
        ax.set_xlabel("")
        st.pyplot(fig, use_container_width=True)

    def show_summary(self):
        st.subheader("📊 Summary Statistics")
        summary_df = (
            self.df.groupby('country')[[self.metric]]
            .agg(['mean', 'median', 'std'])
            .round(2)
        )
        st.dataframe(summary_df, use_container_width=True)

    def show_country_trends(self):
        st.subheader("📉 Daily Irradiance Trends")
        country_df = self.df[self.df['country'] == self.country_option]
        if 'datetime' in country_df.columns:
            country_df = country_df.copy()
            country_df['datetime'] = pd.to_datetime(country_df['datetime'])
            country_df.set_index('datetime', inplace=True)
            daily_avg = country_df[[self.metric]].resample('D').mean()
            fig, ax = plt.subplots(figsize=(10, 3))
            daily_avg.plot(ax=ax, legend=False, color='orange')
            ax.set_title(f"Daily Average {self.metric} in {self.country_option}")
            ax.set_ylabel(f"{self.metric} (kWh/m²/day)")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("ℹ️ No datetime column found for trend visualization.")

    def run(self):
        st.title("🌍 Cross-Country Solar Potential Dashboard")
        st.markdown("Analyze and compare solar irradiance metrics across Benin 🇧🇯, Togo 🇹🇬, and Sierra Leone 🇸🇱.")
        self.sidebar_filters()
        col1, col2 = st.columns(2)
        with col1:
            self.show_summary()
        with col2:
            self.plot_irradiance_distribution()
        st.divider()
        self.show_country_trends()
        st.markdown("---")
        st.markdown("Made with ❤️ for 10 Academy Week 1 Challenge — MoonLight Energy Solutions 🌞")

if __name__ == "__main__":
    dashboard = SolarDashboard()
    dashboard.run()
