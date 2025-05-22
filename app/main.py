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

            # Simulate 30 records per country
            countries = ["Benin", "Togo", "Sierra Leone"]
            rows_per_country = 30

            df = pd.DataFrame({
                "country": np.repeat(countries, rows_per_country),
                "GHI": np.concatenate([
                    np.random.normal(5.1, 0.2, rows_per_country),  # Benin
                    np.random.normal(4.9, 0.2, rows_per_country),  # Togo
                    np.random.normal(5.3, 0.2, rows_per_country),  # Sierra Leone
                ]),
                "DNI": np.concatenate([
                    np.random.normal(6.1, 0.3, rows_per_country),
                    np.random.normal(6.0, 0.3, rows_per_country),
                    np.random.normal(6.2, 0.3, rows_per_country),
                ]),
                "DHI": np.concatenate([
                    np.random.normal(1.1, 0.1, rows_per_country),
                    np.random.normal(1.0, 0.1, rows_per_country),
                    np.random.normal(1.2, 0.1, rows_per_country),
                ]),
                "temperature": np.tile(np.random.normal(28, 1, rows_per_country), 3),
                "month": np.tile(["Jan", "Feb", "Mar"], int(rows_per_country * len(countries) / 3))
            })

        return df


    def plot_irradiance_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='country', y=self.metric, data=self.df, palette='viridis', ax=ax)
        ax.set_title(f"{self.metric} Distribution by Country")
        ax.set_ylabel(f"{self.metric} (kWh/m²/day)")
        ax.set_xlabel("")
        st.pyplot(fig)

    def show_summary(self):
        st.subheader("📊 Summary Statistics")
        summary_df = (
            self.df.groupby('country')[[self.metric]]
            .agg(['mean', 'median', 'std'])
            .round(2)
        )
        st.dataframe(summary_df)

    def show_boxplot(self):
        st.subheader(f"📈 {self.metric} Comparison")
        self.plot_irradiance_distribution()

    def show_country_trends(self):
        st.subheader("📌 Country Trends")
        self.country_option = st.selectbox("Select Country", self.df['country'].unique())

        with st.expander("📉 View Daily Irradiance Trends"):
            country_df = self.df[self.df['country'] == self.country_option]
            if 'datetime' in country_df.columns:
                country_df = country_df.copy()
                country_df['datetime'] = pd.to_datetime(country_df['datetime'])
                country_df.set_index('datetime', inplace=True)
                daily_avg = country_df[[self.metric]].resample('D').mean()

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                daily_avg.plot(ax=ax2, legend=False, color='orange')
                ax2.set_title(f"Daily Average {self.metric} in {self.country_option}")
                ax2.set_ylabel(f"{self.metric} (kWh/m²/day)")
                st.pyplot(fig2)
            else:
                st.info("ℹ️ No datetime column found for trend visualization in this dataset.")

    def sidebar_filters(self):
        st.sidebar.header("🔎 Filters")
        available_metrics = [col for col in ['GHI', 'DNI', 'DHI'] if col in self.df.columns]
        self.metric = st.sidebar.selectbox("Select Irradiance Metric", available_metrics)

    def run(self):
        st.title("🌍 Cross-Country Solar Potential Dashboard")
        st.markdown("Analyze and compare solar irradiance metrics across Benin 🇧🇯, Togo 🇹🇬, and Sierra Leone 🇸🇱.")
        
        self.sidebar_filters()
        self.show_summary()
        self.show_boxplot()
        self.show_country_trends()

        st.markdown("---")
        st.markdown("Made with ❤️ for 10 Academy Week 1 Challenge — MoonLight Energy Solutions 🌞")


if __name__ == "__main__":
    dashboard = SolarDashboard()
    dashboard.run()
