import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from windrose import WindroseAxes
from scipy.stats import zscore


class SolarVisualizer:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df.copy()

    def _ensure_columns_exist(self, required_cols):
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def plot_continuous_histograms(self, n_cols=3, bins=30, color='teal'):
        continuous_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if continuous_cols.empty:
            raise ValueError("No continuous columns found for plotting.")
        n_rows = (len(continuous_cols) + n_cols - 1) // n_cols
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for i, col in enumerate(continuous_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(self.df[col].dropna(), kde=True, bins=bins, color=color)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_scatter_relationships(self):
        self._ensure_columns_exist(['WS', 'GHI', 'WSgust', 'WD', 'RH', 'Tamb'])
        sns.set(style="whitegrid")
        plt.figure(figsize=(16, 12))
        plots = [
            ('WS', 'GHI'),
            ('WSgust', 'GHI'),
            ('WD', 'GHI'),
            ('RH', 'Tamb'),
            ('RH', 'GHI')
        ]
        for i, (x, y) in enumerate(plots, 1):
            plt.subplot(2, 3, i)
            sns.scatterplot(data=self.df, x=x, y=y, alpha=0.5)
            plt.title(f'{x} vs {y}')
        plt.tight_layout()
        plt.show()

    def plot_rh_relationships(self):
        self._ensure_columns_exist(['RH', 'Tamb', 'GHI'])
        sns.set(style="whitegrid", context="notebook")
        plt.figure(figsize=(12, 5))
        for i, y in enumerate(['Tamb', 'GHI'], 1):
            plt.subplot(1, 2, i)
            sns.scatterplot(data=self.df, x='RH', y=y, alpha=0.4)
            sns.regplot(data=self.df, x='RH', y=y, scatter=False, color='red')
            plt.title(f'Relative Humidity vs {y}')
            plt.xlabel('Relative Humidity (%)')
            plt.ylabel(y)
        plt.tight_layout()
        plt.show()

    def plot_bubble_ghi_vs_tamb(self, size_col='RH', color_col='RH'):
        self._ensure_columns_exist(['Tamb', 'GHI', size_col, color_col])
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.df['Tamb'], self.df['GHI'],
            s=self.df[size_col], alpha=0.5,
            c=self.df[color_col], cmap='coolwarm', edgecolors='w'
        )
        plt.title(f'Bubble Chart: GHI vs. Tamb (Bubble size = {size_col})')
        plt.xlabel('Ambient Temperature (°C)')
        plt.ylabel('Global Horizontal Irradiance (GHI) (W/m²)')
        plt.colorbar(scatter, label=f'{color_col} (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_mod_cleaning_effect(self):
        self._ensure_columns_exist(['Cleaning', 'ModA', 'ModB'])
        mod_cleaning_avg = self.df.groupby('Cleaning')[['ModA', 'ModB']].mean().reset_index()
        mod_cleaning_avg['Cleaning'] = mod_cleaning_avg['Cleaning'].map({0: 'Pre-Cleaning', 1: 'Post-Cleaning'})
        mod_cleaning_avg.set_index('Cleaning').plot(kind='bar', figsize=(8, 5), colormap='Paired')
        plt.title('Average ModA & ModB: Pre vs Post Cleaning', fontsize=14)
        plt.ylabel('Average Irradiance (W/m²)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_irradiance_temperature_timeseries(self):
        self._ensure_columns_exist(['Timestamp', 'GHI', 'DNI', 'DHI', 'Tamb'])
        df_plot = self.df.copy()
        df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
        df_plot = df_plot.set_index('Timestamp')
        plt.figure(figsize=(14, 6))
        df_plot[['GHI', 'DNI', 'DHI', 'Tamb']].plot(ax=plt.gca(), alpha=0.8)
        plt.title('Solar Irradiance and Temperature Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Values (W/m² or °C)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_monthly_irradiance_temperature(self):
        self._ensure_columns_exist(['Timestamp', 'GHI', 'DNI', 'DHI', 'Tamb'])
        df_plot = self.df.copy()
        df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
        df_plot = df_plot.set_index('Timestamp')
        df_plot['Month'] = df_plot.index.month_name()
        monthly_avg = df_plot.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()
        monthly_avg = monthly_avg.reindex([
            'January','February','March','April','May','June',
            'July','August','September','October','November','December'
        ])
        monthly_avg.plot(kind='bar', figsize=(14,6), colormap='viridis')
        plt.title('Monthly Average of Irradiance and Temperature')
        plt.ylabel('Average (W/m² or °C)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_hourly_irradiance_temperature(self):
        self._ensure_columns_exist(['Timestamp', 'GHI', 'DNI', 'DHI', 'Tamb'])
        df_plot = self.df.copy()
        df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
        df_plot = df_plot.set_index('Timestamp')
        df_plot['Hour'] = df_plot.index.hour
        hourly_avg = df_plot.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()
        hourly_avg.plot(figsize=(14,6), marker='o')
        plt.title('Average Daily Pattern: Irradiance & Temperature by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_ghi_anomalies(self):
        self._ensure_columns_exist(['Timestamp', 'GHI'])
        df_plot = self.df.copy()
        df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
        df_plot = df_plot.set_index('Timestamp')
        ghi_threshold = df_plot['GHI'].mean() + 3 * df_plot['GHI'].std()
        plt.figure(figsize=(15,5))
        plt.plot(df_plot.index, df_plot['GHI'], label='GHI')
        plt.axhline(ghi_threshold, color='red', linestyle='--', label='Anomaly Threshold')
        plt.title('GHI Over Time with Anomalies')
        plt.ylabel('GHI (W/m²)')
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.show()
        anomalies = df_plot[df_plot['GHI'] > ghi_threshold]
        print(f"\n⚠️ GHI Anomalies Detected:\n{anomalies[['GHI']]}")

    def plot_wind_rose(self):
        self._ensure_columns_exist(['WD', 'WS'])
        df_clean = self.df[['WD', 'WS']].dropna()
        ax = WindroseAxes.from_ax()
        ax.bar(df_clean['WD'], df_clean['WS'], normed=True, opening=0.8, edgecolor='white')
        ax.set_legend()
        plt.title("Wind Rose: Wind Speed vs Direction")
        plt.show()

    def plot_correlation_heatmap(self, columns):
        self._ensure_columns_exist(columns)
        corr_matrix = self.df[columns].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
        plt.title("Correlation Heatmap: Solar and Module Variables", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_pairplot(self, columns):
        self._ensure_columns_exist(columns)
        pairplot_df = self.df[columns].dropna()
        sns.pairplot(pairplot_df, diag_kind='kde', corner=True, plot_kws={'alpha': 0.5})
        plt.suptitle("Pair Plot of Selected Solar Variables", y=1.02, fontsize=14)
        plt.show()

    def plot_outlier_stripplots(self, columns_to_check_for_outliers):
        self._ensure_columns_exist(columns_to_check_for_outliers)
        n_cols = 2
        n_rows = (len(columns_to_check_for_outliers) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 2 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(columns_to_check_for_outliers):
            z_scores = zscore(self.df[col].dropna())
            temp_df = self.df.dropna(subset=[col]).copy()
            temp_df['z'] = z_scores
            sns.stripplot(x='z', data=temp_df, color='orange', ax=axes[i])
            axes[i].axvline(3, color='red', linestyle='--')
            axes[i].axvline(-3, color='red', linestyle='--')
            axes[i].set_title(f'Z-score Strip Plot: {col}')
            axes[i].set_xlabel('Z-score')
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def plot_outlier_boxplots(self, columns_to_check_for_outliers):
        self._ensure_columns_exist(columns_to_check_for_outliers)
        plt.figure(figsize=(max(8, len(columns_to_check_for_outliers) * 1.5), 6))
        sns.set_context("notebook", font_scale=1.1)
        sns.boxplot(data=self.df[columns_to_check_for_outliers], palette="Set2")
        plt.title("Boxplot of Selected Columns (Outliers Visualized)", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_boxplots_comparison(self):
        self._ensure_columns_exist(['country', 'GHI', 'DNI', 'DHI'])
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        metrics = ['GHI', 'DNI', 'DHI']
        for i, metric in enumerate(metrics):
            sns.boxplot(x='country', y=metric, data=self.df, ax=axes[i], palette="Set2")
            axes[i].set_title(f'📦 {metric} Distribution by Country', fontsize=14)
            axes[i].set_xlabel('')
            axes[i].set_ylabel(f'{metric} (W/m²)', fontsize=12)
        plt.suptitle('☀️ Solar Radiation Metrics by Country', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_avg_ghi_by_country(self):
        self._ensure_columns_exist(['country', 'GHI'])
        avg_ghi = self.df.groupby('country')['GHI'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_ghi.values, y=avg_ghi.index, palette='viridis')
        plt.title('Average GHI by Country')
        plt.xlabel('Average GHI')
        plt.ylabel('Country')
        plt.tight_layout()
        plt.show()
