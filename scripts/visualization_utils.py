import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from windrose import WindroseAxes
from scipy.stats import zscore



def plot_continuous_histograms(df, n_cols=3, bins=30, color='teal'):
    """
    Plots histograms with KDE for all continuous numerical columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        n_cols (int): Number of columns in the subplot grid.
        bins (int): Number of bins for the histogram.
        color (str): Color for the histogram bars.

    Returns:
        None
    """

    continuous_cols = df.select_dtypes(include=['float64', 'int64']).columns
    n_rows = (len(continuous_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(continuous_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=bins, color=color)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_scatter_relationships(df):
    """
    Plots scatter plots for key variable relationships in the solar dataset.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))

    # Scatter: Wind Speed vs GHI
    plt.subplot(2, 3, 1)
    sns.scatterplot(data=df, x='WS', y='GHI', alpha=0.5)
    plt.title('Wind Speed (WS) vs GHI')

    # Scatter: Wind Gust vs GHI
    plt.subplot(2, 3, 2)
    sns.scatterplot(data=df, x='WSgust', y='GHI', alpha=0.5, color='orange')
    plt.title('Wind Gust (WSgust) vs GHI')

    # Scatter: Wind Direction vs GHI
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=df, x='WD', y='GHI', alpha=0.5, color='green')
    plt.title('Wind Direction (WD) vs GHI')

    # Scatter: Relative Humidity vs Temperature
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=df, x='RH', y='Tamb', alpha=0.5, color='purple')
    plt.title('Relative Humidity (RH) vs Ambient Temperature (Tamb)')

    # Scatter: Relative Humidity vs GHI
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=df, x='RH', y='GHI', alpha=0.5, color='red')
    plt.title('Relative Humidity (RH) vs GHI')

    plt.tight_layout()
    plt.show()

def plot_rh_relationships(df):
    """
    Plots scatter plots showing the relationship between Relative Humidity (RH) and
    (1) Ambient Temperature (Tamb), and (2) Global Horizontal Irradiance (GHI).

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """
    sns.set(style="whitegrid", context="notebook")
    plt.figure(figsize=(12, 5))

    # Scatter Plot: RH vs Ambient Temperature (Tamb)
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='RH', y='Tamb', alpha=0.4)
    sns.regplot(data=df, x='RH', y='Tamb', scatter=False, color='red')
    plt.title('Relative Humidity vs Ambient Temperature')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Ambient Temperature (°C)')

    # Scatter Plot: RH vs GHI (Global Horizontal Irradiance)
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='RH', y='GHI', alpha=0.4)
    sns.regplot(data=df, x='RH', y='GHI', scatter=False, color='red')
    plt.title('Relative Humidity vs Solar Radiation (GHI)')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('GHI (W/m²)')

    plt.tight_layout()
    plt.show()


def plot_bubble_ghi_vs_tamb(df, size_col='RH', color_col='RH'):
    """
    Plots a bubble chart of GHI vs. Tamb with bubble size and color representing a specified column (default: RH).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        size_col (str): Column name for bubble size (default: 'RH').
        color_col (str): Column name for bubble color (default: 'RH').

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['Tamb'], df['GHI'],
        s=df[size_col],                   # Bubble size
        alpha=0.5,
        c=df[color_col], cmap='coolwarm', # Bubble color
        edgecolors='w'
    )

    plt.title(f'Bubble Chart: GHI vs. Tamb (Bubble size = {size_col})')
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('Global Horizontal Irradiance (GHI) (W/m²)')
    plt.colorbar(scatter, label=f'{color_col} (%)')
    plt.grid(True)
    plt.show()

def plot_mod_cleaning_effect(df):
    """
    Plots the average ModA and ModB values before and after cleaning.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Cleaning', 'ModA', and 'ModB' columns.

    Returns:
        None
    """

    mod_cleaning_avg = df.groupby('Cleaning')[['ModA', 'ModB']].mean().reset_index()
    mod_cleaning_avg['Cleaning'] = mod_cleaning_avg['Cleaning'].map({0: 'Pre-Cleaning', 1: 'Post-Cleaning'})
    mod_cleaning_avg.set_index('Cleaning').plot(kind='bar', figsize=(8, 5), colormap='Paired')

    plt.title('Average ModA & ModB: Pre vs Post Cleaning', fontsize=14)
    plt.ylabel('Average Irradiance (W/m²)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_irradiance_temperature_timeseries(df):
    """
    Plots time series of GHI, DNI, DHI, and Tamb vs. Timestamp.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Timestamp', 'GHI', 'DNI', 'DHI', and 'Tamb' columns.

    Returns:
        None
    """
    df_plot = df.copy()
    # Ensure Timestamp is datetime type
    df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
    # Set Timestamp as index (optional, helps with time series)
    df_plot = df_plot.set_index('Timestamp')
    # Plotting
    plt.figure(figsize=(14, 6))
    df_plot[['GHI', 'DNI', 'DHI', 'Tamb']].plot(ax=plt.gca(), alpha=0.8)
    plt.title('Solar Irradiance and Temperature Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Values (W/m² or °C)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_monthly_irradiance_temperature(df):
    """
    Plots monthly averages of GHI, DNI, DHI, and Tamb as a bar chart.

    Parameters:
        df (pd.DataFrame): DataFrame with a datetime index or a 'Timestamp' column and columns 'GHI', 'DNI', 'DHI', 'Tamb'.

    Returns:
        None
    """

    df_plot = df.copy()
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_plot.index):
        if 'Timestamp' in df_plot.columns:
            df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
            df_plot = df_plot.set_index('Timestamp')
        else:
            raise ValueError("DataFrame must have a datetime index or a 'Timestamp' column.")

    df_plot['Month'] = df_plot.index.month_name()
    monthly_avg = df_plot.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean().reindex([
        'January','February','March','April','May','June',
        'July','August','September','October','November','December'
    ])

    monthly_avg.plot(kind='bar', figsize=(14,6), colormap='viridis')
    plt.title('Monthly Average of Irradiance and Temperature')
    plt.ylabel('Average (W/m² or °C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_hourly_irradiance_temperature(df):
    """
    Plots the average hourly pattern of GHI, DNI, DHI, and Tamb.

    Parameters:
        df (pd.DataFrame): DataFrame with a datetime index or a 'Timestamp' column and columns 'GHI', 'DNI', 'DHI', 'Tamb'.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df_plot = df.copy()
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_plot.index):
        if 'Timestamp' in df_plot.columns:
            df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
            df_plot = df_plot.set_index('Timestamp')
        else:
            raise ValueError("DataFrame must have a datetime index or a 'Timestamp' column.")

    df_plot['Hour'] = df_plot.index.hour
    hourly_avg = df_plot.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

    hourly_avg.plot(figsize=(14,6), marker='o')
    plt.title('Average Daily Pattern: Irradiance & Temperature by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ghi_anomalies(df):
    """
    Plots GHI over time with anomaly threshold and prints anomaly timestamps.

    Parameters:
        df (pd.DataFrame): DataFrame with a datetime index or a 'Timestamp' column and 'GHI' column.

    Returns:
        None
    """

    df_plot = df.copy()
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_plot.index):
        if 'Timestamp' in df_plot.columns:
            df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
            df_plot = df_plot.set_index('Timestamp')
        else:
            raise ValueError("DataFrame must have a datetime index or a 'Timestamp' column.")

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
    print(f"⚠️ GHI Anomalies Detected:\n{anomalies[['GHI']]}")

def plot_wind_rose(df):
    """
    Plots a wind rose using wind direction (WD) and wind speed (WS).

    Parameters:
        df (pd.DataFrame): DataFrame containing 'WD' and 'WS' columns.

    Returns:
        None
    """

    df_clean = df[['WD', 'WS']].dropna()
    ax = WindroseAxes.from_ax()
    ax.bar(df_clean['WD'], df_clean['WS'], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.title("Wind Rose: Wind Speed vs Direction")
    plt.show()

def plot_correlation_heatmap(df, columns):
    """
    Plots a correlation heatmap for the specified columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns.
        columns (list): List of column names to include in the heatmap.

    Returns:
        None
    """

    corr_matrix = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Correlation Heatmap: Solar and Module Variables", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_pairplot(df, columns):
    """
    Plots a seaborn pairplot for the specified columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns.
        columns (list): List of column names to include in the pairplot.

    Returns:
        None
    """

    pairplot_df = df[columns].dropna()
    sns.pairplot(pairplot_df, diag_kind='kde', corner=True, plot_kws={'alpha': 0.5})
    plt.suptitle("Pair Plot of Selected Solar Variables", y=1.02, fontsize=14)
    plt.show()


def plot_outlier_stripplots(df, columns_to_check_for_outliers):
    """
    Plots Z-score strip plots for the specified columns to visualize outliers.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns.
        columns_to_check_for_outliers (list): List of column names to plot.

    Returns:
        None
    """
   
    n_cols = 2  # adjust as needed
    n_rows = (len(columns_to_check_for_outliers) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 2 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_check_for_outliers):
        df['z'] = zscore(df[col].dropna())
        sns.stripplot(x='z', data=df.dropna(subset=[col]), color='orange', ax=axes[i])
        axes[i].axvline(3, color='red', linestyle='--')
        axes[i].axvline(-3, color='red', linestyle='--')
        axes[i].set_title(f'Z-score Strip Plot: {col}')
        axes[i].set_xlabel('Z-score')

    # Remove unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    # Clean up temporary column
    if 'z' in df.columns:
        df.drop(columns=['z'], inplace=True)


def plot_outlier_boxplots(df, columns_to_check_for_outliers):
    """
    Plots boxplots for the specified columns to visualize outliers.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns.
        columns_to_check_for_outliers (list): List of column names to plot.

    Returns:
        None
    """
    plt.figure(figsize=(max(8, len(columns_to_check_for_outliers) * 1.5), 6))  # Auto-adjust width
    sns.set_context("notebook", font_scale=1.1)
    sns.boxplot(data=df[columns_to_check_for_outliers], palette="Set2")
    plt.title("Boxplot of Selected Columns (Outliers Visualized)", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def filter_daytime(df, timestamp_col='Timestamp', start_hour=6, end_hour=18):
    """
    Returns a DataFrame containing only the rows where the timestamp is during daytime hours.

    Parameters:
    - df: pandas DataFrame
    - timestamp_col: name of the column with datetime information (default: 'Timestamp')
    - start_hour: start of the daytime in 24h format (default: 6 for 6 AM)
    - end_hour: end of the daytime in 24h format (default: 18 for 6 PM)

    Returns:
    - Filtered DataFrame with only daytime records
    """
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    daytime_df = df[(df[timestamp_col].dt.hour >= start_hour) & 
                    (df[timestamp_col].dt.hour < end_hour)]
    
    return daytime_df


def plot_boxplots_comparison(df):
    """
    Plots side-by-side boxplots for GHI, DNI, and DHI by country.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'country', 'GHI', 'DNI', and 'DHI' columns.
    """
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot for GHI
    sns.boxplot(x='country', y='GHI', data=df, ax=axes[0], palette="Set2")
    axes[0].set_title('📦 GHI Distribution by Country', fontsize=14)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('GHI (W/m²)', fontsize=12)

    # Plot for DNI
    sns.boxplot(x='country', y='DNI', data=df, ax=axes[1], palette="Set2")
    axes[1].set_title('📦 DNI Distribution by Country', fontsize=14)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('DNI (W/m²)', fontsize=12)

    # Plot for DHI
    sns.boxplot(x='country', y='DHI', data=df, ax=axes[2], palette="Set2")
    axes[2].set_title('📦 DHI Distribution by Country', fontsize=14)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('DHI (W/m²)', fontsize=12)

    # Super title
    plt.suptitle('☀️ Solar Radiation Metrics by Country', fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()
    plt.show()
