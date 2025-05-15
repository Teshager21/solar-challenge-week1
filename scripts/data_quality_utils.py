import pandas as pd
def columns_with_significant_missing_values(df, threshold=5):
    """
    Returns a DataFrame listing columns with missing values above a specified threshold (percentage).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Percentage threshold for missing values (default is 5%).

    Returns:
        pd.DataFrame: Columns with missing values above the threshold, showing counts and percentages.
    """
    # Calculate missing value counts and percentages
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    # Filter columns with missing percentage above threshold
    significant_missing = missing_percentages[missing_percentages > threshold]

    # Create result DataFrame
    missing_values_df = pd.DataFrame({
        '#missing_values': missing_counts[significant_missing.index],
        'percentage': significant_missing.apply(lambda x: f"{x:.2f}%")
    })

    return missing_values_df.sort_values(by='#missing_values', ascending=False)