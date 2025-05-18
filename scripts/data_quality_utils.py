import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


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




def detect_outliers_zscore(df, column_name, threshold=3):
    """
    Detect outliers in a dataframe column based on z-score.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Column to check for outliers.
        threshold (float): Z-score threshold to identify outliers (default=3).

    Returns:
        pd.DataFrame: Subset of rows considered outliers.
    """
    df = df.copy()
    df['zscore'] = zscore(df[column_name])
    outliers = df[(df['zscore'] > threshold) | (df['zscore'] < -threshold)]
    return outliers


def find_columns_with_invalid_values(df: pd.DataFrame, valid_ranges: dict) -> dict:
    """
    Identifies columns with values outside the defined valid ranges and specifies which condition was violated.

    Parameters:
        df (pd.DataFrame): The input DataFrame to check.
        valid_ranges (dict): Dictionary mapping column names to (min, max) valid ranges.

    Returns:
        dict: A dictionary where each key is a column name with invalid values,
              and the value is a list of violated conditions ('< min' or '> max').
    """
    violations = {}

    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            below_min = (df[col] < min_val).any()
            above_max = (df[col] > max_val).any()

            violated = []
            if below_min:
                violated.append(f"< {min_val}")
            if above_max:
                violated.append(f"> {max_val}")

            if violated:
                violations[col] = violated

    return violations


def conditional_impute(df, timestamp_col, conditions, updates):
    """
    Imputes values in a DataFrame based on a dictionary of simple string conditions.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str): Name of timestamp column (must be in datetime format).
        conditions (dict): Dictionary with column names as keys and condition strings as values.
                           Example: {'GHI': '<= 0', 'is_night': '== True'}
        updates (dict): Dictionary of columns to update and their new values.
                        Example: {'GHI': 0, 'DNI': 0}

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    df = df.copy()

    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Create is_night column
    df['is_night'] = df[timestamp_col].dt.hour.apply(lambda h: h < 6 or h >= 18)

    # Build the query string from the conditions dict
    condition_expr = ' & '.join([f'`{col}` {op}' for col, op in conditions.items()])
    
    # Use query to create mask
    mask = df.query(condition_expr).index

    # Apply updates
    for col, val in updates.items():
        df.loc[mask, col] = val
    # Drop the helper column before returning
    df = df.drop(columns=['is_night'])

    return df



def impute_ghi_with_linear_regression(df: pd.DataFrame, features=['ModB', 'ModA'], target='GHI', model=None):
    """
    Imputes missing values in the target column using linear regression on the specified features.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing missing values in the target column.
    - features (list): List of column names to use as predictors.
    - target (str): The name of the column to impute.
    - model: scikit-learn regression model instance (default is LinearRegression).

    Returns:
    - pd.DataFrame: A copy of the input DataFrame with missing target values imputed.
    """

    # Use Linear Regression if no model is provided
    if model is None:
        model = LinearRegression()

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Drop rows with missing values in features or target for training
    train_data = df_copy[features + [target]].dropna()

    # Rows where the target is missing but predictors are available
    predict_data = df_copy[df_copy[target].isna() & df_copy[features].notna().all(axis=1)]

    if len(predict_data) == 0:
        print("No imputable rows found for target:", target)
        return df_copy

    # Fit the model
    model.fit(train_data[features], train_data[target])

    # Predict and fill missing target values (rounded to 1 decimal place)
    predicted_values = np.round(model.predict(predict_data[features]), 1)
    df_copy.loc[predict_data.index, target] = predicted_values

    return df_copy

def impute_multiple_targets_with_model(
    df: pd.DataFrame,
    targets=['DHI', 'DNI'],
    exclude_cols=['Timestamp','Precipitation','WD'],
    model_cls=RandomForestRegressor,
    model_kwargs=None
):
    """
    Impute missing values for multiple target columns using all other columns as features.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - targets (list): List of target columns to impute.
    - exclude_cols (list): Columns to exclude from features (e.g., timestamp).
    - model_cls: Scikit-learn regressor class (default: RandomForestRegressor).
    - model_kwargs (dict): Additional kwargs for the regressor.

    Returns:
    - pd.DataFrame: DataFrame with imputed target values.
    """
    if model_kwargs is None:
        model_kwargs = {'n_estimators': 100, 'random_state': 42}

    df_copy = df.copy()
    feature_cols = [col for col in df.columns if col not in targets + exclude_cols]

    for target in targets:
        available = df_copy[feature_cols + [target]].dropna()
        missing = df_copy[df_copy[target].isna() & df_copy[feature_cols].notna().all(axis=1)]

        if len(missing) == 0 or len(available) == 0:
            continue

        model = model_cls(**model_kwargs)
        model.fit(available[feature_cols], available[target])
        # Predict and round to 1 decimal place
        df_copy.loc[missing.index, target] = np.round(model.predict(missing[feature_cols]), 1)

    return df_copy



def replace_negative_irradiance_with_nan(df):
    """
    Replaces negative values in GHI, DNI, and DHI columns with NaN.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing irradiance columns.

    Returns:
        pd.DataFrame: Updated DataFrame with negative irradiance values set to NaN.
    """
    df = df.copy()
    for col in ['GHI', 'DNI', 'DHI']:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    return df

def get_outlier_counts(df, columns_to_check_for_outliers):
    """
    Returns a DataFrame with the number of outliers (z-score > 3 or < -3) for each specified column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns_to_check_for_outliers (list): List of column names to check for outliers.

    Returns:
    - pd.DataFrame: DataFrame with columns ['column', 'num_outliers'].
    """
    outlier_counts = {
        "column": [],
        "num_outliers": []
    }

    for col in columns_to_check_for_outliers:
        outliers = detect_outliers_zscore(df, col)
        outlier_counts["column"].append(col)
        outlier_counts["num_outliers"].append(len(outliers))

    outlier_df = pd.DataFrame(outlier_counts)
    return outlier_df
