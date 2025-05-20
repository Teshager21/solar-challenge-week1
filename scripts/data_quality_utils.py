import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

class DataQualityUtils:
    def __init__(self, df):
        self.df = df.copy()

    def columns_with_significant_missing_values(self, threshold=5):
        """
        Returns a DataFrame listing columns with missing values above a specified threshold (percentage).
        """
        missing_counts = self.df.isna().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        significant_missing = missing_percentages[missing_percentages > threshold]
        missing_values_df = pd.DataFrame({
            '#missing_values': missing_counts[significant_missing.index],
            'percentage': significant_missing.apply(lambda x: f"{x:.2f}%")
        })
        return missing_values_df.sort_values(by='#missing_values', ascending=False)

    def detect_outliers_zscore(self, column_name, threshold=3):
        """
        Detect outliers in a dataframe column based on z-score.
        """
        df = self.df.copy()
        df['zscore'] = zscore(df[column_name])
        outliers = df[(df['zscore'] > threshold) | (df['zscore'] < -threshold)]
        return outliers

    def find_columns_with_invalid_values(self, valid_ranges: dict) -> dict:
        """
        Identifies columns with values outside the defined valid ranges and specifies which condition was violated.
        """
        violations = {}
        for col, (min_val, max_val) in valid_ranges.items():
            if col in self.df.columns:
                below_min = (self.df[col] < min_val).any()
                above_max = (self.df[col] > max_val).any()
                violated = []
                if below_min:
                    violated.append(f"< {min_val}")
                if above_max:
                    violated.append(f"> {max_val}")
                if violated:
                    violations[col] = violated
        return violations

    def conditional_impute(self, timestamp_col, conditions, updates):
        """
        Imputes values in a DataFrame based on a dictionary of simple string conditions.
        """
        df = self.df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['is_night'] = df[timestamp_col].dt.hour.apply(lambda h: h < 6 or h >= 18)
        condition_expr = ' & '.join([f'`{col}` {op}' for col, op in conditions.items()])
        mask = df.query(condition_expr).index
        for col, val in updates.items():
            df.loc[mask, col] = val
        df = df.drop(columns=['is_night'])
        self.df = df
        return self.df

    def impute_ghi_with_linear_regression(self, features=['ModB', 'ModA'], target='GHI', model=None):
        """
        Imputes missing values in the target column using linear regression on the specified features.
        """
        if model is None:
            model = LinearRegression()
        df_copy = self.df.copy()
        train_data = df_copy[features + [target]].dropna()
        predict_data = df_copy[df_copy[target].isna() & df_copy[features].notna().all(axis=1)]
        if len(predict_data) == 0:
            print("No imputable rows found for target:", target)
            return df_copy
        model.fit(train_data[features], train_data[target])
        predicted_values = np.round(model.predict(predict_data[features]), 1)
        df_copy.loc[predict_data.index, target] = predicted_values
        self.df = df_copy
        return self.df

    def impute_multiple_targets_with_model(
        self,
        targets=['DHI', 'DNI'],
        exclude_cols=['Timestamp','Precipitation','WD'],
        model_cls=RandomForestRegressor,
        model_kwargs=None
    ):
        """
        Impute missing values for multiple target columns using all other columns as features.
        """
        if model_kwargs is None:
            model_kwargs = {'n_estimators': 100, 'random_state': 42}
        df_copy = self.df.copy()
        feature_cols = [col for col in self.df.columns if col not in targets + exclude_cols]
        for target in targets:
            available = df_copy[feature_cols + [target]].dropna()
            missing = df_copy[df_copy[target].isna() & df_copy[feature_cols].notna().all(axis=1)]
            if len(missing) == 0 or len(available) == 0:
                continue
            model = model_cls(**model_kwargs)
            model.fit(available[feature_cols], available[target])
            df_copy.loc[missing.index, target] = np.round(model.predict(missing[feature_cols]), 1)
        self.df = df_copy
        return self.df

    def replace_negative_irradiance_with_nan(self):
        """
        Replaces negative values in GHI, DNI, and DHI columns with NaN.
        """
        df = self.df.copy()
        for col in ['GHI', 'DNI', 'DHI']:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan
        self.df = df
        return self.df

    def get_outlier_counts(self, columns_to_check_for_outliers):
        """
        Returns a DataFrame with the number of outliers (z-score > 3 or < -3) for each specified column.
        """
        outlier_counts = {
            "column": [],
            "num_outliers": []
        }
        for col in columns_to_check_for_outliers:
            outliers = self.detect_outliers_zscore(col)
            outlier_counts["column"].append(col)
            outlier_counts["num_outliers"].append(len(outliers))
        outlier_df = pd.DataFrame(outlier_counts)
        return outlier_df
    def filter_daytime(self, timestamp_col='Timestamp', start_hour=6, end_hour=18):
        """
        Returns a DataFrame containing only the rows where the timestamp is during daytime hours.

        Parameters:
        - timestamp_col: name of the column with datetime information (default: 'Timestamp')
        - start_hour: start of the daytime in 24h format (default: 6 for 6 AM)
        - end_hour: end of the daytime in 24h format (default: 18 for 6 PM)

        Returns:
        - Filtered DataFrame with only daytime records
        """
        df = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        daytime_df = df[(df[timestamp_col].dt.hour >= start_hour) & 
                        (df[timestamp_col].dt.hour < end_hour)]
        return daytime_df