import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class DataQualityUtils:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df.copy()

    def columns_with_significant_missing_values(self, threshold: float = 5.0) -> pd.DataFrame:
        """Return columns with missing values above the given percentage threshold."""
        missing_counts = self.df.isna().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        significant = missing_percent[missing_percent > threshold]
        return pd.DataFrame({
            '#missing_values': missing_counts[significant.index],
            'percentage': significant.apply(lambda x: f"{x:.2f}%")
        }).sort_values(by='#missing_values', ascending=False)

    def detect_outliers_zscore(self, column_name: str, threshold: float = 3.0) -> pd.DataFrame:
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        df = self.df.copy()
        z_scores = zscore(df[column_name], nan_policy='omit')  # ignores NaNs but preserves alignment
        df['zscore'] = z_scores

        return df[(df['zscore'].abs() > threshold)]


    def find_columns_with_invalid_values(self, valid_ranges: dict) -> dict:
        violations = {}
        for col, (min_val, max_val) in valid_ranges.items():
            if col in self.df.columns:
                violated = []
                if (self.df[col] < min_val).any():
                    violated.append(f"< {min_val}")
                if (self.df[col] > max_val).any():
                    violated.append(f"> {max_val}")
                if violated:
                    violations[col] = violated
        return violations

    def conditional_impute(self, timestamp_col: str, conditions: dict, updates: dict) -> pd.DataFrame:
        df = self.df.copy()
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            raise ValueError(f"Error parsing timestamp column '{timestamp_col}': {e}")

        df['is_night'] = df[timestamp_col].dt.hour.apply(lambda h: h < 6 or h >= 18)
        condition_expr = ' & '.join([f'`{col}` {op}' for col, op in conditions.items()])
        try:
            mask = df.query(condition_expr).index
        except Exception as e:
            raise ValueError(f"Invalid condition expression: {e}")

        for col, val in updates.items():
            if col in df.columns:
                df.loc[mask, col] = val

        self.df = df.drop(columns=['is_night'])
        return self.df

    def impute_ghi_with_linear_regression(self, features=None, target='GHI', model=None) -> pd.DataFrame:
        if features is None:
            features = ['ModB', 'ModA']
        if model is None:
            model = LinearRegression()

        df_copy = self.df.copy()
        try:
            train_data = df_copy[features + [target]].dropna()
            predict_data = df_copy[df_copy[target].isna() & df_copy[features].notna().all(axis=1)]

            if predict_data.empty:
                return df_copy

            model.fit(train_data[features], train_data[target])
            predicted = np.round(model.predict(predict_data[features]), 1)
            df_copy.loc[predict_data.index, target] = predicted
        except Exception as e:
            raise RuntimeError(f"Error during GHI imputation: {e}")

        self.df = df_copy
        return self.df

    def impute_multiple_targets_with_model(self, targets=None, exclude_cols=None,
                                           model_cls=RandomForestRegressor, model_kwargs=None) -> pd.DataFrame:
        if targets is None:
            targets = ['DHI', 'DNI']
        if exclude_cols is None:
            exclude_cols = ['Timestamp', 'Precipitation', 'WD']
        if model_kwargs is None:
            model_kwargs = {'n_estimators': 100, 'random_state': 42}

        df_copy = self.df.copy()
        feature_cols = [col for col in self.df.columns if col not in targets + exclude_cols]

        for target in targets:
            available = df_copy[feature_cols + [target]].dropna()
            missing = df_copy[df_copy[target].isna() & df_copy[feature_cols].notna().all(axis=1)]

            if missing.empty or available.empty:
                continue

            try:
                model = model_cls(**model_kwargs)
                model.fit(available[feature_cols], available[target])
                df_copy.loc[missing.index, target] = np.round(model.predict(missing[feature_cols]), 1)
            except Exception as e:
                raise RuntimeError(f"Error imputing {target} using {model_cls.__name__}: {e}")

        self.df = df_copy
        return self.df

    def replace_negative_irradiance_with_nan(self) -> pd.DataFrame:
        df = self.df.copy()
        for col in ['GHI', 'DNI', 'DHI']:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan
        self.df = df
        return self.df

    def get_outlier_counts(self, columns_to_check_for_outliers: list) -> pd.DataFrame:
        outlier_counts = {"column": [], "num_outliers": []}
        for col in columns_to_check_for_outliers:
            try:
                outliers = self.detect_outliers_zscore(col)
                outlier_counts["column"].append(col)
                outlier_counts["num_outliers"].append(len(outliers))
            except Exception as e:
                outlier_counts["column"].append(col)
                outlier_counts["num_outliers"].append(f"Error: {e}")
        return pd.DataFrame(outlier_counts)

    def filter_daytime(self, timestamp_col: str = 'Timestamp', start_hour: int = 6, end_hour: int = 18) -> pd.DataFrame:
        df = self.df.copy()
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            raise ValueError(f"Error parsing timestamp column: {e}")

        return df[(df[timestamp_col].dt.hour >= start_hour) & (df[timestamp_col].dt.hour < end_hour)]
