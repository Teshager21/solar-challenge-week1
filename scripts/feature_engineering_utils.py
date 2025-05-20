import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def log_transform_columns(self, columns, method='log1p'):
        """
        Apply log transformation to specified columns in the DataFrame.

        Parameters:
        - columns (list): List of column names to transform
        - method (str): 'log', 'log1p', or 'log10' (default is 'log1p')

        Returns:
        - pd.DataFrame: DataFrame with transformed columns (rounded to 3 decimals)
        """
        df_transformed = self.df.copy()
        for col in columns:
            if method == 'log':
                df_transformed[col] = np.log(df_transformed[col].clip(lower=1e-8))
            elif method == 'log1p':
                df_transformed[col] = np.log1p(df_transformed[col])
            elif method == 'log10':
                df_transformed[col] = np.log10(df_transformed[col].clip(lower=1e-8))
            else:
                raise ValueError("Invalid method. Choose from 'log', 'log1p', or 'log10'.")
            # Round to 3 decimal places
            df_transformed[col] = df_transformed[col].round(3)
        self.df = df_transformed
        return self.df