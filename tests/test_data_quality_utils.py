import pandas as pd
import numpy as np
import pytest
from scripts.data_quality_utils import DataQualityUtils  # Adjust the import based on your project structure


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'GHI': [100, -5, 500, np.nan, 250],
        'DNI': [50, 600, np.nan, 0, 450],
        'DHI': [np.nan, 200, 300, 150, -10],
        'Timestamp': pd.date_range("2023-01-01", periods=5, freq='H'),
        'ModA': [1, 2, 3, 4, 5],
        'ModB': [5, 4, 3, 2, 1],
        'Noise': [10, 20, np.nan, 40, 50]
    })

def test_columns_with_significant_missing_values(sample_df):
    dq = DataQualityUtils(sample_df)
    result = dq.columns_with_significant_missing_values(threshold=10)
    assert 'DHI' in result.index

def test_detect_outliers_zscore(sample_df):
    dq = DataQualityUtils(sample_df)
    result = dq.detect_outliers_zscore("GHI")
    assert isinstance(result, pd.DataFrame)
    assert 'zscore' in result.columns


def test_find_columns_with_invalid_values(sample_df):
    dq = DataQualityUtils(sample_df)
    result = dq.find_columns_with_invalid_values({"GHI": (0, 400), "DNI": (0, 500)})
    assert 'GHI' in result or 'DNI' in result

def test_conditional_impute(sample_df):
    dq = DataQualityUtils(sample_df)
    updated = dq.conditional_impute(
        timestamp_col='Timestamp',
        conditions={'is_night': '== True'},
        updates={'GHI': 0}
    )
    assert 'GHI' in updated.columns

def test_impute_ghi_with_linear_regression(sample_df):
    sample_df.loc[1, 'GHI'] = np.nan
    dq = DataQualityUtils(sample_df)
    updated = dq.impute_ghi_with_linear_regression()
    assert not pd.isna(updated.loc[1, 'GHI'])

def test_impute_multiple_targets_with_model(sample_df):
    sample_df.loc[0, 'DHI'] = np.nan
    sample_df.loc[1, 'DNI'] = np.nan
    dq = DataQualityUtils(sample_df)
    updated = dq.impute_multiple_targets_with_model()
    assert not pd.isna(updated.loc[0, 'DHI']) or not pd.isna(updated.loc[1, 'DNI'])

def test_replace_negative_irradiance_with_nan(sample_df):
    dq = DataQualityUtils(sample_df)
    updated = dq.replace_negative_irradiance_with_nan()
    assert pd.isna(updated.loc[1, 'GHI'])
    assert pd.isna(updated.loc[4, 'DHI'])

def test_get_outlier_counts(sample_df):
    dq = DataQualityUtils(sample_df)
    result = dq.get_outlier_counts(['GHI', 'DNI'])
    assert 'column' in result.columns
    assert 'num_outliers' in result.columns

def test_filter_daytime(sample_df):
    dq = DataQualityUtils(sample_df)
    result = dq.filter_daytime()
    assert result['Timestamp'].dt.hour.between(6, 17).all()
