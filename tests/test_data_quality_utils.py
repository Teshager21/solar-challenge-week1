import pandas as pd
import numpy as np
import pytest
from scripts.data_quality_utils import DataQualityUtils  # Replace with actual module name

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
    dqu = DataQualityUtils(sample_df)
    result = dqu.columns_with_significant_missing_values(threshold=10)
    assert "DHI" in result.index

def test_detect_outliers_zscore(sample_df):
    dqu = DataQualityUtils(sample_df)
    outliers = dqu.detect_outliers_zscore("GHI")
    assert isinstance(outliers, pd.DataFrame)

def test_find_columns_with_invalid_values(sample_df):
    dqu = DataQualityUtils(sample_df)
    ranges = {"GHI": (0, 400), "DNI": (0, 500)}
    violations = dqu.find_columns_with_invalid_values(ranges)
    assert "GHI" in violations

def test_replace_negative_irradiance_with_nan(sample_df):
    dqu = DataQualityUtils(sample_df)
    df = dqu.replace_negative_irradiance_with_nan()
    assert pd.isna(df.loc[1, "GHI"])  # originally -5
    assert pd.isna(df.loc[4, "DHI"])  # originally -10

def test_get_outlier_counts(sample_df):
    dqu = DataQualityUtils(sample_df)
    result = dqu.get_outlier_counts(["GHI"])
    assert isinstance(result, pd.DataFrame)
    assert "column" in result.columns

def test_filter_daytime(sample_df):
    dqu = DataQualityUtils(sample_df)
    df = dqu.filter_daytime()
    assert df['Timestamp'].dt.hour.between(6, 17).all()

def test_impute_ghi_with_linear_regression(sample_df):
    sample_df.loc[1, 'GHI'] = np.nan
    dqu = DataQualityUtils(sample_df)
    result = dqu.impute_ghi_with_linear_regression()
    assert not pd.isna(result.loc[1, "GHI"])

def test_impute_multiple_targets_with_model(sample_df):
    sample_df.loc[0, 'DHI'] = np.nan
    sample_df.loc[1, 'DNI'] = np.nan
    dqu = DataQualityUtils(sample_df)
    result = dqu.impute_multiple_targets_with_model()
    assert not pd.isna(result.loc[0, "DHI"]) or not pd.isna(result.loc[1, "DNI"])
