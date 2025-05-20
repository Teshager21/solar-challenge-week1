import pandas as pd
import numpy as np
import pytest
from scripts.feature_engineering_utils import FeatureEngineer  # Adjust if your path is different


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 10, 100, 1000],
        'B': [0.1, 1, 10, 100]
    })


def test_log1p_transform(sample_data):
    fe = FeatureEngineer(sample_data)
    df_transformed = fe.log_transform_columns(['A', 'B'], method='log1p')

    expected_A = np.log1p(sample_data['A']).round(3)
    expected_B = np.log1p(sample_data['B']).round(3)

    pd.testing.assert_series_equal(df_transformed['A'], expected_A, check_names=False)
    pd.testing.assert_series_equal(df_transformed['B'], expected_B, check_names=False)


def test_log_transform(sample_data):
    fe = FeatureEngineer(sample_data)
    df_transformed = fe.log_transform_columns(['A'], method='log')

    expected = np.log(sample_data['A'].clip(lower=1e-8)).round(3)
    pd.testing.assert_series_equal(df_transformed['A'], expected, check_names=False)


def test_log10_transform(sample_data):
    fe = FeatureEngineer(sample_data)
    df_transformed = fe.log_transform_columns(['B'], method='log10')

    expected = np.log10(sample_data['B'].clip(lower=1e-8)).round(3)
    pd.testing.assert_series_equal(df_transformed['B'], expected, check_names=False)


def test_invalid_method_raises_error(sample_data):
    fe = FeatureEngineer(sample_data)
    with pytest.raises(ValueError, match="Invalid method. Choose from 'log', 'log1p', or 'log10'"):
        fe.log_transform_columns(['A'], method='invalid')
