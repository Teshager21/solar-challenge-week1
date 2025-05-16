from scripts.data_quality_utils import columns_with_significant_missing_values,detect_outliers_zscore
import pandas as pd

def test_columns_with_significant_missing_values():
    df = pd.DataFrame({
        'A': [1, None, 3],
        'B': [None, None, 2],
        'C': [1, 2, 3]
    })
    result = columns_with_significant_missing_values(df, threshold=20)
    assert 'B' in result.index
    assert 'A' in result.index
    assert 'C' not in result.index

def test_detect_outliers_zscore_basic():
    # Create a DataFrame with clear outliers
    data = {'value': [10]*50 + [12]*50 + [11]*50 + [1000]}
    df = pd.DataFrame(data)

    outliers = detect_outliers_zscore(df, 'value', threshold=3)

    # Check that 1 outlier is returned
    assert len(outliers) == 1
    assert outliers.iloc[0]['value'] == 1000

def test_detect_outliers_zscore_no_outliers():
    # Data with no outliers
    data = {'value': [10, 12, 11, 10, 12, 11]}
    df = pd.DataFrame(data)
