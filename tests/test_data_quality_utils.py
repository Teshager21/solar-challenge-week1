from scripts.data_quality_utils import columns_with_significant_missing_values
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
