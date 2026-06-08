import pandas as pd
import pytest

from src.validation import parse_k_list, validate_simulation_inputs, validate_time_series


def test_parse_k_list_filters_duplicates_and_invalid():
    values = parse_k_list('5, 10, 10, x, 2000, 20', max_k=100)
    assert values == [5, 10, 20]


def test_validate_simulation_inputs_rejects_invalid():
    with pytest.raises(ValueError):
        validate_simulation_inputs('2h', 10)
    with pytest.raises(ValueError):
        validate_simulation_inputs('1h', 0)


def test_validate_time_series_accepts_regular_series():
    idx = pd.date_range('2024-01-01', periods=72, freq='h')
    series = pd.Series(range(72), index=idx)
    out = validate_time_series(series)
    assert out['is_regular'] is True
    assert out['length'] == 72
