import numpy as np
import pandas as pd

from src.fourier_core import compute_error_tables, error_metrics, fourier_analysis, generate_harmonic_series, reconstruct_from_top_k


def make_series(n=24 * 14):
    idx = pd.date_range('2024-01-01', periods=n, freq='h')
    t = np.arange(n)
    values = 10 + 3 * np.cos(2 * np.pi * t / 24) + 1.2 * np.cos(2 * np.pi * t / 168 + 0.3)
    return pd.Series(values, index=idx)


def test_fourier_reconstruction_is_reasonable():
    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    rec = reconstruct_from_top_k(res, 2)
    total = compute_error_tables(series, rec)[1].iloc[0]
    assert total['R2'] > 0.95
    assert total['RMSE'] < 0.5


def test_error_metrics_zero_error():
    actual = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    out = error_metrics(actual, pred)
    assert out['MAE'] == 0.0
    assert out['RMSE'] == 0.0
    assert out['Bias'] == 0.0


def test_generate_harmonic_series_has_expected_columns():
    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    sim = generate_harmonic_series(res, k=2, resolution='1h', total_days=7)
    assert list(sim.columns) == ['timestamp', 'harmonic_value']
    assert len(sim) == 24 * 7 + 1


import numpy as np
import pandas as pd


def test_error_metrics_empty_arrays():
    out = error_metrics(np.array([]), np.array([]))
    assert np.isnan(out['MAE'])
    assert np.isnan(out['Max Abs Error'])


def test_compute_error_tables_all_nan_safe():
    idx = pd.date_range('2024-01-01', periods=4, freq='h')
    actual = pd.Series([np.nan, np.nan, np.nan, np.nan], index=idx)
    reconstructed = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    point_df, total_df, daily_df, monthly_df = compute_error_tables(actual, reconstructed)
    assert point_df.empty
    assert total_df.shape[0] == 1
    assert daily_df.empty
    assert monthly_df.empty
