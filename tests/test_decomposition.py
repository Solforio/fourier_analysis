import pandas as pd
import numpy as np

from src.decomposition import build_temporal_components
from src.fourier_core import fourier_analysis


def make_series(n=24 * 90):
    idx = pd.date_range('2024-01-01', periods=n, freq='h')
    t = np.arange(n)
    values = (
        100
        + 10 * np.cos(2 * np.pi * t / 24)
        + 5 * np.cos(2 * np.pi * t / 168)
        + 2 * np.cos(2 * np.pi * t / (24 * 45))
        + 1.5 * np.cos(2 * np.pi * t / 2)
    )
    return pd.Series(values, index=idx)


def test_build_temporal_components_has_expected_columns():
    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    component_df, summary_df, notes = build_temporal_components(res)
    expected = {'original', 'trend_low_freq', 'annual', 'seasonal', 'weekly', 'daily', 'residual'}
    assert expected.issubset(set(component_df.columns))
    assert not summary_df.empty
    assert 'summary' in notes


def test_variance_shares_are_not_overlapping():
    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    _, summary_df, _ = build_temporal_components(res)
    total_share = summary_df['Variance Share (%)'].sum()
    residual_share = float(summary_df.loc[summary_df['Key'] == 'residual', 'Variance Share (%)'].iloc[0])
    trend_share = float(summary_df.loc[summary_df['Key'] == 'trend_low_freq', 'Variance Share (%)'].iloc[0])
    assert 99.0 <= total_share <= 101.0
    assert residual_share > 0.0
    assert trend_share < 1.0


def test_band_energy_summary_matches_named_bands_plus_other():
    from src.fourier_core import band_energy_summary

    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    df = band_energy_summary(res.freqs, res.powers)
    expected_bands = {
        'Trend / very low (>500 d)',
        'Annual-scale (180-500 d)',
        'Seasonal (10-180 d)',
        'Weekly (3-10 d)',
        'Daily (6-36 h)',
        'Other frequencies',
    }
    assert set(df['Band']) == expected_bands
    total = float(df['Energy Share (%)'].sum())
    assert 99.0 <= total <= 101.0



def test_component_window_figure_builds():
    from src.plotting import build_component_window_figure

    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    component_df, _, _ = build_temporal_components(res)
    fig = build_component_window_figure(
        component_df=component_df,
        unit='kW',
        start_ts=component_df.index.min(),
        horizon_label='7D',
        mode='overlay',
        show_original=True,
    )
    assert len(fig.data) >= 2


def test_cumulative_components_figure_builds():
    from src.plotting import build_cumulative_components_figure

    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    component_df, _, _ = build_temporal_components(res)
    fig = build_cumulative_components_figure(
        component_df=component_df,
        unit='kW',
        start_ts=component_df.index.min(),
        horizon_label='7D',
        show_original=True,
        show_cumulative_lines=True,
    )
    assert len(fig.data) >= 3


def test_temporal_component_summary_contains_extended_stats():
    series = make_series()
    res = fourier_analysis(series, sample_hours=1.0)
    _, summary_df, _ = build_temporal_components(res)
    required = {
        'Component', 'Key', 'Variance Share (%)', 'Mean', 'Std', 'Min', 'Max',
        'Peak-to-Peak', 'Abs Mean', 'RMS', 'Abs Contribution (%)'
    }
    assert required.issubset(set(summary_df.columns))
