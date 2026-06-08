from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.fourier_core import AnalysisResult


HOURS_PER_DAY = 24


COMPONENT_SPECS = [
    ('trend_low_freq', 'Trend / very low frequency (>500 d)', 500 * HOURS_PER_DAY, None, True),
    ('annual', 'Annual-scale (180-500 d)', 180 * HOURS_PER_DAY, 500 * HOURS_PER_DAY, False),
    ('seasonal', 'Seasonal (10-180 d)', 10 * HOURS_PER_DAY, 180 * HOURS_PER_DAY, False),
    ('weekly', 'Weekly (3-10 d)', 3 * HOURS_PER_DAY, 10 * HOURS_PER_DAY, False),
    ('daily', 'Daily (6-36 h)', 6, 36, False),
]


def _band_mask_from_periods(freqs_full: np.ndarray, min_period_h: float | None, max_period_h: float | None, include_dc: bool = False) -> np.ndarray:
    freqs_full = np.asarray(freqs_full, dtype=float)
    abs_freq = np.abs(freqs_full)
    mask = np.zeros(len(freqs_full), dtype=bool)
    nonzero = abs_freq > 0
    periods = np.full(len(freqs_full), np.inf, dtype=float)
    periods[nonzero] = 1.0 / abs_freq[nonzero]
    low_ok = periods >= min_period_h if min_period_h is not None else np.ones(len(freqs_full), dtype=bool)
    high_ok = periods < max_period_h if max_period_h is not None else np.ones(len(freqs_full), dtype=bool)
    mask = nonzero & low_ok & high_ok
    if include_dc:
        mask |= np.isclose(freqs_full, 0.0)
    return mask


def reconstruct_band_component(res: AnalysisResult, min_period_h: float | None, max_period_h: float | None, include_dc: bool = False, name: str = 'component') -> pd.Series:
    mask = _band_mask_from_periods(res.freqs_full, min_period_h, max_period_h, include_dc=include_dc)
    X_filtered = np.where(mask, res.X, 0)
    values = np.real(np.fft.ifft(X_filtered))
    return pd.Series(values, index=res.series.index, name=name)


def build_temporal_components(res: AnalysisResult) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    components: Dict[str, pd.Series] = {'original': res.series.astype(float)}
    used_mask = np.zeros(len(res.freqs_full), dtype=bool)

    for key, label, min_period_h, max_period_h, include_dc in COMPONENT_SPECS:
        series = reconstruct_band_component(res, min_period_h, max_period_h, include_dc=include_dc, name=key)
        components[key] = series
        used_mask |= _band_mask_from_periods(res.freqs_full, min_period_h, max_period_h, include_dc=include_dc)

    residual_mask = ~used_mask
    residual_series = pd.Series(np.real(np.fft.ifft(np.where(residual_mask, res.X, 0))), index=res.series.index, name='residual')
    components['residual'] = residual_series

    component_df = pd.DataFrame(components)

    original_vals = component_df['original'].values.astype(float)
    original_centered = original_vals - original_vals.mean()
    original_var = float(np.var(original_centered))
    original_abs_mean = float(np.mean(np.abs(original_vals))) if len(original_vals) > 0 else 1.0
    if original_abs_mean <= 0:
        original_abs_mean = 1.0

    rows = []
    for key, label, *_ in COMPONENT_SPECS:
        vals = component_df[key].values.astype(float)
        centered = vals - vals.mean()
        variance_share = float(np.var(centered) / original_var * 100) if original_var > 0 else 0.0
        abs_mean = float(np.mean(np.abs(vals)))
        rms = float(np.sqrt(np.mean(vals ** 2)))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        ptp_v = float(max_v - min_v)
        contrib_share = float(abs_mean / original_abs_mean * 100)
        rows.append({
            'Component': label,
            'Key': key,
            'Variance Share (%)': variance_share,
            'Mean': float(np.mean(vals)),
            'Std': float(np.std(vals)),
            'Min': min_v,
            'Max': max_v,
            'Peak-to-Peak': ptp_v,
            'Abs Mean': abs_mean,
            'RMS': rms,
            'Abs Contribution (%)': contrib_share,
        })

    vals = component_df['residual'].values.astype(float)
    centered = vals - vals.mean()
    variance_share = float(np.var(centered) / original_var * 100) if original_var > 0 else 0.0
    abs_mean = float(np.mean(np.abs(vals)))
    rms = float(np.sqrt(np.mean(vals ** 2)))
    min_v = float(np.min(vals))
    max_v = float(np.max(vals))
    ptp_v = float(max_v - min_v)
    contrib_share = float(abs_mean / original_abs_mean * 100)
    rows.append({
        'Component': 'Residual / other frequencies',
        'Key': 'residual',
        'Variance Share (%)': variance_share,
        'Mean': float(np.mean(vals)),
        'Std': float(np.std(vals)),
        'Min': min_v,
        'Max': max_v,
        'Peak-to-Peak': ptp_v,
        'Abs Mean': abs_mean,
        'RMS': rms,
        'Abs Contribution (%)': contrib_share,
    })

    summary_df = pd.DataFrame(rows).sort_values('Variance Share (%)', ascending=False).reset_index(drop=True)
    span_days = float((res.series.index.max() - res.series.index.min()).total_seconds() / 3600 / HOURS_PER_DAY)
    notes = {
        'summary': 'This decomposition separates the signal into non-overlapping Fourier bands associated with daily, weekly, seasonal, annual, low-frequency, and residual scales.',
        'span': f'Dataset span: {span_days:.1f} days. Annual-scale interpretation is much more reliable when the series covers at least 180-365 days.',
        'caution': 'These components are frequency-band reconstructions, not causal end-use disaggregation. They explain temporal structure, not specific appliances or physical subsystems.',
        'mean_note': 'Oscillatory Fourier bands typically have mean near zero. The baseline level is usually carried by the DC / very-low-frequency trend component.',
    }
    return component_df, summary_df, notes
