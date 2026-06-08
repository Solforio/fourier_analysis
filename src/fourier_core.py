from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import welch
from scipy.stats import describe

from src.io_utils import normalize_frequency_alias


@dataclass
class AnalysisResult:
    series: pd.Series
    sample_hours: float
    stats: Dict[str, float]
    n: int
    x: np.ndarray
    X: np.ndarray
    freqs_full: np.ndarray
    freqs: np.ndarray
    amps: np.ndarray
    powers: np.ndarray
    phases: np.ndarray
    sorted_positive_indices: np.ndarray
    f_welch: np.ndarray
    psd_welch: np.ndarray
    dominant_df: pd.DataFrame
    band_energy_df: pd.DataFrame
    autocorr: np.ndarray
    cumvar: np.ndarray
    k90: int
    k95: int
    k99: int
    max_k: int


def infer_sample_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    return float(diffs.median()) if len(diffs) else 1.0


def is_regular_series(index: pd.DatetimeIndex, tol_seconds: float = 1.0) -> bool:
    if len(index) < 3:
        return True
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    return float(diffs.max() - diffs.min()) <= tol_seconds


def freq_to_period_label(freq: float) -> str:
    if freq <= 0:
        return 'inf'
    period_h = 1.0 / freq
    if period_h < 1:
        return f'{period_h * 60:.0f} min'
    if period_h < 48:
        return f'{period_h:.2f} h'
    if period_h < 24 * 14:
        return f'{period_h / 24:.2f} d'
    if period_h < 24 * 90:
        return f'{period_h / 168:.2f} wk'
    return f'{period_h / 8760:.3f} y'


def band_energy_summary(freqs: np.ndarray, powers: np.ndarray) -> pd.DataFrame:
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)

    labels = [
        'Trend / very low (>500 d)',
        'Annual-scale (180-500 d)',
        'Seasonal (10-180 d)',
        'Weekly (3-10 d)',
        'Daily (6-36 h)',
        'Other frequencies',
    ]
    positive = freqs > 0
    if not np.any(positive):
        return pd.DataFrame({'Band': labels, 'Energy Share (%)': [0.0] * len(labels)})

    freqs_pos = freqs[positive]
    powers_pos = powers[positive]
    periods_h = 1.0 / freqs_pos
    total_power = float(np.sum(powers_pos))
    if total_power <= 0:
        total_power = 1.0

    specs = [
        ('Trend / very low (>500 d)', periods_h >= 500 * 24),
        ('Annual-scale (180-500 d)', (periods_h >= 180 * 24) & (periods_h < 500 * 24)),
        ('Seasonal (10-180 d)', (periods_h >= 10 * 24) & (periods_h < 180 * 24)),
        ('Weekly (3-10 d)', (periods_h >= 3 * 24) & (periods_h < 10 * 24)),
        ('Daily (6-36 h)', (periods_h >= 6) & (periods_h < 36)),
    ]

    used = np.zeros_like(freqs_pos, dtype=bool)
    rows = []
    for label, mask in specs:
        used |= mask
        share = float(np.sum(powers_pos[mask]) / total_power * 100.0)
        rows.append({'Band': label, 'Energy Share (%)': share})

    other_mask = ~used
    other_share = float(np.sum(powers_pos[other_mask]) / total_power * 100.0)
    rows.append({'Band': 'Other frequencies', 'Energy Share (%)': other_share})
    return pd.DataFrame(rows)


def fourier_analysis(series: pd.Series, sample_hours: float) -> AnalysisResult:
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < 48:
        raise ValueError('At least 48 samples are required for a meaningful Fourier analysis.')
    stat_desc = describe(x)
    stats = {
        'n': float(n),
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'median': float(np.median(x)),
        'skewness': float(stat_desc.skewness),
        'kurtosis': float(stat_desc.kurtosis),
        'cv_pct': float(np.std(x) / np.mean(x) * 100) if np.mean(x) != 0 else 0.0,
        'p10': float(np.percentile(x, 10)),
        'p25': float(np.percentile(x, 25)),
        'p75': float(np.percentile(x, 75)),
        'p90': float(np.percentile(x, 90)),
    }
    X = fft(x)
    freqs_full = fftfreq(n, d=sample_hours)
    amplitude_full = np.abs(X) / n
    phase_full = np.angle(X)
    half = n // 2
    pos_freqs = freqs_full[:half]
    pos_amp = 2 * amplitude_full[:half]
    pos_power = pos_amp ** 2
    pos_phase = phase_full[:half]
    positive_mask = pos_freqs > 0
    positive_indices = np.where(positive_mask)[0]
    sorted_positive_indices = positive_indices[np.argsort(pos_amp[positive_mask])[::-1]]
    f_nonzero = pos_freqs[positive_mask]
    a_nonzero = pos_amp[positive_mask]
    p_nonzero = pos_power[positive_mask]
    ph_nonzero = pos_phase[positive_mask]
    top_idx = np.argsort(a_nonzero)[::-1][:50]
    total_nonzero_power = np.sum(p_nonzero)
    dominant_df = pd.DataFrame({
        'Rank': range(1, len(top_idx) + 1),
        'Period': [freq_to_period_label(f_nonzero[i]) for i in top_idx],
        'Frequency (cycles/hour)': f_nonzero[top_idx],
        'Amplitude': a_nonzero[top_idx],
        'Energy Share (%)': (p_nonzero[top_idx] / total_nonzero_power * 100 if total_nonzero_power > 0 else 0),
        'Phase (rad)': ph_nonzero[top_idx],
        'Phase (deg)': np.degrees(ph_nonzero[top_idx]),
    }).round(6)
    band_energy_df = band_energy_summary(pos_freqs, pos_power)
    fs = 1.0 / sample_hours
    nperseg = min(max(256, n // 4), n)
    f_welch, psd_welch = welch(x, fs=fs, nperseg=nperseg)
    x_centered = x - np.mean(x)
    ac_full = np.correlate(x_centered, x_centered, mode='full')
    autocorr = ac_full[n - 1:] / ac_full[n - 1]
    sorted_amp = np.sort(a_nonzero)[::-1]
    cumvar = np.cumsum(sorted_amp ** 2) / np.sum(sorted_amp ** 2) * 100 if np.sum(sorted_amp ** 2) > 0 else np.zeros_like(sorted_amp)
    max_k = int(min(1000, len(sorted_positive_indices)))
    return AnalysisResult(series, sample_hours, stats, n, x, X, freqs_full, pos_freqs, pos_amp, pos_power, pos_phase, sorted_positive_indices, f_welch, psd_welch, dominant_df, band_energy_df, autocorr, cumvar, int(np.searchsorted(cumvar, 90)) + 1 if len(cumvar) else 0, int(np.searchsorted(cumvar, 95)) + 1 if len(cumvar) else 0, int(np.searchsorted(cumvar, 99)) + 1 if len(cumvar) else 0, max_k)


def reconstruct_from_top_k(res: AnalysisResult, k: int) -> pd.Series:
    k = int(max(1, min(k, res.max_k)))
    chosen = res.sorted_positive_indices[:k]
    X_filtered = np.zeros(res.n, dtype=complex)
    X_filtered[0] = res.X[0]
    X_filtered[chosen] = res.X[chosen]
    negative_idx = res.n - chosen
    negative_idx = negative_idx[negative_idx < res.n]
    X_filtered[negative_idx] = res.X[negative_idx]
    reconstructed = np.real(ifft(X_filtered))
    return pd.Series(reconstructed, index=res.series.index, name=f'fourier_k_{k}')


def build_harmonic_dataframe(res: AnalysisResult, k: int) -> pd.DataFrame:
    k = int(max(1, min(k, res.max_k)))
    chosen = res.sorted_positive_indices[:k]
    rows = []
    for rank, idx in enumerate(chosen, start=1):
        rows.append({'Rank': rank, 'FFT Index': int(idx), 'Frequency (cycles/hour)': float(res.freqs[idx]), 'Period': freq_to_period_label(float(res.freqs[idx])), 'Amplitude': float(res.amps[idx]), 'Phase (rad)': float(res.phases[idx]), 'Phase (deg)': float(np.degrees(res.phases[idx]))})
    return pd.DataFrame(rows)


def build_harmonic_function_text(res: AnalysisResult, k: int, preview_terms: int = 12) -> Tuple[str, str]:
    coeffs = build_harmonic_dataframe(res, k)
    mean_val = float(res.stats['mean'])
    preview_rows = coeffs.head(preview_terms)
    preview = ['x_hat(t) = ' + f'{mean_val:.6f}']
    for _, row in preview_rows.iterrows():
        preview.append(f" + {row['Amplitude']:.6f} * cos(2*pi*{row['Frequency (cycles/hour)']:.10f}*t + {row['Phase (rad)']:.6f})")
    full = ['Fourier harmonic function', 'Reference: t is expressed in hours from the first timestamp.', '', 'x_hat(t) = ' + f'{mean_val:.10f}']
    for _, row in coeffs.iterrows():
        full.append(f" + {row['Amplitude']:.10f} * cos(2*pi*{row['Frequency (cycles/hour)']:.12f}*t + {row['Phase (rad)']:.10f})")
    preview_text = ''.join(preview)
    if len(coeffs) > preview_terms:
        preview_text += '\n\nPreview truncated to the first terms.'
    return preview_text, '\n'.join(full)


def error_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    empty_result = {
        'MAE': np.nan,
        'RMSE': np.nan,
        'Bias': np.nan,
        'Max Abs Error': np.nan,
        'MAPE filtered (%)': np.nan,
        'sMAPE (%)': np.nan,
        'wMAPE (%)': np.nan,
        'R2': np.nan,
    }

    if actual.size == 0 or pred.size == 0:
        return empty_result

    err = actual - pred
    abs_err = np.abs(err)
    if abs_err.size == 0:
        return empty_result

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    max_abs = float(np.max(abs_err))
    eps = np.finfo(float).eps
    scale = float(np.nanmedian(np.abs(actual)))
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.nanmean(np.abs(actual)))
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0
    min_denom = max(0.05 * scale, eps)
    valid_mape = np.abs(actual) >= min_denom
    mape_filtered = float(np.mean(abs_err[valid_mape] / np.abs(actual[valid_mape])) * 100) if np.any(valid_mape) else np.nan
    smape = float(np.mean(2.0 * abs_err / np.maximum(np.abs(actual) + np.abs(pred), eps)) * 100)
    wmape = float(np.sum(abs_err) / np.sum(np.abs(actual)) * 100) if np.sum(np.abs(actual)) > 0 else np.nan
    sst = np.sum((actual - np.mean(actual)) ** 2)
    sse = np.sum((actual - pred) ** 2)
    r2 = float(1 - sse / sst) if sst > 0 else np.nan
    return {'MAE': mae, 'RMSE': rmse, 'Bias': bias, 'Max Abs Error': max_abs, 'MAPE filtered (%)': mape_filtered, 'sMAPE (%)': smape, 'wMAPE (%)': wmape, 'R2': r2}


def compute_error_tables(actual: pd.Series, reconstructed: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame({'actual': actual, 'reconstructed': reconstructed})
    df = df.dropna(subset=['actual', 'reconstructed']).copy()
    df.index.name = 'timestamp'

    if df.empty:
        empty_point = pd.DataFrame(columns=['timestamp', 'actual', 'reconstructed', 'error', 'abs_error', 'error_pct', 'abs_error_pct', 'smape_pct'])
        empty_total = pd.DataFrame([error_metrics(np.array([]), np.array([]))])
        empty_daily = pd.DataFrame(columns=['timestamp', 'MAE', 'RMSE', 'Bias', 'Max Abs Error', 'MAPE filtered (%)', 'sMAPE (%)', 'wMAPE (%)', 'R2'])
        empty_monthly = pd.DataFrame(columns=['timestamp', 'MAE', 'RMSE', 'Bias', 'Max Abs Error', 'MAPE filtered (%)', 'sMAPE (%)', 'wMAPE (%)', 'R2'])
        return empty_point, empty_total, empty_daily, empty_monthly

    df['error'] = df['actual'] - df['reconstructed']
    df['abs_error'] = df['error'].abs()

    eps = np.finfo(float).eps
    scale = float(np.nanmedian(np.abs(df['actual'].values)))
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.nanmean(np.abs(df['actual'].values)))
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0

    min_denom = max(0.05 * scale, eps)
    df['error_pct'] = np.where(df['actual'].abs() >= min_denom, df['error'] / df['actual'] * 100, np.nan)
    df['abs_error_pct'] = np.where(df['actual'].abs() >= min_denom, df['abs_error'] / df['actual'].abs() * 100, np.nan)
    df['smape_pct'] = 2.0 * df['abs_error'] / np.maximum(df['actual'].abs() + df['reconstructed'].abs(), eps) * 100

    total_metrics = pd.DataFrame([error_metrics(df['actual'].values, df['reconstructed'].values)])

    def group_metrics(group: pd.DataFrame) -> pd.Series:
        if group.empty:
            return pd.Series(error_metrics(np.array([]), np.array([])))
        return pd.Series(error_metrics(group['actual'].values, group['reconstructed'].values))

    daily_metrics = df.groupby(pd.Grouper(freq='D')).apply(group_metrics).reset_index()
    metric_cols = [c for c in daily_metrics.columns if c != 'timestamp']
    if metric_cols:
        daily_metrics = daily_metrics.dropna(how='all', subset=metric_cols)

    monthly_metrics = df.groupby(pd.Grouper(freq='MS')).apply(group_metrics).reset_index()
    metric_cols = [c for c in monthly_metrics.columns if c != 'timestamp']
    if metric_cols:
        monthly_metrics = monthly_metrics.dropna(how='all', subset=metric_cols)

    return df.reset_index(), total_metrics, daily_metrics, monthly_metrics


def compare_k_performance(res: AnalysisResult, k_values: List[int]) -> pd.DataFrame:
    rows = []
    for k in sorted(set(int(k) for k in k_values)):
        total_error_df = compute_error_tables(res.series, reconstruct_from_top_k(res, k))[1]
        row = total_error_df.iloc[0].to_dict()
        row['K'] = int(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['K', 'MAE', 'RMSE', 'wMAPE (%)', 'MAPE filtered (%)', 'sMAPE (%)', 'R2', 'score'])
    best_r2 = float(df['R2'].max()) if df['R2'].notna().any() else np.nan
    min_rmse = float(df['RMSE'].min()) if df['RMSE'].notna().any() else np.nan
    min_wmape = float(df['wMAPE (%)'].min()) if df['wMAPE (%)'].notna().any() else np.nan
    df['score'] = df['K']
    if np.isfinite(min_rmse):
        df.loc[df['RMSE'] > min_rmse * 1.05, 'score'] += 100000
    if np.isfinite(min_wmape):
        df.loc[df['wMAPE (%)'] > min_wmape * 1.05, 'score'] += 10000
    if np.isfinite(best_r2):
        df.loc[df['R2'] < best_r2 - 0.01, 'score'] += 1000
    return df.sort_values('K').reset_index(drop=True)


def select_best_k(compare_df: pd.DataFrame) -> int:
    if compare_df.empty:
        return 1
    return int(compare_df.sort_values(['score', 'K']).iloc[0]['K'])


def generate_harmonic_series(res: AnalysisResult, k: int, resolution: str = '1h', total_days: int = 30) -> pd.DataFrame:
    resolution = normalize_frequency_alias(resolution)
    resolution_to_hours = {'15min': 0.25, '30min': 0.5, '1h': 1.0, '6h': 6.0, '1d': 24.0}
    step_hours = resolution_to_hours[resolution]
    periods = int(total_days * 24 / step_hours) + 1
    t = np.arange(periods) * step_hours
    coeffs = build_harmonic_dataframe(res, k)
    values = np.full(shape=periods, fill_value=float(res.stats['mean']), dtype=float)
    for _, row in coeffs.iterrows():
        values += row['Amplitude'] * np.cos(2 * np.pi * row['Frequency (cycles/hour)'] * t + row['Phase (rad)'])
    timestamps = pd.date_range(start=res.series.index[0], periods=periods, freq=resolution)
    return pd.DataFrame({'timestamp': timestamps, 'harmonic_value': values})


def build_interpretation(res: AnalysisResult, unit: str, k: int, total_error_df: pd.DataFrame) -> Dict[str, str]:
    s = res.stats
    top = res.dominant_df.iloc[0] if not res.dominant_df.empty else None
    err = total_error_df.iloc[0]
    skew_comment = 'roughly symmetric' if abs(s['skewness']) < 0.2 else ('right-skewed' if s['skewness'] > 0 else 'left-skewed')
    kurt_comment = 'close to Gaussian' if abs(s['kurtosis']) < 0.2 else ('heavy-tailed' if s['kurtosis'] > 0 else 'flatter than Gaussian')
    top_text = f"The strongest spectral component is the {top['Period']} cycle with amplitude {top['Amplitude']:.3f} {unit}." if top is not None else 'No dominant harmonic could be extracted.'
    return {
        'summary': f"The series contains {int(s['n']):,} samples. The mean value is {s['mean']:.3f} {unit} and the standard deviation is {s['std']:.3f} {unit}. The coefficient of variation is {s['cv_pct']:.2f}%. The distribution appears {skew_comment} and {kurt_comment}.",
        'spectrum': f"{top_text} Large peaks indicate stable recurring cycles, while smaller components reflect secondary structure or noise.",
        'reconstruction': f"Using K = {k} harmonics, the reconstruction reaches R² = {err['R2']:.4f}, RMSE = {err['RMSE']:.4f} {unit}, wMAPE = {err['wMAPE (%)']:.2f}%, and filtered MAPE = {err['MAPE filtered (%)']:.2f}%.",
        'acf': 'Autocorrelation highlights persistence and repeating temporal patterns. Repeated peaks near 24 h and 168 h usually indicate daily and weekly operating cycles.',
        'harmonic': 'The harmonic function is expressed as a constant term plus a sum of cosine waves. Each term has amplitude, frequency, and phase.',
    }
