from __future__ import annotations

import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.decomposition import build_temporal_components
from src.fourier_core import (
    build_harmonic_dataframe,
    build_harmonic_function_text,
    build_interpretation,
    compare_k_performance,
    compute_error_tables,
    fourier_analysis,
    generate_harmonic_series,
    infer_sample_hours,
    is_regular_series,
    reconstruct_from_top_k,
    select_best_k,
)
from src.io_utils import build_series_from_dataframe, list_excel_sheets, load_dataset_from_bytes, normalize_frequency_alias
from src.logging_utils import get_logger
from src.plotting import (
    apply_app_style,
    build_acf_figure,
    build_band_energy_figure,
    build_component_energy_figure,
    build_cumvar_figure,
    build_daily_error_figure,
    build_distribution_figure,
    build_error_percentage_figure,
    build_error_timeseries_figure,
    build_generated_series_figure,
    build_k_comparison_figure,
    build_monthly_error_figure,
    build_multi_k_reconstruction_figure,
    build_reconstruction_figure,
    build_spectrum_figure,
    build_temporal_components_figure,
    build_time_figure,
    build_component_window_figure,
    build_cumulative_components_figure,
)
from src.reporting import generate_csv_exports, generate_pdf_report, generate_zip_export
from src.validation import parse_k_list, validate_simulation_inputs, validate_time_series

logger = get_logger(__name__)

st.set_page_config(page_title='Fourier Analysis Studio V4.4.2', page_icon='📈', layout='wide')
apply_app_style()

GLOSSARY = [
    ('Frequency', 'How often a sinusoidal component repeats in time; in the app it is expressed in cycles per hour.'),
    ('Period', 'The duration of one complete cycle. It is the inverse of frequency.'),
    ('Amplitude', 'The strength of a harmonic component. Larger amplitude means a stronger oscillation.'),
    ('Phase', 'The time shift of a sinusoidal component relative to a reference origin.'),
    ('Harmonic', 'One sinusoidal term used to reconstruct the signal in Fourier analysis.'),
    ('Dominant frequency', 'A frequency with high spectral magnitude or high contribution to the total signal energy.'),
    ('Residual', 'The difference between the original series and the reconstruction using the selected K harmonics.'),
    ('Residual energy', 'The share of signal variance or squared error not explained by the selected reconstruction.'),
    ('Leakage', 'Spectral spreading caused by finite observation windows or signals that do not align perfectly with the sample window.'),
    ('Regular series', 'A time series with constant sampling interval. FFT works best in this condition.'),
    ('Coverage', 'How much of the expected data is actually present inside a selected year or period.'),
    ('ACF', 'Autocorrelation function. It shows whether the series or residual still has temporal structure at different lags.'),
]


def format_display_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include='number').columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(decimals)
    return out


def chart_key(scope: str, name: str, year: int | None = None) -> str:
    return f'{scope}_{year}_{name}' if year is not None else f'{scope}_{name}'


def download_key(scope: str, name: str, year: int | None = None) -> str:
    return f'dl_{scope}_{year}_{name}' if year is not None else f'dl_{scope}_{name}'


def estimate_expected_samples_per_year(sample_hours: float) -> float:
    return 24.0 * 365.25 / max(sample_hours, 1e-9)


def build_year_coverage_table(series: pd.Series, sample_hours: float, min_coverage_pct: float) -> pd.DataFrame:
    expected = estimate_expected_samples_per_year(sample_hours)
    rows = []
    for year, part in series.groupby(series.index.year):
        valid = int(part.notna().sum())
        total = int(len(part))
        coverage_pct = float(valid / expected * 100.0) if expected > 0 else 0.0
        eligible = bool(valid >= 24 and coverage_pct >= float(min_coverage_pct))
        rows.append(
            {
                'year': int(year),
                'rows_in_dataset': total,
                'valid_samples': valid,
                'expected_samples_full_year': float(expected),
                'coverage_pct': coverage_pct,
                'eligible': eligible,
            }
        )
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True) if rows else pd.DataFrame()


def build_yearly_series_map(series: pd.Series, sample_hours: float, min_coverage_pct: float) -> tuple[dict[int, pd.Series], pd.DataFrame]:
    coverage_df = build_year_coverage_table(series, sample_hours, min_coverage_pct)
    yearly = {}
    if coverage_df.empty:
        return yearly, coverage_df
    eligible_years = set(coverage_df.loc[coverage_df['eligible'], 'year'].astype(int).tolist())
    for year, part in series.groupby(series.index.year):
        if int(year) in eligible_years:
            yearly[int(year)] = part.dropna().copy()
    return yearly, coverage_df


def prepare_component_display_df(component_df: pd.DataFrame, auto_round_low_freq: bool, round_decimals: int, relative_threshold: float) -> tuple[pd.DataFrame, list[str]]:
    display_df = component_df.copy()
    rounded_cols: list[str] = []
    if not auto_round_low_freq or component_df.empty:
        return display_df, rounded_cols
    for col in display_df.columns:
        col_l = str(col).lower()
        if ('low' not in col_l and 'trend' not in col_l) or 'original' in col_l:
            continue
        s = pd.to_numeric(display_df[col], errors='coerce')
        if s.dropna().empty:
            continue
        span = float(s.max() - s.min())
        ref = float(max(abs(s.mean()), abs(s).max(), 1e-12))
        rel_span = span / ref if ref > 0 else 0.0
        if rel_span <= relative_threshold:
            display_df[col] = s.round(round_decimals)
            rounded_cols.append(str(col))
    return display_df, rounded_cols


def filter_series_by_period(series: pd.Series, period_mode: str, start_date, end_date, selected_year: int | None) -> tuple[pd.Series, str]:
    if period_mode == 'Full series':
        return series.copy(), 'Full series'
    if period_mode == 'Single year' and selected_year is not None:
        filtered = series[series.index.year == int(selected_year)].copy()
        return filtered, f'Single year: {selected_year}'
    if period_mode == 'Custom date range' and start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = series.loc[(series.index >= start_ts) & (series.index <= end_ts)].copy()
        return filtered, f'Custom range: {start_ts.date()} → {pd.Timestamp(end_date).date()}'
    return series.copy(), 'Full series'


def compute_series_summary(series: pd.Series, sample_hours: float, label: str) -> pd.DataFrame:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return pd.DataFrame({'metric': ['selection'], 'value': [label]})
    duration_hours = float((series.index.max() - series.index.min()).total_seconds() / 3600.0) if len(series) > 1 else 0.0
    missing_values = int(series.isna().sum())
    missing_pct = float(missing_values / len(series) * 100.0) if len(series) > 0 else 0.0
    expected_points = int(round(duration_hours / max(sample_hours, 1e-9))) + 1 if len(series) > 1 else len(series)
    completeness_pct = float(len(series.dropna()) / expected_points * 100.0) if expected_points > 0 else 100.0
    rows = [
        {'metric': 'selection', 'value': label},
        {'metric': 'start', 'value': str(series.index.min())},
        {'metric': 'end', 'value': str(series.index.max())},
        {'metric': 'samples', 'value': int(len(series))},
        {'metric': 'sampling_step_hours', 'value': float(sample_hours)},
        {'metric': 'duration_days', 'value': float(duration_hours / 24.0)},
        {'metric': 'detected_years', 'value': int(series.index.year.nunique())},
        {'metric': 'missing_values', 'value': missing_values},
        {'metric': 'missing_pct', 'value': missing_pct},
        {'metric': 'estimated_completeness_pct', 'value': completeness_pct},
        {'metric': 'is_regular', 'value': bool(is_regular_series(series.index))},
        {'metric': 'mean', 'value': float(s.mean())},
        {'metric': 'std', 'value': float(s.std()) if len(s) > 1 else 0.0},
        {'metric': 'min', 'value': float(s.min())},
        {'metric': 'p05', 'value': float(s.quantile(0.05))},
        {'metric': 'median', 'value': float(s.quantile(0.50))},
        {'metric': 'p95', 'value': float(s.quantile(0.95))},
        {'metric': 'max', 'value': float(s.max())},
    ]
    return pd.DataFrame(rows)


def maybe_downsample_series(series: pd.Series, max_points: int) -> pd.Series:
    if len(series) <= max_points:
        return series.copy()
    step = max(1, math.ceil(len(series) / max_points))
    return series.iloc[::step].copy()


def build_series_preview_figure(series: pd.Series, unit: str, max_points: int) -> go.Figure:
    sampled = maybe_downsample_series(series, max_points=max_points)
    fig = build_time_figure(sampled, unit)
    fig.update_layout(title=f'Preview of selected series ({len(sampled):,}/{len(series):,} points shown)')
    return fig


def compute_residual_outputs(original: pd.Series, reconstructed: pd.Series, sample_hours: float, max_lag_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame({'original': original, 'reconstructed': reconstructed}).dropna()
    df['residual'] = df['original'] - df['reconstructed']
    residual = df['residual']
    signal_energy = float((df['original'] ** 2).sum())
    residual_energy = float((residual ** 2).sum())
    residual_energy_pct = float(100.0 * residual_energy / signal_energy) if signal_energy > 0 else 0.0
    variance_original = float(df['original'].var()) if len(df) > 1 else 0.0
    variance_residual = float(residual.var()) if len(residual) > 1 else 0.0
    variance_residual_pct = float(100.0 * variance_residual / variance_original) if variance_original > 0 else 0.0
    lags = max(1, int(max_lag_hours / max(sample_hours, 1e-9)))
    acf1 = float(residual.autocorr(lag=1)) if len(residual) > 2 else float('nan')
    acf24 = float(residual.autocorr(lag=int(round(24.0 / max(sample_hours, 1e-9))))) if len(residual) > 24 else float('nan')
    acf168 = float(residual.autocorr(lag=int(round(168.0 / max(sample_hours, 1e-9))))) if len(residual) > 168 else float('nan')
    stats_df = pd.DataFrame(
        [
            {'metric': 'residual_mean', 'value': float(residual.mean()) if not residual.empty else 0.0},
            {'metric': 'residual_std', 'value': float(residual.std()) if len(residual) > 1 else 0.0},
            {'metric': 'residual_min', 'value': float(residual.min()) if not residual.empty else 0.0},
            {'metric': 'residual_p05', 'value': float(residual.quantile(0.05)) if not residual.empty else 0.0},
            {'metric': 'residual_median', 'value': float(residual.quantile(0.50)) if not residual.empty else 0.0},
            {'metric': 'residual_p95', 'value': float(residual.quantile(0.95)) if not residual.empty else 0.0},
            {'metric': 'residual_max', 'value': float(residual.max()) if not residual.empty else 0.0},
            {'metric': 'residual_energy_pct', 'value': residual_energy_pct},
            {'metric': 'residual_variance_pct', 'value': variance_residual_pct},
            {'metric': 'residual_acf_lag1', 'value': acf1},
            {'metric': 'residual_acf_24h', 'value': acf24},
            {'metric': 'residual_acf_168h', 'value': acf168},
            {'metric': 'residual_max_lag_samples_for_analysis', 'value': int(lags)},
        ]
    )
    return stats_df, df.reset_index().rename(columns={'index': 'timestamp'})


def build_residual_time_figure(residual_df: pd.DataFrame, unit: str, max_points: int) -> go.Figure:
    df = residual_df[['timestamp', 'residual']].copy().set_index('timestamp')['residual']
    df = maybe_downsample_series(df, max_points=max_points)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.values, mode='lines', name='Residual'))
    fig.add_hline(y=0.0, line_dash='dash', line_color='gray')
    fig.update_layout(title=f'Residual in time ({len(df):,} points shown)', xaxis_title='Timestamp', yaxis_title=unit, template='plotly_white')
    return fig


def build_residual_distribution_figure(residual_df: pd.DataFrame, unit: str) -> go.Figure:
    fig = px.histogram(residual_df, x='residual', nbins=80, title='Residual distribution', labels={'residual': unit})
    fig.update_layout(template='plotly_white')
    return fig


def build_residual_heatmap_figure(residual_df: pd.DataFrame) -> go.Figure:
    df = residual_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = df.pivot_table(index='hour', columns='weekday', values='residual', aggfunc='mean').reindex(columns=order)
    fig = px.imshow(pivot, aspect='auto', color_continuous_scale='RdBu_r', origin='lower', title='Mean residual by hour and weekday')
    fig.update_layout(template='plotly_white')
    return fig


def build_residual_acf_figure(residual_df: pd.DataFrame, sample_hours: float, max_lag_hours: int) -> go.Figure:
    s = residual_df.set_index('timestamp')['residual'].dropna()
    max_lag = max(1, int(max_lag_hours / max(sample_hours, 1e-9)))
    lags = list(range(1, min(max_lag, max(2, len(s) - 1)) + 1))
    vals = [float(s.autocorr(lag=l)) for l in lags]
    lag_hours = [l * sample_hours for l in lags]
    fig = go.Figure(go.Bar(x=lag_hours, y=vals, name='Residual ACF'))
    fig.update_layout(title='Residual autocorrelation', xaxis_title='Lag (hours)', yaxis_title='ACF', template='plotly_white')
    return fig


def build_top_residual_events(residual_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = residual_df.copy()
    df['abs_residual'] = df['residual'].abs()
    return df.sort_values('abs_residual', ascending=False).head(top_n)


def clear_analysis_state() -> None:
    for key in ['fourier_result', 'annual_series_map', 'annual_coverage_df', 'annual_summary_df', 'annual_payload', 'analysis_signature']:
        if key in st.session_state:
            del st.session_state[key]


def sidebar_block(title: str):
    st.sidebar.markdown(f'### {title}')


@st.cache_data(show_spinner=False, max_entries=6)
def cached_load_dataframe(file_bytes: bytes, file_name: str, sheet_name: str | None):
    return load_dataset_from_bytes(file_bytes, file_name, sheet_name=sheet_name)


@st.cache_data(show_spinner=False, max_entries=10)
def cached_build_series(df: pd.DataFrame, date_col: str | None, value_col: str | None, resample_rule: str | None):
    return build_series_from_dataframe(df, date_col=date_col, value_col=value_col, resample_rule=resample_rule)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_fourier_analysis(series: pd.Series, sample_hours: float):
    return fourier_analysis(series, sample_hours=sample_hours)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_generated_series(result, k: int, resolution: str, total_days: int):
    return generate_harmonic_series(result, k=k, resolution=resolution, total_days=total_days)


def build_annual_summary(series_map: dict[int, pd.Series], sample_hours: float, compare_k_values: list[int], requested_k: int, k_mode: str, acf_horizon_hours: int) -> tuple[pd.DataFrame, dict[int, dict]]:
    rows = []
    payload = {}
    for year, year_series in series_map.items():
        res = cached_fourier_analysis(year_series, sample_hours)
        compare_df = compare_k_performance(res, [k for k in compare_k_values if k <= res.max_k] or [1])
        auto_k = select_best_k(compare_df)
        selected_k = auto_k if k_mode.startswith('Auto') else int(min(max(1, requested_k), res.max_k))
        reconstructed = reconstruct_from_top_k(res, selected_k)
        error_point_df, total_error_df, daily_error_df, monthly_error_df = compute_error_tables(res.series, reconstructed)
        residual_stats_df, residual_timeseries_df = compute_residual_outputs(res.series, reconstructed, infer_sample_hours(res.series.index), max_lag_hours=acf_horizon_hours)
        top_row = res.dominant_df.iloc[0] if not res.dominant_df.empty else None
        top_freq = float(top_row['Frequency (cycles/hour)']) if top_row is not None and 'Frequency (cycles/hour)' in top_row.index else float('nan')
        top_period = float(1.0 / top_freq) if pd.notna(top_freq) and top_freq > 0 else float('nan')
        rmse = float(total_error_df.iloc[0]['RMSE']) if not total_error_df.empty else float('nan')
        mae = float(total_error_df.iloc[0]['MAE']) if not total_error_df.empty else float('nan')
        r2 = float(total_error_df.iloc[0]['R2']) if not total_error_df.empty else float('nan')
        residual_energy_pct = float(residual_stats_df.loc[residual_stats_df['metric'] == 'residual_energy_pct', 'value'].iloc[0]) if 'residual_energy_pct' in residual_stats_df['metric'].values else float('nan')
        rows.append(
            {
                'year': int(year),
                'samples': int(res.stats['n']),
                'mean': float(res.stats['mean']),
                'std': float(res.stats['std']),
                'max_k': int(res.max_k),
                'auto_k': int(auto_k),
                'selected_k': int(selected_k),
                'top_frequency_cph': top_freq,
                'top_period_hours': top_period,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'residual_energy_pct': residual_energy_pct,
            }
        )
        payload[int(year)] = {
            'result': res,
            'compare_df': compare_df,
            'selected_k': selected_k,
            'reconstructed': reconstructed,
            'error_point_df': error_point_df,
            'total_error_df': total_error_df,
            'daily_error_df': daily_error_df,
            'monthly_error_df': monthly_error_df,
            'residual_stats_df': residual_stats_df,
            'residual_timeseries_df': residual_timeseries_df,
        }
    summary_df = pd.DataFrame(rows).sort_values('year').reset_index(drop=True) if rows else pd.DataFrame()
    return summary_df, payload


st.title('📈 Fourier Analysis Studio V4.4.2')
st.caption('Navigation simplified into fewer workspaces, with a cleaner sidebar and the glossary kept collapsible.')

with st.expander('Glossary', expanded=False):
    st.dataframe(pd.DataFrame(GLOSSARY, columns=['term', 'definition']), use_container_width=True, hide_index=True)

with st.sidebar:
    st.header('Configuration')
    sidebar_block('Data source')
    uploaded_file = st.file_uploader('Upload CSV or XLSX', type=['csv', 'xlsx'])
    unit = st.selectbox('Value unit', ['kW', 'MW', 'GW'], index=0, key='sidebar_unit')

    sidebar_block('Preparation')
    resample_rule = st.selectbox('Resample for analysis', ['None', '15min', '30min', '1h'], index=0, key='sidebar_resample')
    min_year_coverage = st.slider('Minimum annual coverage (%)', min_value=50, max_value=100, value=85, step=5, key='sidebar_min_coverage')
    preview_max_points = st.slider('Preview max points', min_value=500, max_value=20000, value=3000, step=500, key='sidebar_preview_max_points')
    acf_max_hours = st.slider('ACF horizon (hours)', min_value=24, max_value=720, value=336, step=24, key='sidebar_acf_hours')

    with st.expander('Fourier settings', expanded=True):
        k_mode = st.radio('K selection mode', ['Manual', 'Auto (best K)'], index=0, key='sidebar_k_mode')
        requested_k = st.number_input('Manual K', min_value=1, max_value=1000, value=20, step=1, key='sidebar_manual_k')
        compare_k_text = st.text_input('K values to compare', value='5, 10, 20, 50, 100, 200', key='sidebar_compare_k')
        max_overlay_k = st.slider('Max K lines in overlay', min_value=1, max_value=20, value=8, step=1, key='sidebar_max_overlay_k')
        preview_terms = st.slider('Displayed terms in harmonic preview', min_value=5, max_value=30, value=12, key='sidebar_preview_terms')

    with st.expander('Display and residuals', expanded=False):
        residual_top_events = st.slider('Top residual events', min_value=5, max_value=50, value=15, step=5, key='sidebar_residual_top_events')
        auto_round_low_freq = st.checkbox('Auto-round nearly flat low-frequency component', value=True, key='sidebar_auto_round_lf')
        low_freq_round_decimals = st.selectbox('Rounded decimals for display', [2, 4, 6, 8], index=2, key='sidebar_lf_decimals')
        low_freq_relative_threshold = st.selectbox('Flatness threshold (relative span)', [1e-3, 1e-4, 1e-5, 1e-6, 1e-8], index=3, format_func=lambda x: f'{x:.0e}', key='sidebar_lf_threshold')

    with st.expander('Synthetic series', expanded=False):
        simulation_resolution = st.selectbox('Synthetic resolution', ['15min', '30min', '1h', '6h', '1d'], index=2, key='sidebar_sim_resolution')
        simulation_days = st.number_input('Synthetic duration (days)', min_value=1, max_value=3650, value=30, step=1, key='sidebar_sim_days')

if uploaded_file is None:
    st.info('Upload a CSV or XLSX file to start the analysis.')
    st.stop()

file_bytes = uploaded_file.getvalue()
file_name = uploaded_file.name
sheet_name = None
if file_name.lower().endswith('.xlsx'):
    try:
        sheets = list_excel_sheets(uploaded_file)
    except Exception as e:
        st.error(f'Failed to inspect Excel sheets: {e}')
        st.stop()
    with st.sidebar:
        sheet_name = st.selectbox('Excel sheet', sheets, index=0, key='sidebar_sheet')

try:
    raw_df = cached_load_dataframe(file_bytes, file_name, sheet_name)
except Exception as e:
    logger.exception('Load failed')
    st.error(f'Failed to load file: {e}')
    st.stop()

st.subheader('Dataset preview')
show_rows = st.slider('Rows to preview', min_value=5, max_value=100, value=20, step=5, key='preview_rows_slider')
st.dataframe(format_display_df(raw_df.head(show_rows)), use_container_width=True)

use_manual_columns = st.checkbox('Use manual column selection', value=False, key='manual_col_select')
date_col = None
value_col = None
if use_manual_columns:
    date_col = st.selectbox('Datetime column', raw_df.columns.tolist(), index=0, key='manual_date_col')
    value_col = st.selectbox('Value column', raw_df.columns.tolist(), index=min(1, len(raw_df.columns) - 1), key='manual_value_col')

try:
    full_series = cached_build_series(raw_df, date_col, value_col, resample_rule)
    validation = validate_time_series(full_series)
except Exception as e:
    logger.exception('Validation failed')
    st.error(f'Parsing/validation error: {e}')
    st.stop()

if not is_regular_series(full_series.index):
    st.warning('The time series is not perfectly regular. Resampling is recommended for a more reliable FFT.')

st.markdown('## Initial analysis window')
period_col1, period_col2, period_col3 = st.columns([1.15, 1.0, 1.25])
with period_col1:
    period_mode = st.selectbox('Period selection mode', ['Full series', 'Custom date range', 'Single year'], index=0, key='initial_period_mode')
with period_col2:
    available_years = sorted(full_series.index.year.unique().tolist())
    year_option = st.selectbox('Year for quick analysis', available_years, index=0 if available_years else None, key='initial_year_select', disabled=(period_mode != 'Single year'))
with period_col3:
    min_date = full_series.index.min().date()
    max_date = full_series.index.max().date()
    selected_range = st.date_input('Custom date range', value=(min_date, max_date), min_value=min_date, max_value=max_date, key='initial_date_range', disabled=(period_mode != 'Custom date range'))
analysis_scope = 'Global'

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
elif isinstance(selected_range, list) and len(selected_range) == 2:
    start_date, end_date = selected_range[0], selected_range[1]
else:
    start_date, end_date = min_date, max_date

analysis_series, selection_label = filter_series_by_period(full_series, period_mode, start_date, end_date, year_option if period_mode == 'Single year' else None)
if analysis_series.empty:
    st.error('The selected period is empty. Choose another date range or year.')
    st.stop()
series = analysis_series
sample_hours = infer_sample_hours(series.index)

analysis_signature = (file_name, sheet_name, analysis_scope, selection_label, len(series), str(series.index.min()), str(series.index.max()), resample_rule, unit, min_year_coverage)
if st.session_state.get('analysis_signature') not in (None, analysis_signature):
    clear_analysis_state()
st.session_state['analysis_signature'] = analysis_signature

st.markdown('## Initial diagnostics')
summary_df = compute_series_summary(series, sample_hours, selection_label)
metric_map = {row['metric']: row['value'] for _, row in summary_df.iterrows()}

m1, m2, m3, m4 = st.columns(4)
m1.metric('Selection', str(metric_map.get('selection', 'n/a')))
m2.metric('Samples', f"{int(metric_map.get('samples', 0)):,}")
m3.metric('Sampling step', f"{float(metric_map.get('sampling_step_hours', 0.0)):.2f} h")
m4.metric('Detected years', str(metric_map.get('detected_years', 0)))

m5, m6, m7, m8 = st.columns(4)
m5.metric('Duration (days)', f"{float(metric_map.get('duration_days', 0.0)):.1f}")
m6.metric('Missing values', str(metric_map.get('missing_values', 0)))
m7.metric('Estimated completeness', f"{float(metric_map.get('estimated_completeness_pct', 0.0)):.1f}%")
m8.metric('Regular index', 'Yes' if bool(metric_map.get('is_regular', False)) else 'No')

with st.expander('Detailed series statistics', expanded=False):
    st.dataframe(format_display_df(summary_df, decimals=4), use_container_width=True)

preview_tabs = st.tabs(['Preview', 'Distribution', 'Coverage'])
with preview_tabs[0]:
    st.plotly_chart(build_series_preview_figure(series, unit, preview_max_points), use_container_width=True, key=chart_key('preview', 'time'))
with preview_tabs[1]:
    st.plotly_chart(build_distribution_figure(maybe_downsample_series(series, preview_max_points), unit), use_container_width=True, key=chart_key('preview', 'distribution'))
with preview_tabs[2]:
    coverage_initial_df = build_year_coverage_table(series, sample_hours, min_year_coverage)
    if coverage_initial_df.empty:
        st.info('No yearly coverage table is available for the selected period.')
    else:
        st.dataframe(format_display_df(coverage_initial_df, decimals=2), use_container_width=True)
        st.caption(f"Eligible years in current selection: {int(coverage_initial_df['eligible'].sum())}/{len(coverage_initial_df)} at threshold {min_year_coverage}%.")

compare_k_values = parse_k_list(compare_k_text, max_k=max(1, len(series) // 2))
if not compare_k_values:
    compare_k_values = [5, 10, 20]

simulation_resolution = normalize_frequency_alias(simulation_resolution)
try:
    validate_simulation_inputs(simulation_resolution, int(simulation_days))
except Exception as e:
    st.error(f'Simulation configuration error: {e}')
    st.stop()

if analysis_scope == 'Annual':
    with st.expander('Annual coverage preview', expanded=False):
        coverage_preview_df = build_year_coverage_table(series, sample_hours, min_year_coverage)
        st.write('Years are eligible only if their valid data coverage reaches the selected threshold within the current analysis window.')
        st.dataframe(format_display_df(coverage_preview_df, decimals=2), use_container_width=True)

run_fourier = st.button('Run Fourier analysis', type='primary', key='run_fourier_button')
if run_fourier:
    st.session_state['analysis_scope'] = analysis_scope
    if analysis_scope == 'Global':
        st.session_state['fourier_result'] = cached_fourier_analysis(series, sample_hours)
    else:
        yearly_map, coverage_df = build_yearly_series_map(series, sample_hours, min_year_coverage)
        summary_annual_df, annual_payload = build_annual_summary(yearly_map, sample_hours, compare_k_values, int(requested_k), k_mode, acf_max_hours)
        st.session_state['annual_coverage_df'] = coverage_df
        st.session_state['annual_summary_df'] = summary_annual_df
        st.session_state['annual_payload'] = annual_payload

if analysis_scope == 'Global' and 'fourier_result' not in st.session_state:
    st.info('Preview the selected period and diagnostics first, then click Run Fourier analysis.')
    st.stop()
if analysis_scope == 'Annual' and 'annual_summary_df' not in st.session_state:
    st.info('Preview the selected period and diagnostics first, then click Run Fourier analysis.')
    st.stop()

st.caption(
    f"Validation: monotonic={validation['is_monotonic']}, unique_index={validation['has_unique_index']}, missing={validation['missing_values']}, regular={validation['is_regular']} | Analysis scope: {analysis_scope} | Selection: {selection_label}"
)

if analysis_scope == 'Annual':
    annual_summary_df = st.session_state.get('annual_summary_df', pd.DataFrame())
    annual_payload = st.session_state.get('annual_payload', {})
    coverage_df = st.session_state.get('annual_coverage_df', pd.DataFrame())
    if annual_summary_df.empty:
        st.warning('No year met the minimum coverage threshold for annual analysis in the selected period.')
        st.stop()

    st.subheader('Annual Fourier analysis')
    if not coverage_df.empty:
        with st.expander('Year eligibility and coverage', expanded=False):
            st.dataframe(format_display_df(coverage_df, decimals=2), use_container_width=True)
            eligible_years = coverage_df.loc[coverage_df['eligible'], 'year'].astype(int).tolist()
            if len(eligible_years) == 1:
                st.info(f'Only one year ({eligible_years[0]}) meets the current coverage threshold inside the selected period.')
            elif len(eligible_years) == 0:
                st.warning('No year meets the current coverage threshold in the selected period.')

    st.dataframe(format_display_df(annual_summary_df), use_container_width=True)
    annual_csv = annual_summary_df.to_csv(index=False).encode('utf-8')
    selected_year = st.selectbox('Year to inspect', annual_summary_df['year'].astype(int).tolist(), index=0, key='annual_year_to_inspect')
    current = annual_payload[int(selected_year)]
    result = current['result']
    compare_k_df = current['compare_df']
    selected_k = current['selected_k']
    reconstructed_series = current['reconstructed']
    error_point_df = current['error_point_df']
    total_error_df = current['total_error_df']
    daily_error_df = current['daily_error_df']
    monthly_error_df = current['monthly_error_df']
    residual_stats_df = current['residual_stats_df']
    residual_timeseries_df = current['residual_timeseries_df']

    harmonic_df = build_harmonic_dataframe(result, selected_k)
    harmonic_preview_text, harmonic_full_text = build_harmonic_function_text(result, selected_k, preview_terms=preview_terms)
    synthetic_df = cached_generated_series(result, selected_k, simulation_resolution, int(simulation_days))
    interpretation = build_interpretation(result, unit, selected_k, total_error_df)
    component_df, component_summary_df, component_notes = build_temporal_components(result)
    component_plot_df, rounded_component_cols = prepare_component_display_df(component_df, auto_round_low_freq, int(low_freq_round_decimals), float(low_freq_relative_threshold))
    overlay_k_values = [k for k in compare_k_values if k <= result.max_k][:max_overlay_k] or [1]

    fig_overview = build_series_preview_figure(result.series, unit, preview_max_points)
    fig_dist = build_distribution_figure(maybe_downsample_series(result.series, preview_max_points), unit)
    fig_spectrum = build_spectrum_figure(result, unit)
    fig_band = build_band_energy_figure(result)
    fig_recon = build_reconstruction_figure(result.series, reconstructed_series, unit, selected_k)
    fig_acf = build_acf_figure(result, acf_max_hours)
    fig_cumvar = build_cumvar_figure(result)
    fig_k_compare = build_k_comparison_figure(compare_k_df)
    fig_multi_k = build_multi_k_reconstruction_figure(result.series, result, overlay_k_values, unit)
    fig_components = build_temporal_components_figure(component_plot_df, unit)
    fig_component_energy = build_component_energy_figure(component_summary_df)
    fig_component_window = build_component_window_figure(component_df=component_plot_df, unit=unit, start_ts=component_plot_df.index.min(), horizon_label='7D', mode='overlay', show_original=True)
    fig_cumulative_components = build_cumulative_components_figure(component_df=component_plot_df, unit=unit, start_ts=component_plot_df.index.min(), horizon_label='7D', show_original=True, show_cumulative_lines=False)
    fig_residual_time = build_residual_time_figure(residual_timeseries_df, unit, preview_max_points)
    fig_residual_dist = build_residual_distribution_figure(residual_timeseries_df, unit)
    fig_residual_heat = build_residual_heatmap_figure(residual_timeseries_df)
    fig_residual_acf = build_residual_acf_figure(residual_timeseries_df, infer_sample_hours(result.series.index), acf_max_hours)
    fig_error_ts = build_error_timeseries_figure(error_point_df)
    fig_error_pct = build_error_percentage_figure(error_point_df)
    fig_error_daily = build_daily_error_figure(daily_error_df)
    fig_error_monthly = build_monthly_error_figure(monthly_error_df)
    fig_generated = build_generated_series_figure(synthetic_df, unit)
    top_residual_events_df = build_top_residual_events(residual_timeseries_df, residual_top_events)

    tabs = st.tabs(['Overview', 'Frequency & Reconstruction', 'Components & Residuals', 'Exports'])
    with tabs[0]:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Selected year', str(selected_year))
        c2.metric('Samples', f"{int(result.stats['n']):,}")
        c3.metric('Mean', f"{result.stats['mean']:.2f} {unit}")
        c4.metric('Selected K', str(selected_k))
        c5.metric('Residual energy', f"{float(residual_stats_df.loc[residual_stats_df['metric']=='residual_energy_pct','value'].iloc[0]):.2f}%")
        st.plotly_chart(fig_overview, use_container_width=True, key=chart_key('annual', 'overview', selected_year))
        st.plotly_chart(fig_dist, use_container_width=True, key=chart_key('annual', 'distribution', selected_year))
        st.write(interpretation['summary'])
        with st.expander('Detailed series statistics', expanded=False):
            st.dataframe(format_display_df(pd.DataFrame({'timestamp': result.series.index, 'value': result.series.values}).head(200)), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(fig_spectrum, use_container_width=True, key=chart_key('annual', 'spectrum', selected_year))
        st.plotly_chart(fig_band, use_container_width=True, key=chart_key('annual', 'band_energy', selected_year))
        st.plotly_chart(fig_recon, use_container_width=True, key=chart_key('annual', 'reconstruction', selected_year))
        st.plotly_chart(fig_acf, use_container_width=True, key=chart_key('annual', 'acf', selected_year))
        st.plotly_chart(fig_cumvar, use_container_width=True, key=chart_key('annual', 'cumvar', selected_year))
        st.plotly_chart(fig_k_compare, use_container_width=True, key=chart_key('annual', 'kcompare', selected_year))
        st.plotly_chart(fig_multi_k, use_container_width=True, key=chart_key('annual', 'multik', selected_year))
        with st.expander('Dominant frequencies and harmonic function', expanded=False):
            st.dataframe(format_display_df(result.dominant_df.head(30)), use_container_width=True)
            st.code(harmonic_preview_text, language='text')
            st.dataframe(format_display_df(harmonic_df), use_container_width=True, height=320)
    with tabs[2]:
        if rounded_component_cols:
            st.info(f'Low-frequency display rounding applied for readability on: {", ".join(rounded_component_cols)} (display only, exports keep original values).')
        st.write(component_notes['summary'])
        st.plotly_chart(fig_components, use_container_width=True, key=chart_key('annual', 'components', selected_year))
        st.plotly_chart(fig_component_energy, use_container_width=True, key=chart_key('annual', 'component_energy', selected_year))
        st.plotly_chart(fig_component_window, use_container_width=True, key=chart_key('annual', 'component_window', selected_year))
        st.plotly_chart(fig_cumulative_components, use_container_width=True, key=chart_key('annual', 'component_cumulative', selected_year))
        st.markdown('#### Residual diagnostics')
        st.dataframe(format_display_df(residual_stats_df, decimals=4), use_container_width=True)
        st.plotly_chart(fig_residual_time, use_container_width=True, key=chart_key('annual', 'residual_time', selected_year))
        st.plotly_chart(fig_residual_dist, use_container_width=True, key=chart_key('annual', 'residual_distribution', selected_year))
        st.plotly_chart(fig_residual_acf, use_container_width=True, key=chart_key('annual', 'residual_acf', selected_year))
        st.plotly_chart(fig_residual_heat, use_container_width=True, key=chart_key('annual', 'residual_heat', selected_year))
        with st.expander('Error analysis and top residual events', expanded=False):
            st.plotly_chart(fig_error_ts, use_container_width=True, key=chart_key('annual', 'errorts', selected_year))
            st.plotly_chart(fig_error_pct, use_container_width=True, key=chart_key('annual', 'errorpct', selected_year))
            st.plotly_chart(fig_error_daily, use_container_width=True, key=chart_key('annual', 'errordaily', selected_year))
            st.plotly_chart(fig_error_monthly, use_container_width=True, key=chart_key('annual', 'errormonthly', selected_year))
            st.dataframe(format_display_df(top_residual_events_df, decimals=4), use_container_width=True)
    with tabs[3]:
        figures = {
            'Overview': fig_overview,
            'Spectrum': fig_spectrum,
            'Reconstruction': fig_recon,
            'Components': fig_components,
            'Residuals': fig_residual_time,
            'Synthetic Series': fig_generated,
        }
        try:
            pdf_bytes = generate_pdf_report(result, unit, f'{file_name} - {selected_year}', selected_k, total_error_df, harmonic_preview_text, figures, compare_k_df)
        except Exception:
            pdf_bytes = None
        csv_files = generate_csv_exports(result, reconstructed_series, harmonic_df, error_point_df, total_error_df, daily_error_df, monthly_error_df, compare_k_df, synthetic_df, component_df=component_df)
        csv_files['annual_summary.csv'] = annual_csv
        csv_files['residual_diagnostics.csv'] = residual_stats_df.to_csv(index=False).encode('utf-8')
        csv_files['residual_timeseries.csv'] = residual_timeseries_df.to_csv(index=False).encode('utf-8')
        txt_files = {'harmonic_function_full.txt': harmonic_full_text.encode('utf-8'), 'harmonic_function_preview.txt': harmonic_preview_text.encode('utf-8')}
        zip_bytes = generate_zip_export(pdf_bytes or b'', csv_files, txt_files, figures)
        st.download_button('Download annual summary CSV', annual_csv, 'annual_fourier_summary.csv', 'text/csv', use_container_width=True, key=download_key('annual', 'summary_tab', selected_year))
        if pdf_bytes is not None:
            st.download_button('Download selected year PDF', pdf_bytes, f'Fourier_Analysis_Report_V4_4_2_{selected_year}.pdf', 'application/pdf', use_container_width=True, key=download_key('annual', 'pdf', selected_year))
        st.download_button('Download selected year ZIP package', zip_bytes, f'fourier_analysis_v4_4_2_year_{selected_year}.zip', 'application/zip', use_container_width=True, key=download_key('annual', 'zip', selected_year))
        with st.expander('Additional exports', expanded=False):
            st.download_button('Download harmonic table CSV', harmonic_df.to_csv(index=False).encode('utf-8'), f'harmonics_{selected_year}.csv', 'text/csv', use_container_width=True, key=download_key('annual', 'harmonics', selected_year))
            st.download_button('Download synthetic series CSV', synthetic_df.to_csv(index=False).encode('utf-8'), f'synthetic_harmonic_series_{selected_year}.csv', 'text/csv', use_container_width=True, key=download_key('annual', 'synthetic', selected_year))
            st.download_button('Download residual diagnostics CSV', residual_stats_df.to_csv(index=False).encode('utf-8'), f'residual_stats_{selected_year}.csv', 'text/csv', use_container_width=True, key=download_key('annual', 'residual_stats', selected_year))

else:
    result = st.session_state['fourier_result']
    compare_k_values = parse_k_list(compare_k_text, max_k=result.max_k)
    if not compare_k_values:
        compare_k_values = [k for k in [5, 10, 20] if k <= result.max_k] or [1]
    overlay_k_values = compare_k_values[:max_overlay_k]
    compare_k_df = compare_k_performance(result, compare_k_values)
    auto_k = select_best_k(compare_k_df)
    selected_k = auto_k if k_mode.startswith('Auto') else int(min(max(1, requested_k), result.max_k))
    reconstructed_series = reconstruct_from_top_k(result, selected_k)
    harmonic_df = build_harmonic_dataframe(result, selected_k)
    harmonic_preview_text, harmonic_full_text = build_harmonic_function_text(result, selected_k, preview_terms=preview_terms)
    error_point_df, total_error_df, daily_error_df, monthly_error_df = compute_error_tables(result.series, reconstructed_series)
    synthetic_df = cached_generated_series(result, selected_k, simulation_resolution, int(simulation_days))
    interpretation = build_interpretation(result, unit, selected_k, total_error_df)
    component_df, component_summary_df, component_notes = build_temporal_components(result)
    component_plot_df, rounded_component_cols = prepare_component_display_df(component_df, auto_round_low_freq, int(low_freq_round_decimals), float(low_freq_relative_threshold))
    residual_stats_df, residual_timeseries_df = compute_residual_outputs(result.series, reconstructed_series, sample_hours, acf_max_hours)

    fig_overview = build_series_preview_figure(result.series, unit, preview_max_points)
    fig_dist = build_distribution_figure(maybe_downsample_series(result.series, preview_max_points), unit)
    fig_spectrum = build_spectrum_figure(result, unit)
    fig_band = build_band_energy_figure(result)
    fig_recon = build_reconstruction_figure(result.series, reconstructed_series, unit, selected_k)
    fig_acf = build_acf_figure(result, acf_max_hours)
    fig_cumvar = build_cumvar_figure(result)
    fig_k_compare = build_k_comparison_figure(compare_k_df)
    fig_multi_k = build_multi_k_reconstruction_figure(result.series, result, overlay_k_values, unit)
    fig_generated = build_generated_series_figure(synthetic_df, unit)
    fig_components = build_temporal_components_figure(component_plot_df, unit)
    fig_component_energy = build_component_energy_figure(component_summary_df)
    fig_residual_time = build_residual_time_figure(residual_timeseries_df, unit, preview_max_points)
    fig_residual_dist = build_residual_distribution_figure(residual_timeseries_df, unit)
    fig_residual_heat = build_residual_heatmap_figure(residual_timeseries_df)
    fig_residual_acf = build_residual_acf_figure(residual_timeseries_df, sample_hours, acf_max_hours)
    fig_error_ts = build_error_timeseries_figure(error_point_df)
    fig_error_pct = build_error_percentage_figure(error_point_df)
    fig_error_daily = build_daily_error_figure(daily_error_df)
    fig_error_monthly = build_monthly_error_figure(monthly_error_df)
    top_residual_events_df = build_top_residual_events(residual_timeseries_df, residual_top_events)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric('Samples', f"{int(result.stats['n']):,}")
    c2.metric('Mean', f"{result.stats['mean']:.2f} {unit}")
    c3.metric('Std', f"{result.stats['std']:.2f} {unit}")
    c4.metric('Sampling step', f'{sample_hours:.2f} h')
    c5.metric('Selected K', str(selected_k))
    c6.metric('Residual energy', f"{float(residual_stats_df.loc[residual_stats_df['metric']=='residual_energy_pct','value'].iloc[0]):.2f}%")

    tabs = st.tabs(['Overview', 'Frequency & Reconstruction', 'Components & Residuals', 'Exports'])
    with tabs[0]:
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(fig_overview, use_container_width=True, key=chart_key('global', 'overview'))
            st.plotly_chart(fig_dist, use_container_width=True, key=chart_key('global', 'distribution'))
        with right:
            st.write(interpretation['summary'])
            st.write(interpretation['acf'])
            with st.expander('Quick sample values', expanded=False):
                st.dataframe(format_display_df(pd.DataFrame({'timestamp': result.series.index, 'value': result.series.values}).head(200)), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(fig_spectrum, use_container_width=True, key=chart_key('global', 'spectrum'))
        st.plotly_chart(fig_band, use_container_width=True, key=chart_key('global', 'band_energy'))
        st.plotly_chart(fig_recon, use_container_width=True, key=chart_key('global', 'reconstruction'))
        st.plotly_chart(fig_acf, use_container_width=True, key=chart_key('global', 'acf'))
        st.plotly_chart(fig_cumvar, use_container_width=True, key=chart_key('global', 'cumvar'))
        st.plotly_chart(fig_k_compare, use_container_width=True, key=chart_key('global', 'kcompare'))
        st.plotly_chart(fig_multi_k, use_container_width=True, key=chart_key('global', 'multik'))
        with st.expander('Dominant frequencies and harmonic function', expanded=False):
            st.dataframe(format_display_df(result.dominant_df.head(30)), use_container_width=True)
            st.dataframe(format_display_df(result.band_energy_df), use_container_width=True)
            st.code(harmonic_preview_text, language='text')
            st.dataframe(format_display_df(harmonic_df), use_container_width=True, height=320)
    with tabs[2]:
        st.write(component_notes['summary'])
        if rounded_component_cols:
            st.info(f'Low-frequency display rounding applied for readability on: {", ".join(rounded_component_cols)} (display only, exports keep original values).')
        st.markdown('#### Temporal components decomposition')
        st.plotly_chart(fig_components, use_container_width=True, key=chart_key('global', 'components'))
        st.markdown('#### Variance share by temporal component')
        st.plotly_chart(fig_component_energy, use_container_width=True, key=chart_key('global', 'component_energy'))
        st.markdown('#### Temporal components window settings')
        st.caption('These controls affect only the window charts shown immediately below.')
        cwin1, cwin2, cwin3 = st.columns(3)
        with cwin1:
            window_mode = st.radio('Window chart mode', ['overlay', 'stacked'], horizontal=True, index=0, key='global_window_mode')
        with cwin2:
            horizon_label = st.selectbox('Window horizon', ['1D', '7D', '30D'], index=1, key='global_window_horizon')
        with cwin3:
            show_original = st.checkbox('Show original series', value=True, key='global_window_show_original')
        horizon_delta_map = {'1D': pd.Timedelta(days=1), '7D': pd.Timedelta(days=7), '30D': pd.Timedelta(days=30)}
        horizon_delta = horizon_delta_map[horizon_label]
        start_min = component_plot_df.index.min().to_pydatetime()
        start_max_ts = component_plot_df.index.max() - horizon_delta
        if start_max_ts < component_plot_df.index.min():
            start_max_ts = component_plot_df.index.min()
        selected_start = st.slider('Window start', min_value=start_min, max_value=start_max_ts.to_pydatetime(), value=start_min, format='YYYY-MM-DD HH:mm', key='global_window_start')
        show_cumulative_lines = st.checkbox('Show cumulative boundary lines', value=False, key='global_show_cumulative_lines')
        fig_component_window = build_component_window_figure(component_df=component_plot_df, unit=unit, start_ts=selected_start, horizon_label=horizon_label, mode=window_mode, show_original=show_original)
        fig_cumulative_components = build_cumulative_components_figure(component_df=component_plot_df, unit=unit, start_ts=selected_start, horizon_label=horizon_label, show_original=show_original, show_cumulative_lines=show_cumulative_lines)
        st.markdown('#### Temporal components window')
        st.plotly_chart(fig_component_window, use_container_width=True, key=chart_key('global', 'component_window'))
        st.plotly_chart(fig_cumulative_components, use_container_width=True, key=chart_key('global', 'component_cumulative'))
        st.markdown('#### Residual diagnostics')
        st.dataframe(format_display_df(residual_stats_df, decimals=4), use_container_width=True)
        st.plotly_chart(fig_residual_time, use_container_width=True, key=chart_key('global', 'residual_time'))
        st.plotly_chart(fig_residual_dist, use_container_width=True, key=chart_key('global', 'residual_distribution'))
        st.plotly_chart(fig_residual_acf, use_container_width=True, key=chart_key('global', 'residual_acf'))
        st.plotly_chart(fig_residual_heat, use_container_width=True, key=chart_key('global', 'residual_heat'))
        with st.expander('Error analysis and top residual events', expanded=False):
            st.plotly_chart(fig_error_ts, use_container_width=True, key=chart_key('global', 'errorts'))
            st.plotly_chart(fig_error_pct, use_container_width=True, key=chart_key('global', 'errorpct'))
            st.plotly_chart(fig_error_daily, use_container_width=True, key=chart_key('global', 'errordaily'))
            st.plotly_chart(fig_error_monthly, use_container_width=True, key=chart_key('global', 'errormonthly'))
            st.dataframe(format_display_df(top_residual_events_df, decimals=4), use_container_width=True)
    with tabs[3]:
        figures = {
            'Overview': fig_overview,
            'Spectrum': fig_spectrum,
            'Reconstruction': fig_recon,
            'Components': fig_components,
            'Residuals': fig_residual_time,
            'Synthetic Series': fig_generated,
        }
        pdf_error = None
        try:
            pdf_bytes = generate_pdf_report(result, unit, file_name, selected_k, total_error_df, harmonic_preview_text, figures, compare_k_df)
        except Exception as e:
            pdf_bytes = None
            pdf_error = str(e)
        csv_files = generate_csv_exports(result, reconstructed_series, harmonic_df, error_point_df, total_error_df, daily_error_df, monthly_error_df, compare_k_df, synthetic_df, component_df=component_df)
        csv_files['residual_diagnostics.csv'] = residual_stats_df.to_csv(index=False).encode('utf-8')
        csv_files['residual_timeseries.csv'] = residual_timeseries_df.to_csv(index=False).encode('utf-8')
        txt_files = {'harmonic_function_full.txt': harmonic_full_text.encode('utf-8'), 'harmonic_function_preview.txt': harmonic_preview_text.encode('utf-8')}
        zip_bytes = generate_zip_export(pdf_bytes or b'', csv_files, txt_files, figures)
        if pdf_bytes is not None:
            st.download_button('Download PDF report', pdf_bytes, 'Fourier_Analysis_Report_V4_4_2.pdf', 'application/pdf', use_container_width=True, key=download_key('global', 'pdf'))
        else:
            st.warning('PDF report with embedded chart images is not available in this environment.')
            if pdf_error:
                st.caption(pdf_error)
        st.download_button('Download ZIP package', zip_bytes, 'fourier_analysis_v4_4_2_outputs.zip', 'application/zip', use_container_width=True, key=download_key('global', 'zip'))
        with st.expander('Additional exports', expanded=False):
            st.download_button('Download K comparison CSV', csv_files['k_comparison.csv'], 'k_comparison.csv', 'text/csv', use_container_width=True, key=download_key('global', 'kcomp_csv'))
            st.download_button('Download residual diagnostics CSV', residual_stats_df.to_csv(index=False).encode('utf-8'), 'residual_diagnostics.csv', 'text/csv', use_container_width=True, key=download_key('global', 'residuals_csv'))
            st.download_button('Download synthetic series CSV', synthetic_df.to_csv(index=False).encode('utf-8'), 'synthetic_harmonic_series.csv', 'text/csv', use_container_width=True, key=download_key('global', 'synthetic_csv'))
            st.download_button('Download full harmonic function', txt_files['harmonic_function_full.txt'], 'harmonic_function_full.txt', 'text/plain', use_container_width=True, key=download_key('global', 'harmonic_full_txt'))
