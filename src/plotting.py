from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.fourier_core import AnalysisResult, infer_sample_hours, reconstruct_from_top_k

KNOWN_FREQS = {1 / 8760: 'Yearly', 1 / 168: 'Weekly', 1 / 24: '24h', 1 / 12: '12h'}


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        .main {background: linear-gradient(180deg, #f6f8fb 0%, #f3f6f9 100%);}
        .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1450px;}
        h1, h2, h3 {color: #16324f;}
        [data-testid='stMetricValue'] {color: #0b6e4f;}
        .small-note {font-size: 0.9rem; color: #4f6d7a;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_time_figure(series: pd.Series, unit: str) -> go.Figure:
    sample_hours = infer_sample_hours(series.index)
    rolling_window = max(3, int(round(168 / sample_hours)))
    rolling = series.rolling(rolling_window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name='Raw series', line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode='lines', name='Rolling mean', line=dict(color='#ff7f0e', width=2)))
    fig.update_layout(title='Time Series Overview', xaxis_title='Timestamp', yaxis_title=f'Value ({unit})', template='plotly_white', height=420)
    return fig


def build_distribution_figure(series: pd.Series, unit: str) -> go.Figure:
    fig = px.histogram(x=series.values, nbins=50, template='plotly_white', title='Distribution of Values', labels={'x': f'Value ({unit})', 'y': 'Count'})
    fig.add_vline(x=float(series.mean()), line_dash='dash', line_color='green')
    fig.add_vline(x=float(series.median()), line_dash='dot', line_color='red')
    fig.update_layout(height=360)
    return fig


def build_spectrum_figure(res: AnalysisResult, unit: str) -> go.Figure:
    mask = res.freqs > 0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=('Amplitude Spectrum', 'Welch PSD'))
    fig.add_trace(go.Scatter(x=res.freqs[mask], y=res.amps[mask], mode='lines', name='Amplitude', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.f_welch[res.f_welch > 0], y=res.psd_welch[res.f_welch > 0], mode='lines', name='Welch PSD', line=dict(color='#ff7f0e')), row=2, col=1)
    ymax = float(np.nanmax(res.amps[mask])) if np.any(mask) else 1.0
    for f_ref, label in KNOWN_FREQS.items():
        fig.add_vline(x=f_ref, line_dash='dot', line_color='gray', row=1, col=1)
        fig.add_annotation(x=f_ref, y=ymax, text=label, showarrow=False, yshift=10, font=dict(size=10), row=1, col=1)
    fig.update_yaxes(title_text=f'Amplitude ({unit})', row=1, col=1)
    fig.update_yaxes(title_text='PSD', type='log', row=2, col=1)
    fig.update_xaxes(title_text='Frequency (cycles/hour)', row=2, col=1)
    fig.update_layout(template='plotly_white', height=680, title='Frequency-Domain Analysis')
    return fig


def build_band_energy_figure(res: AnalysisResult) -> go.Figure:
    plot_df = res.band_energy_df.copy()
    fig = px.bar(
        plot_df,
        x='Energy Share (%)',
        y='Band',
        orientation='h',
        template='plotly_white',
        title='Spectral Energy by Frequency Band',
        text='Energy Share (%)',
    )
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside',
        cliponaxis=False,
        hovertemplate='%{y}<br>Energy Share: %{x:.2f}%<extra></extra>',
    )
    fig.update_layout(height=380, yaxis=dict(categoryorder='total ascending'))
    return fig


def build_reconstruction_figure(actual: pd.Series, reconstructed: pd.Series, unit: str, k: int) -> go.Figure:
    n_show = min(1500, len(actual))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index[:n_show], y=actual.iloc[:n_show], mode='lines', name='Original', line=dict(color='#6c8ebf', width=1)))
    fig.add_trace(go.Scatter(x=reconstructed.index[:n_show], y=reconstructed.iloc[:n_show], mode='lines', name=f'Fourier reconstruction (K={k})', line=dict(color='#0b6e4f', width=2)))
    fig.update_layout(title=f'Signal Reconstruction with K = {k}', xaxis_title='Timestamp', yaxis_title=f'Value ({unit})', template='plotly_white', height=420)
    return fig


def build_acf_figure(res: AnalysisResult, max_hours: int) -> go.Figure:
    max_lag = min(len(res.autocorr), int(max_hours / max(res.sample_hours, 1e-9)) + 1)
    lags = np.arange(max_lag) * res.sample_hours
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags, y=res.autocorr[:max_lag], mode='lines', name='ACF', line=dict(color='#7a3e9d', width=1.5)))
    fig.update_layout(title='Autocorrelation Function', xaxis_title='Lag (hours)', yaxis_title='Autocorrelation', template='plotly_white', height=360)
    return fig


def build_cumvar_figure(res: AnalysisResult) -> go.Figure:
    k = np.arange(1, min(len(res.cumvar), 200) + 1)
    y = res.cumvar[:200]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=y, mode='lines', name='Cumulative variance', line=dict(color='#0b6e4f', width=2)))
    for thr in [90, 95, 99]:
        fig.add_hline(y=thr, line_dash='dash', line_color='gray')
    fig.update_layout(title='Cumulative Explained Variance', xaxis_title='Number of harmonics (K)', yaxis_title='Explained variance (%)', template='plotly_white', height=360)
    return fig


def build_error_timeseries_figure(error_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, subplot_titles=('Pointwise Error', 'Absolute Error'))
    fig.add_trace(go.Scatter(x=error_df['timestamp'], y=error_df['error'], mode='lines', name='Error', line=dict(color='#b22222', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=error_df['timestamp'], y=error_df['abs_error'], mode='lines', name='Absolute error', line=dict(color='#ff8c00', width=1)), row=2, col=1)
    fig.update_layout(template='plotly_white', height=620, title='Reconstruction Error Over Time')
    return fig


def build_error_percentage_figure(error_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, subplot_titles=('Signed Percentage Error', 'Absolute Percentage Error'))
    fig.add_trace(go.Scatter(x=error_df['timestamp'], y=error_df['error_pct'], mode='lines', name='Error (%)', line=dict(color='#8b0000', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=error_df['timestamp'], y=error_df['abs_error_pct'], mode='lines', name='Absolute Error (%)', line=dict(color='#ff8c00', width=1)), row=2, col=1)
    fig.update_layout(template='plotly_white', height=620, title='Percentage Error Over Time')
    return fig


def build_daily_error_figure(daily_error_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_error_df['timestamp'], y=daily_error_df['RMSE'], mode='lines', name='Daily RMSE', line=dict(color='#1f77b4', width=1.5)))
    fig.add_trace(go.Scatter(x=daily_error_df['timestamp'], y=daily_error_df['wMAPE (%)'], mode='lines', name='Daily wMAPE (%)', line=dict(color='#2ca02c', width=1.5), yaxis='y2'))
    fig.update_layout(template='plotly_white', height=380, title='Daily Error Metrics', yaxis=dict(title='RMSE'), yaxis2=dict(title='wMAPE (%)', overlaying='y', side='right'))
    return fig


def build_monthly_error_figure(monthly_error_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_error_df['timestamp'], y=monthly_error_df['MAE'], name='Monthly MAE', marker_color='#1f77b4'))
    fig.add_trace(go.Scatter(x=monthly_error_df['timestamp'], y=monthly_error_df['RMSE'], mode='lines+markers', name='Monthly RMSE', line=dict(color='#d62728', width=2)))
    fig.update_layout(template='plotly_white', height=380, title='Monthly Error Metrics')
    return fig


def build_k_comparison_figure(compare_k_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=compare_k_df['K'], y=compare_k_df['RMSE'], mode='lines+markers', name='RMSE', line=dict(color='#1f77b4')), secondary_y=False)
    fig.add_trace(go.Scatter(x=compare_k_df['K'], y=compare_k_df['R2'], mode='lines+markers', name='R2', line=dict(color='#2ca02c')), secondary_y=True)
    fig.update_layout(template='plotly_white', height=380, title='K Comparison')
    fig.update_yaxes(title_text='RMSE', secondary_y=False)
    fig.update_yaxes(title_text='R2', secondary_y=True)
    return fig


def build_multi_k_reconstruction_figure(actual: pd.Series, result: AnalysisResult, k_values: list[int], unit: str) -> go.Figure:
    n_show = min(1000, len(actual))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index[:n_show], y=actual.iloc[:n_show], mode='lines', name='Original', line=dict(color='black', width=1)))
    palette = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2']
    for i, k in enumerate(k_values):
        recon = reconstruct_from_top_k(result, int(k))
        fig.add_trace(go.Scatter(x=recon.index[:n_show], y=recon.iloc[:n_show], mode='lines', name=f'K={k}', line=dict(color=palette[i % len(palette)], width=1.5)))
    fig.update_layout(template='plotly_white', height=420, title='Multi-K Reconstruction Overlay', yaxis_title=f'Value ({unit})')
    return fig


def build_generated_series_figure(synthetic_df: pd.DataFrame, unit: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=synthetic_df['timestamp'], y=synthetic_df['harmonic_value'], mode='lines', name='Synthetic harmonic series', line=dict(color='#0b6e4f', width=2)))
    fig.update_layout(template='plotly_white', height=380, title='Synthetic Series Generated from Harmonic Model', yaxis_title=f'Value ({unit})')
    return fig


def build_temporal_components_figure(component_df: pd.DataFrame, unit: str) -> go.Figure:
    ordered = ['original', 'trend_low_freq', 'annual', 'seasonal', 'weekly', 'daily', 'residual']
    labels = {
        'original': 'Original',
        'trend_low_freq': 'Trend / low frequency',
        'annual': 'Annual-scale',
        'seasonal': 'Seasonal',
        'weekly': 'Weekly',
        'daily': 'Daily',
        'residual': 'Residual',
    }
    available = [col for col in ordered if col in component_df.columns]
    fig = make_subplots(rows=len(available), cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=[labels[c] for c in available])
    palette = {
        'original': '#1f77b4', 'trend_low_freq': '#2ca02c', 'annual': '#9467bd', 'seasonal': '#8c564b', 'weekly': '#ff7f0e', 'daily': '#d62728', 'residual': '#7f7f7f'
    }
    for i, col in enumerate(available, start=1):
        fig.add_trace(go.Scatter(x=component_df.index, y=component_df[col], mode='lines', name=labels[col], line=dict(color=palette[col], width=1.2)), row=i, col=1)
        fig.update_yaxes(title_text=labels[col], row=i, col=1)
    fig.update_xaxes(title_text='Timestamp', row=len(available), col=1)
    fig.update_layout(template='plotly_white', height=max(900, 210 * len(available)), title=f'Temporal Components Decomposition ({unit})', showlegend=False)
    return fig


def build_component_energy_figure(summary_df: pd.DataFrame) -> go.Figure:
    plot_df = summary_df.copy().sort_values('Variance Share (%)', ascending=True)
    fig = px.bar(
        plot_df,
        x='Variance Share (%)',
        y='Component',
        orientation='h',
        template='plotly_white',
        title='Variance Share by Temporal Component',
        text='Variance Share (%)',
    )
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside',
        cliponaxis=False,
        hovertemplate='%{y}<br>Variance Share: %{x:.2f}%<extra></extra>',
    )
    fig.update_layout(height=380)
    return fig


def build_component_window_figure(
    component_df: pd.DataFrame,
    unit: str,
    start_ts,
    horizon_label: str = '7D',
    mode: str = 'overlay',
    show_original: bool = True,
) -> go.Figure:
    horizon_map = {
        '1D': pd.Timedelta(days=1),
        '7D': pd.Timedelta(days=7),
        '30D': pd.Timedelta(days=30),
    }
    delta = horizon_map.get(horizon_label, pd.Timedelta(days=7))
    start_ts = pd.Timestamp(start_ts)
    end_ts = start_ts + delta

    df = component_df.loc[(component_df.index >= start_ts) & (component_df.index < end_ts)].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_white',
            title='Temporal Components Window',
            xaxis_title='Timestamp',
            yaxis_title=f'Value ({unit})',
            annotations=[
                dict(
                    text='No data available in selected window.',
                    x=0.5,
                    y=0.5,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                )
            ],
        )
        return fig

    colors = {
        'trend_low_freq': 'rgba(44, 160, 44, 0.40)',
        'annual': 'rgba(148, 103, 189, 0.40)',
        'seasonal': 'rgba(140, 86, 75, 0.40)',
        'weekly': 'rgba(255, 127, 14, 0.40)',
        'daily': 'rgba(214, 39, 40, 0.35)',
        'residual': 'rgba(127, 127, 127, 0.28)',
    }
    line_colors = {
        'trend_low_freq': '#2ca02c',
        'annual': '#9467bd',
        'seasonal': '#8c564b',
        'weekly': '#ff7f0e',
        'daily': '#d62728',
        'residual': '#7f7f7f',
        'original': '#1f77b4',
    }
    labels = {
        'trend_low_freq': 'Trend / low frequency',
        'annual': 'Annual-scale',
        'seasonal': 'Seasonal',
        'weekly': 'Weekly',
        'daily': 'Daily',
        'residual': 'Residual',
        'original': 'Original',
    }
    ordered_components = ['trend_low_freq', 'annual', 'seasonal', 'weekly', 'daily', 'residual']

    fig = go.Figure()
    if mode == 'stacked':
        for comp in ordered_components:
            if comp in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[comp],
                        mode='lines',
                        name=labels[comp],
                        stackgroup='components',
                        line=dict(width=1, color=line_colors[comp]),
                        fillcolor=colors[comp],
                        hovertemplate=f'{labels[comp]}<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
                    )
                )
        if show_original and 'original' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['original'],
                    mode='lines',
                    name='Original',
                    line=dict(color=line_colors['original'], width=2.4),
                    hovertemplate=f'Original<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
                )
            )
    else:
        default_visible = {'daily', 'weekly', 'seasonal'}
        for comp in ordered_components:
            if comp in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[comp],
                        mode='lines',
                        name=labels[comp],
                        line=dict(width=1.5, color=line_colors[comp]),
                        fill='tozeroy',
                        fillcolor=colors[comp],
                        visible=True if comp in default_visible else 'legendonly',
                        hovertemplate=f'{labels[comp]}<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
                    )
                )
        if show_original and 'original' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['original'],
                    mode='lines',
                    name='Original',
                    line=dict(color=line_colors['original'], width=2.6),
                    hovertemplate=f'Original<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
                )
            )

    fig.update_layout(
        template='plotly_white',
        height=540,
        title=f'Temporal Components Window ({horizon_label}, {mode})',
        xaxis_title='Timestamp',
        yaxis_title=f'Value ({unit})',
        legend_title='Components',
        hovermode='x unified',
    )
    return fig


def build_cumulative_components_figure(
    component_df: pd.DataFrame,
    unit: str,
    start_ts,
    horizon_label: str = '7D',
    show_original: bool = True,
    show_cumulative_lines: bool = True,
) -> go.Figure:
    horizon_map = {
        '1D': pd.Timedelta(days=1),
        '7D': pd.Timedelta(days=7),
        '30D': pd.Timedelta(days=30),
    }
    delta = horizon_map.get(horizon_label, pd.Timedelta(days=7))
    start_ts = pd.Timestamp(start_ts)
    end_ts = start_ts + delta

    df = component_df.loc[(component_df.index >= start_ts) & (component_df.index < end_ts)].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_white',
            title='Cumulative Reconstruction View',
            xaxis_title='Timestamp',
            yaxis_title=f'Value ({unit})',
            annotations=[
                dict(
                    text='No data available in selected window.',
                    x=0.5,
                    y=0.5,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                )
            ],
        )
        return fig

    ordered_components = ['trend_low_freq', 'annual', 'seasonal', 'weekly', 'daily', 'residual']
    labels = {
        'trend_low_freq': 'Trend / low frequency',
        'annual': 'Annual-scale',
        'seasonal': 'Seasonal',
        'weekly': 'Weekly',
        'daily': 'Daily',
        'residual': 'Residual',
    }
    fill_colors = {
        'trend_low_freq': 'rgba(44, 160, 44, 0.28)',
        'annual': 'rgba(148, 103, 189, 0.28)',
        'seasonal': 'rgba(140, 86, 75, 0.28)',
        'weekly': 'rgba(255, 127, 14, 0.28)',
        'daily': 'rgba(214, 39, 40, 0.22)',
        'residual': 'rgba(127, 127, 127, 0.18)',
    }
    line_colors = {
        'trend_low_freq': '#2ca02c',
        'annual': '#9467bd',
        'seasonal': '#8c564b',
        'weekly': '#ff7f0e',
        'daily': '#d62728',
        'residual': '#7f7f7f',
        'original': '#1f77b4',
        'reconstructed': '#111111',
    }

    cumulative = pd.DataFrame(index=df.index)
    running = pd.Series(0.0, index=df.index)
    for comp in ordered_components:
        if comp in df.columns:
            running = running + df[comp]
            cumulative[comp] = running

    fig = go.Figure()
    prev = pd.Series(0.0, index=df.index)
    for comp in ordered_components:
        if comp not in cumulative.columns:
            continue
        upper = cumulative[comp]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=prev,
                mode='lines',
                line=dict(width=0),
                hoverinfo='skip',
                showlegend=False,
                legendgroup=comp,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=upper,
                mode='lines',
                fill='tonexty',
                fillcolor=fill_colors[comp],
                line=dict(color=line_colors[comp], width=1.4),
                name=labels[comp],
                legendgroup=comp,
                hovertemplate=f'{labels[comp]} cumulative<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
            )
        )
        if show_cumulative_lines:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper,
                    mode='lines',
                    line=dict(color=line_colors[comp], width=1.4, dash='dot'),
                    name=f'{labels[comp]} cumulative line',
                    legendgroup=f'{comp}_cumline',
                    visible='legendonly',
                    hovertemplate=f'{labels[comp]} cumulative line<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
                )
            )
        prev = upper

    if cumulative.shape[1] > 0:
        reconstructed = cumulative.iloc[:, -1]
    else:
        reconstructed = pd.Series(0.0, index=df.index)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=reconstructed,
            mode='lines',
            name='Reconstructed sum',
            line=dict(color=line_colors['reconstructed'], width=2.4),
            hovertemplate=f'Reconstructed sum<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
        )
    )

    if show_original and 'original' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['original'],
                mode='lines',
                name='Original',
                line=dict(color=line_colors['original'], width=2.6),
                hovertemplate=f'Original<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>',
            )
        )

    fig.update_layout(
        template='plotly_white',
        height=560,
        title=f'Cumulative Reconstruction View ({horizon_label})',
        xaxis_title='Timestamp',
        yaxis_title=f'Value ({unit})',
        hovermode='x unified',
        legend_title='Layers',
    )
    return fig
