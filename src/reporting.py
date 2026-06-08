from __future__ import annotations

import io
import zipfile
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.fourier_core import AnalysisResult


def fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    try:
        return pio.to_image(fig, format='png', width=1400, height=800, scale=2)
    except Exception:
        return None


def generate_pdf_report(result: AnalysisResult, unit: str, source_name: str, k: int, total_error_df: pd.DataFrame, harmonic_preview_text: str, figures: Dict[str, go.Figure], compare_k_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph('Fourier Analysis Report', styles['Title']))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f'Source file: {source_name}', styles['BodyText']))
    story.append(Paragraph(f'Unit: {unit}', styles['BodyText']))
    story.append(Paragraph(f'Selected K: {k}', styles['BodyText']))
    story.append(Spacer(1, 0.3 * cm))
    stats_table = Table([
        ['Metric', 'Value'],
        ['Mean', f"{result.stats['mean']:.4f} {unit}"],
        ['Std', f"{result.stats['std']:.4f} {unit}"],
        ['CV', f"{result.stats['cv_pct']:.2f}%"],
        ['Min', f"{result.stats['min']:.4f} {unit}"],
        ['Max', f"{result.stats['max']:.4f} {unit}"],
    ], colWidths=[6 * cm, 8 * cm])
    stats_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbe7f3')), ('GRID', (0, 0), (-1, -1), 0.4, colors.grey)]))
    story.append(Paragraph('Statistics', styles['Heading2']))
    story.append(stats_table)
    story.append(Spacer(1, 0.3 * cm))
    err = total_error_df.iloc[0]
    error_table = Table([
        ['Error metric', 'Value'],
        ['MAE', f"{err['MAE']:.6f}"],
        ['RMSE', f"{err['RMSE']:.6f}"],
        ['Bias', f"{err['Bias']:.6f}"],
        ['wMAPE (%)', f"{err['wMAPE (%)']:.4f}"],
        ['R2', f"{err['R2']:.6f}"],
    ], colWidths=[6 * cm, 8 * cm])
    error_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbe7f3')), ('GRID', (0, 0), (-1, -1), 0.4, colors.grey)]))
    story.append(Paragraph('Reconstruction Error', styles['Heading2']))
    story.append(error_table)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph('Harmonic Function Preview', styles['Heading2']))
    story.append(Paragraph(harmonic_preview_text.replace('\n', '<br/>'), styles['BodyText']))
    story.append(Spacer(1, 0.3 * cm))
    if not compare_k_df.empty:
        topk = compare_k_df[['K', 'RMSE', 'wMAPE (%)', 'R2']].copy().head(10)
        table_data = [['K', 'RMSE', 'wMAPE (%)', 'R2']] + topk.round(6).astype(str).values.tolist()
        k_table = Table(table_data, colWidths=[2.5 * cm, 4 * cm, 4 * cm, 4 * cm])
        k_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbe7f3')), ('GRID', (0, 0), (-1, -1), 0.4, colors.grey)]))
        story.append(Paragraph('K Comparison', styles['Heading2']))
        story.append(k_table)
        story.append(Spacer(1, 0.3 * cm))
    missing_images = []
    for title, fig in figures.items():
        story.append(Paragraph(title, styles['Heading2']))
        png_bytes = fig_to_png_bytes(fig)
        if png_bytes is not None:
            story.append(RLImage(io.BytesIO(png_bytes), width=17 * cm, height=9.2 * cm))
        else:
            missing_images.append(title)
            story.append(Paragraph('Chart image could not be embedded in this environment. The interactive chart remains available in the app.', styles['BodyText']))
        story.append(Spacer(1, 0.2 * cm))
    if missing_images:
        story.append(Paragraph('Notes', styles['Heading2']))
        story.append(Paragraph('Some charts were omitted from the PDF because static image export is unavailable in the current runtime environment.', styles['BodyText']))
    doc.build(story)
    return buffer.getvalue()


def generate_csv_exports(result: AnalysisResult, reconstructed: pd.Series, harmonic_df: pd.DataFrame, error_point_df: pd.DataFrame, total_error_df: pd.DataFrame, daily_error_df: pd.DataFrame, monthly_error_df: pd.DataFrame, compare_k_df: pd.DataFrame, synthetic_df: pd.DataFrame, component_df: pd.DataFrame | None = None) -> Dict[str, bytes]:
    files = {}
    files['statistics.csv'] = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in result.stats.items()]).to_csv(index=False).encode('utf-8')
    files['dominant_frequencies.csv'] = result.dominant_df.to_csv(index=False).encode('utf-8')
    files['band_energy.csv'] = result.band_energy_df.to_csv(index=False).encode('utf-8')
    files['parsed_and_reconstructed_series.csv'] = pd.DataFrame({'timestamp': result.series.index, 'actual': result.series.values, 'reconstructed': reconstructed.values}).to_csv(index=False).encode('utf-8')
    files['harmonic_coefficients.csv'] = harmonic_df.to_csv(index=False).encode('utf-8')
    files['error_pointwise.csv'] = error_point_df.to_csv(index=False).encode('utf-8')
    files['error_total.csv'] = total_error_df.to_csv(index=False).encode('utf-8')
    files['error_daily.csv'] = daily_error_df.to_csv(index=False).encode('utf-8')
    files['error_monthly.csv'] = monthly_error_df.to_csv(index=False).encode('utf-8')
    files['k_comparison.csv'] = compare_k_df.to_csv(index=False).encode('utf-8')
    files['synthetic_harmonic_series.csv'] = synthetic_df.to_csv(index=False).encode('utf-8')
    if component_df is not None:
        files['temporal_components.csv'] = component_df.reset_index().rename(columns={'index': 'timestamp'}).to_csv(index=False).encode('utf-8')
    return files


def generate_zip_export(pdf_bytes: bytes, csv_files: Dict[str, bytes], txt_files: Dict[str, bytes], figures: Dict[str, go.Figure]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        if pdf_bytes:
            zf.writestr('Fourier_Analysis_Report.pdf', pdf_bytes)
        for name, data in csv_files.items():
            zf.writestr(name, data)
        for name, data in txt_files.items():
            zf.writestr(name, data)
        for name, fig in figures.items():
            png_bytes = fig_to_png_bytes(fig)
            if png_bytes is not None:
                zf.writestr(name.lower().replace(' ', '_') + '.png', png_bytes)
    return buffer.getvalue()
