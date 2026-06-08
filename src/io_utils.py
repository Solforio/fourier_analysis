from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd


FREQ_ALIAS_MAP = {
    '1H': '1h',
    '6H': '6h',
    '1D': '1d',
    'H': 'h',
    'D': 'd',
}


def normalize_frequency_alias(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return FREQ_ALIAS_MAP.get(value, value)


def clean_dataframe_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
    return df


def parse_datetime_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    candidates = [
        pd.to_datetime(s, errors='coerce'),
        pd.to_datetime(s, errors='coerce', dayfirst=True),
        pd.to_datetime(s, errors='coerce', format='mixed'),
        pd.to_datetime(s, errors='coerce', format='mixed', dayfirst=True),
    ]
    return max(candidates, key=lambda x: x.notna().sum())


def parse_numeric_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    s = series.astype(str).str.strip().str.replace(' ', '', regex=False).str.replace('\u00A0', '', regex=False)
    euro = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    euro_num = pd.to_numeric(euro, errors='coerce')
    eng = s.str.replace(',', '', regex=False)
    eng_num = pd.to_numeric(eng, errors='coerce')
    return euro_num if euro_num.notna().sum() >= eng_num.notna().sum() else eng_num


def load_dataset_from_bytes(file_bytes: bytes, file_name: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    file_name = file_name.lower()
    if file_name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', dtype=str)
    elif file_name.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, dtype=str)
    else:
        raise ValueError('Unsupported file format. Please upload a CSV or XLSX file.')
    return clean_dataframe_strings(df)


def list_excel_sheets(uploaded_file) -> List[str]:
    if not uploaded_file.name.lower().endswith('.xlsx'):
        return []
    xls = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
    return xls.sheet_names


def build_series_from_dataframe(df: pd.DataFrame, date_col: Optional[str] = None, value_col: Optional[str] = None, resample_rule: Optional[str] = None) -> pd.Series:
    if df.shape[1] < 2:
        raise ValueError('The dataset must contain at least two columns.')
    dt_col = date_col if date_col else df.columns[0]
    val_col = value_col if value_col else df.columns[1]
    dt = parse_datetime_column(df[dt_col])
    val = parse_numeric_column(df[val_col])
    out = pd.DataFrame({'timestamp': dt, 'value': val}).dropna()
    if out.empty:
        raise ValueError('No valid rows found after datetime and numeric parsing.')
    out = out.sort_values('timestamp').drop_duplicates(subset='timestamp').set_index('timestamp')
    series = out['value'].astype(float)
    series.name = 'value'
    rule = normalize_frequency_alias(resample_rule)
    if rule and rule != 'None':
        series = series.resample(rule).mean().interpolate(limit_direction='both')
    return series
