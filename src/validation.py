from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.fourier_core import infer_sample_hours, is_regular_series
from src.io_utils import normalize_frequency_alias

ALLOWED_SIMULATION_RESOLUTIONS = {'15min', '30min', '1h', '6h', '1d'}


def validate_time_series(series: pd.Series) -> Dict[str, object]:
    if series is None or len(series) == 0:
        raise ValueError('The parsed series is empty.')
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError('The series index must be a DatetimeIndex.')
    if series.isna().any():
        raise ValueError('The series still contains missing values after preprocessing.')
    if len(series) < 48:
        raise ValueError('At least 48 samples are required.')
    info = {
        'length': int(len(series)),
        'missing_values': int(series.isna().sum()),
        'has_unique_index': bool(series.index.is_unique),
        'is_monotonic': bool(series.index.is_monotonic_increasing),
        'sample_hours': float(infer_sample_hours(series.index)),
        'is_regular': bool(is_regular_series(series.index)),
    }
    if not info['has_unique_index']:
        raise ValueError('The time index contains duplicated timestamps.')
    if not info['is_monotonic']:
        raise ValueError('The time index must be sorted increasingly.')
    if info['sample_hours'] <= 0:
        raise ValueError('The inferred sample step must be positive.')
    return info


def parse_k_list(text: str, max_k: int) -> List[int]:
    values: List[int] = []
    for raw in text.replace(';', ',').split(','):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 1 <= value <= max_k:
            values.append(value)
    return sorted(set(values))


def validate_simulation_inputs(resolution: str, total_days: int) -> None:
    normalized = normalize_frequency_alias(resolution)
    if normalized not in ALLOWED_SIMULATION_RESOLUTIONS:
        raise ValueError(f'Unsupported resolution: {resolution}')
    if total_days <= 0:
        raise ValueError('Simulation duration must be positive.')
