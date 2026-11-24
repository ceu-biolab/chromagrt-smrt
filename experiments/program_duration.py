"""Metadata-only duration extraction and optional ChromaGRT input encoding.

Prepared RepoRT tables express both RT and gradient-node times in SECONDS,
despite retaining the original ``t [min]_N`` column names. Never derive the
duration from RT, max_rt, or mean_rt, or from already scaled input features.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MODE = 'program-duration'
DURATION = 'program_duration_seconds'
FEATURE = 'program.duration'
DEFINITION = 'maximum recorded gradient-node time, including final holds; seconds'


def program_durations(frame):
    """Return one positive, finite duration per row, using only raw metadata."""
    columns = [c for c in frame if re.fullmatch(r't \[min\]_\d+', c)]
    if not columns:
        raise ValueError('Program-duration normalization requires gradient-node times')
    if 'cc_id' not in frame or frame.cc_id.isna().any():
        raise ValueError('Program-duration normalization requires nonmissing cc_id')
    times = frame[columns].apply(pd.to_numeric, errors='raise')
    populated = times.to_numpy(dtype=float)
    if np.isinf(populated).any() or (populated < 0).any():
        raise ValueError('Gradient-node times must be finite and nonnegative (NaN padding is allowed)')
    duration = times.max(axis=1).rename(DURATION)
    invalid = duration.isna() | ~np.isfinite(duration) | duration.le(0)
    if invalid.any():
        bad = frame.loc[invalid, 'cc_id'].unique().tolist()
        raise ValueError(f'Nonpositive or missing program duration for conditions: {bad[:10]}')
    values = pd.DataFrame({'cc_id': frame.cc_id, DURATION: duration})
    if values.groupby('cc_id')[DURATION].nunique().gt(1).any():
        raise ValueError('Program duration must be constant within each cc_id')
    return duration


def duration_table(frame):
    return pd.DataFrame({'cc_id': frame.cc_id, DURATION: program_durations(frame)}).drop_duplicates('cc_id').sort_values('cc_id').reset_index(drop=True)


def add_duration_inputs(raw, scaled, config):
    """Keep all existing features unchanged; append an explicit duration input.

The private duration used to decode predictions is placed BEFORE the model
features, so input normalization can never change it. The separate duration
feature is standardized using unique training CCs, or divided by 60 for the
deterministic encoding. Neither fitting step uses validation/test RTs.
"""
    tables = [duration_table(frame) for frame in raw.values()]
    combined = pd.concat(tables, ignore_index=True)
    if combined.groupby('cc_id')[DURATION].nunique().gt(1).any():
        raise ValueError('Program duration differs between partitions for the same cc_id')
    train = duration_table(raw['train'])
    minutes = train[DURATION].to_numpy(dtype=float) / 60
    standard = config['condition_normalization'] == 'standard'
    mean = float(minutes.mean()) if standard else 0.0
    scale = float(minutes.std()) if standard else 1.0
    if scale == 0:
        scale = 1.0
    result = {}
    audit = dict(mode=MODE, duration_definition=DEFINITION, target='rt / program_duration_seconds',
                 output_upper_bound=None, loss='MAE in seconds', duration_feature=FEATURE,
                 duration_feature_fit_partition='train', duration_feature_mean_minutes=mean,
                 duration_feature_scale_minutes=scale, partitions={})
    for role, frame in scaled.items():
        if DURATION in frame or FEATURE in frame:
            raise ValueError('Reserved program-duration columns already exist')
        if list(frame.molecule_id) != list(raw[role].molecule_id):
            raise ValueError('Duration metadata no longer matches model input order')
        duration = program_durations(raw[role]).to_numpy(dtype=float)
        result[role] = frame.copy()
        result[role].insert(0, DURATION, duration)
        result[role][FEATURE] = (duration / 60 - mean) / scale
        audit['partitions'][role] = dict(n_rows=len(frame), n_conditions=frame.cc_id.nunique(),
            min_duration_seconds=float(duration.min()), max_duration_seconds=float(duration.max()))
    return result, audit
