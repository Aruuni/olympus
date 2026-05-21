"""Recurrent ORCA-repo observation state.

The ORCA worker normally owns this transform directly through
``algorithms.orca.model``. This plugin makes ``state: orca_repo`` resolvable by
shared state-loading utilities without changing the worker path.
"""

import os
import time

import numpy as np
import torch

from single_agent_olympus.algorithms.orca import model


STATE_FEATURE_VERSION = model.STATE_FEATURE_VERSION
STATE_FEATURES = list(model.STATE_FEATURES)
STATE_DIM = int(model.STATE_DIM)

_BASE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_BASE_HIGH = np.array([2.0, 10.0, 10.0, 4.0, 10.0, 2.0, 1.0], dtype=np.float32)
STATE_LOW = np.tile(_BASE_LOW, model.REC_DIM).astype(np.float32)
STATE_HIGH = np.tile(_BASE_HIGH, model.REC_DIM).astype(np.float32)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)

_transform = None
_history = None
_last_sample_t = None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ('0', 'false', 'no', 'off', '')


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def reset_state() -> None:
    global _transform, _history, _last_sample_t
    delay_margin_coef = _float_env('OC_ORCA_DELAY_MARGIN_COEF', 1.25)
    use_normalizer = _bool_env('SAO_ORCA_USE_NORMALIZER', False)
    _transform = model.OrcaRepoTransform(
        delay_margin_coef=delay_margin_coef,
        use_normalizer=use_normalizer,
    )
    _history = model.HistoryStack(model.REC_DIM)
    _last_sample_t = None


def normalize_state(info: dict) -> np.ndarray:
    global _last_sample_t
    if _transform is None or _history is None:
        reset_state()

    now = time.monotonic()
    if _last_sample_t is None:
        interval_s = _float_env('SAO_INTERVAL_MS', 20.0) / 1000.0
    else:
        interval_s = max(now - _last_sample_t, 1e-6)
    _last_sample_t = now

    target = _float_env('SAO_ORCA_TARGET_MS',
                        _float_env('OC_ORCA_TARGET_MS', 50.0))
    base_state, _reward = _transform.step(
        info,
        interval_s=interval_s,
        target=target,
        evaluation=True,
    )
    state = _history.push(base_state)
    return np.clip(state, STATE_LOW, STATE_HIGH)
