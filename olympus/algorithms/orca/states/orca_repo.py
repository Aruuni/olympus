"""Recurrent ORCA-repo observation state (Soheil-ab/Orca envwrapper.py).

This is Orca's own state, as a self-contained runtime state plugin selected by
``runtime.state: orca_repo``. It is the parallel of ``rewards/orca.py``: the
observation lives here, not in the learner's model. It does not import the
algorithm (states stay independent of algos), so ``orca/model.py`` loads it
through the normal state-plugin system without a circular import.

Per step it builds Orca's 7 compact features and stacks the last ``rec_dim``
of them into the flat policy input (Orca's flat-history analogue of an LSTM):

    thr/max_bw, pacing/max_bw, 5*loss/max_bw, samples/cwnd, delta_t,
    min_rtt/srtt, delay_metric = min(1, coef*min_rtt/srtt)

``max_bw`` is a per-flow running maximum of throughput; one worker process
serves one flow (orchestrator.listener_single_flow), so the module-level
running state below is per-flow. This reproduces the transform's
``use_normalizer=False`` path (Orca's deployed default); the optional upstream
Welford normalizer is not modelled here.
"""

import os
import time

import numpy as np
import torch

from olympus.common import runtime_config


BASE_STATE_FEATURES = [
    'thr_over_max_bw',
    'pacing_over_max_bw',
    'loss_penalty_over_max_bw',
    'samples_over_cwnd',
    'delta_t',
    'min_rtt_over_srtt',
    'delay_metric',
]
BASE_STATE_DIM = len(BASE_STATE_FEATURES)

_RUNTIME_CFG = runtime_config.load_config()
REC_DIM = int(runtime_config.agent_value(
    _RUNTIME_CFG, 'rec_dim', env='SAO_ORCA_REC_DIM',
    default=os.environ.get('SAO_REC_DIM', '10')))

STATE_DIM = BASE_STATE_DIM * REC_DIM
STATE_FEATURE_VERSION = f'orca_repo_state_v1_rec{REC_DIM}'
STATE_FEATURES = [
    f'h{hist}:{name}'
    for hist in range(REC_DIM)
    for name in BASE_STATE_FEATURES
]

_BASE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_BASE_HIGH = np.array([2.0, 10.0, 10.0, 4.0, 10.0, 2.0, 1.0], dtype=np.float32)
STATE_LOW = np.tile(_BASE_LOW, REC_DIM).astype(np.float32)
STATE_HIGH = np.tile(_BASE_HIGH, REC_DIM).astype(np.float32)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)

# Per-flow running state (module-level; one flow per worker process).
_history = None
_max_bw = 0.0
_last_sample_t = None


def _finite(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if v != v or v in (float('inf'), float('-inf')):
        return float(default)
    return v


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def reset_state() -> None:
    global _history, _max_bw, _last_sample_t
    _history = np.zeros(REC_DIM * BASE_STATE_DIM, dtype=np.float32)
    _max_bw = 0.0
    _last_sample_t = None


def _interval_s(info: dict) -> float:
    global _last_sample_t
    for key in ('interval_s', 'delta_t'):
        if key in info:
            dt = _finite(info.get(key), 0.0)
            if dt > 0.0:
                _last_sample_t = time.monotonic()
                return dt
    now = time.monotonic()
    if _last_sample_t is None:
        dt = _float_env('SAO_INTERVAL_MS', 20.0) / 1000.0
    else:
        dt = max(now - _last_sample_t, 1e-6)
    _last_sample_t = now
    return dt


def _base_state(info: dict) -> np.ndarray:
    global _max_bw
    coef = _float_env('OC_ORCA_DELAY_MARGIN_COEF', 1.25)

    throughput = max(_finite(info.get('throughput', info.get('avg_thr', 0.0))), 0.0)
    samples = max(_finite(info.get('count', info.get('cnt', 0.0))), 0.0)
    delta_t = max(_interval_s(info), 1e-6)
    cwnd = max(_finite(info.get('cwnd', 1.0), 1.0), 1.0)
    pacing_rate = max(_finite(info.get('pacing_rate', 0.0)), 0.0)

    loss_rate = max(_finite(info.get('lost_rate', info.get('loss_rate', 0.0))), 0.0)
    if loss_rate <= 0.0:
        loss_bytes = max(_finite(info.get('loss_bytes',
                                          info.get('lost_bytes', 0.0))), 0.0)
        loss_rate = loss_bytes / delta_t

    avg_urtt_us = max(_finite(info.get('avg_urtt', info.get('delay_us', 0.0))), 0.0)
    srtt_raw = _finite(info.get('srtt_us', 0.0))
    srtt_ms = ((srtt_raw / 8.0) if srtt_raw > 0.0 else avg_urtt_us) / 1000.0

    min_rtt_ms = info.get('min_rtt_ms', None)
    if min_rtt_ms is None:
        min_rtt_ms = max(_finite(info.get('min_rtt',
                                          info.get('min_rtt_us', 0.0))), 0.0) / 1000.0
    else:
        min_rtt_ms = max(_finite(min_rtt_ms), 0.0)

    _max_bw = max(_max_bw, throughput)

    if srtt_ms > 1e-12:
        delay_metric = min(1.0, coef * min_rtt_ms / srtt_ms)
        min_over_srtt = min_rtt_ms / srtt_ms
    else:
        delay_metric = 1.0
        min_over_srtt = 0.0

    if _max_bw > 1e-12:
        state0 = throughput / _max_bw
        pacing = min(pacing_rate / _max_bw, 10.0)
        loss = 5.0 * loss_rate / _max_bw
    else:
        state0 = pacing = loss = 0.0

    state = np.array([
        state0,
        pacing,
        loss,
        samples / cwnd,
        delta_t,
        min_over_srtt,
        delay_metric,
    ], dtype=np.float32)
    return np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)


def normalize_state(info: dict) -> np.ndarray:
    global _history
    if _history is None:
        reset_state()
    base_state = _base_state(info).reshape(BASE_STATE_DIM)
    _history = np.concatenate([_history[BASE_STATE_DIM:], base_state]).astype(np.float32)
    return np.clip(_history.copy(), STATE_LOW, STATE_HIGH)
