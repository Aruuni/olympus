"""ORCA-style TCP observation state.

This mirrors Orca's seven features using only values observed from the socket
via TCP_INFO / TCP_DEEPCC_INFO. It does not use link BDP or scheduled RTT.
"""

import os

import numpy as np
import torch


STATE_FEATURE_VERSION = 'orca_observation_v1'
STATE_FEATURES = [
    'thr_over_peak',
    'pacing_over_peak',
    'loss_over_peak',
    'samples_over_cwnd',
    'delta_t',
    'min_rtt_over_srtt',
    'delay_metric',
]

STATE_LOW = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
STATE_HIGH = np.array(
    [2.0, 10.0, 10.0, 4.0, 10.0, 2.0, 1.0],
    dtype=np.float32,
)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)
STATE_DIM = len(STATE_FEATURES)

_DELAY_MARGIN_COEF = 1.25


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _interval_s(info: dict) -> float:
    if 'delta_t' in info:
        return float(info.get('delta_t') or 0.0)
    if 'interval_s' in info:
        return float(info.get('interval_s') or 0.0)
    if 'interval_ms' in info:
        return float(info.get('interval_ms') or 0.0) / 1000.0
    return _float_env('SAO_INTERVAL_MS', 20.0) / 1000.0


def normalize_state(info: dict) -> np.ndarray:
    cwnd = max(float(info.get('cwnd', 1)), 1.0)
    avg_thr = max(float(info.get('avg_thr', 0)), 0.0)
    peak_thr = max(float(info.get('peak_thr', 0)), avg_thr, 1.0)
    pacing_rate = max(float(info.get('pacing_rate', 0)), 0.0)

    delta_t = max(_interval_s(info), 0.0)
    loss_bytes = max(float(info.get('loss_bytes', info.get('lost_bytes', 0))), 0.0)
    loss_rate = max(float(info.get('loss_rate', info.get('lost_rate', 0))), 0.0)
    if loss_rate <= 0.0 and delta_t > 0.0:
        loss_rate = loss_bytes / delta_t

    samples = max(float(info.get('cnt', info.get('samples', 0))), 0.0)
    min_rtt_us = max(float(info.get('min_rtt', info.get('min_rtt_us', 0))), 0.0)
    avg_urtt_us = max(float(info.get('avg_urtt', 0)), 1.0)
    srtt_raw = float(info.get('srtt_us', 0))
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0 else avg_urtt_us
    srtt_us = max(srtt_us, 1.0)

    min_rtt_over_srtt = min_rtt_us / srtt_us
    if min_rtt_us > 0.0 and min_rtt_us * _DELAY_MARGIN_COEF < srtt_us:
        delay_metric = (min_rtt_us * _DELAY_MARGIN_COEF) / srtt_us
    else:
        delay_metric = 1.0

    s = np.array([
        avg_thr / peak_thr,
        min(pacing_rate / peak_thr, 10.0),
        5.0 * loss_rate / peak_thr,
        samples / cwnd,
        delta_t,
        min_rtt_over_srtt,
        delay_metric,
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=STATE_HIGH, neginf=STATE_LOW)
    return np.clip(s, STATE_LOW, STATE_HIGH)
