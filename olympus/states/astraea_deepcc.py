"""Astraea/DeepCC TCP observation state.

This is the exact 10-feature transform used by ``astraea.agent.definitions``
for per-flow state. It expects real DeepCC/TCP_INFO-style fields, not
CleanSlate's seven normalized observations.
"""

import numpy as np
import torch


STATE_FEATURE_VERSION = 'astraea_deepcc_observation_v1'
STATE_FEATURES = [
    'avg_thr_over_max_tput',
    'avg_urtt_over_min_rtt',
    'srtt_over_min_rtt',
    'cwnd_bdp_ratio',
    'max_tput_scaled',
    'min_rtt_scaled',
    'loss_ratio_over_max_tput',
    'packets_out_over_cwnd',
    'pacing_over_max_tput',
    'retrans_out_over_cwnd',
]

STATE_LOW = np.zeros(len(STATE_FEATURES), dtype=np.float32)
STATE_HIGH = np.array(
    [10.0, 2.0, 2.0, 2.0, 1000.0, 1000.0, 10.0, 1000.0, 2.0, 1000.0],
    dtype=np.float32,
)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)
STATE_DIM = len(STATE_FEATURES)

_REQUIRED_RAW_FIELDS = (
    'avg_thr',
    'avg_urtt',
    'min_rtt',
    'srtt_us',
    'cwnd',
    'pacing_rate',
    'packets_out',
    'retrans_out',
)


def _require_deepcc_fields(info: dict) -> None:
    missing = [key for key in _REQUIRED_RAW_FIELDS if key not in info]
    if missing:
        raise ValueError(
            'astraea_deepcc state requires raw DeepCC/Astraea fields; missing '
            + ', '.join(missing)
            + '. CleanSlate normalized observations cannot be converted into '
            'the Astraea state space without changing the RayNet runner to '
            'emit these fields.')


def _finite_float(info: dict, key: str, default=0.0) -> float:
    try:
        value = float(info.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _loss_ratio(info: dict) -> float:
    for key in ('loss_ratio', 'loss_rate'):
        if key in info:
            return max(_finite_float(info, key), 0.0)

    loss_bytes = None
    for key in ('loss_bytes', 'lost_bytes'):
        if key in info:
            loss_bytes = max(_finite_float(info, key), 0.0)
            break
    if loss_bytes is None:
        return 0.0

    interval_s = _finite_float(info, 'interval_s')
    if interval_s <= 0.0:
        interval_s = _finite_float(info, 'delta_t')
    if interval_s <= 0.0:
        interval_s = _finite_float(info, 'interval_ms') / 1000.0
    return loss_bytes / max(interval_s, 1e-9)


def normalize_state(info: dict) -> np.ndarray:
    _require_deepcc_fields(info)

    avg_thr = _finite_float(info, 'avg_thr')
    avg_urtt = _finite_float(info, 'avg_urtt')
    min_rtt = _finite_float(info, 'min_rtt')
    srtt_us = _finite_float(info, 'srtt_us')
    cwnd = max(_finite_float(info, 'cwnd'), 1.0)
    pacing_rate = _finite_float(info, 'pacing_rate')
    packets_out = _finite_float(info, 'packets_out')
    retrans_out = _finite_float(info, 'retrans_out')
    max_tput = max(
        _finite_float(info, 'max_tput'),
        _finite_float(info, 'peak_thr'),
        avg_thr,
        0.0,
    )
    loss_ratio = _loss_ratio(info)

    if avg_thr == 0.0:
        thr_feature = 0.5
    else:
        thr_feature = avg_thr / max_tput if max_tput > 0.0 else 0.0

    if avg_urtt == 0.0:
        urtt_feature = 2.0
    elif min_rtt == 0.0:
        urtt_feature = 0.0
    else:
        urtt_feature = avg_urtt / min_rtt

    if srtt_us == 0.0:
        srtt_feature = 2.0
    elif min_rtt == 0.0:
        srtt_feature = 0.0
    else:
        srtt_feature = srtt_us / 8.0 / min_rtt

    if min_rtt == 0.0 or max_tput == 0.0:
        cwnd_feature = 0.0
    else:
        cwnd_feature = (
            cwnd * 1460.0 * 8.0 / (min_rtt / 1e6) / max_tput / 10.0
        )

    s = np.array([
        thr_feature,
        urtt_feature,
        srtt_feature,
        cwnd_feature,
        max_tput / 1e7,
        min_rtt / 5e5,
        loss_ratio / max_tput if max_tput > 0.0 else 0.0,
        packets_out / cwnd,
        pacing_rate / max_tput if max_tput > 0.0 else 0.0,
        retrans_out / cwnd,
    ], dtype=np.float32)

    s = np.nan_to_num(s, nan=0.0, posinf=STATE_HIGH, neginf=STATE_LOW)
    for index in (1, 2, 3, 8):
        if s[index] > 2.0:
            s[index] = 2.0
    return s
