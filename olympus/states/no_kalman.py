"""Observable TCP state without the Kalman RTT feature."""

import math

import numpy as np
import torch


STATE_FEATURE_VERSION = 'no_kalman_observation_v1'
STATE_FEATURES = [
    'delta_cwnd',
    'avg_urtt',
    'cwnd_log',
    'thr_over_peak',
    'pacing_over_peak',
    'inflight_over_cwnd',
    'delta_rtt',
    'retrans_ratio',
    'avg_thr',
    'srtt',
]

STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0],
    dtype=np.float32,
)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)
STATE_DIM = len(STATE_FEATURES)


def normalize_state(info: dict) -> np.ndarray:
    cwnd = max(int(info.get('cwnd', 1)), 1)
    avg_thr = float(info.get('avg_thr', 0))
    avg_urtt = float(info.get('avg_urtt', 0))
    srtt_raw = float(info.get('srtt_us', 0))
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0 else avg_urtt
    srtt_us = max(srtt_us, 1.0)
    pacing_rate = float(info.get('pacing_rate', 0))
    packets_out = float(info.get('packets_out', 0))
    retrans_out = float(info.get('retrans_out', 0))
    prev_urtt = float(info.get('prev_urtt', avg_urtt))
    prev_cwnd = float(info.get('prev_cwnd', cwnd))
    peak_thr = float(info.get('peak_thr', 0))

    bw_ref = max(peak_thr, pacing_rate, 1.0)
    delta_rtt = float(np.clip((avg_urtt - prev_urtt) / max(prev_urtt, 1.0),
                              -1.0, 1.0))
    delta_cwnd = float(np.clip((cwnd - prev_cwnd) / max(prev_cwnd, 1.0),
                               -1.0, 1.0))
    cwnd_log = math.log1p(float(cwnd)) / math.log1p(10_000.0)

    s = np.array([
        delta_cwnd,
        avg_urtt / 1e5,
        cwnd_log,
        max(avg_thr / bw_ref, 0.0),
        max(pacing_rate / bw_ref, 0.0),
        max(packets_out / max(cwnd, 1), 0.0),
        delta_rtt,
        min(retrans_out / max(packets_out, 1.0), 1.0),
        avg_thr / 1e7,
        srtt_us / 1e5,
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=STATE_HIGH, neginf=STATE_LOW)
    return np.clip(s, STATE_LOW, STATE_HIGH)
