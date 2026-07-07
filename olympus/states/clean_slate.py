"""CleanSlate RayNet observation state.

CleanSlate emits seven already-normalized features from its OMNeT++ agent:
throughput, pacing rate, loss, ACK/cwnd ratio, interval duration, normalized
sRTT, and its delay metric. Keep them as-is so Olympus does not reinterpret
them as Orca/Astraea byte counters.
"""

import numpy as np
import torch


STATE_FEATURE_VERSION = 'clean_slate_observation_v1'
STATE_FEATURES = [
    'throughput_norm',
    'pacing_norm',
    'loss_norm',
    'acks_over_cwnd',
    'interval_s',
    'srtt_norm',
    'delay_metric',
]

STATE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
STATE_HIGH = np.array([1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float32)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)
STATE_DIM = len(STATE_FEATURES)


def normalize_state(info: dict) -> np.ndarray:
    s = np.array([
        float(info.get('throughput_norm', info.get('avg_thr', 0.0)) or 0.0),
        float(info.get('pacing_norm', info.get('pacing_rate', 0.0)) or 0.0),
        float(info.get('loss_norm', info.get('loss_rate', 0.0)) or 0.0),
        float(info.get('acks_over_cwnd', info.get('samples', 0.0)) or 0.0),
        float(info.get('interval_s', info.get('delta_t', 0.0)) or 0.0),
        float(info.get('srtt_norm', 0.0) or 0.0),
        float(info.get('delay_metric', 0.0) or 0.0),
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=STATE_HIGH, neginf=STATE_LOW)
    return np.clip(s, STATE_LOW, STATE_HIGH)
