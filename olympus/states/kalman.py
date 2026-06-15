"""Kalman-observation TCP state used by the older single-agent models.

Old shared_no_bdp checkpoints are accepted through the compatibility map in
state_plugins.py.
"""

import math

import numpy as np
import torch


STATE_FEATURE_VERSION = 'kalman_observation_v1_srtt_unshifted'
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
    'kalman_min_rtt',
]

STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    dtype=np.float32,
)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)
STATE_DIM = len(STATE_FEATURES)


class KalmanMinRTT:
    """
    1-D asymmetric Kalman tracker for the observed propagation RTT floor.

    The state module owns this because the estimate is a state feature. The
    update input and public output use microseconds to match TCP_INFO fields.
    """

    def __init__(self, init_us: float = 20_000.0):
        self._x = init_us / 1000.0
        self._Q = 5.0e-4 # was 1.0e-3
        self._R_dn = 1.0e-3 # was 1.0e-3
        self._R_up = 5 # was 0.15
        self._jump_thresh = 1.5 # was 1.25
        self._P = math.sqrt(self._Q * self._R_up)
        self._ready = False

    def update(self, obs_us: float) -> float:
        obs_ms = obs_us / 1000.0
        if not self._ready:
            self._x = obs_ms
            self._ready = True
            return self.estimate

        p_pred = self._P + self._Q
        if obs_ms < self._x:
            r = self._R_dn
        else:
            frac = obs_ms / max(self._x, 1e-3)
            r = self._R_dn / frac if frac > self._jump_thresh else self._R_up / frac

        k = p_pred / (p_pred + r)
        self._x = self._x + k * (obs_ms - self._x)
        self._P = (1.0 - k) * p_pred
        return self.estimate

    @property
    def estimate(self) -> float:
        return self._x * 1000.0


_KALMAN = None


def reset_kalman(initial_rtt_us: float = None) -> None:
    global _KALMAN
    _KALMAN = KalmanMinRTT(20_000.0 if initial_rtt_us is None
                           else float(initial_rtt_us))


def update_kalman_min_rtt(info: dict) -> float:
    global _KALMAN
    if _KALMAN is None:
        reset_kalman()

    avg_urtt = float(info.get('avg_urtt', 0))
    if avg_urtt > 0:
        return _KALMAN.update(avg_urtt)
    return _KALMAN.estimate


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
    kalman_min_rtt = update_kalman_min_rtt(info)
    info['kalman_min_rtt_us'] = kalman_min_rtt

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
        kalman_min_rtt / 1e5,
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=STATE_HIGH, neginf=STATE_LOW)
    return np.clip(s, STATE_LOW, STATE_HIGH)
