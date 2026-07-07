"""Fully-normalised Tempest observation state.

Every feature is dimensionless and bounded, so the observation distribution is
(approximately) invariant to the path's RTT and bottleneck bandwidth. This is
the generalisation-oriented sibling of ``tempest.py``:

* RTT signals (``avg_urtt``, ``srtt``) are expressed as *inflation over the
  Kalman propagation-RTT floor* rather than in absolute microseconds, so a
  queue that doubles the RTT reads the same on a 5 ms path and a 200 ms path.
* Throughput and pacing are ratios against the self-measured bandwidth
  reference ``bw_ref`` (already regime-invariant in ``tempest.py``).
* ``cwnd`` is normalised by an estimated BDP instead of a fixed log reference.
* The one deliberately-absolute quantity, the Kalman RTT floor, is passed
  through a saturating ``rtt / (rtt + C)`` squash so it stays in [0, 1] while
  still giving the policy a soft sense of the path timescale. See the module
  docstring notes below for alternative ways to normalise it.

Contract matches the state-plugin protocol used by ``common/state_plugins.py``
(STATE_FEATURES / STATE_LOW / STATE_HIGH / STATE_DIM / normalize_state), so it
is selectable via ``state: tempest_ratio`` (config.yaml) or ``SAO_STATE``.

The Kalman tracker is inlined (not imported from ``tempest.py``) so each state
representation stays self-contained and independently tunable.
"""

import math
import os

import numpy as np
import torch

STATE_FEATURE_VERSION = 'tempest_ratio_v1_kalman_norm'
STATE_FEATURES = [
    'delta_cwnd',
    'rtt_infl',           # avg_urtt inflation over kalman min-rtt
    'cwnd_over_bdp',      # cwnd / (headroom * estimated BDP)
    'thr_over_peak',      # avg_thr / bw_ref
    'pacing_over_peak',   # pacing_rate / bw_ref
    'inflight_over_cwnd',
    'delta_rtt',
    'retrans_ratio',
    'srtt_infl',          # srtt inflation over kalman min-rtt
    'rtt_scale',          # saturating absolute-timescale anchor
]

# All magnitudes live in [0, 1]; the two signed deltas in [-1, 1].
STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
STATE_HIGH = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)
STATE_LOW_T = torch.from_numpy(STATE_LOW)
STATE_HIGH_T = torch.from_numpy(STATE_HIGH)
STATE_DIM = len(STATE_FEATURES)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return val if math.isfinite(val) else float(default)


class TempestKalmanMinRTT:
    """Asymmetric Kalman tracker for the propagation-RTT floor (in us)."""

    def __init__(self,
                 init_us: float = 20_000.0,
                 q: float = 5.0e-4,
                 r_down: float = 1.0e-3,
                 r_up: float = 5.0,
                 jump_thresh: float = 1.5):
        self._x = max(float(init_us), 1.0) / 1000.0
        self._Q = max(float(q), 1e-12)
        self._R_dn = max(float(r_down), 1e-12)
        self._R_up = max(float(r_up), 1e-12)
        self._jump_thresh = max(float(jump_thresh), 1.0)
        self._P = math.sqrt(self._Q * self._R_up)
        self._ready = False

    def update(self, obs_us: float) -> float:
        obs_ms = max(float(obs_us), 1.0) / 1000.0
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


def reset_tempest_kalman(initial_rtt_us: float = None) -> None:
    global _KALMAN
    init_us = (20_000.0 if initial_rtt_us is None else float(initial_rtt_us))
    init_us = _float_env('SAO_TEMPEST_KALMAN_INIT_US', init_us)
    _KALMAN = TempestKalmanMinRTT(
        init_us=init_us,
        q=_float_env('SAO_TEMPEST_KALMAN_Q', 5.0e-4),
        r_down=_float_env('SAO_TEMPEST_KALMAN_R_DOWN', 1.0e-3),
        r_up=_float_env('SAO_TEMPEST_KALMAN_R_UP', 5.0),
        jump_thresh=_float_env('SAO_TEMPEST_KALMAN_JUMP_THRESH', 1.5),
    )


def update_tempest_kalman_min_rtt(info: dict) -> float:
    global _KALMAN
    if _KALMAN is None:
        reset_tempest_kalman()

    avg_urtt = float(info.get('avg_urtt', 0.0) or 0.0)
    if avg_urtt > 0:
        return _KALMAN.update(avg_urtt)
    return _KALMAN.estimate


def _inflation(rtt_us: float, floor_us: float, max_infl: float) -> float:
    """Map RTT/floor in [1, max_infl] to [0, 1]; 0 == no queue."""
    if floor_us <= 0.0:
        return 0.0
    infl = rtt_us / floor_us
    return float(np.clip((infl - 1.0) / max(max_infl - 1.0, 1e-6), 0.0, 1.0))


def normalize_state(info: dict) -> np.ndarray:
    cwnd = max(int(info.get('cwnd', 1) or 1), 1)
    avg_thr = float(info.get('avg_thr', 0.0) or 0.0)
    avg_urtt = float(info.get('avg_urtt', 0.0) or 0.0)
    srtt_raw = float(info.get('srtt_us', 0.0) or 0.0)
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0 else avg_urtt
    srtt_us = max(srtt_us, 1.0)
    pacing_rate = float(info.get('pacing_rate', 0.0) or 0.0)
    packets_out = float(info.get('packets_out', 0.0) or 0.0)
    retrans_out = float(info.get('retrans_out', 0.0) or 0.0)
    prev_urtt = float(info.get('prev_urtt', 0.0) or 0.0)
    prev_cwnd = float(info.get('prev_cwnd', cwnd) or cwnd)
    peak_thr = float(info.get('peak_thr', 0.0) or 0.0)
    mss = float(info.get('mss', 0.0) or 0.0)
    if mss <= 0.0:
        mss = _float_env('SAO_TEMPEST_MSS_BYTES', 1448.0)

    # Tunables.
    max_infl = _float_env('SAO_TEMPEST_MAX_RTT_INFL', 8.0)
    max_cwnd_bdp = _float_env('SAO_TEMPEST_MAX_CWND_BDP', 8.0)
    rtt_scale_c_us = _float_env('SAO_TEMPEST_RTT_SCALE_US', 50_000.0)

    kalman_min_rtt = max(update_tempest_kalman_min_rtt(info), 1.0)
    info['kalman_min_rtt_us'] = kalman_min_rtt

    bw_ref = max(peak_thr, pacing_rate, 1.0)

    delta_rtt = float(np.clip((avg_urtt - prev_urtt) / max(prev_urtt, 1.0),
                              -1.0, 1.0))
    delta_cwnd = float(np.clip((cwnd - prev_cwnd) / max(prev_cwnd, 1.0),
                               -1.0, 1.0))

    # cwnd normalised by an estimated BDP (bytes). bw_ref is bytes/s, the
    # Kalman floor is us; headroom covers the env's buffer (queue can hold a
    # few BDP), so a full pipe+buffer maps well below 1.0 with room to spare.
    bdp_bytes = bw_ref * (kalman_min_rtt / 1e6)
    cwnd_bytes = cwnd * mss
    cwnd_over_bdp = float(np.clip(
        cwnd_bytes / max(max_cwnd_bdp * bdp_bytes, 1.0), 0.0, 1.0))

    inflight_over_cwnd = float(np.clip(packets_out / max(cwnd, 1), 0.0, 2.0)) / 2.0

    s = np.array([
        delta_cwnd,
        _inflation(avg_urtt, kalman_min_rtt, max_infl),
        cwnd_over_bdp,
        float(np.clip(avg_thr / bw_ref, 0.0, 1.0)),
        float(np.clip(pacing_rate / bw_ref, 0.0, 1.0)),
        inflight_over_cwnd,
        delta_rtt,
        min(retrans_out / max(packets_out, 1.0), 1.0),
        _inflation(srtt_us, kalman_min_rtt, max_infl),
        kalman_min_rtt / (kalman_min_rtt + rtt_scale_c_us),
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(s, STATE_LOW, STATE_HIGH)
