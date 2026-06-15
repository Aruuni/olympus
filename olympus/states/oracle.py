"""Oracle TCP state — kalman.py with the Kalman filter swapped for the
scheduled RTT.

Identical observation layout and bounds as kalman.py; the only behavioural
difference is the final feature slot. Where kalman.py runs a 1-D asymmetric
Kalman tracker over the noisy ``avg_urtt`` samples, this plugin reads the
emulated link's scheduled RTT directly from dedicated env vars
(``SAO_ORACLE_BASE_RTT_US`` + ``SAO_ORACLE_LINK_SCHEDULE`` +
``SAO_ORACLE_EPISODE_START``) that the orchestrator only sets when
``runtime.state == 'oracle'``. That is, the agent receives the *ground
truth* propagation RTT for the current link phase, not an estimate — and
no other state plugin can read these env vars because they aren't set
when a different state is configured.

For drop-in compatibility with rewards that read ``info['kalman_min_rtt_us']``
(notably the tempest family), we still populate that key — with the oracle
value, since by construction it is what a perfect Kalman tracker would
report.
"""

import json
import math
import os
import time

import numpy as np
import torch


STATE_FEATURE_VERSION = 'oracle_observation_v1'
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
    'sched_rtt',
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


def _parse_schedule():
    """Parse SAO_ORACLE_LINK_SCHEDULE into a sorted list of (t_abs, rtt_us)."""
    raw = os.environ.get('SAO_ORACLE_LINK_SCHEDULE', '')
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except (TypeError, ValueError):
        return []
    try:
        episode_start = float(os.environ.get('SAO_ORACLE_EPISODE_START', '0') or 0.0)
    except (TypeError, ValueError):
        episode_start = 0.0
    out = []
    for entry in entries or []:
        if 't' not in entry or 'delay' not in entry:
            continue
        try:
            t_abs = episode_start + float(entry['t'])
            rtt_us = float(entry['delay']) * 1_000.0
        except (TypeError, ValueError):
            continue
        out.append((t_abs, rtt_us))
    out.sort()
    return out


def _initial_rtt_us() -> float:
    raw = os.environ.get('SAO_ORACLE_BASE_RTT_US')
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 20_000.0
    return v if v > 0.0 and math.isfinite(v) else 20_000.0


_INITIAL_RTT_US = _initial_rtt_us()
_SCHEDULE = _parse_schedule()


def reset_oracle(initial_rtt_us: float = None) -> None:
    """Re-read OC_BASE_RTT_US / OC_LINK_SCHEDULE / OC_EPISODE_START.

    Mirrors kalman.reset_kalman so callers that re-init the state pipeline
    between episodes get a fresh schedule snapshot.
    """
    global _INITIAL_RTT_US, _SCHEDULE
    _INITIAL_RTT_US = (float(initial_rtt_us)
                       if initial_rtt_us is not None else _initial_rtt_us())
    _SCHEDULE = _parse_schedule()


def scheduled_rtt_us() -> float:
    """Current scheduled RTT in microseconds, honouring link-schedule changes."""
    if not _SCHEDULE:
        return _INITIAL_RTT_US
    now = time.monotonic()
    val = _INITIAL_RTT_US
    for t_abs, rtt_us in _SCHEDULE:
        if now >= t_abs:
            val = rtt_us
        else:
            break
    return val


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
    sched_rtt = scheduled_rtt_us()
    # Drop-in for rewards that still read kalman_min_rtt_us (tempest etc.):
    # the oracle value IS the ground truth, so feed it through that key too.
    info['kalman_min_rtt_us'] = sched_rtt

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
        sched_rtt / 1e5,
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=STATE_HIGH, neginf=STATE_LOW)
    return np.clip(s, STATE_LOW, STATE_HIGH)
