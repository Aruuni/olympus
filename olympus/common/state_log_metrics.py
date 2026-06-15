"""Per-episode metric aggregation over a single-agent state-log CSV.

Reads a state log written by any of the SAO worker scripts (column header
includes t_s, avg_thr_mbps, avg_urtt_ms, srtt_ms, min_rtt_ms, cwnd,
cwnd_mult, loss_ratio, reward, kalman_rtt_ms, option, ...) and returns
the summary dict consumed by benchmark _metric_row() builders.
"""

import csv
import math
import os
from collections import Counter

import numpy as np


_METRIC_KEYS_DEFAULT = {
    'return': '',
    'duration_s': '',
    'score_per_second': '',
    'steps': 0,
    'mean_thr_mbps': '',
    'p50_thr_mbps': '',
    'p05_thr_mbps': '',
    'mean_rtt_ms': '',
    'p95_rtt_ms': '',
    'mean_srtt_ms': '',
    'p95_srtt_ms': '',
    'mean_min_rtt_ms': '',
    'mean_kalman_rtt_ms': '',
    'mean_cwnd': '',
    'mean_cwnd_mult': '',
    'frac_mult_below_1': '',
    'frac_mult_above_1': '',
    'mean_loss_ratio': '',
    'mean_reward': '',
    'switch_rate': '',
    'dominant_option_share': '',
    'option_entropy': '',
    'n_options_seen': 0,
}


def _to_float_array(rows, key):
    out = []
    for row in rows:
        raw = row.get(key, '')
        if raw is None or raw == '':
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return np.asarray(out, dtype=float)


def _mean(arr):
    return float(np.mean(arr)) if arr.size else ''


def _percentile(arr, q):
    return float(np.percentile(arr, q)) if arr.size else ''


def _to_int_array(rows, key):
    out = []
    for row in rows:
        raw = row.get(key, '')
        if raw is None or raw == '':
            continue
        try:
            val = int(float(raw))
        except (TypeError, ValueError):
            continue
        out.append(val)
    return out


def _option_stats(options):
    if not options:
        return {'switch_rate': '', 'dominant_option_share': '',
                'option_entropy': '', 'n_options_seen': 0}
    switches = sum(1 for a, b in zip(options[:-1], options[1:]) if a != b)
    switch_rate = switches / max(len(options) - 1, 1)
    counts = Counter(options)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    dominant_share = max(probs)
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    return {
        'switch_rate': float(switch_rate),
        'dominant_option_share': float(dominant_share),
        'option_entropy': float(entropy),
        'n_options_seen': len(counts),
    }


def state_log_metrics(state_log_path: str, ep_return=None) -> dict:
    """Summarise an SAO single-agent state-log CSV.

    Returns a dict keyed to match the benchmark METRIC_FIELDS schema.
    Missing or unreadable inputs yield an empty dict so the caller fills
    each METRIC_FIELDS entry with the default empty string via setdefault.
    """
    if not state_log_path or not os.path.exists(state_log_path):
        return {}

    try:
        with open(state_log_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (OSError, csv.Error):
        return {}

    if not rows:
        return {}

    thr = _to_float_array(rows, 'avg_thr_mbps')
    rtt = _to_float_array(rows, 'avg_urtt_ms')
    srtt = _to_float_array(rows, 'srtt_ms')
    min_rtt = _to_float_array(rows, 'min_rtt_ms')
    kalman_rtt = _to_float_array(rows, 'kalman_rtt_ms')
    cwnd = _to_float_array(rows, 'cwnd')
    mult = _to_float_array(rows, 'cwnd_mult')
    loss = _to_float_array(rows, 'loss_ratio')
    reward = _to_float_array(rows, 'reward')
    t_s = _to_float_array(rows, 't_s')

    if srtt.size == 0:
        srtt = rtt

    duration_s = float(t_s[-1] - t_s[0]) if t_s.size >= 2 else 0.0
    if ep_return is not None:
        try:
            ret = float(ep_return)
        except (TypeError, ValueError):
            ret = float(np.nansum(reward)) if reward.size else 0.0
    else:
        ret = float(np.nansum(reward)) if reward.size else 0.0
    score_per_second = ret / duration_s if duration_s > 0 else ''

    if mult.size:
        frac_below = float(np.mean(mult < 1.0))
        frac_above = float(np.mean(mult > 1.0))
    else:
        frac_below = ''
        frac_above = ''

    metrics = dict(_METRIC_KEYS_DEFAULT)
    metrics.update({
        'return': ret,
        'duration_s': duration_s,
        'score_per_second': score_per_second,
        'steps': len(rows),
        'mean_thr_mbps': _mean(thr),
        'p50_thr_mbps': _percentile(thr, 50),
        'p05_thr_mbps': _percentile(thr, 5),
        'mean_rtt_ms': _mean(rtt),
        'p95_rtt_ms': _percentile(rtt, 95),
        'mean_srtt_ms': _mean(srtt),
        'p95_srtt_ms': _percentile(srtt, 95),
        'mean_min_rtt_ms': _mean(min_rtt),
        'mean_kalman_rtt_ms': _mean(kalman_rtt),
        'mean_cwnd': _mean(cwnd),
        'mean_cwnd_mult': _mean(mult),
        'frac_mult_below_1': frac_below,
        'frac_mult_above_1': frac_above,
        'mean_loss_ratio': _mean(loss),
        'mean_reward': _mean(reward),
    })
    metrics.update(_option_stats(_to_int_array(rows, 'option')))
    return metrics


_state_log_metrics = state_log_metrics
