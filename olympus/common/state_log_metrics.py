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


# ── Multi-agent variant ──────────────────────────────────────────────────────
# Aggregates across N per-agent traces (summed goodput, averaged RTT, Jain
# fairness). Relocated from the former inference_benchmark.py; consumed by the
# responsiveness benchmark, which passes the flow count as n_agents.

def _nanmean(values, default=math.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def _nanpercentile(values, pct, default=math.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.percentile(arr, pct))


def _aligned_matrix(traces, key):
    if not traces:
        return np.asarray([]), np.asarray([[]])
    min_len = min(len(t['data'][key]) for t in traces)
    if min_len <= 0:
        return np.asarray([]), np.asarray([[]])
    t = traces[0]['data']['t_s'][:min_len]
    mat = np.vstack([item['data'][key][:min_len] for item in traces])
    return t, mat


def multiflow_state_log_metrics(state_log_path: str, n_agents: int, ep_return):
    """Summarise a multi-agent state log across n_agents per-agent traces."""
    # Lazy import keeps the single-agent path free of the matplotlib-backed
    # plot library that supplies the trace loader / fairness series.
    from olympus.plots.multi_flow_episode_plot import fairness_series, load_agent_traces

    traces = load_agent_traces(state_log_path, n_agents=n_agents)
    if not traces:
        return {}

    t, thr = _aligned_matrix(traces, 'avg_thr_mbps')
    _, rtt = _aligned_matrix(traces, 'avg_urtt_ms')
    _, srtt = _aligned_matrix(traces, 'srtt_ms')
    _, min_rtt = _aligned_matrix(traces, 'min_rtt_ms')
    _, kalman_rtt = _aligned_matrix(traces, 'kalman_rtt_ms')
    _, cwnd = _aligned_matrix(traces, 'cwnd')
    _, mult = _aligned_matrix(traces, 'cwnd_mult')
    _, loss = _aligned_matrix(traces, 'loss_ratio')
    _, reward = _aligned_matrix(traces, 'reward')
    _, fair = fairness_series(state_log_path, n_agents=n_agents)

    if thr.size == 0:
        return {}
    if not np.isfinite(srtt).any():
        srtt = rtt

    sum_thr = thr.sum(axis=0)
    mean_reward_step = reward.mean(axis=0) if reward.size else np.asarray([])
    ret = float(ep_return) if ep_return is not None else float(np.nansum(reward))
    duration_s = float(max(t[-1] - t[0], 0.0)) if len(t) > 1 else 0.0
    total_steps = int(sum(len(item['data']['t_s']) for item in traces))

    return {
        'return': ret,
        'duration_s': duration_s,
        'score_per_second': ret / duration_s if duration_s > 0 else math.nan,
        'steps': total_steps,
        'n_agents_seen': int(len(traces)),
        'mean_thr_mbps': _nanmean(sum_thr),
        'p50_thr_mbps': _nanpercentile(sum_thr, 50),
        'p05_thr_mbps': _nanpercentile(sum_thr, 5),
        'mean_agent_thr_mbps': _nanmean(thr),
        'mean_rtt_ms': _nanmean(rtt),
        'p95_rtt_ms': _nanpercentile(rtt, 95),
        'mean_srtt_ms': _nanmean(srtt),
        'p95_srtt_ms': _nanpercentile(srtt, 95),
        'mean_min_rtt_ms': _nanmean(min_rtt),
        'mean_kalman_rtt_ms': _nanmean(kalman_rtt),
        'mean_cwnd': _nanmean(cwnd),
        'mean_cwnd_mult': _nanmean(mult),
        'frac_mult_below_1': float(np.nanmean(mult < 1.0)),
        'frac_mult_above_1': float(np.nanmean(mult > 1.0)),
        'mean_loss_ratio': _nanmean(loss),
        'mean_reward': _nanmean(mean_reward_step),
        'fairness_mean': _nanmean(fair),
        'fairness_p05': _nanpercentile(fair, 5),
        'switch_rate': 0.0,
        'dominant_option_share': 1.0,
        'option_entropy': 0.0,
        'n_options_seen': 1,
    }
