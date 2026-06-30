"""
Per-episode plotter for olympus.

Each MAT worker writes a per-flow trace beside the requested state log:

    mat_state_ep000123_a0.csv
    mat_state_ep000123_a1.csv
    ...

This module overlays all available agents in one PDF and returns the summed
episode return. By default it trims the final five seconds to match the
single-agent plotter's iperf teardown handling; simulation callers can disable
that trim.
"""

import csv
import glob
import os
import re

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np


_COLORS = [
    '#4878cf', '#f28e2c', '#59a14f', '#e15759',
    '#b07aa1', '#76b7b2', '#ff9da7', '#9c755f',
    '#bab0ac', '#edc948', '#d37295', '#54a24b',
]


def _agent_id_from_path(path: str):
    m = re.search(r'_a(\d+)\.[^.]+$', os.path.basename(path))
    return int(m.group(1)) if m else None


def agent_trace_paths(state_log_path: str, n_agents: int = None):
    base, ext = os.path.splitext(state_log_path)
    paths = []
    if n_agents is not None:
        for agent_id in range(int(n_agents)):
            path = f'{base}_a{agent_id}{ext}'
            if os.path.exists(path):
                paths.append((agent_id, path))
    else:
        for path in glob.glob(f'{base}_a*{ext}'):
            agent_id = _agent_id_from_path(path)
            if agent_id is not None:
                paths.append((agent_id, path))

    if os.path.exists(state_log_path) and not paths:
        paths.append((0, state_log_path))
    return sorted(paths, key=lambda item: item[0])


def _load(path: str) -> dict:
    cols = {k: [] for k in [
        't_s', 'option', 'cwnd_mult', 'cwnd',
        'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
        'loss_ratio', 'reward', 'kalman_rtt_ms',
        'act_mu', 'act_sig', 'agent_id',
        'fair_bw_mbps', 'active_flows',
    ]}
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    cols['t_s'].append(float(row['t_s']))
                    cols['option'].append(int(float(row.get('option', 0))))
                    cols['cwnd_mult'].append(float(row['cwnd_mult']))
                    cols['cwnd'].append(float(row['cwnd']))
                    cols['avg_thr_mbps'].append(float(row['avg_thr_mbps']))
                    avg_urtt_ms = float(row['avg_urtt_ms'])
                    cols['avg_urtt_ms'].append(avg_urtt_ms)
                    srtt_raw = row.get('srtt_ms', '')
                    cols['srtt_ms'].append(
                        float(srtt_raw) if srtt_raw not in ('', None) else np.nan)
                    cols['min_rtt_ms'].append(float(row['min_rtt_ms']))
                    cols['loss_ratio'].append(float(row.get('loss_ratio', 0)))
                    cols['reward'].append(float(row['reward']))
                    cols['kalman_rtt_ms'].append(float(row.get('kalman_rtt_ms', 0)))
                    cols['act_mu'].append(float(row.get('act_mu', np.nan)))
                    cols['act_sig'].append(float(row.get('act_sig', np.nan)))
                    cols['agent_id'].append(float(row.get('agent_id', np.nan)))
                    fair_bw = row.get('fair_bw_mbps', '')
                    active = row.get('active_flows', '')
                    cols['fair_bw_mbps'].append(
                        float(fair_bw) if fair_bw not in ('', None) else np.nan)
                    cols['active_flows'].append(
                        float(active) if active not in ('', None) else np.nan)
                except (TypeError, ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}


def load_agent_traces(state_log_path: str, n_agents: int = None,
                      trim_tail_s: float = 5.0):
    traces = []
    for agent_id, path in agent_trace_paths(state_log_path, n_agents=n_agents):
        data = _load(path)
        if len(data['t_s']) == 0:
            continue
        if trim_tail_s and len(data['t_s']) > 1:
            mask = data['t_s'] <= data['t_s'][-1] - float(trim_tail_s)
            if mask.any():
                data = {k: v[mask] for k, v in data.items()}
        traces.append({'agent_id': agent_id, 'path': path, 'data': data})
    return traces


def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    w = min(max(1, int(window)), len(arr))
    out = np.convolve(arr, np.ones(w) / w, mode='valid')
    return np.concatenate([np.full(w - 1, np.nan), out])


def _step_series(t_s, base_val, schedule, key, scale=1.0):
    if len(t_s) == 0:
        return np.asarray([], dtype=np.float32)
    bps = [(0.0, float(base_val))]
    for entry in schedule or []:
        if key in entry:
            bps.append((float(entry['t']), float(entry[key]) * scale))
    bps.sort()
    out = np.full(len(t_s), float(base_val), dtype=np.float32)
    for t_off, val in bps:
        out[t_s >= t_off] = val
    return out


def _aligned_matrix(traces, key):
    if not traces:
        return np.asarray([]), np.asarray([[]])
    min_len = min(len(t['data'][key]) for t in traces)
    if min_len == 0:
        return np.asarray([]), np.asarray([[]])
    t_ref = traces[0]['data']['t_s'][:min_len]
    mat = np.vstack([t['data'][key][:min_len] for t in traces])
    return t_ref, mat


def _jains_matrix(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    x = np.clip(values.astype(np.float64), 0.0, None)
    n = float(x.shape[0])
    num = np.square(x.sum(axis=0))
    den = np.maximum(n * np.square(x).sum(axis=0), 1e-9)
    return (num / den).astype(np.float32)


def _r_fair_matrix(values: np.ndarray) -> np.ndarray:
    """Attached unfairness metric over each column of average throughputs."""
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    x = np.clip(values.astype(np.float64), 0.0, None)
    n = float(x.shape[0])
    total = x.sum(axis=0)
    mean = total / max(n, 1.0)
    numerator = np.square(x - mean).sum(axis=0)
    denominator = n * np.square(total)
    out = np.zeros_like(total, dtype=np.float64)
    valid = (n > 1.0) & (total > 1e-9)
    out[valid] = np.sqrt(
        numerator[valid] / np.maximum(denominator[valid], 1e-9))
    return out.astype(np.float32)


def fairness_series(state_log_path: str, n_agents: int = None,
                    trim_tail_s: float = 5.0):
    traces = load_agent_traces(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    t, thr = _aligned_matrix(traces, 'avg_thr_mbps')
    return t, _jains_matrix(thr)


def r_fair_series(state_log_path: str, n_agents: int = None,
                  trim_tail_s: float = 5.0):
    traces = load_agent_traces(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    t, thr = _aligned_matrix(traces, 'avg_thr_mbps')
    return t, _r_fair_matrix(thr)


def episode_return(state_log_path: str, n_agents: int = None,
                   trim_tail_s: float = 5.0) -> float:
    traces = load_agent_traces(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    if not traces:
        return None
    total = 0.0
    found = False
    for item in traces:
        rewards = item['data']['reward']
        if len(rewards) == 0:
            continue
        total += float(np.nansum(rewards))
        found = True
    return total if found else None


def _sched_lines(ax, link_schedule):
    for entry in link_schedule or []:
        ax.axvline(float(entry['t']), color='black',
                   linewidth=0.8, linestyle=':', alpha=0.55)


def _agent_label(agent_id):
    return f'agent {int(agent_id)}'


def plot(state_log_path: str, output: str, bw: float, delay: float,
         title: str = None, link_schedule: list = None, n_agents: int = None,
         trim_tail_s: float = 5.0):
    traces = load_agent_traces(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    if not traces:
        print(f'[multi_ep_plot] no agent traces for: {state_log_path}', flush=True)
        return None

    ep_return = episode_return(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    t_ref = max((item['data']['t_s'] for item in traces),
                key=lambda arr: len(arr))
    bw_ref = _step_series(t_ref, bw, link_schedule or [], 'bw')
    delay_ref = _step_series(t_ref, delay, link_schedule or [], 'delay')
    fair_t, fair = r_fair_series(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    fair_mean = float(np.nanmean(fair)) if len(fair) else np.nan

    fig, axes = plt.subplots(8, 1, figsize=(14, 22), sharex=False)
    if title is None:
        title = f'MAT multi-agent  flows={len(traces)}  bw={bw} Mbps  delay={delay} ms'
    fair_text = f' R_fair={fair_mean:.3f}' if np.isfinite(fair_mean) else ''
    fig.suptitle(f'{title}  [sum return={ep_return:.1f}{fair_text}]',
                 fontsize=13, fontweight='bold')

    colors = {item['agent_id']: _COLORS[i % len(_COLORS)]
              for i, item in enumerate(traces)}

    ax = axes[0]
    ax_bw = ax
    ax_bw.step(t_ref, bw_ref, color='#4878cf', linewidth=1.8,
               where='post', label='BW (Mbps)')
    ax_bw.set_ylabel('Link BW (Mbps)', fontsize=9, color='#4878cf')
    ax_bw.tick_params(axis='y', labelcolor='#4878cf')
    ax_bw.set_ylim(bottom=0)
    ax_delay = ax_bw.twinx()
    ax_delay.step(t_ref, delay_ref, color='#e15759', linewidth=1.8,
                  where='post', label='RTT (ms)')
    ax_delay.set_ylabel('Sched RTT (ms)', fontsize=9, color='#e15759')
    ax_delay.tick_params(axis='y', labelcolor='#e15759')
    ax_delay.set_ylim(bottom=0)
    lines = ax_bw.get_lines() + ax_delay.get_lines()
    ax_bw.legend(lines, [line.get_label() for line in lines],
                 fontsize=8, loc='upper right')
    ax_bw.set_title('Scheduled network conditions', fontsize=9, loc='left')
    ax_bw.grid(True, alpha=0.3)
    _sched_lines(ax_bw, link_schedule)

    ax = axes[1]
    for item in traces:
        d = item['data']
        aid = item['agent_id']
        ax.plot(d['t_s'], d['avg_thr_mbps'], linewidth=0.8,
                color=colors[aid], alpha=0.75, label=_agent_label(aid))
    agg_t, agg_thr = _aligned_matrix(traces, 'avg_thr_mbps')
    if agg_thr.size:
        ax.plot(agg_t, agg_thr.sum(axis=0), color='black', linewidth=1.4,
                label='sum throughput')
    fair_bw = traces[0]['data'].get('fair_bw_mbps', np.asarray([]))
    fair_bw_t = traces[0]['data']['t_s']
    if (len(fair_bw) == len(fair_bw_t)
            and np.isfinite(fair_bw).any()):
        ax.step(
            fair_bw_t, fair_bw,
            color='#59a14f', linewidth=1.2, linestyle='--',
            where='post', label='ground-truth fair BW / flow',
        )
    ax.plot(t_ref, bw_ref, color='red', linewidth=1.4, label='link BW')
    ax.set_ylabel('Throughput (Mbps)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax, link_schedule)

    ax = axes[2]
    for item in traces:
        d = item['data']
        aid = item['agent_id']
        ax.plot(d['t_s'], d['cwnd'], linewidth=0.8,
                color=colors[aid], alpha=0.8, label=_agent_label(aid))
    ax.set_ylabel('CWND (pkts)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax, link_schedule)

    ax = axes[3]
    for item in traces:
        d = item['data']
        aid = item['agent_id']
        ax.plot(d['t_s'], d['avg_urtt_ms'], linewidth=0.8,
                color=colors[aid], alpha=0.45, linestyle=':',
                label=f'{_agent_label(aid)} avg')
        srtt_mask = np.isfinite(d['srtt_ms']) & (d['srtt_ms'] > 0)
        if srtt_mask.any():
            ax.plot(d['t_s'][srtt_mask], d['srtt_ms'][srtt_mask], linewidth=0.9,
                    color=colors[aid], alpha=0.9, label=f'{_agent_label(aid)} srtt')
        ax.plot(d['t_s'], d['min_rtt_ms'], linewidth=0.6,
                color=colors[aid], alpha=0.35, linestyle='--')
    ax.plot(t_ref, delay_ref, color='red', linewidth=1.4, label='sched RTT')
    ax.set_ylabel('RTT (ms)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax, link_schedule)

    ax = axes[4]
    for item in traces:
        d = item['data']
        aid = item['agent_id']
        rtt = d['avg_urtt_ms']
        grad = np.zeros(len(rtt), dtype=np.float32)
        if len(rtt) > 1:
            prev = np.maximum(rtt[:-1], 1e-6)
            grad[1:] = np.clip((rtt[1:] - prev) / prev, -1.0, 1.0)
        ax.plot(d['t_s'], grad, linewidth=0.8, color=colors[aid],
                alpha=0.8, label=_agent_label(aid))
    ax.axhline(0, color='red', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_ylabel('RTT delta\n(fraction)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax, link_schedule)

    ax = axes[5]
    for item in traces:
        d = item['data']
        aid = item['agent_id']
        t = d['t_s']
        dt = np.diff(t).mean() if len(t) > 1 else 0.02
        win = max(1, int(5.0 / max(float(dt), 0.001)))
        ax.scatter(t, d['cwnd_mult'], s=4, alpha=0.25, color=colors[aid])
        ax.plot(t, _rolling(d['cwnd_mult'], win), linewidth=1.0,
                color=colors[aid], alpha=0.9, label=_agent_label(aid))
    ax.axhline(1.0, color='red', linewidth=0.6, linestyle='--')
    ax.set_ylabel('CWND mult', fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax, link_schedule)

    ax = axes[6]
    for item in traces:
        d = item['data']
        aid = item['agent_id']
        t = d['t_s']
        dt = np.diff(t).mean() if len(t) > 1 else 0.02
        win = max(1, int(5.0 / max(float(dt), 0.001)))
        ax.plot(t, d['reward'], color=colors[aid], linewidth=0.45,
                alpha=0.25)
        ax.plot(t, _rolling(d['reward'], win), color=colors[aid],
                linewidth=1.1, label=_agent_label(aid))
    ax.axhline(0, color='red', linewidth=0.6, linestyle='--')
    ax.set_ylabel('Reward', fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax, link_schedule)

    ax = axes[7]
    for item in traces:
        d = item['data']
        aid = item['agent_id']
        ax.plot(d['t_s'], np.cumsum(d['reward']), color=colors[aid],
                linewidth=0.9, label=f'{_agent_label(aid)} return')
    if len(fair):
        ax_f = ax.twinx()
        ax_f.plot(fair_t, fair, color='black', linewidth=1.0,
                  linestyle='--', alpha=0.75, label='R_fair')
        ax_f.set_ylabel('R_fair (lower is fairer)', fontsize=8, color='black')
        ax_f.set_ylim(0, max(0.55, float(np.nanmax(fair)) * 1.1))
        ax_f.tick_params(axis='y', labelsize=8)
        lines = ax.get_lines() + ax_f.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, fontsize=7, loc='upper left',
                  ncol=2, framealpha=0.85)
    else:
        ax.legend(fontsize=7, loc='upper left', ncol=2, framealpha=0.85)
    ax.set_ylabel('Cum. return', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax, link_schedule)

    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'[multi_ep_plot] saved -> {output}  return={ep_return:.1f}',
          flush=True)
    return ep_return
