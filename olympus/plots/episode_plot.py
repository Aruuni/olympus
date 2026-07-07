"""
episode_plot.py — per-episode plot for opinion-critic_clean_slate.

Panels:
  1. Throughput (Mbps) + link BW reference
  2. CWND (packets)
  3. RTT: avg_urtt + srtt + min_rtt + scheduled RTT reference
  4. RTT gradient — fractional change (rtt − prev_rtt) / prev_rtt, clipped ±1
  5. CWND multiplier + rolling mean
  6. Reward (raw + 5s rolling mean)
  7. Cumulative return

Columns in state log CSV:
  t_s, option, cwnd_mult, cwnd, avg_thr_mbps, avg_urtt_ms, srtt_ms,
  min_rtt_ms, loss_ratio, reward
"""

import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family']        = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

_BASE_COLORS = ['#4878cf', '#f28e2c', '#59a14f', '#e15759',
                '#b07aa1', '#76b7b2', '#ff9da7', '#9c755f',
                '#bab0ac', '#edc948', '#d37295', '#59a14f']


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    cols = {k: [] for k in
            ['t_s','option','cwnd_mult','cwnd',
             'avg_thr_mbps','avg_urtt_ms','srtt_ms','min_rtt_ms',
             'loss_ratio','reward','kalman_rtt_ms',
             'fair_bw_mbps','active_flows','fairness_cost']}

    def _optional_float(row, key):
        raw = row.get(key, '')
        if raw in ('', None):
            return np.nan
        return float(raw)

    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    cols['t_s'].append(float(row['t_s']))
                    cols['option'].append(int(row['option']))
                    cols['cwnd_mult'].append(float(row['cwnd_mult']))
                    cols['cwnd'].append(float(row['cwnd']))
                    cols['avg_thr_mbps'].append(float(row['avg_thr_mbps']))
                    avg_urtt_ms = float(row['avg_urtt_ms'])
                    cols['avg_urtt_ms'].append(avg_urtt_ms)
                    srtt_raw = row.get('srtt_ms', '')
                    cols['srtt_ms'].append(
                        float(srtt_raw) if srtt_raw not in ('', None) else np.nan)
                    cols['min_rtt_ms'].append(float(row['min_rtt_ms']))
                    cols['loss_ratio'].append(float(row['loss_ratio']))
                    cols['reward'].append(float(row['reward']))
                    cols['kalman_rtt_ms'].append(float(row.get('kalman_rtt_ms', 0)))
                    cols['fair_bw_mbps'].append(_optional_float(row, 'fair_bw_mbps'))
                    cols['active_flows'].append(_optional_float(row, 'active_flows'))
                    cols['fairness_cost'].append(_optional_float(row, 'fairness_cost'))
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return {k: np.array(v) for k, v in cols.items()}


def _has_signal(values: np.ndarray) -> bool:
    if values is None or len(values) == 0:
        return False
    finite = values[np.isfinite(values)]
    return finite.size > 0 and np.nanmax(np.abs(finite)) > 1e-12


def _episode_id_from_path(path: str):
    match = re.search(r'_ep(\d+)', os.path.basename(path))
    return match.group(1) if match else None


def _raynet_trace_paths(state_log_path: str) -> list:
    episode_id = _episode_id_from_path(state_log_path)
    if not episode_id:
        return []
    directory = os.path.dirname(os.path.abspath(state_log_path))
    return sorted(glob.glob(os.path.join(
        directory, f'raynet_trace_ep{episode_id}_slot*.jsonl')))


def _extract_clean_slate_trace(message: dict):
    observations = message.get('observations') or {}
    if not observations:
        return None
    if len(observations) == 1:
        return next(iter(observations.values()))
    for key in ('CleanSlate', '0', 0):
        if key in observations:
            return observations[key]
    return next(iter(observations.values()))


def _load_raynet_trace(state_log_path: str, bw: float, delay: float,
                       link_schedule: list = None):
    rows = []
    for path in _raynet_trace_paths(state_log_path):
        try:
            with open(path) as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = event.get('message') or {}
                    obs = _extract_clean_slate_trace(message)
                    if not obs:
                        continue
                    info = message.get('info') or {}
                    if 'time_s' not in info:
                        continue
                    try:
                        t_s = float(info['time_s'])
                    except (TypeError, ValueError):
                        continue
                    rewards = message.get('rewards') or {}
                    reward = None
                    if rewards:
                        reward = rewards.get('CleanSlate')
                        if reward is None and len(rewards) == 1:
                            reward = next(iter(rewards.values()))
                    rows.append((t_s, obs, reward))
        except OSError:
            continue
    if not rows:
        return None

    rows.sort(key=lambda item: item[0])
    t = np.asarray([item[0] for item in rows], dtype=float)

    def _obs_float(obs, *keys, default=0.0):
        for key in keys:
            raw = obs.get(key)
            if raw not in ('', None):
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        return float(default)

    throughput_norm = np.asarray([
        _obs_float(obs, 'throughput_norm', 'avg_thr')
        for _, obs, _ in rows
    ], dtype=float)
    srtt_norm = np.asarray([
        _obs_float(obs, 'srtt_norm', 'delay_metric')
        for _, obs, _ in rows
    ], dtype=float)
    reward = np.asarray([
        float(reward) if reward not in ('', None) else np.nan
        for _, _, reward in rows
    ], dtype=float)
    loss = np.asarray([
        _obs_float(obs, 'loss_norm', 'loss_rate')
        for _, obs, _ in rows
    ], dtype=float)

    return {
        't_s': t,
        # CleanSlate exposes normalized observations, not physical Mbps/ms.
        # Keep these values on their native scale and label the panels as such.
        'avg_thr_mbps': np.clip(throughput_norm, 0.0, None),
        'avg_urtt_ms': np.clip(srtt_norm, 0.0, None),
        'srtt_ms': np.asarray([
            _obs_float(obs, 'delay_metric')
            for _, obs, _ in rows
        ], dtype=float),
        'min_rtt_ms': np.full(len(t), np.nan, dtype=float),
        'loss_ratio': loss,
        'reward': np.nan_to_num(reward, nan=0.0),
    }


def _apply_raynet_trace_fallback(d: dict, state_log_path: str, bw: float,
                                 delay: float, link_schedule: list = None):
    needs_trace = not (
        _has_signal(d.get('avg_thr_mbps'))
        or _has_signal(d.get('avg_urtt_ms'))
        or _has_signal(d.get('srtt_ms'))
        or _has_signal(d.get('reward'))
    )
    if not needs_trace:
        return d, False
    trace = _load_raynet_trace(
        state_log_path, bw=bw, delay=delay, link_schedule=link_schedule)
    if trace is None:
        return d, False

    t_src = trace['t_s']
    if len(t_src) == 0:
        return d, False
    t_dst = d['t_s']
    for key in ('avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms',
                'min_rtt_ms', 'loss_ratio', 'reward'):
        d[key] = np.interp(t_dst, t_src, trace[key])
    return d, True


def _flow_id_from_path(path: str):
    match = re.search(r'_flow(\d+)(?:\.[^.]+)?$', os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _sibling_flow_logs(path: str) -> list:
    directory = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    ext = ext or '.csv'
    base_stem = re.sub(r'_flow\d+$', '', stem)
    return sorted(
        glob.glob(os.path.join(directory, f'{base_stem}_flow*{ext}')),
        key=lambda p: (
            _flow_id_from_path(p) if _flow_id_from_path(p) is not None else 10**9,
            p,
        ),
    )


def _load_flow_throughput(path: str, trim_tail_s: float = 5.0):
    t_s, thr = [], []
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    t_s.append(float(row['t_s']))
                    thr.append(float(row['avg_thr_mbps']))
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        return None
    if not t_s:
        return None
    t_s = np.asarray(t_s, dtype=float)
    thr = np.asarray(thr, dtype=float)
    if trim_tail_s and len(t_s) > 1:
        mask = t_s <= t_s[-1] - float(trim_tail_s)
        if mask.any():
            t_s = t_s[mask]
            thr = thr[mask]
    flow_id = _flow_id_from_path(path)
    return {
        'path': os.path.abspath(path),
        'flow_id': flow_id,
        't_s': t_s,
        'avg_thr_mbps': thr,
    }


def _sibling_flow_throughputs(path: str, trim_tail_s: float = 5.0) -> list:
    current = os.path.abspath(path)
    flows = []
    for flow_path in _sibling_flow_logs(path):
        if os.path.abspath(flow_path) == current:
            continue
        item = _load_flow_throughput(flow_path, trim_tail_s=trim_tail_s)
        if item is not None and len(item['t_s']):
            flows.append(item)
    return flows


def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < 2:
        return arr
    w = min(window, len(arr))
    out = np.convolve(arr, np.ones(w) / w, mode='valid')
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, out])


def episode_return(state_log_path: str, trim_tail_s: float = 5.0) -> float:
    """Compute the episode return (sum of reward column) without rendering.

    Mirrors plot()'s tail trim so the returned value matches what the
    figure would have shown when per-episode plotting is disabled. Streams just
    `t_s` and `reward` instead of materialising all eleven trace columns.
    """
    t_s, reward = [], []
    try:
        with open(state_log_path, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    t_s.append(float(row['t_s']))
                    reward.append(float(row['reward']))
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        return None
    if not t_s:
        return None
    t_s    = np.array(t_s)
    reward = np.array(reward)
    if trim_tail_s and len(t_s) > 1:
        return float(reward[t_s <= t_s[-1] - float(trim_tail_s)].sum())
    return float(reward.sum())


def _step_series(t_s, base_val, schedule, key, scale=1.0):
    """Build a step-function aligned to t_s from a link schedule list."""
    if not schedule:
        return np.full(len(t_s), base_val)
    bps = [(0.0, base_val)]
    for entry in schedule:
        if key in entry:
            bps.append((float(entry['t']), float(entry[key]) * scale))
    bps.sort()
    out = np.full(len(t_s), base_val)
    for t_off, val in bps:
        out[t_s >= t_off] = val
    return out


# ── Main entry point ──────────────────────────────────────────────────────────

def plot(state_log_path: str, output: str,
         bw: float, delay: float,
         title: str = None,
         link_schedule: list = None,
         trim_tail_s: float = 5.0):
    """
    Read state_log_path, render figure, save to output (.pdf or .png).

    Panels:
      0 — Network conditions: scheduled BW + RTT step functions
      1 — Throughput vs link BW
      2 — CWND
      3 — RTT (avg_urtt + srtt + min_rtt + scheduled RTT)
      4 — RTT gradient (fractional change vs previous sample, ±1 clip)
      5 — CWND multiplier
      6 — Reward
      7 — Cumulative return

    Returns the episode return (sum of rewards), or None if no data.
    """
    d = _load(state_log_path)
    if len(d['t_s']) == 0:
        print(f'[ep_plot] state log empty or missing: {state_log_path}', flush=True)
        return None
    d, used_raynet_trace = _apply_raynet_trace_fallback(
        d, state_log_path, bw=bw, delay=delay, link_schedule=link_schedule)
    sibling_flows = _sibling_flow_throughputs(
        state_log_path, trim_tail_s=trim_tail_s)

    # Trim the iperf teardown tail for emulation runs. Simulations pass 0.
    if trim_tail_s and len(d['t_s']) > 1:
        mask = d['t_s'] <= d['t_s'][-1] - float(trim_tail_s)
        if mask.any():
            d = {k: v[mask] for k, v in d.items()}

    t         = d['t_s']
    x_max = float(t[-1]) if len(t) else 0.0
    for flow in sibling_flows:
        if len(flow['t_s']):
            x_max = max(x_max, float(flow['t_s'][-1]))
    if link_schedule:
        x_max = max(x_max, max(float(e.get('t', 0.0)) for e in link_schedule))
    ep_return = float(d['reward'].sum())

    bw_ref    = _step_series(t, bw,    link_schedule or [], 'bw')
    delay_ref = _step_series(t, delay, link_schedule or [], 'delay')
    fair_bw = d.get('fair_bw_mbps', np.asarray([]))
    fair_bw_mask = (
        len(fair_bw) == len(t)
        and np.isfinite(fair_bw).any()
        and np.nanmax(fair_bw) > 0
    )
    active_flows = d.get('active_flows', np.asarray([]))
    active_flow_mask = (
        len(active_flows) == len(t)
        and np.isfinite(active_flows).any()
        and np.nanmax(active_flows) > 0
    )
    fairness_cost = d.get('fairness_cost', np.asarray([]))
    fairness_cost_mask = (
        len(fairness_cost) == len(t)
        and np.isfinite(fairness_cost).any()
        and np.nanmax(fairness_cost) > 0
    )

    # 5s rolling window
    dt   = np.diff(t).mean() if len(t) > 1 else 0.02
    win  = max(1, int(5.0 / max(dt, 0.001)))

    fig, axes = plt.subplots(8, 1, figsize=(14, 23), sharex=True)
    if title is None:
        title = f'OC-Clean  bw={bw} Mbps  delay={delay} ms'
    fig.suptitle(f'{title}   [return={ep_return:.1f}]',
                 fontsize=12, fontweight='bold', y=0.992)

    def _sched_lines(ax):
        if not link_schedule:
            return
        for e in link_schedule:
            ax.axvline(float(e['t']), color='black',
                       linewidth=0.8, linestyle=':', alpha=0.6)

    # ── 0. Network conditions (scheduled BW + RTT) ───────────────────────────
    ax = axes[0]
    ax_bw = ax
    ax_bw.step(t, bw_ref,    color='#4878cf', linewidth=1.8, where='post', label='BW (Mbps)')
    if fair_bw_mask:
        ax_bw.step(t, fair_bw, color='#59a14f', linewidth=1.4,
                   linestyle='--', where='post', label='fair BW')
    ax_bw.set_ylabel('Link BW (Mbps)', fontsize=9, color='#4878cf')
    ax_bw.tick_params(axis='y', labelcolor='#4878cf')
    ax_bw.set_ylim(bottom=0)
    ax_delay = ax_bw.twinx()
    ax_delay.step(t, delay_ref, color='#e15759', linewidth=1.8, where='post', label='RTT (ms)')
    ax_delay.set_ylabel('Sched RTT (ms)', fontsize=9, color='#e15759')
    ax_delay.tick_params(axis='y', labelcolor='#e15759')
    ax_delay.set_ylim(bottom=0)
    # Combined legend
    lines = ax_bw.get_lines() + ax_delay.get_lines()
    labels = [l.get_label() for l in lines]
    ax_bw.legend(lines, labels, fontsize=8, loc='upper right')
    ax_bw.grid(True, alpha=0.3)
    _sched_lines(ax_bw)

    # ── 1. Throughput ─────────────────────────────────────────────────────────
    ax = axes[1]
    current_flow_id = _flow_id_from_path(state_log_path)
    current_label = (
        f'flow {current_flow_id} thr'
        if current_flow_id is not None else 'avg_thr'
    )
    if used_raynet_trace:
        current_label = 'throughput_norm'
    ax.plot(t, d['avg_thr_mbps'], color='0.35', linewidth=0.35,
            alpha=0.28, label=current_label)
    ax.plot(t, _rolling(d['avg_thr_mbps'], win), color='black',
            linewidth=1.1, label='5s mean')
    for idx, flow in enumerate(sibling_flows):
        flow_id = flow['flow_id']
        label = f'flow {flow_id} thr' if flow_id is not None else 'other flow thr'
        ax.plot(
            flow['t_s'], flow['avg_thr_mbps'],
            color=_BASE_COLORS[idx % len(_BASE_COLORS)],
            linewidth=0.8, alpha=0.55, label=label,
        )
    if not used_raynet_trace:
        ax.plot(t, bw_ref, color='red', linewidth=1.5, label='link BW')
    if fair_bw_mask and not used_raynet_trace:
        ax.plot(t, fair_bw, color='#59a14f', linewidth=1.2,
                linestyle='--', label='fair BW')
    ax.set_ylabel(
        'Throughput obs (norm)' if used_raynet_trace else 'Throughput (Mbps)',
        fontsize=9)
    ax.set_ylim(bottom=0)
    if active_flow_mask:
        ax_flow = ax.twinx()
        ax_flow.step(t, active_flows, color='#7f7f7f', linewidth=0.9,
                     linestyle=':', where='post', label='active flows')
        ax_flow.set_ylabel('Active flows', fontsize=8, color='#7f7f7f')
        ax_flow.tick_params(axis='y', labelcolor='#7f7f7f', labelsize=8)
        ax_flow.set_ylim(0, max(2.0, float(np.nanmax(active_flows)) + 1.0))
        lines = ax.get_lines() + ax_flow.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='upper right')
    else:
        ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _sched_lines(ax)
    if not _has_signal(d['avg_thr_mbps']):
        ax.text(0.5, 0.5, 'throughput signal is all zero',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color='0.35')

    # ── 2. CWND ───────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, d['cwnd'], color='black', linewidth=0.8)
    ax.set_ylabel('CWND (pkts)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    _sched_lines(ax)

    # ── 3. RTT ────────────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, d['avg_urtt_ms'], color='0.35', linewidth=0.35,
            alpha=0.28, label='srtt_norm' if used_raynet_trace else 'avg_urtt')
    ax.plot(t, _rolling(d['avg_urtt_ms'], win), color='black',
            linewidth=1.1,
            label='srtt_norm 5s mean' if used_raynet_trace else 'avg_urtt 5s mean')
    srtt_mask = np.isfinite(d['srtt_ms']) & (d['srtt_ms'] > 0)
    if srtt_mask.any():
        ax.plot(t[srtt_mask], d['srtt_ms'][srtt_mask], color='#f28e2c',
                linewidth=0.45, alpha=0.35,
                label='delay_metric' if used_raynet_trace else 'srtt')
    if not used_raynet_trace:
        ax.plot(t, d['min_rtt_ms'], color='green', linewidth=0.8,
                linestyle='--', alpha=0.7, label='min_rtt')
        ax.plot(t, delay_ref, color='red', linewidth=1.5, label='sched RTT')
    if d['kalman_rtt_ms'].any() and not used_raynet_trace:
        ax.plot(t, d['kalman_rtt_ms'], color='purple', linewidth=1.0,
                linestyle='-.', alpha=0.8, label='kalman min')
    ax.set_ylabel('RTT obs (norm)' if used_raynet_trace else 'RTT (ms)',
                  fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _sched_lines(ax)
    if not (_has_signal(d['avg_urtt_ms']) or _has_signal(d['srtt_ms'])):
        ax.text(0.5, 0.5, 'RTT signal is all zero',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color='0.35')

    # ── 4. RTT gradient (fractional change vs previous sample) ───────────────
    # Matches state dim 6 (delta_rtt): (rtt - prev_rtt) / prev_rtt, clipped to
    # [-1, 1].  Derived from avg_urtt_ms only — not logged separately.
    ax = axes[4]
    rtt_grad = np.zeros(len(t))
    if len(t) > 1:
        prev = np.maximum(d['avg_urtt_ms'][:-1], 1e-6)
        rtt_grad[1:] = np.clip((d['avg_urtt_ms'][1:] - prev) / prev, -1.0, 1.0)
    ax.plot(t, rtt_grad, color='0.35', linewidth=0.35,
            alpha=0.25, label='Δrtt/prev')
    ax.plot(t, _rolling(rtt_grad, win), color='black',
            linewidth=1.1, label='5s mean')
    ax.axhline(0, color='red', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_ylabel(('Obs Δ\n(fraction)' if used_raynet_trace
                   else 'RTT Δ\n(fraction)'), fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _sched_lines(ax)
    if not _has_signal(rtt_grad):
        ax.text(0.5, 0.5, 'RTT fraction has no variation',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color='0.35')

    # ── 5. CWND multiplier ────────────────────────────────────────────────────
    ax = axes[5]
    ax.scatter(t, d['cwnd_mult'], s=4, alpha=0.4, color='steelblue', label='mult')
    ax.plot(t, _rolling(d['cwnd_mult'], win), color='black', linewidth=1.2,
            label='5s mean')
    ax.axhline(1.0, color='red', linewidth=0.6, linestyle='--')
    ax.set_ylabel('CWND mult', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _sched_lines(ax)
    if not _has_signal(d['reward']):
        label = (
            'reward signal is all zero'
            if not used_raynet_trace else 'RayNet reward signal is all zero'
        )
        ax.text(0.5, 0.5, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='0.35')

    # ── 6. Reward ─────────────────────────────────────────────────────────────
    ax = axes[6]
    ax.plot(t, d['reward'],              color='grey',  linewidth=0.5,
            alpha=0.5, label='reward')
    ax.plot(t, _rolling(d['reward'], win), color='black', linewidth=1.2,
            label='5s mean')
    if fairness_cost_mask:
        ax.plot(t, -fairness_cost, color='#e15759', linewidth=0.8,
                alpha=0.8, label='-fairness cost')
    ax.axhline(0,    color='red',   linewidth=0.6, linestyle='--')
    ax.set_ylabel('Reward', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _sched_lines(ax)

    # ── 7. Cumulative return ─────────────────────────────────────────────────
    ax = axes[7]
    _sched_lines(ax)
    ax.set_xlabel('Time (s)', fontsize=9)
    if x_max > 0:
        ax.set_xlim(0, x_max)

    cum_r = np.cumsum(d['reward'])
    ax.plot(t, cum_r, color='navy', linewidth=0.9, linestyle='--', alpha=0.8)
    ax.set_ylabel('Cum. return', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'[ep_plot] saved → {output}  return={ep_return:.1f}', flush=True)
    return ep_return
