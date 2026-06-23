"""
episode_plot.py — per-episode plot for opinion-critic_clean_slate.

Panels:
  1. Throughput (Mbps) + link BW reference
  2. CWND (packets)
  3. RTT: avg_urtt + srtt + min_rtt + scheduled RTT reference
  4. RTT gradient — fractional change (rtt − prev_rtt) / prev_rtt, clipped ±1
  5. CWND multiplier + rolling mean
  6. Reward (raw + 5s rolling mean)
  7. Option timeline + cumulative return

Columns in state log CSV:
  t_s, option, cwnd_mult, cwnd, avg_thr_mbps, avg_urtt_ms, srtt_ms,
  min_rtt_ms, loss_ratio, reward
"""

import csv
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family']        = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Option colours (extend automatically for any n_options) ──────────────────

_BASE_COLORS = ['#4878cf', '#f28e2c', '#59a14f', '#e15759',
                '#b07aa1', '#76b7b2', '#ff9da7', '#9c755f',
                '#bab0ac', '#edc948', '#d37295', '#59a14f']

def _opt_colors(n):
    import itertools
    return list(itertools.islice(itertools.cycle(_BASE_COLORS), n))


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


def _option_spans(t_s, options):
    if len(t_s) == 0:
        return []
    spans, start = [], 0
    for i in range(1, len(options)):
        if options[i] != options[i - 1]:
            spans.append((t_s[start], t_s[i], int(options[start])))
            start = i
    spans.append((t_s[start], t_s[-1], int(options[start])))
    return spans


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
      7 — Option timeline + cumulative return

    Returns the episode return (sum of rewards), or None if no data.
    """
    d = _load(state_log_path)
    if len(d['t_s']) == 0:
        print(f'[ep_plot] state log empty or missing: {state_log_path}', flush=True)
        return None
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
    spans     = _option_spans(t, d['option'])

    # Derive n_options from data so the plot works for any config value
    n_options  = max(int(d['option'].max()) + 1, 1) if len(d['option']) else 4
    opt_colors = _opt_colors(n_options)

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

    fig, axes = plt.subplots(8, 1, figsize=(14, 22), sharex=True)
    if title is None:
        title = f'OC-Clean  bw={bw} Mbps  delay={delay} ms'
    fig.suptitle(f'{title}   [return={ep_return:.1f}]',
                 fontsize=13, fontweight='bold')

    def _shade(ax):
        for ts, te, opt in spans:
            ax.axvspan(ts, te, color=opt_colors[opt % n_options],
                       alpha=0.10, linewidth=0)

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
    ax_bw.set_title('Scheduled network conditions', fontsize=9, loc='left')
    ax_bw.grid(True, alpha=0.3)
    _sched_lines(ax_bw)

    # ── 1. Throughput ─────────────────────────────────────────────────────────
    ax = axes[1]
    current_flow_id = _flow_id_from_path(state_log_path)
    current_label = (
        f'flow {current_flow_id} thr'
        if current_flow_id is not None else 'avg_thr'
    )
    ax.plot(t, d['avg_thr_mbps'], color='black', linewidth=0.8,
            label=current_label)
    for idx, flow in enumerate(sibling_flows):
        flow_id = flow['flow_id']
        label = f'flow {flow_id} thr' if flow_id is not None else 'other flow thr'
        ax.plot(
            flow['t_s'], flow['avg_thr_mbps'],
            color=_BASE_COLORS[idx % len(_BASE_COLORS)],
            linewidth=0.8, alpha=0.55, label=label,
        )
    ax.plot(t, bw_ref,                  color='red',   linewidth=1.5, label='link BW')
    if fair_bw_mask:
        ax.plot(t, fair_bw, color='#59a14f', linewidth=1.2,
                linestyle='--', label='fair BW')
    ax.set_ylabel('Throughput (Mbps)', fontsize=9)
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
    _shade(ax); _sched_lines(ax)

    # ── 2. CWND ───────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, d['cwnd'], color='black', linewidth=0.8)
    ax.set_ylabel('CWND (pkts)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    _shade(ax); _sched_lines(ax)

    # ── 3. RTT ────────────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, d['avg_urtt_ms'], color='black', linewidth=0.8, label='avg_urtt')
    srtt_mask = np.isfinite(d['srtt_ms']) & (d['srtt_ms'] > 0)
    if srtt_mask.any():
        ax.plot(t[srtt_mask], d['srtt_ms'][srtt_mask], color='#f28e2c',
                linewidth=0.9, alpha=0.9, label='srtt')
    ax.plot(t, d['min_rtt_ms'], color='green', linewidth=0.8,
            linestyle='--', alpha=0.7, label='min_rtt')
    ax.plot(t, delay_ref,       color='red',   linewidth=1.5, label='sched RTT')
    if d['kalman_rtt_ms'].any():
        ax.plot(t, d['kalman_rtt_ms'], color='purple', linewidth=1.0,
                linestyle='-.', alpha=0.8, label='kalman min')
    ax.set_ylabel('RTT (ms)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _shade(ax); _sched_lines(ax)

    # ── 4. RTT gradient (fractional change vs previous sample) ───────────────
    # Matches state dim 6 (delta_rtt): (rtt - prev_rtt) / prev_rtt, clipped to
    # [-1, 1].  Derived from avg_urtt_ms only — not logged separately.
    ax = axes[4]
    rtt_grad = np.zeros(len(t))
    if len(t) > 1:
        prev = np.maximum(d['avg_urtt_ms'][:-1], 1e-6)
        rtt_grad[1:] = np.clip((d['avg_urtt_ms'][1:] - prev) / prev, -1.0, 1.0)
    ax.plot(t, rtt_grad, color='black', linewidth=0.8, label='Δrtt/prev')
    ax.axhline(0, color='red', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_ylabel('RTT Δ\n(fraction)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _shade(ax); _sched_lines(ax)

    # ── 5. CWND multiplier ────────────────────────────────────────────────────
    ax = axes[5]
    ax.scatter(t, d['cwnd_mult'], s=4, alpha=0.4, color='steelblue', label='mult')
    ax.plot(t, _rolling(d['cwnd_mult'], win), color='black', linewidth=1.2,
            label='5s mean')
    ax.axhline(1.0, color='red', linewidth=0.6, linestyle='--')
    ax.set_ylabel('CWND mult', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    _shade(ax); _sched_lines(ax)

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
    _shade(ax); _sched_lines(ax)

    # ── 7. Option timeline + cumulative return ────────────────────────────────
    ax = axes[7]
    for ts, te, opt in spans:
        c = opt_colors[opt % n_options]
        ax.barh(0, te - ts, left=ts, height=0.6, color=c, align='center')
        if te - ts > 1.0:
            ax.text((ts + te) / 2, 0, f'opt-{opt}',
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold', clip_on=True)
    _sched_lines(ax)
    ax.set_ylabel('Option', fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel('Time (s)', fontsize=9)
    if x_max > 0:
        ax.set_xlim(0, x_max)

    cum_r = np.cumsum(d['reward'])
    ax_r  = ax.twinx()
    ax_r.plot(t, cum_r, color='navy', linewidth=0.8, linestyle='--', alpha=0.7)
    ax_r.set_ylabel('Cum. return', fontsize=7, color='navy')
    ax_r.tick_params(axis='y', labelcolor='navy', labelsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=opt_colors[i], label=f'opt-{i}')
               for i in range(n_options)]
    fig.legend(handles=patches, loc='lower center', ncol=n_options,
               fontsize=9, title='Option', framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'[ep_plot] saved → {output}  return={ep_return:.1f}', flush=True)
    return ep_return
