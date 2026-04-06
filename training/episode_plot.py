"""
training/episode_plot.py — shared plotting helper used by both the training
orchestrator (per-episode PDFs) and inference_test.py.

Call plot() with a state-log CSV and an iperf3 JSON to produce a 4-panel
figure:  goodput  /  CWND  /  RTT  /  arm timeline.
"""

import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Arm colours ───────────────────────────────────────────────────────────────
ARM_IDS    = [0,     2,      5,       12,     13]
ARM_NAMES  = ['cubic', 'bbr1', 'vegas', 'bbr3', 'astraea']
ARM_COLORS = {
    'cubic':   '#4878cf',
    'bbr1':    '#f28e2c',
    'vegas':   '#59a14f',
    'bbr3':    '#e15759',
    'astraea': '#b07aa1',
}
_ID_TO_NAME = dict(zip(ARM_IDS, ARM_NAMES))


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_state_log(path: str) -> dict:
    cols = {
        't_s': [], 'arm_id': [], 'arm_name': [],
        'rtt_us': [], 'min_rtt_us': [], 'cwnd_mss': [],
        'delivery_rate': [], 'reward': [], 'lost': [],
    }
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cols['t_s'].append(float(row['t_s']))
                cols['arm_id'].append(int(row['arm_id']))
                cols['arm_name'].append(row['arm_name'])
                cols['rtt_us'].append(float(row['rtt_us']))
                cols['min_rtt_us'].append(float(row.get('min_rtt_us', row.get('min_rtt_us', 0))))
                cols['cwnd_mss'].append(int(row['cwnd_mss']))
                cols['delivery_rate'].append(float(row['delivery_rate']))
                cols['reward'].append(float(row['reward']))
                cols['lost'].append(int(row['lost']))
            except (ValueError, KeyError):
                pass
    return {k: np.array(v) for k, v in cols.items()}


def _load_iperf_json(path: str):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return np.array([]), np.array([])
    times, bps = [], []
    for iv in data.get('intervals', []):
        s = iv.get('sum') or (iv.get('streams') or [{}])[0]
        times.append(s.get('end', s.get('start', 0)))
        bps.append(s.get('bits_per_second', 0))
    return np.array(times), np.array(bps)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arm_spans(t_s, arm_names):
    if len(t_s) == 0:
        return []
    spans, start_i = [], 0
    for i in range(1, len(arm_names)):
        if arm_names[i] != arm_names[i - 1]:
            spans.append((t_s[start_i], t_s[i], arm_names[start_i]))
            start_i = i
    spans.append((t_s[start_i], t_s[-1], arm_names[start_i]))
    return spans


# ── Main entry point ──────────────────────────────────────────────────────────

def _step_series(t_s: np.ndarray, base_val: float,
                 link_schedule: list, key: str,
                 scale: float = 1.0):
    """
    Build a step-function array aligned to t_s from the link schedule.
    `scale` is applied to every value (use 2.0 to convert one-way delay → RTT).
    """
    if not link_schedule:
        return np.full(len(t_s), base_val)
    breakpoints = [(0.0, base_val)]
    for entry in link_schedule:
        if key in entry:
            breakpoints.append((float(entry['t']), float(entry[key]) * scale))
    breakpoints.sort(key=lambda x: x[0])
    out = np.full(len(t_s), base_val)
    for t_offset, val in breakpoints:
        out[t_s >= t_offset] = val
    return out


def plot(state_log_path: str, iperf_json_path: str, output: str,
         bw: float, delay: float,
         title: str = None,
         link_schedule: list = None) -> None:
    """
    Generate a 4-panel episode plot and save to `output` (PNG or PDF).

    Parameters
    ----------
    state_log_path  : CSV written by oc_worker / oc_inference_worker
    iperf_json_path : iperf3 --json output file
    output          : destination file path (.pdf or .png)
    bw              : base link bandwidth in Mbps (for reference line)
    delay           : base one-way delay in ms (for reference line)
    title           : figure suptitle (auto-generated if None)
    link_schedule   : list of {t, bw?, delay?} dicts — drawn as vertical lines
    """
    if not os.path.exists(state_log_path):
        print(f'[plot] state log not found: {state_log_path}', flush=True)
        return

    sl = _load_state_log(state_log_path)
    t_iperf, bps = _load_iperf_json(iperf_json_path)

    if len(sl['t_s']) == 0:
        print('[plot] state log is empty — nothing to plot', flush=True)
        return

    spans = _arm_spans(sl['t_s'], sl['arm_name'].tolist())

    # ── Episode return ────────────────────────────────────────────────────────
    ep_return = float(sl['reward'].sum())

    # ── Pre-compute link step functions ───────────────────────────────────────
    bw_step    = _step_series(sl['t_s'], bw,        link_schedule, 'bw')
    delay_step = _step_series(sl['t_s'], delay, link_schedule, 'delay')

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    if title is None:
        title = f'Olympus v2  —  bw={bw} Mbps  delay={delay} ms'
    fig.suptitle(f'{title}   [return={ep_return:.1f}]',
                 fontsize=13, fontweight='bold')

    def _shade(ax):
        for ts, te, arm in spans:
            ax.axvspan(ts, te, color=ARM_COLORS.get(arm, 'grey'),
                       alpha=0.12, linewidth=0)

    def _sched_lines(ax):
        if not link_schedule:
            return
        for entry in link_schedule:
            ax.axvline(float(entry['t']), color='black',
                       linewidth=0.8, linestyle=':', alpha=0.6)

    # ── 1. Goodput (Mbps) + max link BW ──────────────────────────────────────
    ax = axes[0]
    if len(t_iperf) > 0:
        ax.plot(t_iperf, bps / 1e6, color='black', linewidth=1.2, label='goodput')
    ax.plot(sl['t_s'], bw_step, color='red', linewidth=1.5, label='max link bw')
    ax.legend(fontsize=8, loc='upper right')
    _shade(ax); _sched_lines(ax)
    ax.set_ylabel('Goodput (Mbps)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ── 2. CWND (MSS) ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(sl['t_s'], sl['cwnd_mss'], color='black', linewidth=0.8)
    _shade(ax); _sched_lines(ax)
    ax.set_ylabel('CWND (MSS)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ── 3. RTT (ms) + min propagation RTT ────────────────────────────────────
    ax = axes[2]
    rtt_ms     = sl['rtt_us']     / 1000.0
    min_rtt_ms = sl['min_rtt_us'] / 1000.0
    ax.plot(sl['t_s'], rtt_ms,     color='black', linewidth=0.8, label='srtt')
    ax.plot(sl['t_s'], min_rtt_ms, color='green', linewidth=0.8,
            linestyle='--', alpha=0.7, label='min_rtt')
    ax.plot(sl['t_s'], delay_step, color='red', linewidth=1.5, label='scheduled RTT')
    ax.legend(fontsize=8, loc='upper right')
    _shade(ax); _sched_lines(ax)
    ax.set_ylabel('RTT (ms)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ── 4. Reward ─────────────────────────────────────────────────────────────
    ax = axes[3]
    reward = sl['reward']
    # 5-second rolling mean (window = 5s / 20ms = 250 frames)
    window = max(1, int(5.0 / max(np.diff(sl['t_s']).mean(), 0.001)))
    kernel = np.ones(window) / window
    reward_smooth = np.convolve(reward, kernel, mode='same')
    ax.plot(sl['t_s'], reward, color='grey',  linewidth=0.5, alpha=0.5, label='reward')
    ax.plot(sl['t_s'], reward_smooth, color='black', linewidth=1.2, label='5s mean')
    ax.axhline(0, color='red', linewidth=0.6, linestyle='--')
    ax.legend(fontsize=8, loc='upper right')
    _shade(ax); _sched_lines(ax)
    ax.set_ylabel('Reward', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 5. Arm timeline ───────────────────────────────────────────────────────
    ax = axes[4]
    for ts, te, arm in spans:
        ax.barh(0, te - ts, left=ts, height=0.6,
                color=ARM_COLORS.get(arm, 'grey'), align='center')
        if te - ts > 1.0:
            ax.text((ts + te) / 2, 0, arm, ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold', clip_on=True)
    _sched_lines(ax)
    ax.set_ylabel('Arm', fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel('Time (s)', fontsize=9)

    # ── Cumulative return annotation ──────────────────────────────────────────
    cum_reward = np.cumsum(reward)
    ax_r = ax.twinx()
    ax_r.plot(sl['t_s'], cum_reward, color='navy', linewidth=0.8,
              linestyle='--', alpha=0.7)
    ax_r.set_ylabel('Cum. return', fontsize=7, color='navy')
    ax_r.tick_params(axis='y', labelcolor='navy', labelsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=ARM_COLORS[n], label=n) for n in ARM_NAMES]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=9, title='CC arm', framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved → {output}  return={ep_return:.1f}', flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n── Summary ────────────────────────────────────────────────────────')
    if len(bps) > 0:
        print(f'  Goodput  mean={bps.mean()/1e6:.2f}  '
              f'p50={np.percentile(bps,50)/1e6:.2f}  '
              f'p5={np.percentile(bps,5)/1e6:.2f} Mbps')
    if len(rtt_ms) > 0:
        print(f'  RTT      mean={rtt_ms.mean():.1f}  '
              f'p50={np.percentile(rtt_ms,50):.1f}  '
              f'p95={np.percentile(rtt_ms,95):.1f} ms')
    arm_counts = {}
    for a in sl['arm_name'].tolist():
        arm_counts[a] = arm_counts.get(a, 0) + 1
    total = sum(arm_counts.values()) or 1
    print('  Arm usage (% of frames):')
    for arm in ARM_NAMES:
        pct = arm_counts.get(arm, 0) / total * 100
        print(f'    {arm:8s}  {pct:5.1f}%  {"█" * int(pct / 2)}')
    print(f'  Episode return (sum of frame rewards): {ep_return:.2f}')
    print('───────────────────────────────────────────────────────────────────\n')
    return ep_return
