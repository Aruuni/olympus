"""
plot_returns_watcher.py — schema-agnostic episode/learner plot.

The learner writes telemetry as a CSV at `cfg["training"]["log_path"]`
(typically `<run>/telemetry/learner_metrics.csv`). Each algorithm chooses its
own column set; this plotter discovers columns from the CSV header and renders
one panel per column. Vector columns serialised as ';'-separated floats
(e.g. `option_mass=0.33;0.33;0.34`) are auto-detected and plotted as one line
per component.

Pages:
  1 — Episode returns coloured by BW
  2 — Episode returns coloured by propagation delay
  3 — Episode returns grouped by schedule-change count
  4 — Return summary (trend + histogram)
  5 — Learner metrics: auto grid, one panel per CSV column

Written atomically (temp file + os.replace) so a concurrent reader can't see
a half-written PDF.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml


_HERE        = os.path.dirname(os.path.abspath(__file__))
_REPO        = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_CSV_PATH    = os.path.join(_HERE, 'episode_returns.csv')
_OUTPUT      = os.path.join(_HERE, 'episode_returns.pdf')
_LEARNER_LOG = os.path.join(_HERE, 'learner_metrics.csv')
_PLOT_EVERY  = 10


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_returns(csv_path: str):
    episodes, returns, bws, delays = [], [], [], []
    scheduled, changes, episode_types = [], [], []
    flows, elapsed_s, episode_wall_s = [], [], []
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ret = row.get('return', '')
                    if ret == '':
                        continue
                    episodes.append(int(row['episode']))
                    returns.append(float(ret))
                    bws.append(float(row.get('bw', 0)))
                    delays.append(float(row.get('delay', 0)))
                    flow_raw = row.get('flows', '')
                    flows.append(float(flow_raw) if flow_raw not in ('', None) else np.nan)
                    scheduled_i = int(row.get('scheduled', 0))
                    scheduled.append(scheduled_i)
                    if row.get('schedule_changes', '') != '':
                        change_count = int(row.get('schedule_changes', 0))
                    else:
                        # Backward compatibility for older episode_returns.csv.
                        change_count = 1 if scheduled_i else 0
                    changes.append(change_count)
                    episode_types.append(row.get('episode_type') or _episode_type_label(change_count))
                    elapsed_raw = row.get('elapsed_s', '')
                    wall_raw = row.get('episode_wall_s', '')
                    elapsed_s.append(float(elapsed_raw) if elapsed_raw not in ('', None) else np.nan)
                    episode_wall_s.append(float(wall_raw) if wall_raw not in ('', None) else np.nan)
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    episodes      = np.array(episodes)
    returns       = np.array(returns)
    bws           = np.array(bws)
    delays        = np.array(delays)
    scheduled     = np.array(scheduled)
    changes       = np.array(changes)
    episode_types = np.array(episode_types, dtype=object)
    flows         = np.array(flows, dtype=np.float64)
    elapsed_s     = np.array(elapsed_s, dtype=np.float64)
    episode_wall_s = np.array(episode_wall_s, dtype=np.float64)
    order = np.argsort(episodes)
    return (
        episodes[order], returns[order], bws[order], delays[order],
        scheduled[order], changes[order], episode_types[order],
        flows[order], elapsed_s[order], episode_wall_s[order],
    )


def _load_metrics_csv(path: str):
    """
    Generic learner-CSV loader. Returns a dict keyed by column name:
      - scalar columns → 1D np.float32 array of length n_rows (NaN where empty)
      - vector columns (';'-separated floats) → 2D (n_rows, n_components)
    Returns None if file is missing or empty.
    """
    rows = []
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
    except (FileNotFoundError, OSError, csv.Error):
        return None
    if not rows or not cols:
        return None

    out = {}
    for col in cols:
        # Find first non-empty sample to decide scalar vs vector.
        sample = ''
        for r in rows:
            v = r.get(col, '')
            if v != '' and v is not None:
                sample = v
                break
        if sample == '':
            continue

        if ';' in sample:
            # Vector column — fix component count from first parseable row.
            n_components = None
            parsed = []
            for r in rows:
                v = r.get(col, '')
                if v == '' or v is None:
                    parsed.append(None)
                    continue
                try:
                    parts = [float(p) for p in v.split(';')]
                except ValueError:
                    parsed.append(None)
                    continue
                if n_components is None:
                    n_components = len(parts)
                if len(parts) == n_components:
                    parsed.append(parts)
                else:
                    parsed.append(None)
            if not n_components:
                continue
            arr = np.full((len(parsed), n_components), np.nan, dtype=np.float32)
            for i, p in enumerate(parsed):
                if p is not None:
                    arr[i] = p
            out[col] = arr
        else:
            arr = np.full(len(rows), np.nan, dtype=np.float32)
            for i, r in enumerate(rows):
                v = r.get(col, '')
                if v == '' or v is None:
                    continue
                try:
                    arr[i] = float(v)
                except ValueError:
                    pass
            if np.all(np.isnan(arr)):
                continue
            out[col] = arr
    return out if out else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    pad  = np.full(w - 1, np.nan)
    conv = np.convolve(arr, kernel, mode='valid')
    return np.concatenate([pad, conv])


def _smooth(arr: np.ndarray, window: int = 50) -> np.ndarray:
    return _rolling(arr, window)


def _episode_type_label(change_count: int) -> str:
    n = int(change_count)
    return f'{n}_change' if n == 1 else f'{n}_changes'


def _episode_type_display(label: str) -> str:
    text = str(label).replace('_', ' ')
    return text


def _last_finite(arr: np.ndarray):
    if arr is None or len(arr) == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def _max_finite(arr: np.ndarray):
    if arr is None or len(arr) == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(finite.max())


def _format_duration(seconds) -> str:
    if seconds is None or not np.isfinite(seconds):
        return ''
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}h {minutes:02d}m {secs:02d}s'
    if minutes:
        return f'{minutes}m {secs:02d}s'
    return f'{secs}s'


def _elapsed_title(elapsed_s: np.ndarray) -> str:
    text = _format_duration(_max_finite(elapsed_s))
    return f'  elapsed={text}' if text else ''


def _alg_name_from_output(output_path: str) -> str:
    """Resolve the run's algorithm name for plot titles.

    The orchestrator writes `telemetry/run_meta.json` next to the output PDF
    with an authoritative `algorithm` field; fall back to parsing the run-dir
    name only if the meta file is missing.
    """
    out_abs   = os.path.abspath(output_path)
    meta_path = os.path.join(os.path.dirname(out_abs), 'run_meta.json')
    try:
        with open(meta_path) as f:
            alg = (json.load(f) or {}).get('algorithm')
        if alg:
            return str(alg)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass

    parent = os.path.basename(os.path.dirname(os.path.dirname(out_abs)))
    if '_' in parent:
        prefix = parent.rsplit('_', 1)[0]
        if prefix:
            return prefix
    return 'learner'


def _reward_config_from_output(output_path: str) -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        'config.resolved.yaml',
    )
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return {}
    return dict(cfg.get('reward') or {})


def _wants_log(name: str) -> bool:
    n = name.lower()
    if any(tok in n for tok in ('mass', 'usage', 'fraction', 'frac', 'ratio', 'mean_beta')):
        return False
    return ('loss' in n) or n.startswith('lr') or n.endswith('_lr') or n in ('lr_a', 'lr_c')


# ── Pages ─────────────────────────────────────────────────────────────────────

def _page_returns_bw(episodes, returns, bws, n, alg_name, elapsed_s=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'{alg_name} — Episode returns by BW  ({n} episodes'
                 f'{_elapsed_title(elapsed_s)})',
                 fontsize=12, fontweight='bold')
    bw_vals = np.unique(bws)
    cmap = plt.get_cmap('tab10')
    color = {bw: cmap(i % 10) for i, bw in enumerate(sorted(bw_vals))}
    for bw in sorted(bw_vals):
        m = bws == bw
        ax.scatter(episodes[m], returns[m],
                   s=18, alpha=0.55, color=color[bw],
                   label=f'{int(bw)} Mbps', zorder=3)
    ax.plot(episodes, _rolling(returns, 20), color='black', linewidth=1.8,
            label='20-ep rolling mean', zorder=4)
    ax.axhline(returns.mean(), color='red', linewidth=1.0, linestyle='--',
               alpha=0.7, label=f'mean={returns.mean():.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8, ncol=3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    return fig


def _page_returns_delay(episodes, returns, delays, n, alg_name, elapsed_s=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'{alg_name} — Episode returns by delay  ({n} episodes'
                 f'{_elapsed_title(elapsed_s)})',
                 fontsize=12, fontweight='bold')
    delay_vals = np.unique(delays)
    dcmap = plt.get_cmap('Set2')
    dcolor = {d: dcmap(i % 8) for i, d in enumerate(sorted(delay_vals))}
    for d in sorted(delay_vals):
        m = delays == d
        ax.scatter(episodes[m], returns[m],
                   s=10, alpha=0.45, color=dcolor[d], label=f'{int(d)} ms')
    ax.plot(episodes, _rolling(returns, 20), color='black', linewidth=1.6,
            label='20-ep rolling mean', zorder=4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8, ncol=3)
    ax.set_xlim(left=0)
    fig.subplots_adjust(top=0.84, bottom=0.18)
    return fig


def _page_returns_type(episodes, returns, changes, episode_types, n, alg_name,
                       elapsed_s=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                             gridspec_kw={'wspace': 0.28})
    fig.suptitle(f'{alg_name} — Episode returns by schedule changes  '
                 f'({n} episodes{_elapsed_title(elapsed_s)})',
                 fontsize=12, fontweight='bold')

    # Sort labels numerically by change count, then by label for stability.
    labels = sorted(
        np.unique(episode_types),
        key=lambda label: (
            int(changes[episode_types == label][0]) if np.any(episode_types == label) else 0,
            str(label),
        ),
    )
    cmap = plt.get_cmap('tab10')
    color = {label: cmap(i % 10) for i, label in enumerate(labels)}

    ax = axes[0]
    for label in labels:
        m = episode_types == label
        if not m.any():
            continue
        label_text = _episode_type_display(label)
        ax.scatter(episodes[m], returns[m], s=18, alpha=0.55,
                   color=color[label],
                   label=f'{label_text} n={int(m.sum())} mean={returns[m].mean():.1f}')
        if m.sum() >= 5:
            order = np.argsort(episodes[m])
            x = episodes[m][order]
            y = returns[m][order]
            ax.plot(x, _rolling(y, min(20, max(2, int(m.sum())))),
                    color=color[label], linewidth=1.6, alpha=0.9)
    ax.set_title('Trend by episode type')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best', framealpha=0.8)
    ax.set_xlim(left=0)

    ax = axes[1]
    box_data = [returns[episode_types == label] for label in labels
                if np.any(episode_types == label)]
    box_labels = [_episode_type_display(label) for label in labels
                  if np.any(episode_types == label)]
    if box_data:
        ax.boxplot(box_data, showmeans=True)
        ax.set_xticks(np.arange(1, len(box_labels) + 1))
        ax.set_xticklabels(box_labels)
    ax.set_title('Distribution by episode type')
    ax.set_xlabel('Schedule changes')
    ax.set_ylabel('Return')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=25)

    fig.subplots_adjust(top=0.84, bottom=0.18)
    return fig


def _page_returns_summary(episodes, returns, n, alg_name, elapsed_s=None,
                          episode_wall_s=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                             gridspec_kw={'wspace': 0.24})
    elapsed_text = _elapsed_title(elapsed_s)
    fig.suptitle(f'{alg_name} — Return summary  ({n} episodes{elapsed_text})',
                 fontsize=12, fontweight='bold')
    ax = axes[0]
    ax.plot(episodes, returns, color='#9fbbe6', linewidth=0.6, alpha=0.5,
            label='episode return')
    ax.plot(episodes, _rolling(returns, 20), color='black', linewidth=1.5,
            label='20-ep mean')
    ax.plot(episodes, _rolling(returns, 100), color='#e15759', linewidth=1.5,
            label='100-ep mean')
    ax.axhline(returns.mean(), color='#666', linestyle='--', linewidth=1.0,
               label=f'mean={returns.mean():.1f}')
    ax.set_title('Return trend')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.8)

    ax = axes[1]
    ax.hist(returns, bins=min(60, max(20, len(returns) // 80)),
            color='#4878cf', alpha=0.8)
    ax.axvline(np.median(returns), color='black', linewidth=1.2,
               label=f'median={np.median(returns):.1f}')
    ax.axvline(returns.mean(), color='#e15759', linewidth=1.2,
               label=f'mean={returns.mean():.1f}')
    ax.set_title('Return distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.8)

    wall = _last_finite(episode_wall_s) if episode_wall_s is not None else None
    total = _max_finite(elapsed_s) if elapsed_s is not None else None
    if wall is not None or total is not None:
        parts = []
        if total is not None:
            parts.append(f'total elapsed: {_format_duration(total)}')
        finite_wall = (episode_wall_s[np.isfinite(episode_wall_s)]
                       if episode_wall_s is not None else np.array([]))
        if finite_wall.size:
            parts.append(f'mean episode wall: {_format_duration(float(finite_wall.mean()))}')
        if parts:
            ax.text(0.02, 0.96, '\n'.join(parts), transform=ax.transAxes,
                    va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.75, linewidth=0.5))
    return fig


def _page_returns_time(episodes, returns, elapsed_s, n, alg_name):
    if elapsed_s is None or len(elapsed_s) == 0:
        return None
    finite = np.isfinite(elapsed_s)
    if not finite.any():
        return None

    x_raw = elapsed_s[finite]
    y = returns[finite]
    ep = episodes[finite]
    order = np.argsort(x_raw)
    x_raw = x_raw[order]
    y = y[order]
    ep = ep[order]

    total = float(x_raw[-1]) if x_raw.size else 0.0
    if total >= 2 * 3600:
        x = x_raw / 3600.0
        xlabel = 'Elapsed training time (hours)'
    elif total >= 120:
        x = x_raw / 60.0
        xlabel = 'Elapsed training time (minutes)'
    else:
        x = x_raw
        xlabel = 'Elapsed training time (seconds)'

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f'{alg_name} — Episode returns over wall-clock training time  '
        f'({n} episodes, elapsed={_format_duration(total)})',
        fontsize=12, fontweight='bold',
    )
    sc = ax.scatter(x, y, c=ep, cmap='viridis', s=18, alpha=0.65,
                    label='episode return', zorder=3)
    if len(y) >= 2:
        ax.plot(x, _rolling(y, min(20, len(y))), color='black',
                linewidth=1.6, label='20-completion rolling mean', zorder=4)
    ax.axhline(y.mean(), color='#e15759', linewidth=1.0, linestyle='--',
               alpha=0.8, label=f'mean={y.mean():.1f}')
    cbar = fig.colorbar(sc, ax=ax, pad=0.012)
    cbar.set_label('Episode')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.8, loc='best')
    ax.set_xlim(left=0)
    return fig


def _state_log_path_for_episode(csv_path: str, alg_name: str, episode: int):
    episodes_dir = os.path.dirname(os.path.abspath(csv_path))
    preferred = os.path.join(
        episodes_dir, f'{alg_name}_state_ep{int(episode):06d}.csv')
    if os.path.exists(preferred):
        return preferred
    pattern = os.path.join(episodes_dir, f'*_state_ep{int(episode):06d}_a*.csv')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    first = matches[0]
    root, ext = os.path.splitext(first)
    return root.rsplit('_a', 1)[0] + ext


def _episode_team_return(state_log_path: str, reward_config: dict,
                         n_agents: int = None, trim_tail_s: float = 5.0):
    from olympus.common.marl_team_reward import (
        per_agent_team_reward,
        shared_team_reward,
    )
    from olympus.plots.multi_flow_episode_plot import (
        _aligned_matrix,
        load_agent_traces,
    )

    traces = load_agent_traces(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    if len(traces) < 2:
        return None
    _, local_reward = _aligned_matrix(traces, 'reward')
    _, avg_thr_mbps = _aligned_matrix(traces, 'avg_thr_mbps')
    if local_reward.size == 0 or avg_thr_mbps.size == 0:
        return None

    # Team-reward helpers expect timestep-major arrays.
    local_reward = local_reward.T
    avg_thr_mbps = avg_thr_mbps.T
    mask = np.isfinite(local_reward) & np.isfinite(avg_thr_mbps)
    local_reward = np.nan_to_num(local_reward, nan=0.0)
    avg_thr_mbps = np.nan_to_num(avg_thr_mbps, nan=0.0)

    if bool(reward_config.get('per_agent_credit', False)):
        team_rewards, _, _ = per_agent_team_reward(
            local_reward,
            avg_thr_mbps,
            mask,
            team_alpha=float(reward_config.get('team_alpha', 0.5)),
            fairness_weight=float(reward_config.get('fairness_weight', 25.0)),
            fairness_cap=float(reward_config.get('fairness_cap', 1.0)),
            over_weight=float(reward_config.get('over_weight', 50.0)),
        )
        return float(np.nansum(team_rewards))

    shared_rewards, _, _ = shared_team_reward(
        local_reward,
        avg_thr_mbps,
        mask,
        fairness_weight=float(reward_config.get('fairness_weight', 25.0)),
        fairness_cap=float(reward_config.get('fairness_cap', 1.0)),
    )
    return float(np.nansum(shared_rewards))


def _load_episode_team_returns(csv_path: str, episodes: np.ndarray,
                               flows: np.ndarray, alg_name: str,
                               reward_config: dict):
    out = np.full(len(episodes), np.nan, dtype=np.float64)
    for i, episode in enumerate(episodes):
        state_log = _state_log_path_for_episode(csv_path, alg_name, int(episode))
        if not state_log:
            continue
        n_agents = None
        if flows is not None and i < len(flows) and np.isfinite(flows[i]):
            n_agents = int(flows[i])
        value = _episode_team_return(
            state_log, reward_config, n_agents=n_agents, trim_tail_s=5.0)
        if value is not None and np.isfinite(value):
            out[i] = value
    return out


def _page_team_returns_bw(episodes, team_returns, bws, n, alg_name,
                          elapsed_s=None):
    finite = np.isfinite(team_returns)
    if not finite.any():
        return None
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'{alg_name} — Episode team reward by BW  '
                 f'({int(finite.sum())}/{n} episodes'
                 f'{_elapsed_title(elapsed_s)})',
                 fontsize=12, fontweight='bold')
    bw_vals = np.unique(bws[finite])
    cmap = plt.get_cmap('tab10')
    color = {bw: cmap(i % 10) for i, bw in enumerate(sorted(bw_vals))}
    for bw in sorted(bw_vals):
        m = finite & (bws == bw)
        ax.scatter(episodes[m], team_returns[m],
                   s=18, alpha=0.55, color=color[bw],
                   label=f'{int(bw)} Mbps', zorder=3)
    ax.plot(episodes[finite], _rolling(team_returns[finite], 20),
            color='black', linewidth=1.8, label='20-ep rolling mean', zorder=4)
    ax.axhline(team_returns[finite].mean(), color='red', linewidth=1.0,
               linestyle='--', alpha=0.7,
               label=f'mean={team_returns[finite].mean():.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Team reward')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    return fig


def _page_team_returns_delay(episodes, team_returns, delays, n, alg_name,
                             elapsed_s=None):
    finite = np.isfinite(team_returns)
    if not finite.any():
        return None
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'{alg_name} — Episode team reward by delay  '
                 f'({int(finite.sum())}/{n} episodes'
                 f'{_elapsed_title(elapsed_s)})',
                 fontsize=12, fontweight='bold')
    delay_vals = np.unique(delays[finite])
    dcmap = plt.get_cmap('Set2')
    dcolor = {d: dcmap(i % 8) for i, d in enumerate(sorted(delay_vals))}
    for d in sorted(delay_vals):
        m = finite & (delays == d)
        ax.scatter(episodes[m], team_returns[m],
                   s=18, alpha=0.6, color=dcolor[d],
                   label=f'{int(d)} ms', zorder=3)
    ax.plot(episodes[finite], _rolling(team_returns[finite], 20),
            color='black', linewidth=1.8, label='20-ep rolling mean', zorder=4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Team reward')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    return fig


def _page_team_returns_type(episodes, team_returns, changes, episode_types, n,
                            alg_name, elapsed_s=None):
    finite = np.isfinite(team_returns)
    if not finite.any():
        return None
    labels = list(dict.fromkeys(str(v) for v in episode_types[finite]))
    cmap = plt.get_cmap('tab10')
    colors = {label: cmap(i % 10) for i, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'{alg_name} — Episode team reward by type  '
                 f'({int(finite.sum())}/{n} episodes'
                 f'{_elapsed_title(elapsed_s)})',
                 fontsize=12, fontweight='bold')
    for label in labels:
        m = finite & (episode_types == label)
        ax.scatter(episodes[m], team_returns[m],
                   s=20, alpha=0.62, color=colors[label],
                   label=_episode_type_display(label), zorder=3)
    ax.plot(episodes[finite], _rolling(team_returns[finite], 20),
            color='black', linewidth=1.8, label='20-ep rolling mean', zorder=4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Team reward')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    return fig


def _page_learner_metrics_auto(ld: dict, n_ep: int, alg_name: str,
                               panels_per_page: int = 12):
    """
    Auto-laid-out grid of one panel per CSV column. Splits across multiple
    pages so a learner with many columns stays readable. Vector columns
    (`;`-separated) are plotted as one line per component.
    """
    if not ld:
        return []
    if 'step' in ld and ld['step'].ndim == 1:
        step_full = ld['step']
        cols = [c for c in ld if c != 'step']
    else:
        # No 'step' column — fall back to row index as x-axis.
        first_key = next(iter(ld))
        step_full = np.arange(len(ld[first_key]))
        cols = list(ld.keys())
    if not cols:
        return []
    n_steps_seen = int(step_full[~np.isnan(step_full)][-1]) if step_full.size else 0

    figs = []
    for page_idx in range(0, len(cols), panels_per_page):
        page_cols = cols[page_idx:page_idx + panels_per_page]
        n_panels  = len(page_cols)
        n_grid_c  = min(3, max(1, int(math.ceil(math.sqrt(n_panels)))))
        n_grid_r  = int(math.ceil(n_panels / n_grid_c))

        fig, axes = plt.subplots(n_grid_r, n_grid_c,
                                 figsize=(5.2 * n_grid_c, 3.0 * n_grid_r),
                                 gridspec_kw={'hspace': 0.6, 'wspace': 0.32})
        page_n = page_idx // panels_per_page + 1
        n_pages = int(math.ceil(len(cols) / panels_per_page))
        suffix = f' (page {page_n}/{n_pages})' if n_pages > 1 else ''
        fig.suptitle(
            f'{alg_name} — Learner metrics  '
            f'(steps={n_steps_seen}  episodes={n_ep}){suffix}',
            fontsize=12, fontweight='bold',
        )

        # Normalise axes to a 2D array regardless of grid shape.
        if n_grid_r * n_grid_c == 1:
            axes_grid = np.array([[axes]])
        elif n_grid_r == 1:
            axes_grid = np.array([axes])
        elif n_grid_c == 1:
            axes_grid = np.array([[ax] for ax in axes])
        else:
            axes_grid = axes

        for i, col in enumerate(page_cols):
            ax = axes_grid[i // n_grid_c, i % n_grid_c]
            y = ld[col]
            x = step_full[:len(y)] if step_full.size else np.arange(len(y))
            if y.ndim == 2:
                cmap = plt.get_cmap('tab10')
                for k in range(y.shape[1]):
                    yk = y[:, k]
                    m  = ~np.isnan(yk)
                    if m.any():
                        ax.plot(x[m], yk[m], color=cmap(k % 10),
                                linewidth=1.1, alpha=0.9, label=f'[{k}]')
                if y.shape[1] <= 8:
                    ax.legend(fontsize=7, ncol=min(y.shape[1], 4), loc='best',
                              framealpha=0.7)
            else:
                m = ~np.isnan(y)
                if m.any():
                    ax.plot(x[m], y[m], color='#999', linewidth=0.4, alpha=0.45)
                    ax.plot(x[m], _smooth(y[m], 50), color='#4878cf',
                            linewidth=1.4)
                if _wants_log(col):
                    try:
                        ax.set_yscale('symlog', linthresh=1e-3)
                    except Exception:
                        pass

            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.set_xlabel('step', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide trailing empty axes on the last page.
        for j in range(n_panels, n_grid_r * n_grid_c):
            axes_grid[j // n_grid_c, j % n_grid_c].set_visible(False)

        figs.append(fig)

    return figs


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_plot(csv_path: str = _CSV_PATH, output: str = _OUTPUT,
                  learner_log: str = _LEARNER_LOG) -> int:
    (episodes, returns, bws, delays, scheduled,
     changes, episode_types, flows, elapsed_s, episode_wall_s) = _load_returns(csv_path)
    n = len(episodes)
    if n == 0:
        print('[plot] no episode data yet — skipping', flush=True)
        return 0

    alg_name = _alg_name_from_output(output)
    reward_config = _reward_config_from_output(output)
    ld       = _load_metrics_csv(learner_log)
    n_steps  = 0
    if ld and 'step' in ld and ld['step'].size:
        finite = ld['step'][~np.isnan(ld['step'])]
        n_steps = int(finite[-1]) if finite.size else 0

    out_abs = os.path.abspath(output)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    tmp_path = f'{out_abs}.tmp.{os.getpid()}'
    try:
        pdf = PdfPages(tmp_path)
    except PermissionError:
        fallback = os.path.join(os.path.dirname(os.path.dirname(out_abs)),
                                os.path.basename(out_abs))
        out_abs  = fallback
        tmp_path = f'{out_abs}.tmp.{os.getpid()}'
        pdf      = PdfPages(tmp_path)

    try:
        return_figs = [
            _page_returns_bw(episodes, returns, bws, n, alg_name, elapsed_s),
            _page_returns_delay(episodes, returns, delays, n, alg_name, elapsed_s),
            _page_returns_type(episodes, returns, changes, episode_types, n,
                               alg_name, elapsed_s),
            _page_returns_summary(episodes, returns, n, alg_name, elapsed_s,
                                  episode_wall_s),
            _page_returns_time(episodes, returns, elapsed_s, n, alg_name),
        ]
        for fig in return_figs:
            if fig is None:
                continue
            pdf.savefig(fig, dpi=120, bbox_inches='tight')
            plt.close(fig)

        team_returns = _load_episode_team_returns(
            csv_path, episodes, flows, alg_name, reward_config)
        team_return_figs = [
            _page_team_returns_bw(
                episodes, team_returns, bws, n, alg_name, elapsed_s),
            _page_team_returns_delay(
                episodes, team_returns, delays, n, alg_name, elapsed_s),
            _page_team_returns_type(
                episodes, team_returns, changes, episode_types, n,
                alg_name, elapsed_s),
        ]
        for fig in team_return_figs:
            if fig is None:
                continue
            pdf.savefig(fig, dpi=120, bbox_inches='tight')
            plt.close(fig)

        for fig in _page_learner_metrics_auto(ld, n, alg_name):
            pdf.savefig(fig, dpi=120, bbox_inches='tight')
            plt.close(fig)
    finally:
        pdf.close()

    os.replace(tmp_path, out_abs)

    elapsed_text = _format_duration(_max_finite(elapsed_s))
    elapsed_part = f'  elapsed={elapsed_text}' if elapsed_text else ''
    print(f'[plot] saved → {out_abs}  '
          f'(ep={n}  mean={returns.mean():.1f}  '
          f'last={returns[-1]:.1f}  steps={n_steps}{elapsed_part})',
          flush=True)
    return n


def watch(csv_path: str = _CSV_PATH, output: str = _OUTPUT,
          learner_log: str = _LEARNER_LOG, interval: float = 30.0) -> None:
    last = -1
    print(f'[plot] watching {csv_path}  '
          f'(plot every {_PLOT_EVERY} episodes, poll every {interval}s)',
          flush=True)
    while True:
        episodes, *_ = _load_returns(csv_path)
        n = len(episodes)
        if n > 0 and (n - last) >= _PLOT_EVERY:
            generate_plot(csv_path, output, learner_log)
            last = n
        time.sleep(interval)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',         default=_CSV_PATH)
    ap.add_argument('--output',      default=_OUTPUT)
    ap.add_argument('--learner-log', default=_LEARNER_LOG)
    ap.add_argument('--interval',    type=float, default=30.0)
    ap.add_argument('--once',        action='store_true')
    args = ap.parse_args()

    if args.once:
        generate_plot(args.csv, args.output, args.learner_log)
    else:
        watch(args.csv, args.output, args.learner_log, args.interval)
