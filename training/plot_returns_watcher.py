"""
training/plot_returns_watcher.py

Watches plots/episode_returns.csv and regenerates
plots/episode_returns.pdf whenever 10 new episodes have been appended.

Usage:
    python training/plot_returns_watcher.py            # poll every 30s
    python training/plot_returns_watcher.py --once     # single shot, then exit
    python training/plot_returns_watcher.py --interval 60
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE       = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH   = os.path.join(_HERE, 'plots', 'episode_returns.csv')
_OUTPUT     = os.path.join(_HERE, 'plots', 'episode_returns.pdf')
_PLOT_EVERY = 10   # re-draw after this many new episodes


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(csv_path: str):
    episodes, returns, bws, delays, scheduled = [], [], [], [], []
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
                    scheduled.append(int(row.get('scheduled', 0)))
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return (np.array(episodes), np.array(returns),
            np.array(bws), np.array(delays), np.array(scheduled))


# ── Plotting ──────────────────────────────────────────────────────────────────

def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    # 'valid' mode shortens; pad the front with nan instead
    pad = np.full(w - 1, np.nan)
    conv = np.convolve(arr, kernel, mode='valid')
    return np.concatenate([pad, conv])


def generate_plot(csv_path: str = _CSV_PATH, output: str = _OUTPUT) -> int:
    """Read CSV, draw returns plot, save to *output*. Returns episode count."""
    episodes, returns, bws, delays, scheduled = _load(csv_path)
    n = len(episodes)
    if n == 0:
        print('[returns_watcher] no data yet — skipping', flush=True)
        return 0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        f'Olympus v2 — Episodic Returns  ({n} episodes)',
        fontsize=13, fontweight='bold',
    )

    # ── Top panel: returns + rolling mean ────────────────────────────────────
    ax = axes[0]

    # Colour points by bandwidth bucket
    bw_vals  = np.unique(bws)
    cmap     = plt.get_cmap('tab10')
    bw_color = {bw: cmap(i % 10) for i, bw in enumerate(sorted(bw_vals))}

    for bw_val in sorted(bw_vals):
        mask = bws == bw_val
        ax.scatter(episodes[mask], returns[mask],
                   s=18, alpha=0.55,
                   color=bw_color[bw_val],
                   label=f'{int(bw_val)} Mbps',
                   zorder=3)

    # Rolling mean (window = 20 episodes)
    roll = _rolling(returns, 20)
    ax.plot(episodes, roll, color='black', linewidth=1.8,
            label='20-ep rolling mean', zorder=4)

    # Overall horizontal mean
    ax.axhline(returns.mean(), color='red', linewidth=1.0,
               linestyle='--', alpha=0.7, label=f'mean={returns.mean():.1f}')

    ax.set_ylabel('Episode Return', fontsize=10)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Annotate last value
    ax.annotate(f'{returns[-1]:.0f}',
                xy=(episodes[-1], returns[-1]),
                xytext=(6, 0), textcoords='offset points',
                fontsize=7, color='black', va='center')

    # ── Bottom panel: delay scatter ───────────────────────────────────────────
    ax2 = axes[1]
    delay_vals  = np.unique(delays)
    delay_cmap  = plt.get_cmap('Set2')
    delay_color = {d: delay_cmap(i % 8) for i, d in enumerate(sorted(delay_vals))}

    for d_val in sorted(delay_vals):
        mask = delays == d_val
        ax2.scatter(episodes[mask], returns[mask],
                    s=10, alpha=0.45,
                    color=delay_color[d_val],
                    label=f'{int(d_val)} ms')

    ax2.set_ylabel('Return\n(by delay)', fontsize=9)
    ax2.set_xlabel('Episode', fontsize=10)
    ax2.legend(fontsize=7, loc='upper left', framealpha=0.8, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    try:
        plt.savefig(output, dpi=120, bbox_inches='tight')
        print(f'[returns_watcher] saved → {output}  '
              f'(ep={n}  mean={returns.mean():.1f}  '
              f'last={returns[-1]:.1f})', flush=True)
    except PermissionError:
        # plots/ may be root-owned during training; fall back to parent dir
        fallback = os.path.join(os.path.dirname(os.path.dirname(output)),
                                os.path.basename(output))
        plt.savefig(fallback, dpi=120, bbox_inches='tight')
        print(f'[returns_watcher] saved → {fallback} (fallback, no write perms on plots/)  '
              f'(ep={n}  mean={returns.mean():.1f}  '
              f'last={returns[-1]:.1f})', flush=True)
    plt.close(fig)
    return n


# ── Watcher loop ──────────────────────────────────────────────────────────────

def watch(csv_path: str = _CSV_PATH, output: str = _OUTPUT,
          interval: float = 30.0) -> None:
    last_plotted_at = -1   # episode count at the last plot
    print(f'[returns_watcher] watching {csv_path}  '
          f'(plot every {_PLOT_EVERY} episodes, poll every {interval}s)',
          flush=True)
    while True:
        episodes, returns, *_ = _load(csv_path)
        n = len(episodes)
        if n > 0 and (n - last_plotted_at) >= _PLOT_EVERY:
            generate_plot(csv_path, output)
            last_plotted_at = n
        time.sleep(interval)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Watch episode_returns.csv and auto-plot every 10 episodes.')
    ap.add_argument('--csv',      default=_CSV_PATH,
                    help='path to episode_returns.csv')
    ap.add_argument('--output',   default=_OUTPUT,
                    help='output PDF path')
    ap.add_argument('--interval', type=float, default=30.0,
                    help='poll interval in seconds (default 30)')
    ap.add_argument('--once',     action='store_true',
                    help='generate one plot then exit')
    args = ap.parse_args()

    if args.once:
        generate_plot(args.csv, args.output)
    else:
        watch(args.csv, args.output, args.interval)
