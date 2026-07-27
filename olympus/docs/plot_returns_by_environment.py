#!/usr/bin/env python3
"""Standalone plot: episode return coloured by collection environment.

Reads an Olympus ``episode_returns.csv`` and renders a single clean panel of
return vs. episode, with each point coloured by the environment it was collected
in (RayNet **simulation** vs. Mininet **emulation**). A per-environment rolling
mean is overlaid, and — for a warmup-style run where all simulation episodes
precede all emulation episodes — the two phases are shaded and a divider marks
the sim -> emulation switch, so it is obvious at a glance which episodes are
which.

This file is intentionally self-contained (only stdlib + matplotlib/numpy) so it
can be copied and run anywhere, independent of the olympus package.

Usage
-----
    python docs/plot_returns_by_environment.py <run_dir_or_csv> [-o out.pdf]

``<run_dir_or_csv>`` may be either an ``episode_returns.csv`` or a run directory
(the script then looks for ``episodes/episode_returns.csv`` beneath it). The
output defaults to ``returns_by_environment.png`` next to the CSV.

Example
-------
    python docs/plot_returns_by_environment.py \
        olympus/models/olympus/dreamer_v3_20260724-192800
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Canonical environment buckets. `label` is what the legend shows; `color` the
# scatter/line colour; `shade` the light band colour for a contiguous phase.
ENVIRONMENTS = {
    'simulation': {'label': 'RayNet (simulation)', 'color': '#1f77b4', 'shade': '#1f77b4'},
    'emulation':  {'label': 'Mininet (emulation)', 'color': '#ff7f0e', 'shade': '#ff7f0e'},
}
_ORDER = ['simulation', 'emulation']


def _bucket(row: dict) -> str:
    """Map a CSV row to 'simulation' / 'emulation' (or a raw fallback label).

    Prefers the explicit ``backend`` column (raynet/mininet), then ``source``,
    then the environment name — mirroring how the orchestrator tags rows.
    """
    backend = (row.get('backend') or '').strip().lower()
    if backend == 'raynet':
        return 'simulation'
    if backend == 'mininet':
        return 'emulation'
    src = (row.get('source') or '').strip().lower()
    if src == 'raynet':
        return 'simulation'
    if src == 'mininet':
        return 'emulation'
    if src in ('simulation', 'emulation'):
        return src
    return src or 'unknown'


def _rolling(x: np.ndarray, y: np.ndarray, window: int):
    """Rolling mean of y over its own points, returned aligned to x."""
    if len(y) == 0:
        return x, y
    w = min(window, len(y))
    kernel = np.ones(w) / w
    conv = np.convolve(y, kernel, mode='valid')
    return x[w - 1:], conv


def _resolve_csv(path: str) -> str:
    if os.path.isdir(path):
        cand = os.path.join(path, 'episodes', 'episode_returns.csv')
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(path, 'episode_returns.csv')
        if os.path.exists(cand2):
            return cand2
        raise SystemExit(f'no episode_returns.csv found under {path!r}')
    return path


def _load(csv_path: str):
    episodes, returns, buckets = [], [], []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            ret = (row.get('return') or '').strip()
            if ret == '':
                continue  # skip errored / return-less episodes
            try:
                episodes.append(int(float(row['episode'])))
                returns.append(float(ret))
            except (KeyError, ValueError):
                continue
            buckets.append(_bucket(row))
    if not episodes:
        raise SystemExit(f'no plottable rows in {csv_path!r}')
    order = np.argsort(episodes)
    return (np.array(episodes)[order],
            np.array(returns)[order],
            np.array(buckets, dtype=object)[order])


def _contiguous_phase_split(episodes, buckets):
    """If every simulation episode precedes every emulation episode, return the
    boundary episode (first emulation episode); else None. Used to decide
    whether to shade the two phases and draw a divider."""
    sim_mask = buckets == 'simulation'
    emu_mask = buckets == 'emulation'
    if not sim_mask.any() or not emu_mask.any():
        return None
    if episodes[sim_mask].max() < episodes[emu_mask].min():
        return int(episodes[emu_mask].min())
    return None


def make_plot(csv_path: str, output: str, window: int = 20, xmax: float = 600):
    episodes, returns, buckets = _load(csv_path)

    # Restrict everything (points, trend, means, axis) to the visible window so
    # the figure and its stats stay consistent.
    if xmax is not None:
        keep = episodes <= xmax
        episodes, returns, buckets = episodes[keep], returns[keep], buckets[keep]

    present = [b for b in _ORDER if (buckets == b).any()]
    extras = sorted(set(buckets) - set(_ORDER) - {'unknown'})

    # Single-column paper figure: compact (short) size, large fonts so it stays
    # legible when scaled to one column width.
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    boundary = _contiguous_phase_split(episodes, buckets)
    if boundary is not None:
        lo, hi = episodes.min(), episodes.max()
        ax.axvspan(lo, boundary, color=ENVIRONMENTS['simulation']['shade'],
                   alpha=0.06, zorder=0)
        ax.axvspan(boundary, hi, color=ENVIRONMENTS['emulation']['shade'],
                   alpha=0.06, zorder=0)
        ax.axvline(boundary, color='#444', linestyle='--', linewidth=1.2,
                   alpha=0.8, zorder=5)
        # Anchor the label to the top of the axes (x in data coords, y in axes
        # fraction) so it never collides with the x-axis regardless of autoscale.
        ax.annotate('simulation → emulation',
                    xy=(boundary, 1.0), xycoords=ax.get_xaxis_transform(),
                    xytext=(6, -6), textcoords='offset points',
                    va='top', ha='left', fontsize=12, color='#444', zorder=6)

    for b in present:
        meta = ENVIRONMENTS[b]
        m = buckets == b
        ax.scatter(episodes[m], returns[m], s=16, alpha=0.55,
                   color=meta['color'], edgecolors='none',
                   label=f"{meta['label']}  n={int(m.sum())}  "
                         f"mean={returns[m].mean():.0f}", zorder=3)
        rx, ry = _rolling(episodes[m], returns[m], window)
        ax.plot(rx, ry, color=meta['color'], linewidth=2.0, alpha=0.95,
                zorder=4)

    # Any unexpected extra sources still get drawn so nothing is silently hidden.
    for i, b in enumerate(extras):
        m = buckets == b
        ax.scatter(episodes[m], returns[m], s=16, alpha=0.55,
                   color=f'C{(i + 2) % 10}', edgecolors='none',
                   label=f'{b}  n={int(m.sum())}  mean={returns[m].mean():.0f}',
                   zorder=3)

    ax.set_xlabel('Episodes', fontsize=15)
    ax.set_ylabel('Cumulative reward', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    if xmax is not None:
        ax.set_xlim(episodes.min(), xmax)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f'wrote {output}  ({len(episodes)} episodes: '
          + ', '.join(f'{b}={int((buckets == b).sum())}' for b in present) + ')')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='episode_returns.csv or a run directory')
    ap.add_argument('-o', '--output', default=None,
                    help='output path (default: returns_by_environment.pdf '
                         'next to the CSV)')
    ap.add_argument('-w', '--window', type=int, default=20,
                    help='rolling-mean window in episodes (default: 20)')
    ap.add_argument('-x', '--xmax', type=float, default=600,
                    help='cut the x-axis (and data) off at this episode '
                         '(default: 600; pass 0 for no limit)')
    args = ap.parse_args()

    csv_path = _resolve_csv(args.input)
    output = args.output or os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                                         'returns_by_environment.pdf')
    xmax = None if not args.xmax or args.xmax <= 0 else args.xmax
    make_plot(csv_path, output, window=args.window, xmax=xmax)


if __name__ == '__main__':
    main()
