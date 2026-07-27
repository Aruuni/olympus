#!/usr/bin/env python3
"""Render the efficiency scatter in Mihai Mazilu's IFIP style.

One figure per BDP multiplier: normalized throughput vs normalized delay,
colour per approach, marker per RTT, and a 1-sigma confidence ellipse over the
per-run points — matching ``aggregate_plots/figure9_efficiency/plot.py`` from
the mininettestbed paper repo (x axis inverted, so up-and-right is better).
"""

import argparse
import csv
import math
import os
from pathlib import Path
import sys

import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper import paper_style as style
import matplotlib.pyplot as plt


SUITE = 'efficiency'
RTT_MARKERS = ['^', '*', 's', 'D', 'o']


def _read_csv(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _approach_order(cfg: dict, rows: list) -> list:
    present = {row.get('approach') for row in rows}
    ordered = []
    for raw in cfg.get('approaches') or []:
        key = str(raw.get('data_folder') or raw.get('name') or '')
        if key in present and key not in ordered:
            ordered.append(key)
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _approach_cfg(cfg: dict) -> dict:
    out = {}
    for raw in cfg.get('approaches') or []:
        key = str(raw.get('data_folder') or raw.get('name') or '')
        out[key] = raw
    return out


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """1-sigma covariance ellipse, as in the paper's figure-9 plot."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return None
    cov = np.cov(x, y)
    denom = math.sqrt(cov[0, 0] * cov[1, 1])
    pearson = cov[0, 1] / denom if denom > 0 else 0.0
    pearson = min(1.0, max(-1.0, pearson))
    ellipse = Ellipse(
        (0, 0), width=math.sqrt(1 + pearson) * 2,
        height=math.sqrt(1 - pearson) * 2, facecolor=facecolor, **kwargs)
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(math.sqrt(cov[0, 0]) * n_std, math.sqrt(cov[1, 1]) * n_std)
              .translate(float(np.mean(x)), float(np.mean(y))))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def _run_points(rows: list, approach: str, qmult: float, rtt: float) -> tuple:
    xs, ys = [], []
    for row in rows:
        if (row.get('suite') != SUITE or row.get('approach') != approach
                or style.finite(row.get('bdp_multiplier')) != float(qmult)
                or style.finite(row.get('rtt_ms')) != float(rtt)
                or str(row.get('error', '')).strip()):
            continue
        x = style.finite(row.get('norm_delay_mean'))
        y = style.finite(row.get('norm_throughput_mean'))
        if x == x and y == y:
            xs.append(x)
            ys.append(y)
    return xs, ys


def plot_from_config(config_path: str) -> int:
    config_path = os.path.abspath(config_path)
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle) or {}
    output_root = Path(str(cfg.get('output_root', 'data')))
    if not output_root.is_absolute():
        output_root = Path(config_path).parent / output_root
    rows = [row for row in _read_csv(output_root / 'metrics.csv')
            if row.get('suite') == SUITE]
    if not rows:
        print(f'[efficiency_plot] no metrics data at {output_root / "metrics.csv"}')
        return 1

    style.use_science()
    approaches = _approach_order(cfg, rows)
    approach_cfg = _approach_cfg(cfg)
    bench = cfg.get('benchmark', {}) or {}
    qmults = list((bench.get('efficiency') or {}).get('bdp_multipliers',
                                                      [0.2, 1, 4]))
    rtts = sorted({style.finite(row.get('rtt_ms')) for row in rows
                   if style.finite(row.get('rtt_ms')) == style.finite(row.get('rtt_ms'))})
    figure_dir = output_root / 'figures'

    for qmult in qmults:
        fig, ax = plt.subplots(figsize=(3, 1.5))
        drew = False
        for index, approach in enumerate(approaches):
            color, _, label = style.style_for(
                approach_cfg.get(approach, {'plot_label': approach}), index)
            for rtt_index, rtt in enumerate(rtts):
                xs, ys = _run_points(rows, approach, float(qmult), rtt)
                if not xs:
                    continue
                marker = RTT_MARKERS[rtt_index % len(RTT_MARKERS)]
                ax.scatter(float(np.mean(xs)), float(np.mean(ys)),
                           edgecolors=color, marker=marker,
                           facecolors='none', s=22, linewidths=0.8,
                           label=f'{label} {rtt:g}ms')
                confidence_ellipse(xs, ys, ax, facecolor=color,
                                   edgecolor='none', alpha=0.25)
                drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set(xlabel='Norm. Delay', ylabel='Norm. Throughput',
               ylim=[0, 1.05])
        ax.invert_xaxis()
        rtt_handles = [
            Line2D([], [], marker=RTT_MARKERS[i % len(RTT_MARKERS)],
                   color='black', linestyle='None', markersize=5,
                   markeredgewidth=0.8, markerfacecolor='none')
            for i in range(len(rtts))
        ]
        ax.legend(rtt_handles, [f'{rtt:g} ms' for rtt in rtts],
                  loc='lower right', frameon=False, fontsize=6,
                  handlelength=0.8, handletextpad=0.6, labelspacing=0.2,
                  title='RTT', title_fontsize=6)
        approach_handles = [
            Line2D([], [], color=style.style_for(
                approach_cfg.get(a, {'plot_label': a}), i)[0], linewidth=1)
            for i, a in enumerate(approaches)
        ]
        approach_labels = [style.style_for(
            approach_cfg.get(a, {'plot_label': a}), i)[2]
            for i, a in enumerate(approaches)]
        fig.legend(approach_handles, approach_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.14), ncol=min(3, len(approaches)),
                   frameon=False, fontsize=6, columnspacing=0.8,
                   handlelength=1.6, handletextpad=0.5)
        stem = figure_dir / f'efficiency_scatter_qmult{float(qmult):g}'
        style.save(fig, stem)
        print(f'[efficiency_plot] wrote {stem}.pdf')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    args = parser.parse_args(argv)
    return plot_from_config(args.config)


if __name__ == '__main__':
    raise SystemExit(main())
