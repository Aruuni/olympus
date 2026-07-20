#!/usr/bin/env python3
"""Render the inter-RTT goodput-ratio figures in Mihai Mazilu's IFIP style.

One figure per BDP multiplier (0.2x/1x/4x), matching
``aggregate_plots/figure_inter_rtt/plot.py`` from the mininettestbed paper repo.
"""

import argparse
import csv
import os
from pathlib import Path
import sys

import yaml
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper import paper_style as style
import matplotlib.pyplot as plt


SUITE = 'inter_rtt'


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


def _series(rows: list, approach: str, qmult: float) -> tuple:
    subset = [row for row in rows
              if row.get('suite') == SUITE and row.get('approach') == approach
              and style.finite(row.get('bdp_multiplier')) == float(qmult)]
    subset.sort(key=lambda row: style.finite(row.get('rtt_ms')))
    x = [style.finite(row.get('rtt_ms')) for row in subset]
    y = [style.finite(row.get('goodput_ratio_total_mean')) for row in subset]
    e = [style.finite(row.get('goodput_ratio_total_std'), 0.0) for row in subset]
    return x, y, e


def _legend(fig, ax, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, (list, tuple)) else h for h in handles]
    if handles:
        fig.legend(handles, labels, ncol=ncol or len(handles),
                   loc='upper center', bbox_to_anchor=(0.5, 1.12),
                   frameon=False, fontsize=7, columnspacing=0.8,
                   handlelength=2.0, handletextpad=0.5)


def plot_from_config(config_path: str) -> int:
    config_path = os.path.abspath(config_path)
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle) or {}
    output_root = Path(str(cfg.get('output_root', 'data')))
    if not output_root.is_absolute():
        output_root = Path(config_path).parent / output_root
    rows = _read_csv(output_root / 'summary.csv')
    if not rows:
        print(f'[inter_plot] no summary data at {output_root / "summary.csv"}')
        return 1

    style.use_science()
    approaches = _approach_order(cfg, rows)
    approach_cfg = _approach_cfg(cfg)
    qmults = list(cfg.get('benchmark', {}).get('inter_rtt', {})
                  .get('bdp_multipliers', [0.2, 1, 4]))
    figure_dir = output_root / 'figures'

    for qmult in qmults:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 1.1))
        for index, approach in enumerate(approaches):
            x, y, e = _series(rows, approach, float(qmult))
            color, marker, label = style.style_for(
                approach_cfg.get(approach, {'plot_label': approach}), index)
            style.plot_points(ax, x, y, e, marker, color, label)
        ax.set(yscale='linear', xlabel='RTT (ms)', ylabel='Goodput Ratio',
               ylim=[-0.1, 1.1])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        _legend(fig, ax, ncol=len(approaches))
        style.save(fig, figure_dir / f'goodput_inter_rtt_qmult{float(qmult):g}')
        print(f'[inter_plot] wrote {figure_dir}/goodput_inter_rtt_qmult{float(qmult):g}.pdf')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    args = parser.parse_args(argv)
    return plot_from_config(args.config)


if __name__ == '__main__':
    raise SystemExit(main())
