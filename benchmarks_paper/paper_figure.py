#!/usr/bin/env python3
"""Build the paper-ready, three-panel benchmark figure.

The panels use the same plotting primitives as the individual benchmark
figures, but are drawn on one 1x3 grid so every axes has exactly the same
physical size.  The default output is ``benchmarks_paper/paper_figure.pdf``
and a matching high-resolution PNG.
"""

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')

import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper import paper_style as style  # noqa: E402
from benchmarks_paper.efficiency.plot import confidence_ellipse  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_CONFIGS = {
    'intra_rtt': PAPER_ROOT / 'intra' / 'config.yaml',
    'inter_rtt': PAPER_ROOT / 'inter' / 'config.yaml',
    'efficiency': PAPER_ROOT / 'efficiency' / 'config.yaml',
}

# Deliberately unique in both colour and shape.  This figure compares closely
# related implementations, so reusing the protocol colours from the source
# plots would make the two Astraea variants needlessly hard to distinguish.
# Circles are excluded to keep every approach visually distinct at small size.
DISTINCT_COLORS = ('#0C5DA5', '#00B945', '#FF9500', '#845B97', '#FF2C00')
DISTINCT_MARKERS = ('^', 's', 'D', 'X', 'P')
EFFICIENCY_RTT_MARKERS = ('v', 's')


def _figure_style(approach: str, approach_cfg: dict, index: int) -> tuple:
    row = approach_cfg.get(approach, {'plot_label': approach})
    _, _, label = style.style_for(row, index)
    return (
        DISTINCT_COLORS[index % len(DISTINCT_COLORS)],
        DISTINCT_MARKERS[index % len(DISTINCT_MARKERS)],
        label,
    )


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path) -> dict:
    """Load a suite config and its optional shared approach config."""
    path = path.resolve()
    with path.open() as handle:
        cfg = yaml.safe_load(handle) or {}
    parent_ref = cfg.pop('approaches_config', None) or cfg.pop('extends', None)
    if not parent_ref:
        return cfg
    parent_path = Path(str(parent_ref))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_config(parent_path), cfg)


def _data_root(config_path: Path, cfg: dict) -> Path:
    root = Path(str(cfg.get('output_root', 'data')))
    return root if root.is_absolute() else config_path.resolve().parent / root


def _read_csv(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(f'benchmark data not found: {path}')
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _approaches(configs: dict, rows_by_suite: dict) -> tuple:
    present = {
        row.get('approach')
        for rows in rows_by_suite.values()
        for row in rows
        if row.get('approach')
    }
    approach_rows = {}
    ordered = []
    for cfg in configs.values():
        for raw in cfg.get('approaches') or []:
            key = str(raw.get('data_folder') or raw.get('name') or '')
            if not key:
                continue
            approach_rows.setdefault(key, raw)
            if key in present and key not in ordered:
                ordered.append(key)
    ordered.extend(sorted(present - set(ordered)))
    return ordered, approach_rows


def _fairness_series(rows: list, suite: str, approach: str,
                     qmult: float) -> tuple:
    subset = [
        row for row in rows
        if row.get('suite') == suite
        and row.get('approach') == approach
        and style.finite(row.get('bdp_multiplier')) == qmult
    ]
    subset.sort(key=lambda row: style.finite(row.get('rtt_ms')))
    xvals, yvals, errors = [], [], []
    for row in subset:
        xval = style.finite(row.get('rtt_ms'))
        yval = style.finite(row.get('goodput_ratio_total_mean'))
        if xval != xval or yval != yval:
            continue
        xvals.append(xval)
        yvals.append(yval)
        errors.append(style.finite(row.get('goodput_ratio_total_std'), 0.0))
    return xvals, yvals, errors


def _draw_fairness(ax, rows: list, suite: str, approaches: list,
                   approach_cfg: dict, qmult: float) -> None:
    drew = False
    for index, approach in enumerate(approaches):
        xvals, yvals, errors = _fairness_series(
            rows, suite, approach, qmult)
        if not xvals:
            continue
        color, marker, label = _figure_style(
            approach, approach_cfg, index)
        _, caps, bars = ax.errorbar(
            xvals, yvals, yerr=errors,
            marker=marker, markersize=4.2, markeredgewidth=1.1,
            linewidth=style.LINEWIDTH, elinewidth=style.ELINEWIDTH,
            capsize=style.CAPSIZE, capthick=style.CAPTHICK,
            color=color, label=label,
        )
        for bar in bars:
            bar.set_alpha(0.5)
        for cap in caps:
            cap.set_alpha(0.5)
        drew = True
    if not drew:
        raise ValueError(f'no {suite} rows found for {qmult:g}x BDP')
    ax.set(
        xlabel='RTT (ms)',
        ylabel='Goodput Ratio',
        ylim=(-0.1, 1.1),
    )
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())


def _efficiency_points(rows: list, approach: str, qmult: float,
                       rtt: float) -> tuple:
    xvals, yvals = [], []
    for row in rows:
        if (row.get('suite') != 'efficiency'
                or row.get('approach') != approach
                or style.finite(row.get('bdp_multiplier')) != qmult
                or style.finite(row.get('rtt_ms')) != rtt
                or str(row.get('error', '')).strip()):
            continue
        xval = style.finite(row.get('norm_delay_mean'))
        yval = style.finite(row.get('norm_throughput_mean'))
        if xval == xval and yval == yval:
            xvals.append(xval)
            yvals.append(yval)
    return xvals, yvals


def _draw_efficiency(ax, rows: list, approaches: list,
                     approach_cfg: dict, qmult: float) -> None:
    rtts = sorted({
        style.finite(row.get('rtt_ms'))
        for row in rows
        if row.get('suite') == 'efficiency'
        and style.finite(row.get('rtt_ms')) == style.finite(row.get('rtt_ms'))
    })
    drew = False
    for index, approach in enumerate(approaches):
        color, _, _ = _figure_style(approach, approach_cfg, index)
        for rtt_index, rtt in enumerate(rtts):
            xvals, yvals = _efficiency_points(rows, approach, qmult, rtt)
            if not xvals:
                continue
            # In this panel colour identifies the approach and a pair of
            # simple, unfilled shapes identifies RTT.  The in-panel key thus
            # uses exactly the same symbols as the plotted data.
            marker = EFFICIENCY_RTT_MARKERS[
                rtt_index % len(EFFICIENCY_RTT_MARKERS)]
            ax.scatter(
                float(np.mean(xvals)), float(np.mean(yvals)),
                edgecolors=color, marker=marker, facecolors='none',
                s=16, linewidths=1.1,
            )
            confidence_ellipse(
                xvals, yvals, ax, facecolor=color,
                edgecolor='none', alpha=0.25,
            )
            drew = True
    if not drew:
        raise ValueError(f'no efficiency rows found for {qmult:g}x BDP')

    ax.set(
        xlabel='Norm. Delay',
        ylabel='Norm. Thr.',
        ylim=(0.4, 1.05),
    )
    ax.invert_xaxis()
    rtt_handles = [
        Line2D(
            [], [], marker=EFFICIENCY_RTT_MARKERS[
                index % len(EFFICIENCY_RTT_MARKERS)],
            color='black', linestyle='None', markersize=4,
            markeredgewidth=1.1,
            markerfacecolor='none',
        )
        for index in range(len(rtts))
    ]
    ax.legend(
        rtt_handles, [f'{rtt:g} ms' for rtt in rtts],
        loc='lower right', frameon=False, fontsize=6,
        handlelength=0.8, handletextpad=0.6, labelspacing=0.2,
        title='RTT', title_fontsize=6,
    )


def _shared_legend(fig, approaches: list, approach_cfg: dict) -> None:
    handles, labels = [], []
    for index, approach in enumerate(approaches):
        color, marker, label = _figure_style(
            approach, approach_cfg, index)
        handles.append(Line2D(
            [], [], color=color, marker=marker,
            linewidth=style.LINEWIDTH, markersize=3.5,
            markeredgewidth=1.0,
        ))
        labels.append(label)
    fig.legend(
        handles, labels, ncol=len(handles), loc='upper center',
        bbox_to_anchor=(0.5, 0.975), frameon=False, fontsize=7,
        columnspacing=1.2, handlelength=1.8, handletextpad=0.5,
    )


def build_figure(config_paths: dict, output_stem: Path,
                 qmult: float = 1.0) -> None:
    configs = {
        suite: _load_config(path)
        for suite, path in config_paths.items()
    }
    rows_by_suite = {
        'intra_rtt': _read_csv(
            _data_root(config_paths['intra_rtt'], configs['intra_rtt'])
            / 'summary.csv'),
        'inter_rtt': _read_csv(
            _data_root(config_paths['inter_rtt'], configs['inter_rtt'])
            / 'summary.csv'),
        'efficiency': _read_csv(
            _data_root(config_paths['efficiency'], configs['efficiency'])
            / 'metrics.csv'),
    }
    approaches, approach_cfg = _approaches(configs, rows_by_suite)
    if not approaches:
        raise ValueError('no benchmark approaches found')

    style.use_science()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.48))
    _draw_fairness(
        axes[0], rows_by_suite['intra_rtt'], 'intra_rtt',
        approaches, approach_cfg, qmult)
    _draw_fairness(
        axes[1], rows_by_suite['inter_rtt'], 'inter_rtt',
        approaches, approach_cfg, qmult)
    _draw_efficiency(
        axes[2], rows_by_suite['efficiency'],
        approaches, approach_cfg, qmult)

    captions = (
        r'\textbf{(a) Intra-RTT Fairness}',
        r'\textbf{(b) Inter-RTT Fairness}',
        r'\textbf{(c) Network Efficiency}',
    )
    for ax, caption in zip(axes, captions):
        ax.text(
            0.5, -0.42, caption, transform=ax.transAxes,
            ha='center', va='top', fontsize=8,
        )

    _shared_legend(fig, approaches, approach_cfg)
    # Fixed margins keep all three axes boxes identical; the captions mimic
    # the subfigure labels in the supplied paper example.
    fig.subplots_adjust(
        left=0.065, right=0.995, top=0.81, bottom=0.29, wspace=0.38)
    output_stem = output_stem.resolve()
    style.save(fig, output_stem)
    print(f'[paper_figure] wrote {output_stem}.pdf')
    print(f'[paper_figure] wrote {output_stem}.png')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Render equal-size intra/inter/efficiency paper panels.')
    parser.add_argument('--qmult', type=float, default=1.0,
                        help='BDP queue multiplier to plot (default: 1)')
    parser.add_argument('--intra-config', type=Path,
                        default=DEFAULT_CONFIGS['intra_rtt'])
    parser.add_argument('--inter-config', type=Path,
                        default=DEFAULT_CONFIGS['inter_rtt'])
    parser.add_argument('--efficiency-config', type=Path,
                        default=DEFAULT_CONFIGS['efficiency'])
    parser.add_argument(
        '--output', type=Path, default=PAPER_ROOT / 'paper_figure',
        help='output path without extension (default: benchmarks_paper/paper_figure)',
    )
    args = parser.parse_args(argv)
    build_figure(
        {
            'intra_rtt': args.intra_config,
            'inter_rtt': args.inter_config,
            'efficiency': args.efficiency_config,
        },
        args.output,
        args.qmult,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
