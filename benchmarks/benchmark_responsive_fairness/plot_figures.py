#!/usr/bin/env python3
"""Multi-figure responsive-fairness plotter.

Reads figures.yaml — each figure names a list of benchmark data_folders and
gets one 2-panel PDF: aggregate Goodput CDF | SRTT CDF, pooled across every
flow count. Mirrors benchmark_responsiveness/plot_figures.py so the kernel /
Astraea comparison looks the same in both benchmarks.
"""

import argparse
import math
import os
import sys
import textwrap

import yaml

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.benchmark_responsive_fairness.plot import (
    COLORS,
    _completed_rows,
    _load_samples,
    _load_srtt_cache,
    _plot_ecdf,
    _resolve_data_dir,
    _row_values,
    _run_avg_srtt_ms,
    _run_key,
    _sample_values,
)

# Fairness CDF is drawn for this flow count only (fairness is most meaningful
# with the most concurrent flows in the sweep).
_FAIRNESS_FLOW_COUNT = 7
from benchmarks.benchmark_responsive_fairness.responsive_fairness import (
    _bench_cfg,
    _collect_rows,
    _load_benchmark_config,
    _schedule_mean,
    _slug,
)
from olympus.common.bench_utils import _load_yaml

_PLOT_APPROACHES = os.path.join(_HERE, 'plot_approaches.yaml')


# ── figure config ────────────────────────────────────────────────────────────

def _load_figures_config(path: str) -> list:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    figures = cfg.get('figures') or []
    out = []
    for entry in figures:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name', '')).strip()
        if not name:
            continue
        folders = [str(d).strip() for d in (entry.get('data_folders') or [])
                   if str(d).strip()]
        if not folders:
            continue
        out.append({
            'name': name,
            'description': str(entry.get('description', '')).strip(),
            'data_folders': folders,
        })
    return out


# ── one-figure rendering ─────────────────────────────────────────────────────

def _schedule_refs(rows: list, bench: dict):
    """Per-run scheduled bandwidth and base RTT, deduped across flow counts."""
    start = float(bench['arrival_window_s'])
    end = float(bench['duration_s'])
    seen = set()
    capacity = []
    base_rtt = []
    for row in rows:
        try:
            key = (int(float(row['flow_count'])), int(float(row['run'])))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        capacity.append(_row_values([row], 'mean_capacity_mbps'))
        base_rtt.append(_schedule_mean(row, 'delay', start, end))
    cap = np.concatenate(capacity) if capacity else np.asarray([], dtype=float)
    rtt = np.asarray(base_rtt, dtype=float)
    return cap, rtt[np.isfinite(rtt)]


def _render_figure(figure: dict, all_rows: list, bench: dict,
                   output_path: str) -> bool:
    name = figure['name']
    description = figure['description']
    wanted = list(dict.fromkeys(figure['data_folders']))

    rows = [r for r in all_rows
            if str(r.get('data_folder') or r.get('approach') or '') in set(wanted)]
    if not rows:
        print(f'[plot_figures] {name}: no rows for {wanted}; skipping', flush=True)
        return False

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.8))
    title = name
    if description:
        title = f'{name}  —  {description}'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    folders = sorted(set(str(r.get('data_folder') or r.get('approach')) for r in rows))
    goodput_plotted = False
    srtt_plotted = False
    fairness_plotted = False
    srtt_all = []
    for index, folder in enumerate(folders):
        subset = [r for r in rows
                  if str(r.get('data_folder') or r.get('approach')) == folder]
        plot_label = next((r.get('plot_label') for r in subset if r.get('plot_label')),
                          folder)
        wrapped = '\n'.join(textwrap.wrap(plot_label, width=24)) or plot_label
        color = COLORS[index % len(COLORS)]

        srtt_vals = _row_values(subset, 'avg_srtt_ms')
        srtt_all.extend(float(v) for v in srtt_vals if math.isfinite(float(v)))
        goodput_plotted |= _plot_ecdf(
            axes[0], _row_values(subset, 'mean_aggregate_goodput_mbps'),
            wrapped, color)
        srtt_plotted |= _plot_ecdf(axes[1], srtt_vals, wrapped, color)

        seven_flow = [r for r in subset
                      if int(float(r.get('flow_count', 0))) == _FAIRNESS_FLOW_COUNT]
        fairness_plotted |= _plot_ecdf(
            axes[2], _sample_values(_load_samples(seven_flow), 'jain_fairness'),
            wrapped, color)

    cap, base_rtt = _schedule_refs(rows, bench)
    srtt_all.extend(float(v) for v in base_rtt if math.isfinite(float(v)))
    goodput_plotted |= _plot_ecdf(
        axes[0], cap, 'Scheduled BW', 'black', linewidth=1.7, zorder=10)
    srtt_plotted |= _plot_ecdf(
        axes[1], base_rtt, 'Base RTT', 'black', linewidth=1.7, zorder=10)
    axes[2].axvline(1.0, color='black', linestyle='--', linewidth=1.0,
                    label='Optimal')

    axes[0].set_xlabel('Average Aggregate Goodput (Mbps)')
    axes[0].set_ylabel('CDF (%)')
    axes[0].set_title('Goodput CDF')
    axes[1].set_xlabel('Average SRTT (ms)')
    axes[1].set_ylabel('CDF (%)')
    axes[1].set_title('SRTT CDF')
    axes[2].set_xlabel('Jain fairness ratio')
    axes[2].set_ylabel('CDF (%)')
    axes[2].set_title(f'Jain Fairness CDF ({_FAIRNESS_FLOW_COUNT} flows)')
    axes[2].set_xlim(0, 1.04)
    # Adapt the SRTT window to the data, capped at 250 ms so a few extreme
    # kernel/FRCC tail runs don't squash everything.
    if srtt_all:
        lo = max(0.0, min(30.0, min(srtt_all) - 5.0))
        hi = min(250.0, max(90.0, max(srtt_all) * 1.05))
        axes[1].set_xlim(lo, hi)
    else:
        axes[1].set_xlim(30, 250)
    if not goodput_plotted:
        axes[0].text(0.5, 0.5, 'No goodput data',
                     transform=axes[0].transAxes, ha='center', va='center')
    if not srtt_plotted:
        axes[1].text(0.5, 0.5, 'No SRTT data',
                     transform=axes[1].transAxes, ha='center', va='center')
    if not fairness_plotted:
        axes[2].text(0.5, 0.5, f'No {_FAIRNESS_FLOW_COUNT}-flow fairness data',
                     transform=axes[2].transAxes, ha='center', va='center')
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)

    unique = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    if unique:
        n = len(unique)
        ncol = min(8, max(1, n))
        legend_rows = int(math.ceil(n / float(ncol)))
        bottom = 0.18 + 0.07 * legend_rows
        fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=bottom)
        fig.legend(unique.values(), unique.keys(), loc='lower center',
                   ncol=ncol, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.0))
    else:
        fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return True


# ── row loading ──────────────────────────────────────────────────────────────

def _load_rows(config_path: str, data_dir: str, bench: dict, cfg: dict,
               aggregate_dir: str, wanted: set) -> list:
    selected = {_slug(f) for f in wanted}
    rows = _completed_rows(_collect_rows(data_dir, cfg), bench, selected or None)
    srtt_cache = _load_srtt_cache(
        os.path.join(aggregate_dir, 'fairness_runs.csv'))
    if not srtt_cache:
        srtt_cache = _load_srtt_cache(
            os.path.join(data_dir, 'fairness_runs.csv'))
    for row in rows:
        cached = srtt_cache.get(_run_key(row), float('nan'))
        row['avg_srtt_ms'] = (
            cached if np.isfinite(cached) else _run_avg_srtt_ms(row, bench))
    return rows


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Render configured responsive-fairness comparison figures.')
    ap.add_argument('--figures', default=os.path.join(_HERE, 'figures.yaml'))
    ap.add_argument('--config', default=os.path.join(_HERE, 'config.yaml'))
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = _load_benchmark_config(config_path)
    bench = _bench_cfg(cfg)
    # Reuse the curated plot labels (CUBIC, BBRv3, ...) from plot_approaches.yaml.
    cfg['approaches'] = _load_yaml(_PLOT_APPROACHES).get('approaches') or []
    data_dir = _resolve_data_dir(config_path, cfg, args.data_dir)
    aggregate_dir = os.path.abspath(os.path.join(_HERE, 'aggregate'))
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(data_dir, 'figures'))

    figures = _load_figures_config(os.path.abspath(args.figures))
    if not figures:
        raise SystemExit(f'[plot_figures] {args.figures}: no figures defined')

    wanted = {f for fig in figures for f in fig['data_folders']}
    all_rows = _load_rows(config_path, data_dir, bench, cfg, aggregate_dir, wanted)
    if not all_rows:
        raise SystemExit(f'[plot_figures] no completed rows under {data_dir}')

    rendered = skipped = 0
    for figure in figures:
        target = os.path.join(output_dir, f"{figure['name']}.pdf")
        if _render_figure(figure, all_rows, bench, target):
            rendered += 1
            print(f'[plot_figures] wrote {target}', flush=True)
        else:
            skipped += 1
    print(f'[plot_figures] figures rendered={rendered} skipped={skipped} '
          f'out={output_dir}', flush=True)


if __name__ == '__main__':
    main()
