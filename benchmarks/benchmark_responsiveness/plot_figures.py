#!/usr/bin/env python3
"""Multi-figure responsiveness plotter.

Reads figures.yaml — each figure names a list of benchmark data_folders and
gets one 2-panel PDF: Goodput CDF | SRTT CDF.

The avg_urtt panel from the legacy summary plot is intentionally dropped.
"""

import argparse
import math
import os
import sys
import textwrap

import yaml

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
_DEFAULT_VARIANT_DIR = _HERE
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.benchmark_responsiveness.responsiveness import (
    APPROACH_COLOR_PALETTE,
    _approach_color_palette,
    _approach_label_map,
    _collect_approach_rows_with_labels,
    _enrich_kernel_rtt_rows,
    _float_values,
    _load_benchmark_config,
    _plot_cdf,
    _plot_histogram_cdf,
    _rtt_values_for_plot,
    _schedule_reference_rows,
)


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
        folders = [str(d).strip() for d in (entry.get('data_folders') or []) if str(d).strip()]
        if not folders:
            continue
        out.append({
            'name': name,
            'description': str(entry.get('description', '')).strip(),
            'data_folders': folders,
        })
    return out


# ── one-figure rendering ─────────────────────────────────────────────────────

def _color_map_for(data_folders: list) -> dict:
    palette = _approach_color_palette(len(data_folders))
    return {folder: palette[i] for i, folder in enumerate(data_folders)}


def _render_figure(figure: dict, all_rows: list, output_root: str, output_path: str) -> bool:
    name = figure['name']
    description = figure['description']
    wanted = list(dict.fromkeys(figure['data_folders']))   # preserve order, dedupe

    rows = [r for r in all_rows if str(r.get('data_folder') or r.get('approach') or '') in set(wanted)]
    if not rows:
        print(f'[plot_figures] {name}: no rows for {wanted}; skipping', flush=True)
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8))
    title = name
    if description:
        title = f'{name}  —  {description}'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Sort by data_folder name (the user asked for this explicitly).
    sorted_folders = sorted(set(r.get('data_folder') for r in rows))
    color_map = _color_map_for(sorted_folders)

    goodput_plotted = False
    srtt_plotted = False
    srtt_all = []   # every finite SRTT value plotted, for adaptive x-limits
    for folder in sorted_folders:
        subset = [r for r in rows if r.get('data_folder') == folder]
        plot_label = next((r.get('plot_label') for r in subset if r.get('plot_label')), folder)
        wrapped = '\n'.join(textwrap.wrap(plot_label, width=24)) or plot_label
        color = color_map[folder]

        srtt_vals = _rtt_values_for_plot(subset, output_root)
        srtt_all.extend(float(v) for v in srtt_vals if math.isfinite(float(v)))
        goodput_plotted |= _plot_cdf(
            axes[0], _float_values(subset, 'mean_thr_mbps'), wrapped, color=color)
        srtt_plotted |= _plot_histogram_cdf(
            axes[1], srtt_vals, wrapped, color=color)

    schedule_refs = _schedule_reference_rows(rows)
    base_rtt_vals = _float_values(schedule_refs, 'mean_scheduled_rtt_ms')
    srtt_all.extend(float(v) for v in base_rtt_vals if math.isfinite(float(v)))
    goodput_plotted |= _plot_cdf(
        axes[0], _float_values(schedule_refs, 'mean_scheduled_bw_mbps'),
        'Scheduled BW', color='black', linewidth=1.7, zorder=10)
    srtt_plotted |= _plot_histogram_cdf(
        axes[1], base_rtt_vals,
        'Base RTT', color='black', linewidth=1.7, zorder=10)

    axes[0].set_xlabel('Average Goodput (Mbps)')
    axes[0].set_ylabel('% of Runs')
    axes[0].set_title('Goodput CDF')
    axes[1].set_xlabel('Average SRTT (ms)')
    axes[1].set_ylabel('Percent of Trials (%)')
    axes[1].set_title('SRTT CDF')
    # Adapt the SRTT window to the data: the legacy 30-90 ms window clipped
    # kernel CCAs (BBR/CUBIC/FRCC) whose queueing RTT runs well past 90 ms.
    # Cap at 250 ms so a few extreme FRCC tail runs don't squash everything.
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
    for ax in axes:
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


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Render configured responsiveness comparison figures.')
    ap.add_argument('--figures', default=os.path.join(_DEFAULT_VARIANT_DIR, 'figures.yaml'),
                    help='figures.yaml describing which data_folders go on each PDF')
    ap.add_argument('--bench-config',
                    default=os.path.join(_DEFAULT_VARIANT_DIR, 'config.yaml'),
                    help='benchmark config for label_map and default data dir')
    ap.add_argument('--data-dir', default=None,
                    help='benchmark data directory; defaults to bench config output_root')
    ap.add_argument('--output-dir', default=None,
                    help='where to write the PDFs; defaults to <data-dir>/figures/')
    args = ap.parse_args()

    bench_cfg = _load_benchmark_config(os.path.abspath(args.bench_config))

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = bench_cfg.get('output_root', 'data')
        if not os.path.isabs(str(data_dir)):
            data_dir = os.path.join(os.path.dirname(os.path.abspath(args.bench_config)),
                                    str(data_dir))
    data_dir = os.path.abspath(data_dir)

    output_dir = os.path.abspath(args.output_dir or os.path.join(data_dir, 'figures'))

    figures = _load_figures_config(os.path.abspath(args.figures))
    if not figures:
        raise SystemExit(f'[plot_figures] {args.figures}: no figures defined')

    label_map = _approach_label_map(bench_cfg)
    all_rows = _enrich_kernel_rtt_rows(
        _collect_approach_rows_with_labels(data_dir, label_map))
    if not all_rows:
        raise SystemExit(f'[plot_figures] no approach metrics found under {data_dir}')

    rendered = 0
    skipped = 0
    for figure in figures:
        target = os.path.join(output_dir, f"{figure['name']}.pdf")
        ok = _render_figure(figure, all_rows, data_dir, target)
        if ok:
            rendered += 1
            print(f'[plot_figures] wrote {target}', flush=True)
        else:
            skipped += 1
    print(f'[plot_figures] figures rendered={rendered} skipped={skipped} '
          f'out={output_dir}', flush=True)


if __name__ == '__main__':
    main()
