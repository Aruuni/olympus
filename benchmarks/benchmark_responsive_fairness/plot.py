#!/usr/bin/env python3
"""Responsive-fairness plotter.

Single, self-contained plot script for this benchmark. It renders exactly the
figures declared in ``figures.yaml`` and nothing else — each figure names a
list of benchmark ``data_folders`` and gets one 3-panel PDF: aggregate Goodput
CDF | SRTT CDF | Jain fairness CDF (7 flows), pooled across every flow count.

Plot labels are taken from the shared approaches registry
(``benchmarks/config.yaml`` via ``approaches_config``), so a folder is drawn
with the same label used everywhere else. Experiment execution never calls this
script; run it by hand after a benchmark completes:

    ./venv_training/bin/python \
        benchmarks/benchmark_responsive_fairness/plot.py
"""

import argparse
import csv
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

from benchmarks.benchmark_responsive_fairness.responsive_fairness import (
    _bench_cfg,
    _collect_rows,
    _finite_float,
    _flow_plan,
    _load_benchmark_config,
    _measured_rtt_flows,
    _row_complete,
    _schedule_mean,
    _slug,
)

_DEFAULT_CONFIG = os.path.join(_HERE, 'config.yaml')
_DEFAULT_FIGURES = os.path.join(_HERE, 'figures.yaml')

# Fairness CDF is drawn for this flow count only (fairness is most meaningful
# with the most concurrent flows in the sweep).
_FAIRNESS_FLOW_COUNT = 7

COLORS = [
    '#4878cf', '#e1812c', '#59a14f', '#e15759', '#b279a2', '#9c755f',
    '#76b7b2', '#ff9da7', '#5e7ce2', '#d37295',
]


# ── row / sample helpers ─────────────────────────────────────────────────────

def _resolve_data_dir(config_path: str, cfg: dict,
                      requested: str = None) -> str:
    data_dir = requested if requested is not None else cfg.get(
        'output_root', 'data')
    if not os.path.isabs(str(data_dir)):
        data_dir = os.path.join(os.path.dirname(config_path), str(data_dir))
    return os.path.abspath(data_dir)


def _completed_rows(rows: list, bench: dict, selected: set = None) -> list:
    completed = []
    for row in rows:
        if selected and _slug(row.get('approach', '')) not in selected:
            continue
        if not _row_complete(row, bench):
            continue
        completed.append(row)
    return sorted(
        completed,
        key=lambda row: (
            str(row.get('approach', '')),
            int(float(row.get('flow_count', 0))),
            int(float(row.get('run', 0))),
        ))


def _row_values(rows: list, field: str) -> np.ndarray:
    values = np.asarray(
        [_finite_float(row.get(field)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _load_samples(rows: list) -> list:
    """Pool the scored per-second sample rows of every run in ``rows``."""
    samples = []
    for row in rows:
        sample_csv = str(row.get('sample_csv', '')).strip()
        if not sample_csv or not os.path.isfile(sample_csv):
            sample_csv = os.path.join(
                str(row.get('run_dir', '')),
                'responsive_fairness_samples.csv',
            )
        try:
            with open(sample_csv, newline='') as handle:
                samples.extend(csv.DictReader(handle))
        except OSError:
            continue
    return samples


def _sample_values(samples: list, metric: str) -> np.ndarray:
    values = np.asarray(
        [_finite_float(sample.get(metric)) for sample in samples],
        dtype=float)
    return values[np.isfinite(values)]


def _run_avg_srtt_ms(row: dict, bench: dict) -> float:
    """Mean measured SRTT across flows during the scored window of one run.

    Model runs read srtt_ms from the per-flow state logs; kernel runs fall
    back to the iperf client's TCP RTT samples (tcpi srtt).
    """
    flow_count = int(float(row['flow_count']))
    state_logs = [
        path for path in str(row.get('state_logs', '')).split(';') if path
    ]
    traces = _measured_rtt_flows(
        str(row.get('run_dir', '')),
        _flow_plan(bench, flow_count),
        state_logs,
    )
    start = float(bench['arrival_window_s'])
    end = float(bench['duration_s'])
    flow_means = []
    for trace in traces:
        mask = (trace['time'] > start) & (trace['time'] <= end)
        values = trace['rtt'][mask]
        values = values[np.isfinite(values)]
        if values.size:
            flow_means.append(float(np.mean(values)))
    return float(np.mean(flow_means)) if flow_means else float('nan')


def _run_key(row: dict):
    try:
        return (
            _slug(row.get('approach') or row.get('data_folder') or ''),
            int(float(row['flow_count'])),
            int(float(row['run'])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_srtt_cache(path: str) -> dict:
    cache = {}
    try:
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                key = _run_key(row)
                value = _finite_float(row.get('avg_srtt_ms'))
                if key is not None and np.isfinite(value):
                    cache[key] = value
    except OSError:
        pass
    return cache


def _plot_ecdf(ax, values: np.ndarray, label: str, color: str,
               **plot_kwargs) -> bool:
    values = np.sort(values[np.isfinite(values)])
    if not values.size:
        return False
    pct = np.arange(1, len(values) + 1, dtype=float) / len(values) * 100.0
    kwargs = {'color': color, 'linewidth': 1.6, 'label': label}
    kwargs.update(plot_kwargs)
    ax.plot(values, pct, **kwargs)
    return True


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
        print(f'[plot] {name}: no rows for {wanted}; skipping', flush=True)
        return False

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.8))
    title = name
    if description:
        title = f'{name}  —  {description}'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Draw folders in the order they are listed in figures.yaml so colors and
    # legend order are predictable.
    present = {str(r.get('data_folder') or r.get('approach')) for r in rows}
    folders = [f for f in wanted if f in present]
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

def _load_rows(data_dir: str, bench: dict, cfg: dict,
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
        description='Render the responsive-fairness figures declared in '
                    'figures.yaml (and nothing else).')
    ap.add_argument('--figures', default=_DEFAULT_FIGURES)
    ap.add_argument('--config', default=_DEFAULT_CONFIG)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    # Merges the shared benchmarks/config.yaml approaches (with their plot
    # labels) via approaches_config, so folders are labelled consistently.
    cfg = _load_benchmark_config(config_path)
    bench = _bench_cfg(cfg)
    data_dir = _resolve_data_dir(config_path, cfg, args.data_dir)
    aggregate_dir = os.path.abspath(os.path.join(_HERE, 'aggregate'))
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(_HERE, 'figures'))

    figures = _load_figures_config(os.path.abspath(args.figures))
    if not figures:
        raise SystemExit(f'[plot] {args.figures}: no figures defined')

    wanted = {f for fig in figures for f in fig['data_folders']}
    all_rows = _load_rows(data_dir, bench, cfg, aggregate_dir, wanted)
    if not all_rows:
        raise SystemExit(f'[plot] no completed rows under {data_dir}')

    rendered = skipped = 0
    for figure in figures:
        target = os.path.join(output_dir, f"{figure['name']}.pdf")
        if _render_figure(figure, all_rows, bench, target):
            rendered += 1
            print(f'[plot] wrote {target}', flush=True)
        else:
            skipped += 1
    print(f'[plot] figures rendered={rendered} skipped={skipped} '
          f'out={output_dir}', flush=True)


if __name__ == '__main__':
    main()
