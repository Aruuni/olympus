#!/usr/bin/env python3
"""Plot responsive-fairness results without running any experiments."""

import argparse
import csv
import os
import sys
import textwrap
import zlib

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
_DEFAULT_CONFIG = os.path.join(_HERE, 'config.yaml')
# Curated approaches drawn into the aggregate plots, kept separate from the
# benchmark approaches config (which controls what gets run). Hardcoded so the
# default `plot.py` invocation needs no extra flags.
_DEFAULT_PLOT_APPROACHES = os.path.join(_HERE, 'plot_approaches.yaml')
sys.path.insert(0, _ROOT)

from benchmarks.benchmark_responsive_fairness.responsive_fairness import (
    _bench_cfg,
    _collect_rows,
    _finite_float,
    _flow_plan,
    _load_benchmark_config,
    _measured_rtt_flows,
    _restore_sudo_user_ownership,
    _row_complete,
    _schedule_mean,
    _slug,
)
from olympus.common.bench_utils import _approach_label, _load_yaml
from olympus.common.marl_team_reward import r_fair_from_avg_throughput


COLORS = [
    '#4878cf', '#e1812c', '#59a14f', '#e15759', '#b279a2', '#9c755f',
    '#76b7b2', '#ff9da7', '#5e7ce2', '#d37295',
]

RUN_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'flow_count', 'run', 'seed',
    'mean_jain_fairness', 'p05_jain_fairness',
    'mean_min_max_goodput_ratio', 'p05_min_max_goodput_ratio',
    'avg_srtt_ms',
]

SUMMARY_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'flow_count', 'completed_runs',
    'metric', 'mean', 'std', 'p05', 'p25', 'median', 'p75', 'p95',
]

PLOT_METRICS = [
    ('mean_jain_fairness', 'Mean Jain fairness'),
    ('p05_jain_fairness', '5th-percentile Jain fairness'),
    ('mean_min_max_goodput_ratio', 'Mean min / max goodput'),
    ('p05_min_max_goodput_ratio', '5th-percentile min / max goodput'),
]


def _resolve_data_dir(config_path: str, cfg: dict,
                      requested: str = None) -> str:
    data_dir = requested if requested is not None else cfg.get(
        'output_root', 'data')
    if not os.path.isabs(str(data_dir)):
        data_dir = os.path.join(os.path.dirname(config_path), str(data_dir))
    return os.path.abspath(data_dir)


def _write_csv(path: str, fields: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


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


def _run_rows(rows: list) -> list:
    return [
        {field: row.get(field, '') for field in RUN_FIELDS}
        for row in rows
    ]


def _summary_rows(rows: list, flow_counts: list) -> list:
    summaries = []
    approaches = sorted({str(row['approach']) for row in rows})
    for approach in approaches:
        approach_rows = [row for row in rows if row['approach'] == approach]
        plot_label = next(
            (row.get('plot_label') for row in approach_rows
             if row.get('plot_label')),
            approach,
        )
        data_folder = next(
            (row.get('data_folder') for row in approach_rows
             if row.get('data_folder')),
            approach,
        )
        for flow_count in flow_counts:
            group = [
                row for row in approach_rows
                if int(float(row['flow_count'])) == flow_count
            ]
            for metric, _ in PLOT_METRICS:
                values = np.asarray(
                    [_finite_float(row.get(metric)) for row in group],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                summaries.append({
                    'approach': approach,
                    'data_folder': data_folder,
                    'plot_label': plot_label,
                    'flow_count': flow_count,
                    'completed_runs': len(values),
                    'metric': metric,
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'p05': float(np.percentile(values, 5)),
                    'p25': float(np.percentile(values, 25)),
                    'median': float(np.median(values)),
                    'p75': float(np.percentile(values, 75)),
                    'p95': float(np.percentile(values, 95)),
                })
    return summaries


def _scatter_with_medians(ax, rows: list, approaches: list,
                          flow_counts: list, values_for, tag: str) -> bool:
    """Jittered per-run scatter plus median/IQR error bars per approach.

    ``values_for(group)`` returns the finite per-run values for one
    (approach, flow_count) group of metric rows.
    """
    width = 0.72 / max(1, len(approaches))
    centered = (len(approaches) - 1) / 2.0
    plotted = False
    for approach_index, approach in enumerate(approaches):
        approach_rows = [row for row in rows if row['approach'] == approach]
        label = next(
            (row.get('plot_label') for row in approach_rows
             if row.get('plot_label')),
            approach,
        )
        color = COLORS[approach_index % len(COLORS)]
        x_values = []
        medians = []
        lower = []
        upper = []
        for flow_index, flow_count in enumerate(flow_counts):
            group = [
                row for row in approach_rows
                if int(float(row['flow_count'])) == flow_count
            ]
            values = values_for(group)
            if not values.size:
                continue
            x = flow_index + (approach_index - centered) * width
            seed = zlib.crc32(
                f'{approach}:{flow_count}:{tag}'.encode('utf-8'))
            rng = np.random.default_rng(seed)
            jitter = rng.uniform(
                -width * 0.24, width * 0.24, size=len(values))
            ax.scatter(
                np.full(len(values), x) + jitter, values,
                s=12, color=color, alpha=0.22, linewidths=0, zorder=2,
            )
            median = float(np.median(values))
            p25 = float(np.percentile(values, 25))
            p75 = float(np.percentile(values, 75))
            x_values.append(x)
            medians.append(median)
            lower.append(median - p25)
            upper.append(p75 - median)
        if not x_values:
            continue
        plotted = True
        ax.errorbar(
            x_values, medians, yerr=[lower, upper],
            color=color, marker='o', markersize=5, linewidth=1.4,
            capsize=3, label='\n'.join(textwrap.wrap(str(label), 34)),
            zorder=3,
        )
    ax.set_xticks(range(len(flow_counts)))
    ax.set_xticklabels([str(value) for value in flow_counts])
    ax.set_xlabel('Number of flows')
    ax.grid(True, alpha=0.25)
    return plotted


def _row_values(rows: list, field: str) -> np.ndarray:
    values = np.asarray(
        [_finite_float(row.get(field)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _plot_metric(ax, rows: list, approaches: list, flow_counts: list,
                 metric: str, title: str) -> None:
    plotted = _scatter_with_medians(
        ax, rows, approaches, flow_counts,
        lambda group: _row_values(group, metric), metric)
    ax.set_title(title)
    ax.set_ylabel('Fairness ratio')
    ax.set_ylim(0, 1.02)
    if not plotted:
        ax.text(
            0.5, 0.5, 'No completed fairness data',
            transform=ax.transAxes, ha='center', va='center',
        )


def _plot_fairness(rows: list, flow_counts: list, output: str) -> None:
    approaches = sorted({str(row['approach']) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=True)
    fig.suptitle(
        'Responsive Fairness by Number of Flows per Learner',
        fontsize=13, fontweight='bold',
    )
    for ax, (metric, title) in zip(axes.flat, PLOT_METRICS):
        _plot_metric(ax, rows, approaches, flow_counts, metric, title)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc='lower center',
            ncol=min(3, len(handles)), frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.subplots_adjust(
        left=0.08, right=0.985, top=0.92,
        bottom=0.13 if handles else 0.08, hspace=0.27, wspace=0.18,
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = f'{output}.tmp.{os.getpid()}.pdf'
    try:
        fig.savefig(tmp, dpi=240, bbox_inches='tight')
    finally:
        plt.close(fig)
    os.replace(tmp, output)


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


def _sample_r_fair_values(samples: list) -> np.ndarray:
    """Astraea R_fair per scored second from the per-flow goodput columns."""
    if not samples:
        return np.empty(0)
    goodput = np.zeros((len(samples), 7), dtype=float)
    active = np.zeros((len(samples), 7), dtype=float)
    for index, sample in enumerate(samples):
        flow_count = int(float(sample.get('flow_count', 0) or 0))
        for flow in range(min(flow_count, 7)):
            value = _finite_float(sample.get(f'flow{flow + 1}_goodput_mbps'))
            if np.isfinite(value):
                goodput[index, flow] = max(value, 0.0)
                active[index, flow] = 1.0
    r_fair = np.asarray(
        r_fair_from_avg_throughput(goodput, mask=active), dtype=float)
    return r_fair[active.sum(axis=1) >= 2]


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


def _plot_learner_cdfs(rows: list, flow_counts: list, output: str) -> int:
    approaches = sorted({str(row['approach']) for row in rows})
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = f'{output}.tmp.{os.getpid()}.pdf'
    pages = 0
    with PdfPages(tmp) as pdf:
        for approach in approaches:
            approach_rows = [
                row for row in rows if row['approach'] == approach]
            plot_label = next(
                (row.get('plot_label') for row in approach_rows
                 if row.get('plot_label')),
                approach,
            )
            fig, axes = plt.subplots(
                1, 3, figsize=(16.0, 5.4), sharey=True)
            fig.suptitle(str(plot_label), fontsize=12, fontweight='bold')
            plotted = [False, False, False]
            for index, flow_count in enumerate(flow_counts):
                group = [
                    row for row in approach_rows
                    if int(float(row['flow_count'])) == flow_count
                ]
                samples = _load_samples(group)
                label = f'{flow_count} flows ({len(group)} runs)'
                color = COLORS[index % len(COLORS)]
                plotted[0] |= _plot_ecdf(
                    axes[0],
                    _sample_values(samples, 'jain_fairness'),
                    label,
                    color,
                )
                plotted[1] |= _plot_ecdf(
                    axes[1],
                    _sample_r_fair_values(samples),
                    label,
                    color,
                )
                plotted[2] |= _plot_ecdf(
                    axes[2],
                    _sample_values(samples, 'min_max_goodput_ratio'),
                    label,
                    color,
                )

            axes[0].set_title('Jain fairness')
            axes[0].set_xlabel('Fairness ratio')
            axes[0].set_xlim(0, 1.04)
            axes[1].set_title('R_fair (Astraea CoV, lower is fairer)')
            axes[1].set_xlabel('R_fair')
            # One flow hogging the link gives sqrt(n-1)/n, at most 0.5 (n=2).
            axes[1].set_xlim(-0.02, 0.55)
            axes[2].set_title('Minimum / maximum goodput')
            axes[2].set_xlabel('Fairness ratio')
            axes[2].set_xlim(0, 1.04)
            for ax, optimal in zip(axes, (1.0, 0.0, 1.0)):
                ax.axvline(
                    optimal, color='black', linestyle='--', linewidth=1.1,
                    label='Optimal')
            for index, ax in enumerate(axes):
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.25)
                if plotted[index]:
                    ax.legend(frameon=False, fontsize=9, loc='best')
                else:
                    ax.text(
                        0.5, 0.5, 'No completed fairness samples',
                        transform=ax.transAxes, ha='center', va='center',
                    )
            axes[0].set_ylabel('CDF (%)')
            fig.subplots_adjust(
                left=0.06, right=0.99, top=0.88,
                bottom=0.13, wspace=0.08)
            pdf.savefig(fig, dpi=240, bbox_inches='tight')
            plt.close(fig)
            pages += 1
    os.replace(tmp, output)
    return pages


def _plot_aggregate_reports(rows: list, flow_counts: list, bench: dict,
                            aggregate_dir: str) -> list:
    """Write one aggregate report PDF per approach into ``aggregate_dir``.

    Each report is a responsiveness-summary-style page: per-run goodput
    and SRTT CDFs with one curve per flow count (plus the pooled
    Scheduled BW / Base RTT references), and the rightmost column stacks
    CDFs of the three fairness metrics computed from the scored
    per-second samples. Rows need ``avg_srtt_ms`` filled in (see
    ``_run_avg_srtt_ms``). Returns the written paths.
    """
    approaches = sorted({str(row['approach']) for row in rows})
    start = float(bench['arrival_window_s'])
    end = float(bench['duration_s'])
    os.makedirs(aggregate_dir, exist_ok=True)
    outputs = []
    for approach in approaches:
        approach_rows = [
            row for row in rows if row['approach'] == approach]
        plot_label = next(
            (row.get('plot_label') for row in approach_rows
             if row.get('plot_label')),
            approach,
        )
        fig = plt.figure(figsize=(16.0, 6.4))
        grid = fig.add_gridspec(
            3, 3, wspace=0.24, hspace=0.55,
            left=0.05, right=0.99, top=0.86, bottom=0.2)
        ax_goodput = fig.add_subplot(grid[:, 0])
        ax_srtt = fig.add_subplot(grid[:, 1])
        fairness_axes = [
            fig.add_subplot(grid[index, 2]) for index in range(3)]
        fig.suptitle(
            f'{plot_label} — {approach}',
            fontsize=12, fontweight='bold')

        for index, flow_count in enumerate(flow_counts):
            group = [
                row for row in approach_rows
                if int(float(row['flow_count'])) == flow_count
            ]
            samples = _load_samples(group)
            label = f'{flow_count} flows ({len(group)} runs)'
            color = COLORS[index % len(COLORS)]
            _plot_ecdf(
                ax_goodput,
                _row_values(group, 'mean_aggregate_goodput_mbps'),
                label, color)
            _plot_ecdf(
                ax_srtt, _row_values(group, 'avg_srtt_ms'), label, color)
            _plot_ecdf(
                fairness_axes[0],
                _sample_values(samples, 'jain_fairness'), label, color)
            _plot_ecdf(
                fairness_axes[1],
                _sample_r_fair_values(samples), label, color)
            _plot_ecdf(
                fairness_axes[2],
                _sample_values(samples, 'min_max_goodput_ratio'),
                label, color)

        _plot_ecdf(
            ax_goodput,
            _row_values(approach_rows, 'mean_capacity_mbps'),
            'Scheduled BW', 'black')
        base_rtt = np.asarray(
            [_schedule_mean(row, 'delay', start, end)
             for row in approach_rows],
            dtype=float)
        _plot_ecdf(
            ax_srtt, base_rtt[np.isfinite(base_rtt)],
            'Base RTT', 'black')

        ax_goodput.set_title('Goodput CDF')
        ax_goodput.set_xlabel('Average aggregate goodput (Mbps)')
        ax_goodput.set_ylabel('% of runs')
        ax_srtt.set_title('SRTT CDF')
        ax_srtt.set_xlabel('Average SRTT (ms)')
        ax_srtt.set_ylabel('% of runs')
        for ax in (ax_goodput, ax_srtt):
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
        # x-limits padded past the optimal values (Jain/min-max 1, R_fair 0)
        # so the dashed optimal line stands clear of the axis frame.
        fairness_specs = (
            ('Jain fairness', (0, 1.04), 1.0),
            ('R_fair (lower is fairer)', (-0.02, 0.55), 0.0),
            ('Min / max goodput', (0, 1.04), 1.0),
        )
        for ax, (title, xlim, optimal) in zip(fairness_axes, fairness_specs):
            ax.axvline(
                optimal, color='black', linestyle='--', linewidth=1.1,
                label='Optimal')
            ax.set_title(title, fontsize=9)
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
        fairness_axes[1].set_ylabel('CDF (%)', fontsize=8)
        fairness_axes[2].set_xlabel('Fairness metric', fontsize=8)

        unique = {}
        for ax in (ax_goodput, ax_srtt, *fairness_axes):
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                unique.setdefault(label, handle)
        fig.legend(
            list(unique.values()), list(unique.keys()),
            loc='lower center', bbox_to_anchor=(0.5, 0.02),
            frameon=False, fontsize=8.4, ncol=min(6, len(unique)),
        )
        output = os.path.join(aggregate_dir, f'{_slug(approach)}.pdf')
        tmp = f'{output}.tmp.{os.getpid()}.pdf'
        try:
            fig.savefig(tmp, dpi=240)
        finally:
            plt.close(fig)
        os.replace(tmp, output)
        outputs.append(output)
    return outputs


def _plot_all_report(rows: list, flow_counts: list, bench: dict,
                     output: str) -> int:
    """Write one page per flow count with every approach overlaid."""
    approaches = sorted({str(row['approach']) for row in rows})
    start = float(bench['arrival_window_s'])
    end = float(bench['duration_s'])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = f'{output}.tmp.{os.getpid()}.pdf'
    pages = 0
    with PdfPages(tmp) as pdf:
        for flow_count in flow_counts:
            flow_rows = [
                row for row in rows
                if int(float(row['flow_count'])) == flow_count
            ]
            fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0))
            fig.suptitle(
                f'Responsive Fairness: {flow_count} Flows, All Approaches',
                fontsize=13, fontweight='bold')
            metric_axes = {
                'responsiveness': axes[0, 0],
                'srtt': axes[0, 1],
                'jain': axes[0, 2],
                'r_fair': axes[1, 0],
                'min_max': axes[1, 1],
            }
            legend_ax = axes[1, 2]
            legend_ax.axis('off')

            for index, approach in enumerate(approaches):
                group = [
                    row for row in flow_rows
                    if row['approach'] == approach
                ]
                if not group:
                    continue
                label = next(
                    (row.get('plot_label') for row in group
                     if row.get('plot_label')),
                    approach,
                )
                color = COLORS[index % len(COLORS)]
                samples = _load_samples(group)
                _plot_ecdf(
                    metric_axes['responsiveness'],
                    _row_values(group, 'mean_aggregate_goodput_mbps'),
                    label, color)
                _plot_ecdf(
                    metric_axes['srtt'],
                    _row_values(group, 'avg_srtt_ms'),
                    label, color)
                _plot_ecdf(
                    metric_axes['jain'],
                    _sample_values(samples, 'jain_fairness'),
                    label, color)
                _plot_ecdf(
                    metric_axes['r_fair'],
                    _sample_r_fair_values(samples),
                    label, color)
                _plot_ecdf(
                    metric_axes['min_max'],
                    _sample_values(samples, 'min_max_goodput_ratio'),
                    label, color)

            schedule_by_run = {}
            for row in flow_rows:
                schedule_by_run.setdefault(int(float(row['run'])), row)
            _plot_ecdf(
                metric_axes['responsiveness'],
                _row_values(
                    list(schedule_by_run.values()), 'mean_capacity_mbps'),
                'Scheduled BW', 'black', linewidth=1.7, zorder=10)

            base_rtt = np.asarray(
                [_schedule_mean(row, 'delay', start, end)
                 for row in schedule_by_run.values()],
                dtype=float)
            _plot_ecdf(
                metric_axes['srtt'], base_rtt[np.isfinite(base_rtt)],
                'Base RTT', 'black')

            specs = (
                ('responsiveness', 'Aggregate Goodput CDF',
                 'Average Aggregate Goodput (Mbps)', None, None),
                ('srtt', 'Average SRTT', 'SRTT (ms)', (0, None), None),
                ('jain', 'Jain Fairness', 'Fairness ratio', (0, 1.04), 1.0),
                ('r_fair', 'R_fair', 'R_fair (lower is fairer)',
                 (-0.02, 0.55), 0.0),
                ('min_max', 'Minimum / Maximum Goodput',
                 'Fairness ratio', (0, 1.04), 1.0),
            )
            for key, title, xlabel, xlim, optimal in specs:
                ax = metric_axes[key]
                if optimal is not None:
                    ax.axvline(
                        optimal, color='black', linestyle='--',
                        linewidth=1.0, label='Optimal')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('CDF (%)')
                ax.set_ylim(0, 100)
                if xlim is not None:
                    ax.set_xlim(*xlim)
                ax.grid(True, alpha=0.25)

            goodput = _row_values(
                flow_rows, 'mean_aggregate_goodput_mbps')
            scheduled_bw = _row_values(
                list(schedule_by_run.values()), 'mean_capacity_mbps')
            responsiveness_scale = np.concatenate((goodput, scheduled_bw))
            if responsiveness_scale.size:
                p01, p99 = np.percentile(responsiveness_scale, [1, 99])
                padding = max(1.0, float(p99 - p01) * 0.08)
                metric_axes['responsiveness'].set_xlim(
                    max(0.0, float(p01) - padding),
                    float(p99) + padding)

            unique = {}
            for ax in metric_axes.values():
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    unique.setdefault(label, handle)
            legend_ax.legend(
                list(unique.values()), list(unique.keys()),
                loc='center', frameon=False, fontsize=9,
                title='Approaches and references', title_fontsize=10)
            fig.subplots_adjust(
                left=0.06, right=0.985, top=0.91,
                bottom=0.08, hspace=0.3, wspace=0.24)
            pdf.savefig(fig, dpi=240, bbox_inches='tight')
            plt.close(fig)
            pages += 1
    os.replace(tmp, output)
    return pages


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


def _plot_srtt_by_flow_count(rows: list, flow_counts: list,
                             output: str) -> None:
    approaches = sorted({str(row['approach']) for row in rows})
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fig.suptitle(
        'Average SRTT by Number of Flows',
        fontsize=12, fontweight='bold')
    plotted = _scatter_with_medians(
        ax, rows, approaches, flow_counts,
        lambda group: _row_values(group, 'avg_srtt_ms'), 'avg_srtt_ms')
    ax.set_ylabel('Average SRTT (ms)')
    ax.set_ylim(bottom=0)
    if plotted:
        ax.legend(frameon=False, fontsize=9, loc='best')
    else:
        ax.text(
            0.5, 0.5, 'No measured SRTT samples',
            transform=ax.transAxes, ha='center', va='center',
        )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = f'{output}.tmp.{os.getpid()}.pdf'
    try:
        fig.savefig(tmp, dpi=240, bbox_inches='tight')
    finally:
        plt.close(fig)
    os.replace(tmp, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot responsive-fairness data from completed runs.')
    parser.add_argument('--config', default=_DEFAULT_CONFIG)
    parser.add_argument(
        '--plot-config', default=_DEFAULT_PLOT_APPROACHES,
        help='YAML listing the approaches (data_folder + plot_label) to '
             'include in the aggregate plots; defaults to plot_approaches.yaml')
    parser.add_argument(
        '--data-dir', default=None,
        help='directory containing learner metrics; defaults to output_root')
    parser.add_argument(
        '--output-dir', default=None,
        help='aggregate output directory; defaults to <benchmark>/aggregate')
    parser.add_argument(
        '--approach', action='append', default=None,
        help='learner data_folder/name to include; repeatable')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = _load_benchmark_config(config_path)
    bench = _bench_cfg(cfg)
    data_dir = _resolve_data_dir(config_path, cfg, args.data_dir)
    aggregate_dir = os.path.abspath(
        args.output_dir or os.path.join(_HERE, 'aggregate'))

    # The curated plot config drives both which data folders are drawn and the
    # labels they get, independent of the benchmark approaches config. A data
    # folder present under data/ but absent here is left out of the plots.
    plot_approaches = _load_yaml(
        os.path.abspath(args.plot_config)).get('approaches') or []
    cfg['approaches'] = plot_approaches
    if args.approach:
        selected = {_slug(value) for value in args.approach}
    else:
        selected = {_approach_label(approach) for approach in plot_approaches}

    rows = _completed_rows(
        _collect_rows(data_dir, cfg), bench, selected or None)
    if not rows:
        raise SystemExit(
            f'[responsive_fairness_plot] no completed rows under {data_dir}')

    run_csv = os.path.join(aggregate_dir, 'fairness_runs.csv')
    summary_csv = os.path.join(
        aggregate_dir, 'fairness_by_flow_count.csv')
    output_pdf = os.path.join(
        aggregate_dir, 'fairness_by_flow_count.pdf')
    cdf_pdf = os.path.join(aggregate_dir, 'fairness_cdfs.pdf')
    srtt_pdf = os.path.join(aggregate_dir, 'srtt_by_flow_count.pdf')
    all_pdf = os.path.join(aggregate_dir, 'all.pdf')
    srtt_cache = _load_srtt_cache(run_csv)
    if not srtt_cache:
        srtt_cache = _load_srtt_cache(
            os.path.join(data_dir, 'fairness_runs.csv'))
    for row in rows:
        cached = srtt_cache.get(_run_key(row), float('nan'))
        row['avg_srtt_ms'] = (
            cached if np.isfinite(cached)
            else _run_avg_srtt_ms(row, bench))
    _write_csv(run_csv, RUN_FIELDS, _run_rows(rows))
    _write_csv(
        summary_csv, SUMMARY_FIELDS,
        _summary_rows(rows, bench['flow_counts']))
    _plot_fairness(rows, bench['flow_counts'], output_pdf)
    cdf_pages = _plot_learner_cdfs(
        rows, bench['flow_counts'], cdf_pdf)
    aggregate_reports = _plot_aggregate_reports(
        rows, bench['flow_counts'], bench, aggregate_dir)
    _plot_srtt_by_flow_count(rows, bench['flow_counts'], srtt_pdf)
    all_pages = _plot_all_report(
        rows, bench['flow_counts'], bench, all_pdf)
    _restore_sudo_user_ownership(aggregate_dir)

    print(
        f'[responsive_fairness_plot] learners='
        f'{len({row["approach"] for row in rows})} rows={len(rows)}',
        flush=True)
    print(f'[responsive_fairness_plot] runs={run_csv}', flush=True)
    print(f'[responsive_fairness_plot] aggregate={summary_csv}', flush=True)
    print(f'[responsive_fairness_plot] plot={output_pdf}', flush=True)
    print(
        f'[responsive_fairness_plot] fairness_cdfs={cdf_pdf} '
        f'pages={cdf_pages}',
        flush=True)
    for report in aggregate_reports:
        print(f'[responsive_fairness_plot] aggregate={report}', flush=True)
    print(f'[responsive_fairness_plot] srtt={srtt_pdf}', flush=True)
    print(
        f'[responsive_fairness_plot] all={all_pdf} pages={all_pages}',
        flush=True)


if __name__ == '__main__':
    main()
