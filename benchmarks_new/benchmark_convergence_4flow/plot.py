#!/usr/bin/env python3
"""Plots for the staggered four-flow convergence benchmark."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.transforms as mtransforms
import numpy as np

from benchmarks_new.common import load_scenario
from benchmarks_new.plot_data import (
    aligned_throughput,
    episode_traces,
    load_benchmark,
    metadata_index,
    number,
    output_root,
    return_rows,
    run_label,
)


_APPROACH_COLORS = [
    '#4878cf', '#e1812c', '#59a14f', '#e15759', '#b279a2', '#9c755f',
    '#76b7b2', '#ff9da7', '#5e7ce2', '#d37295', '#7f7f7f', '#bcbd22',
]
_CONDITION_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'p']


def join_times(config, manifest, overlay):
    scenario_value = (manifest.get('matrix') or {}).get('scenarios', [None])[0]
    if not scenario_value:
        return []
    path = Path(scenario_value)
    path = path if path.is_absolute() else config.parent / path
    resolved = load_scenario(path, overlay)
    sweep = resolved.get('sweep') or {}
    flow_schedule = sweep.get('flow_schedule') or {}
    arrival = flow_schedule.get('arrival') or {}
    if arrival.get('start_delays') is not None:
        return [float(value) for value in arrival['start_delays']]
    if arrival.get('evenly_spaced_over_s') is not None:
        flows = (sweep.get('flows') or [1])[0]
        window = float(arrival['evenly_spaced_over_s'])
        return [window * i / max(int(flows) - 1, 1) for i in range(int(flows))]
    return []


def _episode_efficiency_point(traces, bw_mbps, base_rtt_ms):
    """Return normalized delay/goodput over the all-flows-active period.

    `benchmarks_new` records worker observations rather than iperf client JSON,
    so avg_urtt_ms is the closest common RTT measurement for both backends.
    Zeros are drain-on-read/no-sample values and deliberately excluded.
    """
    aligned = aligned_throughput(traces)
    if aligned is None or bw_mbps <= 0 or base_rtt_ms <= 0:
        return None
    time_s, throughput = aligned
    active = np.isfinite(throughput)
    active_count = active.sum(axis=0)
    mask = active_count >= throughput.shape[0]
    if not mask.any():
        mask = active_count > 0
    if not mask.any():
        return None

    total_goodput = np.nansum(throughput, axis=0)[mask]
    total_goodput = total_goodput[np.isfinite(total_goodput)]
    if not total_goodput.size:
        return None

    start, end = float(time_s[mask].min()), float(time_s[mask].max())
    rtt_samples = []
    for _, data in traces:
        sample_time = np.asarray(data.get('t_s', []), dtype=float)
        avg_urtt = np.asarray(data.get('avg_urtt_ms', []), dtype=float)
        srtt = np.asarray(data.get('srtt_ms', []), dtype=float)
        rtt = np.where(np.isfinite(avg_urtt) & (avg_urtt > 0), avg_urtt, srtt)
        keep = ((sample_time >= start - 0.5) & (sample_time <= end + 0.5)
                & np.isfinite(rtt) & (rtt > 0))
        rtt_samples.extend(rtt[keep])
    if not rtt_samples:
        return None

    return (
        float(np.mean(rtt_samples)) / base_rtt_ms,
        float(np.mean(total_goodput)) / bw_mbps,
    )


def efficiency_records(root):
    """Group per-episode efficiency points by approach and link condition."""
    metadata = metadata_index(return_rows(root))
    grouped = {}
    for key, traces in episode_traces(root).items():
        meta = metadata.get(key, {})
        bw_mbps = number(meta, 'bw')
        base_rtt_ms = number(meta, 'delay')
        if not (np.isfinite(bw_mbps) and bw_mbps > 0
                and np.isfinite(base_rtt_ms) and base_rtt_ms > 0):
            continue
        point = _episode_efficiency_point(traces, bw_mbps, base_rtt_ms)
        if point is not None:
            grouped.setdefault((key[0], bw_mbps, base_rtt_ms), []).append(point)
    return grouped


def _confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """Draw the same Pearson covariance ellipse as the legacy benchmark."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2 or x.size != y.size:
        return None
    covariance = np.cov(x, y)
    if (not np.all(np.isfinite(covariance)) or covariance[0, 0] <= 0
            or covariance[1, 1] <= 0):
        return None
    pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    pearson = float(np.clip(pearson, -0.999999, 0.999999))
    ellipse = Ellipse(
        (0, 0), width=2 * np.sqrt(1 + pearson),
        height=2 * np.sqrt(1 - pearson), facecolor=facecolor, **kwargs)
    transform = (mtransforms.Affine2D().rotate_deg(45).scale(
        np.sqrt(covariance[0, 0]) * n_std,
        np.sqrt(covariance[1, 1]) * n_std,
    ).translate(float(np.mean(x)), float(np.mean(y))))
    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)


def plot_efficiency(root, output):
    """Write Fig.9-style delay/throughput efficiency scatter to ``output``."""
    grouped = efficiency_records(root)
    if not grouped:
        print(f'[convergence_plot] no efficiency observations beneath {root}')
        return False

    runs = sorted({key[0] for key in grouped})
    conditions = sorted({(key[1], key[2]) for key in grouped})
    colors = {run: _APPROACH_COLORS[index % len(_APPROACH_COLORS)]
              for index, run in enumerate(runs)}
    markers = {condition: _CONDITION_MARKERS[index % len(_CONDITION_MARKERS)]
               for index, condition in enumerate(conditions)}
    all_goodput = np.concatenate([
        np.asarray(points, dtype=float)[:, 1] for points in grouped.values()
    ])

    fig, (ax, legend_ax) = plt.subplots(
        2, 1, figsize=(6.0, 4.4),
        gridspec_kw={'height_ratios': [1.0, 0.26]}, constrained_layout=True)
    fig.suptitle(
        'Efficiency: competing-flow interval, mean +/- 1 sigma\n'
        'colour = approach, marker = BW/RTT condition',
        fontsize=9, fontweight='bold')
    for (run, bw_mbps, base_rtt_ms), points in sorted(grouped.items()):
        values = np.asarray(points, dtype=float)
        color = colors[run]
        _confidence_ellipse(values[:, 0], values[:, 1], ax, n_std=1.0,
                            facecolor=color, edgecolor='none', alpha=0.25)
        ax.scatter(
            [float(np.mean(values[:, 0]))], [float(np.mean(values[:, 1]))],
            s=46, edgecolors=color, facecolors='none',
            marker=markers[(bw_mbps, base_rtt_ms)], linewidths=1.3, zorder=3)

    ax.set(xlabel='Norm. Delay', ylabel='Norm. Throughput')
    ax.set_ylim(min(0.5, float(np.nanmin(all_goodput)) - 0.03),
                max(1.0, float(np.nanmax(all_goodput)) + 0.03))
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    legend_ax.axis('off')
    approach_handles = [
        Line2D([], [], color=colors[run], marker='o', linestyle='None',
               markersize=6, markeredgewidth=1.3, markerfacecolor='none',
               label=run_label(run))
        for run in runs
    ]
    condition_handles = [
        Line2D([], [], color='black', marker=markers[condition],
               linestyle='None', markersize=6, markeredgewidth=1.2,
               markerfacecolor='none',
               label=f'BW={condition[0]:g} Mbps, RTT={condition[1]:g} ms')
        for condition in conditions
    ]
    first_legend = legend_ax.legend(
        handles=approach_handles, loc='center left', bbox_to_anchor=(0.0, 0.5),
        ncol=max(1, min(2, len(approach_handles))), frameon=False, fontsize=7,
        title='Approach', title_fontsize=7)
    legend_ax.add_artist(first_legend)
    legend_ax.legend(
        handles=condition_handles, loc='center right', bbox_to_anchor=(1.0, 0.5),
        ncol=max(1, min(3, len(condition_handles))), frameon=False, fontsize=7,
        title='Condition', title_fontsize=7)
    fig.savefig(output, dpi=600)
    plt.close(fig)
    print(f'[convergence_plot] wrote {output}')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--output')
    parser.add_argument('--efficiency-output')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    config = Path(args.config).resolve()
    manifest, overlay = load_benchmark(config, debug=args.debug)
    joins = join_times(config, manifest, overlay)
    root = output_root(config, manifest)
    output = Path(args.output) if args.output else root / 'benchmark_summary.pdf'
    efficiency_output = (Path(args.efficiency_output) if args.efficiency_output
                         else output.with_name('efficiency.pdf'))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata_index(return_rows(root))
    records = []
    for key, traces in episode_traces(root).items():
        aligned = aligned_throughput(traces)
        meta = metadata.get(key, {})
        if (aligned is None or not np.isfinite(number(meta, 'bw'))
                or number(meta, 'bw') <= 0):
            continue
        time_s, throughput = aligned
        active = np.isfinite(throughput)
        total = np.nansum(throughput, axis=0)
        squares = np.nansum(throughput ** 2, axis=0)
        count = active.sum(axis=0)
        jain = np.ones_like(total)
        np.divide(total ** 2, count * squares, out=jain,
                  where=(count > 1) & (squares > 0))
        records.append((key[0], time_s, total / number(meta, 'bw'), jain))
    if not records:
        print(f'[convergence_plot] no per-flow traces beneath {root}')
        return 1

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    for run in sorted({record[0] for record in records}):
        subset = [record for record in records if record[0] == run]
        end = min(record[1][-1] for record in subset)
        grid = np.arange(0, end + 1e-9)
        utilization = np.vstack([
            np.interp(grid, record[1], record[2]) for record in subset])
        fairness = np.vstack([
            np.interp(grid, record[1], record[3]) for record in subset])
        label = run_label(run)
        axes[0].plot(grid, np.nanmean(utilization, axis=0), label=label)
        axes[1].plot(grid, np.nanmean(fairness, axis=0), label=label)
    axes[0].set(ylabel='Total goodput / capacity', title='Capacity utilization')
    axes[1].set(xlabel='Episode time (s)', ylabel='Jain index',
                title='Flow convergence fairness', ylim=(0, 1.05))
    for axis in axes:
        for join in joins[1:]:
            axis.axvline(join, color='black', linestyle=':', linewidth=.8, alpha=.5)
        axis.grid(alpha=.25)
        axis.legend(frameon=False)
    fig.savefig(output)
    plt.close(fig)
    print(f'[convergence_plot] wrote {output}')
    plot_efficiency(root, efficiency_output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
