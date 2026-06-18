#!/usr/bin/env python3
"""Dynamic responsiveness and staggered-flow fairness experiment runner.

Runs separate 2-, 4-, 5-, and 7-flow experiments. Flows start at equal
intervals over the first 45 seconds, then all remain active for a 300-second
scoring window. Bandwidth and RTT change every 15 seconds for the full
345-second episode.

Receiver iperf3 goodput is used for every approach. Scored one-second samples:

* aggregate responsiveness = aggregate goodput / scheduled BW(t)
* Jain fairness = Jain(per-flow goodput normalized by BW(t) / N)
* max-min ratio = minimum per-flow goodput / maximum per-flow goodput

Plotting is intentionally handled by ``plot.py`` so experiment execution only
collects raw run data, per-second samples, and summary metrics.
"""

import argparse
import csv
import glob
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
import numpy as np
import yaml


_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
_DEFAULT_CONFIG = os.path.join(_HERE, 'config.yaml')
sys.path.insert(0, _ROOT)

from olympus.environments.mininet.env import MininetEnv
from olympus.orchestrator import (
    _as_bool,
    _final_runtime_cleanup,
    _resolve_repo_path,
    run_episode_auto,
)
from olympus.common.bench_utils import (
    _append_csv,
    _approach_label_map,
    _approach_plot_label,
    _approach_selection_keys,
    _binned_series,
    _collect_approach_rows_with_labels,
    _configured_approaches,
    _copy_if_exists,
    _deterministic_env,
    _ensure_kernel_cc_available,
    _finite_float,
    _kernel_flows,
    _load_approach_runtime_config,
    _load_benchmark_config,
    _orca_receiver_to_csvs,
    _orca_settings,
    _parse_iperf_json,
    _prepare_runtime_cfg as _prepare_runtime_cfg_base,
    _resolve_from,
    _restore_sudo_user_ownership,
    _run_orca_on_env,
    _safe_overwrite_dir,
    _safe_unlink,
    _slug,
    _start_astraea_listener,
    _stop_astraea_listener,
    _validate_approach,
    _validate_unique_data_folders,
    _write_flow_plan,
    _write_rows,
    _write_schedule,
)


GOODPUT_SOURCE = 'iperf3_receiver_scored_1s'

METRIC_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'kind', 'algorithm', 'kernel_cc',
    'state', 'reward', 'training_environment', 'benchmark_environment',
    'flow_count', 'run', 'seed', 'duration_s', 'arrival_window_s',
    'score_duration_s', 'change_interval_s', 'flow_starts_s',
    'mean_capacity_mbps', 'mean_aggregate_goodput_mbps',
    'mean_aggregate_responsiveness', 'p05_aggregate_responsiveness',
    'mean_jain_fairness', 'p05_jain_fairness',
    'mean_min_max_goodput_ratio', 'p05_min_max_goodput_ratio',
    'score_samples', 'goodput_source', 'schedule_csv', 'flow_schedule_csv',
    'sample_csv', 'state_logs', 'individual_plot', 'run_dir', 'checkpoint',
    'error',
]

SAMPLE_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'flow_count', 'run', 'seed',
    'time_s', 'capacity_mbps', 'aggregate_goodput_mbps',
    'aggregate_responsiveness', 'jain_fairness',
    'min_max_goodput_ratio',
] + [f'flow{flow}_goodput_mbps' for flow in range(1, 8)]

COLORS = [
    '#4878cf', '#e1812c', '#59a14f', '#e15759', '#b279a2', '#9c755f',
    '#76b7b2', '#ff9da7', '#5e7ce2', '#d37295',
]

_AGENT_LOG_RE = re.compile(r'_a(\d+)\.csv$')
_PLOT_BACKEND = None


def _plot_backend():
    """Load legacy plotting dependencies only when a plot helper is called."""
    global _PLOT_BACKEND
    if _PLOT_BACKEND is None:
        os.environ.setdefault(
            'MPLCONFIGDIR',
            os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
        os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pyplot
        from matplotlib.backends.backend_pdf import PdfPages
        _PLOT_BACKEND = pyplot, PdfPages
    return _PLOT_BACKEND


def _bench_cfg(cfg: dict) -> dict:
    bench = dict(cfg.get('benchmark') or {})
    bench.setdefault('name', 'responsive_fairness_join45_score300')
    bench.setdefault('runs', 50)
    bench.setdefault('n_parallel', 10)
    bench.setdefault('flow_counts', [2, 4, 5, 7])
    bench.setdefault('arrival_window_s', 45)
    bench.setdefault('score_duration_s', 300)
    bench.setdefault(
        'duration_s',
        float(bench['arrival_window_s']) + float(bench['score_duration_s']))
    bench.setdefault('change_interval_s', 15)
    bench.setdefault('bw_min_mbps', 10)
    bench.setdefault('bw_max_mbps', 100)
    bench.setdefault('rtt_min_ms', 10)
    bench.setdefault('rtt_max_ms', 100)
    bench.setdefault('seed_offset', 0)
    bench.setdefault('bdp_mult', 4)
    bench.setdefault('measure_interval_s', 1.0)
    bench.setdefault('kernel_cport_base', 56000)
    bench.setdefault('deterministic_inference', True)
    bench['flow_counts'] = sorted({
        int(value) for value in bench['flow_counts'] if int(value) >= 2
    })
    if not bench['flow_counts'] or max(bench['flow_counts']) > 7:
        raise SystemExit(
            '[responsive_fairness] flow_counts must be between 2 and 7')
    expected = (
        float(bench['arrival_window_s']) + float(bench['score_duration_s']))
    if abs(float(bench['duration_s']) - expected) > 1e-9:
        raise SystemExit(
            '[responsive_fairness] duration_s must equal arrival_window_s + '
            f'score_duration_s ({expected:g})')
    if int(bench['change_interval_s']) <= 0:
        raise SystemExit(
            '[responsive_fairness] change_interval_s must be positive')
    return bench


def _flow_plan(bench: dict, flow_count: int) -> list:
    arrival = float(bench['arrival_window_s'])
    duration = float(bench['duration_s'])
    spacing = arrival / float(flow_count - 1)
    return [
        {
            'flow': idx + 1,
            'start': idx * spacing,
            'duration': duration - idx * spacing,
            # End-of-bin timestamps need the last physical second included.
            'end': duration + 1e-6,
        }
        for idx in range(flow_count)
    ]


def _sample_schedule(run_number: int, bench: dict) -> dict:
    seed = int(bench.get('seed_offset', 0)) + int(run_number)
    rng = random.Random(seed)

    def draw():
        return {
            'bw': int(rng.randint(
                int(bench['bw_min_mbps']), int(bench['bw_max_mbps']))),
            'delay': int(rng.randint(
                int(bench['rtt_min_ms']), int(bench['rtt_max_ms']))),
        }

    initial = draw()
    changes = []
    for t_s in range(
            int(bench['change_interval_s']),
            int(bench['duration_s']),
            int(bench['change_interval_s'])):
        changes.append({'t': t_s, **draw()})
    return {
        'seed': seed,
        'initial': initial,
        'changes': changes,
        'rows': [{'t': 0, **initial}] + changes,
    }


def _run_link_schedule(env, changes: list, episode_start: float, stop) -> None:
    for entry in changes:
        target = episode_start + float(entry['t'])
        while not stop.is_set():
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            stop.wait(timeout=min(remaining, 0.05))
        if stop.is_set():
            return
        env.set_link(bw=entry.get('bw'), delay=entry.get('delay'))


def _iperf_client_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_{int(cport)}_{int(flow)}.json'


def _iperf_receiver_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_server_{int(cport)}_{int(flow)}.json'


def _clear_iperf_tmp_outputs(cport: int, flow_count: int) -> None:
    for flow in range(1, flow_count + 1):
        _safe_unlink(_iperf_client_tmp(cport, flow))
        _safe_unlink(_iperf_receiver_tmp(cport, flow))


def _copy_receiver_outputs(cport: int, run_dir: str, flow_plan: list,
                           bench: dict) -> str:
    errors = []
    for item in flow_plan:
        flow = int(item['flow'])
        receiver_dst = os.path.join(
            run_dir, f'iperf_receiver_flow{flow}.json')
        _copy_if_exists(
            _iperf_client_tmp(cport, flow),
            os.path.join(run_dir, f'iperf_client_flow{flow}.json'))
        if not _copy_if_exists(
                _iperf_receiver_tmp(cport, flow), receiver_dst):
            errors.append(f'missing receiver iperf JSON for flow {flow}')
            continue
        parsed = _parse_iperf_json(
            receiver_dst,
            os.path.join(run_dir, 'csvs', f'x{flow}.csv'),
            measure_interval_s=float(bench['measure_interval_s']))
        if not parsed.get('samples'):
            errors.append(f'no receiver goodput samples for flow {flow}')
    return '; '.join(errors)


def _flows_have_goodput(flows: list, flow_count: int) -> bool:
    if len(flows) < flow_count:
        return False
    return all(np.isfinite(np.asarray(flow.get('thr', []), dtype=float)).any()
               for flow in flows[:flow_count])


def _capacity_at(times: np.ndarray, schedule_rows: list,
                 field: str = 'bw') -> np.ndarray:
    change_times = np.asarray(
        [float(row['t']) for row in schedule_rows], dtype=float)
    values = np.asarray(
        [float(row[field]) for row in schedule_rows], dtype=float)
    indices = np.searchsorted(change_times, times, side='right') - 1
    return values[np.clip(indices, 0, len(values) - 1)]


def _full_series(flows: list, flow_plan: list, schedule: dict,
                 bench: dict) -> dict:
    binned = _binned_series(flows, flow_plan, bench)
    times = np.asarray(binned['time'], dtype=float)
    per_flow = np.asarray(binned['per_flow'], dtype=float)
    capacity = _capacity_at(times, schedule['rows'], 'bw')
    aggregate = np.sum(per_flow, axis=0)
    responsiveness = np.divide(
        aggregate, capacity, out=np.full_like(aggregate, np.nan),
        where=capacity > 0)
    fair_share = capacity / float(len(flow_plan))
    normalized = np.divide(
        per_flow, fair_share[np.newaxis, :], out=np.zeros_like(per_flow),
        where=fair_share[np.newaxis, :] > 0)
    sums = np.sum(normalized, axis=0)
    sumsq = np.sum(normalized * normalized, axis=0)
    jain = np.divide(
        sums * sums, float(len(flow_plan)) * sumsq,
        out=np.full_like(sums, np.nan), where=sumsq > 0)
    minimum = np.min(per_flow, axis=0)
    maximum = np.max(per_flow, axis=0)
    min_max = np.divide(
        minimum, maximum, out=np.full_like(maximum, np.nan),
        where=maximum > 0)
    return {
        'time': times,
        'per_flow': per_flow,
        'capacity': capacity,
        'aggregate': aggregate,
        'responsiveness': responsiveness,
        'jain': jain,
        'min_max': min_max,
    }


def _score_series(flows: list, flow_plan: list, schedule: dict,
                  bench: dict) -> dict:
    full = _full_series(flows, flow_plan, schedule, bench)
    mask = (
        (full['time'] > float(bench['arrival_window_s']))
        & (full['time'] <= float(bench['duration_s']))
    )
    return {
        key: value[:, mask] if key == 'per_flow' else value[mask]
        for key, value in full.items()
    }


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _finite_percentile(values, percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else math.nan


def _write_sample_csv(path: str, approach: dict, flow_count: int,
                      run_number: int, seed: int, scored: dict) -> str:
    rows = []
    for index, time_s in enumerate(scored['time']):
        row = {
            'approach': approach['_label'],
            'data_folder': approach['_label'],
            'plot_label': _approach_plot_label(approach),
            'flow_count': flow_count,
            'run': run_number,
            'seed': seed,
            'time_s': time_s,
            'capacity_mbps': scored['capacity'][index],
            'aggregate_goodput_mbps': scored['aggregate'][index],
            'aggregate_responsiveness': scored['responsiveness'][index],
            'jain_fairness': scored['jain'][index],
            'min_max_goodput_ratio': scored['min_max'][index],
        }
        for flow_index in range(7):
            row[f'flow{flow_index + 1}_goodput_mbps'] = (
                scored['per_flow'][flow_index, index]
                if flow_index < scored['per_flow'].shape[0] else '')
        rows.append(row)
    _write_rows(path, SAMPLE_FIELDS, rows)
    return path


def _metrics(scored: dict) -> dict:
    return {
        'mean_capacity_mbps': _finite_mean(scored['capacity']),
        'mean_aggregate_goodput_mbps': _finite_mean(scored['aggregate']),
        'mean_aggregate_responsiveness': _finite_mean(
            scored['responsiveness']),
        'p05_aggregate_responsiveness': _finite_percentile(
            scored['responsiveness'], 5),
        'mean_jain_fairness': _finite_mean(scored['jain']),
        'p05_jain_fairness': _finite_percentile(scored['jain'], 5),
        'mean_min_max_goodput_ratio': _finite_mean(scored['min_max']),
        'p05_min_max_goodput_ratio': _finite_percentile(
            scored['min_max'], 5),
        'score_samples': int(len(scored['time'])),
        'goodput_source': GOODPUT_SOURCE,
    }


def _metric_row(approach: dict, bench: dict, flow_count: int,
                run_number: int, schedule: dict, flow_plan: list,
                run_dir: str, schedule_csv: str, flow_schedule_csv: str,
                checkpoint: str, state_logs: list, sample_csv: str,
                individual_plot: str, metrics: dict,
                error: str = '') -> dict:
    row = {
        'approach': approach['_label'],
        'data_folder': approach['_label'],
        'plot_label': _approach_plot_label(approach),
        'kind': str(approach.get('kind', 'model')),
        'algorithm': approach.get('algorithm', ''),
        'kernel_cc': approach.get('kernel_cc', ''),
        'state': approach.get('state', ''),
        'reward': approach.get('reward', ''),
        'training_environment': approach.get('environment', ''),
        'benchmark_environment': bench['name'],
        'flow_count': flow_count,
        'run': run_number,
        'seed': schedule['seed'],
        'duration_s': bench['duration_s'],
        'arrival_window_s': bench['arrival_window_s'],
        'score_duration_s': bench['score_duration_s'],
        'change_interval_s': bench['change_interval_s'],
        'flow_starts_s': ';'.join(
            f'{item["start"]:g}' for item in flow_plan),
        'schedule_csv': schedule_csv,
        'flow_schedule_csv': flow_schedule_csv,
        'sample_csv': sample_csv,
        'state_logs': ';'.join(state_logs),
        'individual_plot': individual_plot,
        'run_dir': run_dir,
        'checkpoint': checkpoint,
        'error': error,
    }
    row.update(metrics or {})
    for field in METRIC_FIELDS:
        row.setdefault(field, '')
    return row


def _runtime_cfg(base_cfg: dict, approach: dict, checkpoint: str,
                 run_dir: str) -> dict:
    scratch = os.path.join(
        '/tmp', f'olympus_responsive_fairness_{os.getpid()}',
        approach['_label'])
    cfg = _prepare_runtime_cfg_base(
        base_cfg, approach, checkpoint, scratch, run_dir,
        plot_episodes=True)
    cfg['outputs']['run_dir'] = scratch
    cfg['outputs']['checkpoints_dir'] = os.path.join(
        scratch, 'checkpoints')
    cfg['outputs']['telemetry_dir'] = os.path.join(scratch, 'telemetry')
    cfg['training']['log_path'] = os.path.join(
        scratch, 'telemetry', 'benchmark_metrics.csv')
    return cfg


def _state_log_rtt(path: str) -> tuple:
    times = []
    values = []
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                time_s = _finite_float(row.get('t_s'))
                rtt_ms = _finite_float(row.get('srtt_ms'))
                if not math.isfinite(rtt_ms) or rtt_ms <= 0:
                    rtt_ms = _finite_float(row.get('avg_urtt_ms'))
                if (math.isfinite(time_s) and math.isfinite(rtt_ms)
                        and rtt_ms > 0):
                    times.append(time_s)
                    values.append(rtt_ms)
    except OSError:
        pass
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _measured_rtt_flows(run_dir: str, flow_plan: list,
                        state_logs: list = None) -> list:
    """Return one measured RTT trace per flow on the episode time axis."""
    traces = {}
    candidates = list(state_logs or [])
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(
            run_dir, '*_state_ep*_a*.csv')))
    for index, path in enumerate(candidates):
        match = _AGENT_LOG_RE.search(os.path.basename(path))
        agent_id = int(match.group(1)) if match else index
        times, values = _state_log_rtt(path)
        if len(times):
            traces[agent_id + 1] = {
                'flow': agent_id + 1,
                'time': times,
                'rtt': values,
                'source': 'state srtt',
            }

    for item in flow_plan:
        flow = int(item['flow'])
        if flow in traces:
            continue
        parsed = _parse_iperf_json(
            os.path.join(run_dir, f'iperf_client_flow{flow}.json'))
        samples = parsed.get('rtt_samples_ms', []) or []
        if not samples:
            continue
        start = float(item['start'])
        traces[flow] = {
            'flow': flow,
            'time': np.asarray(
                [float(time_s) + start for time_s, _ in samples],
                dtype=float),
            'rtt': np.asarray(
                [float(rtt_ms) for _, rtt_ms in samples], dtype=float),
            'source': 'iperf TCP RTT',
        }
    return [traces[flow] for flow in sorted(traces)]


def _plot_run(full: dict, schedule: dict, flow_plan: list, label: str,
              output: str, rtt_flows: list = None) -> str:
    """Plot the complete uncropped experiment and per-flow measured RTT."""
    plt, _ = _plot_backend()
    times = full['time']
    duration = float(times[-1])
    score_start = max(float(item['start']) for item in flow_plan)

    schedule_times = np.asarray(
        [float(row['t']) for row in schedule['rows']], dtype=float)
    scheduled_rtt = np.asarray(
        [float(row['delay']) for row in schedule['rows']], dtype=float)
    schedule_times = np.append(schedule_times, duration)
    scheduled_rtt = np.append(scheduled_rtt, scheduled_rtt[-1])

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 9.2), sharex=True)
    fig.suptitle(label, fontsize=11, fontweight='bold')

    axes[0].plot(
        times, full['capacity'], color='black', linewidth=1.2,
        label='scheduled BW')
    axes[0].plot(
        times, full['aggregate'], color='#4878cf', linewidth=1.2,
        label='aggregate goodput')
    axes[0].set_ylabel('Mbps')
    delay_ax = axes[0].twinx()
    delay_ax.step(
        schedule_times, scheduled_rtt, where='post', color='#7f7f7f',
        linestyle='--', linewidth=1.0, label='scheduled RTT')
    delay_ax.set_ylabel('RTT (ms)')
    delay_ax.set_xlim(0, duration)
    lines, labels = axes[0].get_legend_handles_labels()
    delay_lines, delay_labels = delay_ax.get_legend_handles_labels()
    axes[0].legend(
        lines + delay_lines, labels + delay_labels,
        frameon=False, ncol=3)

    flow_colors = []
    for index in range(full['per_flow'].shape[0]):
        line, = axes[1].plot(
            times, full['per_flow'][index], linewidth=1.0,
            label=f'flow {index + 1}')
        flow_colors.append(line.get_color())
    axes[1].set_ylabel('Per-flow Mbps')
    axes[1].legend(frameon=False, ncol=len(flow_plan))

    plotted_rtt = False
    for trace in rtt_flows or []:
        flow = int(trace['flow'])
        color = flow_colors[(flow - 1) % len(flow_colors)]
        axes[2].plot(
            trace['time'], trace['rtt'], color=color, linewidth=0.9,
            label=f'flow {flow} measured RTT')
        plotted_rtt = True
    axes[2].step(
        schedule_times, scheduled_rtt, where='post', color='#7f7f7f',
        linestyle='--', linewidth=1.0, label='scheduled RTT')
    axes[2].set_ylabel('RTT (ms)')
    axes[2].legend(
        frameon=False, ncol=min(4, len(flow_plan) + 1), fontsize=8)
    if not plotted_rtt:
        axes[2].text(
            0.5, 0.5, 'No measured RTT samples',
            transform=axes[2].transAxes, ha='center', va='center')

    score_mask = times > score_start
    axes[3].plot(
        times[score_mask], full['jain'][score_mask],
        label='Jain fairness', linewidth=1.0)
    axes[3].plot(
        times[score_mask], full['min_max'][score_mask],
        label='min / max goodput', linewidth=1.0)
    axes[3].axhline(1.0, color='black', alpha=0.25, linewidth=0.8)
    axes[3].set_ylabel('Ratio')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(frameon=False, ncol=2)

    for ax in axes:
        ax.axvspan(
            0, score_start, color='#d9d9d9', alpha=0.22, zorder=0)
        for item, color in zip(flow_plan, flow_colors):
            ax.axvline(
                float(item['start']), color=color, linestyle=':',
                linewidth=0.9, alpha=0.75)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, duration)
    axes[0].text(
        score_start / 2.0, axes[0].get_ylim()[1] * 0.96,
        'staggered arrivals', ha='center', va='top', fontsize=8,
        color='#555555')
    axes[3].text(
        score_start + 2.0, axes[3].get_ylim()[0] + 0.04,
        'scoring starts', ha='left', va='bottom', fontsize=8,
        color='#555555')

    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp_output = f'{output}.tmp.{os.getpid()}.pdf'
    try:
        fig.savefig(tmp_output, dpi=180, bbox_inches='tight')
    finally:
        plt.close(fig)
    os.replace(tmp_output, output)
    return output


def _finish_trial(approach: dict, bench: dict, flow_count: int,
                  run_number: int, schedule: dict, flow_plan: list,
                  run_dir: str, schedule_csv: str, flow_schedule_csv: str,
                  checkpoint: str, state_logs: list,
                  parse_error: str = '') -> dict:
    flows = _kernel_flows(run_dir, flow_plan)
    error = parse_error
    sample_csv = os.path.join(
        run_dir, 'responsive_fairness_samples.csv')
    individual_plot = ''
    metrics = {}
    if not error and not _flows_have_goodput(flows, flow_count):
        error = 'missing receiver iperf goodput samples'
    if not error:
        scored = _score_series(flows, flow_plan, schedule, bench)
        if len(scored['time']) != int(bench['score_duration_s']):
            error = (
                f'expected {int(bench["score_duration_s"])} scored samples, '
                f'got {len(scored["time"])}')
        else:
            _write_sample_csv(
                sample_csv, approach, flow_count, run_number,
                schedule['seed'], scored)
            metrics = _metrics(scored)
    return _metric_row(
        approach, bench, flow_count, run_number, schedule, flow_plan,
        run_dir, schedule_csv, flow_schedule_csv, checkpoint, state_logs,
        sample_csv, individual_plot, metrics, error)


def _failed_row(approach: dict, bench: dict, flow_count: int,
                run_number: int, schedule: dict, flow_plan: list,
                run_dir: str, schedule_csv: str, flow_schedule_csv: str,
                checkpoint: str, exc: Exception) -> dict:
    return _metric_row(
        approach, bench, flow_count, run_number, schedule, flow_plan,
        run_dir, schedule_csv, flow_schedule_csv, checkpoint, [], '', '',
        {}, f'{exc}\n{traceback.format_exc()}')


def _run_kernel_trial(instance_id: int, approach: dict, bench: dict,
                      flow_count: int, run_number: int, schedule: dict,
                      run_dir: str, schedule_csv: str,
                      flow_schedule_csv: str) -> dict:
    flow_plan = _flow_plan(bench, flow_count)
    cport = int(bench['kernel_cport_base']) + instance_id * 100
    _clear_iperf_tmp_outputs(cport, flow_count)
    kind = str(approach.get('kind', 'kernel')).lower()
    cc = 'astraea' if kind == 'astraea' else str(approach['kernel_cc'])
    try:
        env = MininetEnv(
            n=flow_count,
            bw=float(schedule['initial']['bw']),
            delay=float(schedule['initial']['delay']),
            bdp_mult=float(bench['bdp_mult']),
            duration=int(bench['duration_s']),
            cport=cport,
            cc_algo=cc,
            instance_id=instance_id)
        stop = threading.Event()
        thread = None
        try:
            env.start()
            env.run_iperf(
                monitor_interval=float(bench['measure_interval_s']),
                start_delays=[item['start'] for item in flow_plan],
                flow_durations=[item['duration'] for item in flow_plan])
            episode_start = time.monotonic()
            thread = threading.Thread(
                target=_run_link_schedule,
                args=(env, schedule['changes'], episode_start, stop),
                daemon=True)
            thread.start()
            time.sleep(float(bench['duration_s']) + 3.0)
        finally:
            stop.set()
            if thread:
                thread.join(timeout=2)
            env.stop()
            time.sleep(0.5)
        parse_error = _copy_receiver_outputs(
            cport, run_dir, flow_plan, bench)
        return _finish_trial(
            approach, bench, flow_count, run_number, schedule, flow_plan,
            run_dir, schedule_csv, flow_schedule_csv, '', [],
            parse_error=parse_error)
    except Exception as exc:
        return _failed_row(
            approach, bench, flow_count, run_number, schedule, flow_plan,
            run_dir, schedule_csv, flow_schedule_csv, '', exc)


def _run_orca_trial(instance_id: int, approach: dict, bench: dict,
                    flow_count: int, run_number: int, schedule: dict,
                    run_dir: str, schedule_csv: str,
                    flow_schedule_csv: str) -> dict:
    flow_plan = _flow_plan(bench, flow_count)
    settings = _orca_settings(approach, bench)
    try:
        env = MininetEnv(
            n=flow_count,
            bw=float(schedule['initial']['bw']),
            delay=float(schedule['initial']['delay']),
            bdp_mult=float(bench['bdp_mult']),
            duration=int(bench['duration_s']),
            cport=int(bench['kernel_cport_base']) + instance_id * 100,
            cc_algo='cubic',
            instance_id=instance_id)
        stop = threading.Event()
        thread = None
        try:
            env.start()
            episode_start = time.monotonic()
            thread = threading.Thread(
                target=_run_link_schedule,
                args=(env, schedule['changes'], episode_start, stop),
                daemon=True)
            thread.start()
            _run_orca_on_env(
                env, settings, flow_plan, instance_id, run_dir,
                int(bench['duration_s']))
        finally:
            stop.set()
            if thread:
                thread.join(timeout=2)
            env.stop()
            time.sleep(0.5)
        parse_error = _orca_receiver_to_csvs(
            run_dir, flow_plan,
            measure_interval_s=float(bench['measure_interval_s']),
            settings=settings)
        return _finish_trial(
            approach, bench, flow_count, run_number, schedule, flow_plan,
            run_dir, schedule_csv, flow_schedule_csv, '', [],
            parse_error=parse_error)
    except Exception as exc:
        return _failed_row(
            approach, bench, flow_count, run_number, schedule, flow_plan,
            run_dir, schedule_csv, flow_schedule_csv, '', exc)


def _run_model_trial(instance_id: int, base_cfg: dict, approach: dict,
                     bench: dict, checkpoint: str, listener_bin: str,
                     python_bin: str, flow_count: int, run_number: int,
                     schedule: dict, run_dir: str, schedule_csv: str,
                     flow_schedule_csv: str) -> dict:
    flow_plan = _flow_plan(bench, flow_count)
    cfg = _runtime_cfg(base_cfg, approach, checkpoint, run_dir)
    cfg['listener_single_flow'] = True
    if _as_bool(bench.get('deterministic_inference'), default=True):
        _deterministic_env(cfg)
    with open(os.path.join(run_dir, 'config.resolved.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    cport = int(cfg.get('cport_base', 21000)) + instance_id * 100
    _clear_iperf_tmp_outputs(cport, flow_count)
    ecfg = {
        'bw': float(schedule['initial']['bw']),
        'delay': float(schedule['initial']['delay']),
        'bdp_mult': float(bench['bdp_mult']),
        'flows': flow_count,
        'duration': int(bench['duration_s']),
        'start_delays': [item['start'] for item in flow_plan],
        'flow_durations': [item['duration'] for item in flow_plan],
        'measure_interval_s': float(bench['measure_interval_s']),
        'per_flow_state_logs': True,
        'link_schedule': schedule['changes'],
        'environment': str(bench['name']),
        'training_environment': str(approach['environment']),
    }
    try:
        run_episode_auto(
            cfg, ecfg, run_number, listener_bin, python_bin,
            '', '', instance_id)
        parse_error = _copy_receiver_outputs(
            cport, run_dir, flow_plan, bench)
        state_logs = sorted(glob.glob(os.path.join(
            run_dir,
            f'{approach["algorithm"]}_state_ep{run_number:06d}_a*.csv')))
        return _finish_trial(
            approach, bench, flow_count, run_number, schedule, flow_plan,
            run_dir, schedule_csv, flow_schedule_csv, checkpoint, state_logs,
            parse_error=parse_error)
    except Exception as exc:
        return _failed_row(
            approach, bench, flow_count, run_number, schedule, flow_plan,
            run_dir, schedule_csv, flow_schedule_csv, checkpoint, exc)


def _slot(instance_id: int, work_q, result_q, base_cfg: dict,
          approach: dict, bench: dict, checkpoint: str,
          listener_bin: str, python_bin: str) -> None:
    while True:
        item = work_q.get()
        if item is None:
            return
        (flow_count, run_number, schedule, run_dir,
         schedule_csv, flow_schedule_csv) = item
        print(
            f'[responsive_fairness] slot={instance_id} '
            f'{approach["_label"]} flows={flow_count} run={run_number}',
            flush=True)
        kind = str(approach.get('kind', 'model')).lower()
        if kind in ('kernel', 'astraea'):
            row = _run_kernel_trial(
                instance_id, approach, bench, flow_count, run_number,
                schedule, run_dir, schedule_csv, flow_schedule_csv)
        elif kind == 'orca':
            row = _run_orca_trial(
                instance_id, approach, bench, flow_count, run_number,
                schedule, run_dir, schedule_csv, flow_schedule_csv)
        else:
            row = _run_model_trial(
                instance_id, base_cfg, approach, bench, checkpoint,
                listener_bin, python_bin, flow_count, run_number, schedule,
                run_dir, schedule_csv, flow_schedule_csv)
        result_q.put(row)


def _trial_key(row: dict):
    try:
        return (
            int(float(row['flow_count'])),
            int(float(row['run'])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _expected_trial_keys(bench: dict) -> set:
    return {
        (flow_count, run_number)
        for flow_count in bench['flow_counts']
        for run_number in range(1, int(bench['runs']) + 1)
    }


def _row_complete(row: dict, bench: dict) -> bool:
    if _trial_key(row) is None:
        return False
    if str(row.get('error', '')).strip():
        return False
    if row.get('goodput_source') != GOODPUT_SOURCE:
        return False
    try:
        return int(float(row.get('score_samples', 0))) == int(
            bench['score_duration_s'])
    except (TypeError, ValueError):
        return False


def _read_trial_rows(path: str) -> dict:
    """Return the newest metrics row for each (flow_count, run) trial."""
    rows = {}
    if not os.path.isfile(path):
        return rows
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key = _trial_key(row)
                if key is None:
                    continue
                for field in METRIC_FIELDS:
                    row.setdefault(field, '')
                rows[key] = row
    except OSError:
        return {}
    return rows


def _missing_trial_keys(output_root: str, approach: dict,
                        bench: dict) -> set:
    path = os.path.join(output_root, approach['_label'], 'metrics.csv')
    rows = _read_trial_rows(path)
    complete = {
        key for key, row in rows.items() if _row_complete(row, bench)
    }
    return _expected_trial_keys(bench) - complete


def _missing_trials_summary(missing: set) -> str:
    counts = {}
    for flow_count, _ in missing:
        counts[flow_count] = counts.get(flow_count, 0) + 1
    return ', '.join(
        f'{flow_count}:{counts[flow_count]}'
        for flow_count in sorted(counts)
    ) or 'none'


def _approach_complete(output_root: str, approach: dict,
                       bench: dict) -> bool:
    return not _missing_trial_keys(output_root, approach, bench)


def _write_run_meta(path: str, approach: dict, bench: dict,
                    config_path: str, checkpoint: str) -> None:
    now = time.strftime('%Y-%m-%d %H:%M:%S %z')
    meta = {}
    if os.path.isfile(path):
        try:
            with open(path) as f:
                meta = json.load(f) or {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            meta = {}
    meta.update({
        'approach': {
            key: value for key, value in approach.items()
            if not key.startswith('_')
        },
        'benchmark': bench,
        'benchmark_config': os.path.abspath(config_path),
        'approach_config': approach.get('_config_path', ''),
        'checkpoint': checkpoint,
        'last_resumed_at': now,
    })
    meta.setdefault('created_at', now)
    tmp_path = f'{path}.tmp.{os.getpid()}'
    with open(tmp_path, 'w') as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, path)


def _run_approach(base_cfg: dict, approach: dict, bench: dict,
                  output_root: str, listener_bin: str, python_bin: str,
                  config_path: str) -> list:
    _validate_approach(approach)
    kind = str(approach.get('kind', 'model')).lower()
    _ensure_kernel_cc_available(approach)
    checkpoint = ''
    if kind not in ('kernel', 'astraea', 'orca'):
        checkpoint = _resolve_from(
            os.path.dirname(config_path), str(approach['checkpoint']))
        if not os.path.exists(checkpoint):
            raise SystemExit(
                f'[responsive_fairness] checkpoint not found: {checkpoint}')

    approach_dir = os.path.abspath(
        os.path.join(output_root, approach['_label']))
    os.makedirs(approach_dir, exist_ok=True)
    metrics_csv = os.path.join(approach_dir, 'metrics.csv')
    rows_by_key = _read_trial_rows(metrics_csv)
    missing = sorted(
        _expected_trial_keys(bench) - {
            key for key, row in rows_by_key.items()
            if _row_complete(row, bench)
        })
    if not missing:
        return list(rows_by_key.values())
    _write_run_meta(
        os.path.join(approach_dir, 'run_meta.json'),
        approach, bench, config_path, checkpoint)

    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    total = len(missing)
    expected_total = len(_expected_trial_keys(bench))
    print(
        f'[responsive_fairness] {approach["_label"]} resuming '
        f'missing={total}/{expected_total} '
        f'by_flow={_missing_trials_summary(set(missing))}',
        flush=True)
    for flow_count, run_number in missing:
        schedule = _sample_schedule(run_number, bench)
        run_dir = os.path.join(
            approach_dir, f'{flow_count}_flows', f'run{run_number}')
        _safe_overwrite_dir(run_dir, approach_dir)
        schedule_csv, _ = _write_schedule(run_dir, schedule, bench)
        flow_schedule_csv = _write_flow_plan(
            run_dir, _flow_plan(bench, flow_count))
        work_q.put((
            flow_count, run_number, schedule, run_dir,
            schedule_csv, flow_schedule_csv))

    n_parallel = max(1, min(int(bench['n_parallel']), total))
    for _ in range(n_parallel):
        work_q.put(None)

    astraea_handle = None
    if kind == 'astraea':
        astraea_handle = _start_astraea_listener(
            approach, bench, approach_dir)
    procs = []
    for instance_id in range(n_parallel):
        proc = multiprocessing.Process(
            target=_slot,
            args=(
                instance_id, work_q, result_q, base_cfg, approach, bench,
                checkpoint, listener_bin, python_bin),
            daemon=True)
        proc.start()
        procs.append(proc)

    new_rows = []
    try:
        for completed in range(1, total + 1):
            row = result_q.get()
            new_rows.append(row)
            rows_by_key[_trial_key(row)] = row
            _append_csv(metrics_csv, METRIC_FIELDS, row)
            status = 'failed' if row.get('error') else 'done'
            print(
                f'[responsive_fairness] {approach["_label"]} '
                f'flows={row["flow_count"]} run={row["run"]} {status} '
                f'({completed}/{total})',
                flush=True)
    finally:
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
        _stop_astraea_listener(astraea_handle)
    rows = [
        rows_by_key[key] for key in sorted(rows_by_key)
    ]
    _write_rows(metrics_csv, METRIC_FIELDS, rows)
    _restore_sudo_user_ownership(approach_dir)
    return new_rows


def _collect_rows(output_root: str, cfg: dict) -> list:
    rows = _collect_approach_rows_with_labels(
        output_root, _approach_label_map(cfg))
    out = []
    for row in rows:
        if {'approach', 'flow_count', 'run', 'run_dir'}.issubset(row):
            for field in METRIC_FIELDS:
                row.setdefault(field, '')
            out.append(row)
    return out


def _schedule_from_row(row: dict) -> dict:
    try:
        with open(str(row['schedule_csv']), newline='') as f:
            schedule_rows = [
                {
                    't': float(item['t']),
                    'bw': float(item['bw']),
                    'delay': float(item['delay']),
                }
                for item in csv.DictReader(f)
            ]
    except (OSError, TypeError, ValueError, KeyError):
        return {}
    if not schedule_rows:
        return {}
    return {
        'seed': int(float(row.get('seed', 0))),
        'initial': schedule_rows[0],
        'changes': schedule_rows[1:],
        'rows': schedule_rows,
    }


def _read_samples(row: dict, bench: dict) -> list:
    path = str(row.get('sample_csv', '')).strip()
    if path and os.path.isfile(path):
        try:
            with open(path, newline='') as f:
                return list(csv.DictReader(f))
        except OSError:
            pass

    flow_count = int(float(row['flow_count']))
    flow_plan = _flow_plan(bench, flow_count)
    flows = _kernel_flows(str(row['run_dir']), flow_plan)
    schedule = _schedule_from_row(row)
    if not schedule or not _flows_have_goodput(flows, flow_count):
        return []
    scored = _score_series(flows, flow_plan, schedule, bench)
    path = os.path.join(
        str(row['run_dir']), 'responsive_fairness_samples.csv')
    _write_sample_csv(
        path,
        {
            '_label': row['approach'],
            'plot_label': row.get('plot_label', row['approach']),
        },
        flow_count, int(float(row['run'])), int(float(row['seed'])), scored)
    row['sample_csv'] = path
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _model_report_dir(output_root: str, approach: str) -> str:
    return os.path.join(
        output_root, 'responsive_fairness', 'legacy', str(approach))


def _model_individual_path(report_dir: str, row: dict,
                           flow_count: int) -> str:
    return os.path.join(
        report_dir, 'individual', f'{flow_count}_flows',
        f'run{int(float(row["run"]))}',
        'responsive_fairness_run.pdf')


def _replot_individual_rows(rows: list, bench: dict,
                            report_dir: str = None) -> tuple:
    plotted = 0
    skipped = 0
    failed = 0
    for row in rows:
        if str(row.get('error', '')).strip():
            skipped += 1
            continue
        try:
            flow_count = int(float(row['flow_count']))
            flow_plan = _flow_plan(bench, flow_count)
            flows = _kernel_flows(str(row['run_dir']), flow_plan)
            schedule = _schedule_from_row(row)
            if not schedule or not _flows_have_goodput(flows, flow_count):
                skipped += 1
                continue
            output = str(row.get('individual_plot') or os.path.join(
                str(row['run_dir']), 'responsive_fairness_run.pdf'))
            if report_dir:
                output = _model_individual_path(
                    report_dir, row, flow_count)
            _plot_run(
                _full_series(flows, flow_plan, schedule, bench),
                schedule, flow_plan,
                f'{row.get("plot_label") or row["approach"]} | '
                f'{flow_count} flows | run {int(float(row["run"]))}',
                output,
                rtt_flows=_measured_rtt_flows(
                    str(row['run_dir']), flow_plan,
                    [
                        path for path in str(row.get('state_logs', '')).split(';')
                        if path
                    ]))
            row['individual_plot'] = output
            plotted += 1
        except Exception as exc:
            failed += 1
            print(
                '[responsive_fairness] individual replot failed for '
                f'{row.get("run_dir", "")}: {exc}',
                flush=True)
    return plotted, skipped, failed


def _sample_array(samples: list, key: str,
                  keep_nan: bool = False) -> np.ndarray:
    values = np.asarray(
        [_finite_float(row.get(key)) for row in samples], dtype=float)
    return values if keep_nan else values[np.isfinite(values)]


def _plot_cdf(ax, values: np.ndarray, label: str, color: str,
              **plot_kwargs) -> bool:
    values = values[np.isfinite(values)]
    if not values.size:
        return False
    values = np.sort(values)
    pct = np.arange(1, len(values) + 1, dtype=float) / len(values) * 100.0
    kwargs = {'linewidth': 1.35, 'label': label, 'color': color}
    kwargs.update(plot_kwargs)
    ax.plot(values, pct, **kwargs)
    return True


def _plot_histogram_cdf(ax, values: np.ndarray, label: str, color: str,
                        bins: int = 50, **plot_kwargs) -> bool:
    values = values[np.isfinite(values)]
    if not values.size:
        return False
    counts, edges = np.histogram(values, bins=bins)
    pct = np.cumsum(counts, dtype=float) / values.size * 100.0
    kwargs = {'linewidth': 1.35, 'label': label, 'color': color}
    kwargs.update(plot_kwargs)
    ax.plot(edges[:-1], pct, **kwargs)
    return True


def _plot_one_cdf(rows: list, bench: dict, flow_count: int, metric: str,
                  title: str, xlabel: str, output: str,
                  xlim: tuple = None) -> None:
    plt, _ = _plot_backend()
    completed = [
        row for row in rows if not str(row.get('error', '')).strip()
        and int(float(row['flow_count'])) == flow_count
    ]
    approaches = sorted({row['approach'] for row in completed})
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.suptitle(
        f'{flow_count}-flow experiment | {title}',
        fontsize=12, fontweight='bold')
    plotted = False
    for index, approach in enumerate(approaches):
        group = [row for row in completed if row['approach'] == approach]
        samples = []
        for row in group:
            samples.extend(_read_samples(row, bench))
        label = next(
            (row.get('plot_label') for row in group if row.get('plot_label')),
            approach)
        plotted |= _plot_cdf(
            ax, _sample_array(samples, metric), label,
            COLORS[index % len(COLORS)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF (%)')
    ax.set_ylim(0, 100)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(
            0.5, 0.5, 'No completed samples',
            transform=ax.transAxes, ha='center', va='center')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp_output = f'{output}.tmp.{os.getpid()}.pdf'
    try:
        fig.savefig(tmp_output, dpi=240, bbox_inches='tight')
    finally:
        plt.close(fig)
    os.replace(tmp_output, output)


def _plot_cdfs(rows: list, bench: dict, output_root: str) -> None:
    for flow_count in bench['flow_counts']:
        flow_dir = os.path.join(output_root, f'{flow_count}_flows')
        _plot_one_cdf(
            rows, bench, flow_count, 'aggregate_responsiveness',
            'Aggregate Throughput Responsiveness CDF',
            'Aggregate goodput / scheduled BW(t)',
            os.path.join(flow_dir, 'aggregate_responsiveness_cdf.pdf'))
        _plot_one_cdf(
            rows, bench, flow_count, 'jain_fairness',
            'Capacity-Normalized Jain Fairness CDF', 'Jain index',
            os.path.join(flow_dir, 'jain_fairness_cdf.pdf'),
            xlim=(0, 1.02))
        _plot_one_cdf(
            rows, bench, flow_count, 'min_max_goodput_ratio',
            'Max-Min Goodput Ratio CDF', 'Minimum flow / maximum flow',
            os.path.join(flow_dir, 'min_max_goodput_cdf.pdf'),
            xlim=(0, 1.02))


def _plot_fairness_cdfs_all_flows(rows: list, bench: dict,
                                  output: str) -> None:
    plt, _ = _plot_backend()
    completed = [
        row for row in rows if not str(row.get('error', '')).strip()
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.6, 8.6), sharex=True,
        gridspec_kw={'hspace': 0.12})
    fig.suptitle(
        'Fairness CDFs Across Flow Counts',
        fontsize=12, fontweight='bold')
    plotted = [False, False]
    for index, flow_count in enumerate(bench['flow_counts']):
        group = [
            row for row in completed
            if int(float(row['flow_count'])) == flow_count
        ]
        samples = []
        for row in group:
            samples.extend(_read_samples(row, bench))
        label = f'{flow_count} flows ({len(group)} runs)'
        color = COLORS[index % len(COLORS)]
        plotted[0] |= _plot_cdf(
            axes[0],
            _sample_array(samples, 'jain_fairness'),
            label,
            color,
        )
        plotted[1] |= _plot_cdf(
            axes[1],
            _sample_array(samples, 'min_max_goodput_ratio'),
            label,
            color,
        )
    axes[0].set_title('Jain fairness')
    axes[1].set_title('Minimum / maximum goodput')
    axes[1].set_xlabel('Fairness ratio')
    for index, ax in enumerate(axes):
        ax.set_ylabel('CDF (%)')
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)
        if plotted[index]:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(
                0.5, 0.5, 'No completed samples',
                transform=ax.transAxes, ha='center', va='center')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp_output = f'{output}.tmp.{os.getpid()}.pdf'
    try:
        fig.savefig(tmp_output, dpi=240, bbox_inches='tight')
    finally:
        plt.close(fig)
    os.replace(tmp_output, output)


def _state_log_mean(row: dict, column: str, start_s: float,
                    end_s: float) -> float:
    paths = [
        path for path in str(row.get('state_logs', '')).split(';') if path
    ]
    if not paths:
        paths = sorted(glob.glob(os.path.join(
            str(row.get('run_dir', '')), '*_state_ep*_a*.csv')))
    total = 0.0
    count = 0
    for path in paths:
        try:
            with open(path, newline='') as handle:
                for item in csv.DictReader(handle):
                    t_s = _finite_float(item.get('t_s'))
                    value = _finite_float(item.get(column))
                    if (start_s < t_s <= end_s
                            and math.isfinite(value) and value > 0.0):
                        total += value
                        count += 1
        except OSError:
            continue
    return total / count if count else math.nan


def _schedule_mean(row: dict, key: str, start_s: float,
                   end_s: float) -> float:
    schedule = _schedule_from_row(row)
    entries = list(schedule.get('rows') or [])
    if not entries or end_s <= start_s:
        return math.nan
    weighted = 0.0
    covered = 0.0
    for index, entry in enumerate(entries):
        left = max(start_s, float(entry['t']))
        right = min(
            end_s,
            float(entries[index + 1]['t'])
            if index + 1 < len(entries) else end_s,
        )
        if right <= left:
            continue
        width = right - left
        weighted += float(entry[key]) * width
        covered += width
    return weighted / covered if covered > 0.0 else math.nan


def _plot_responsiveness_cdfs_all_flows(rows: list, bench: dict,
                                        output: str) -> None:
    plt, _ = _plot_backend()
    completed = [
        row for row in rows if not str(row.get('error', '')).strip()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.2))
    fig.suptitle(
        'Responsive Fairness: Throughput and RTT',
        fontsize=12, fontweight='bold')
    start_s = float(bench['arrival_window_s'])
    end_s = float(bench['duration_s'])
    plotted = [False, False]
    for index, flow_count in enumerate(bench['flow_counts']):
        group = [
            row for row in completed
            if int(float(row['flow_count'])) == flow_count
        ]
        label = f'{flow_count} flows ({len(group)} runs)'
        color = COLORS[index % len(COLORS)]
        goodput = np.asarray([
            _finite_float(row.get('mean_aggregate_goodput_mbps'))
            for row in group
        ], dtype=float)
        srtt = np.asarray([
            _state_log_mean(row, 'srtt_ms', start_s, end_s)
            for row in group
        ], dtype=float)
        plotted[0] |= _plot_cdf(axes[0], goodput, label, color)
        plotted[1] |= _plot_histogram_cdf(
            axes[1], srtt, label, color)

    schedule_by_run = {}
    for row in completed:
        schedule_by_run.setdefault(int(float(row['run'])), row)
    schedule_rows = list(schedule_by_run.values())
    scheduled_bw = np.asarray([
        _finite_float(row.get('mean_capacity_mbps'))
        for row in schedule_rows
    ], dtype=float)
    scheduled_rtt = np.asarray([
        _schedule_mean(row, 'delay', start_s, end_s)
        for row in schedule_rows
    ], dtype=float)
    plotted[0] |= _plot_cdf(
        axes[0], scheduled_bw, 'Scheduled BW', 'black',
        linewidth=1.7, zorder=10)
    plotted[1] |= _plot_histogram_cdf(
        axes[1], scheduled_rtt, 'Scheduled RTT', 'black',
        linewidth=1.7, zorder=10)

    axes[0].set_xlabel('Average Aggregate Goodput (Mbps)')
    axes[0].set_ylabel('% of Runs')
    axes[0].set_title('Goodput CDF')
    axes[1].set_xlabel('Average SRTT (ms)')
    axes[1].set_ylabel('Percent of Trials (%)')
    axes[1].set_title('SRTT CDF')
    for index, ax in enumerate(axes):
        ax.grid(True, alpha=0.25)
        if not plotted[index]:
            ax.text(
                0.5, 0.5, 'No completed samples',
                transform=ax.transAxes, ha='center', va='center')

    unique = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.23)
    if unique:
        fig.legend(
            unique.values(), unique.keys(), loc='lower center',
            ncol=min(5, len(unique)), frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, 0.03))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp_output = f'{output}.tmp.{os.getpid()}.pdf'
    try:
        fig.savefig(tmp_output, dpi=240, bbox_inches='tight')
    finally:
        plt.close(fig)
    os.replace(tmp_output, output)


def _mean_stack(arrays: list) -> tuple:
    if not arrays:
        return np.asarray([]), np.asarray([])
    stack = np.stack(arrays, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def _plot_averages(rows: list, bench: dict, output: str) -> None:
    plt, PdfPages = _plot_backend()
    completed = [
        row for row in rows if not str(row.get('error', '')).strip()]
    groups = sorted({
        (row['approach'], int(float(row['flow_count'])))
        for row in completed
    })
    tmp = f'{output}.tmp.{os.getpid()}.pdf'
    with PdfPages(tmp) as pdf:
        for approach, flow_count in groups:
            group = [
                row for row in completed
                if row['approach'] == approach
                and int(float(row['flow_count'])) == flow_count
            ]
            runs = [_read_samples(row, bench) for row in group]
            runs = [samples for samples in runs if samples]
            if not runs:
                continue
            times = _sample_array(runs[0], 'time_s', keep_nan=True)
            capacity, capacity_std = _mean_stack([
                _sample_array(samples, 'capacity_mbps', keep_nan=True)
                for samples in runs])
            aggregate, aggregate_std = _mean_stack([
                _sample_array(
                    samples, 'aggregate_goodput_mbps', keep_nan=True)
                for samples in runs])
            jain, jain_std = _mean_stack([
                _sample_array(samples, 'jain_fairness', keep_nan=True)
                for samples in runs])
            min_max, min_max_std = _mean_stack([
                _sample_array(
                    samples, 'min_max_goodput_ratio', keep_nan=True)
                for samples in runs])
            per_flow = [
                _mean_stack([
                    _sample_array(
                        samples, f'flow{flow + 1}_goodput_mbps',
                        keep_nan=True)
                    for samples in runs])
                for flow in range(flow_count)
            ]

            fig, axes = plt.subplots(
                3, 1, figsize=(11.0, 8.0), sharex=True,
                gridspec_kw={'hspace': 0.12})
            plot_label = next(
                (row.get('plot_label') for row in group
                 if row.get('plot_label')),
                approach)
            fig.suptitle(
                f'{plot_label} | {flow_count} flows | '
                f'mean +/- 1 std over {len(runs)} runs',
                fontsize=12, fontweight='bold')
            axes[0].plot(
                times, capacity, color='black', linewidth=1.3,
                label='scheduled BW')
            axes[0].fill_between(
                times, np.maximum(0, capacity - capacity_std),
                capacity + capacity_std, color='black', alpha=0.08)
            axes[0].plot(
                times, aggregate, color='#4878cf', linewidth=1.3,
                label='aggregate goodput')
            axes[0].fill_between(
                times, np.maximum(0, aggregate - aggregate_std),
                aggregate + aggregate_std, color='#4878cf', alpha=0.18)
            axes[0].set_ylabel('Mbps')
            axes[0].legend(frameon=False, ncol=2)

            for flow, (mean, std) in enumerate(per_flow):
                color = COLORS[flow % len(COLORS)]
                axes[1].plot(
                    times, mean, color=color, linewidth=1.15,
                    label=f'flow {flow + 1}')
                axes[1].fill_between(
                    times, np.maximum(0, mean - std), mean + std,
                    color=color, alpha=0.12)
            axes[1].set_ylabel('Per-flow Mbps')
            axes[1].legend(frameon=False, ncol=flow_count)

            for label, mean, std, color in (
                    ('Jain fairness', jain, jain_std, '#59a14f'),
                    ('min / max goodput', min_max, min_max_std, '#e1812c')):
                axes[2].plot(
                    times, mean, color=color, linewidth=1.15, label=label)
                axes[2].fill_between(
                    times, np.maximum(0, mean - std), mean + std,
                    color=color, alpha=0.12)
            axes[2].axhline(
                1.0, color='black', alpha=0.25, linewidth=0.8)
            axes[2].set_ylabel('Ratio')
            axes[2].set_xlabel('Time (s)')
            axes[2].legend(frameon=False, ncol=2)
            for ax in axes:
                ax.grid(True, alpha=0.25)
                ax.set_xlim(
                    float(bench['arrival_window_s']),
                    float(bench['duration_s']))
            pdf.savefig(fig, dpi=220, bbox_inches='tight')
            plt.close(fig)
    os.replace(tmp, output)


def _combine_pdfs(inputs: list, output: str) -> None:
    inputs = [
        os.path.abspath(path) for path in inputs
        if path and os.path.isfile(path) and os.path.getsize(path) > 0
    ]
    if not inputs:
        return
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = f'{output}.tmp.{os.getpid()}.pdf'
    pdfunite = shutil.which('pdfunite')
    if pdfunite:
        subprocess.run(
            [pdfunite, *inputs, tmp], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    else:
        gs = shutil.which('gs')
        if not gs:
            raise RuntimeError(
                'cannot combine PDFs: neither pdfunite nor gs is installed')
        subprocess.run(
            [
                gs, '-dBATCH', '-dNOPAUSE', '-q',
                '-sDEVICE=pdfwrite', f'-sOutputFile={tmp}', *inputs,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.replace(tmp, output)


def _model_pdf_inputs(report_dir: str, rows: list, bench: dict) -> list:
    paths = [
        os.path.join(report_dir, 'responsiveness_all_flows_cdfs.pdf'),
        os.path.join(report_dir, 'fairness_all_flows_cdfs.pdf'),
    ]
    for flow_count in bench['flow_counts']:
        flow_dir = os.path.join(report_dir, f'{flow_count}_flows')
        paths.extend([
            os.path.join(flow_dir, 'aggregate_responsiveness_cdf.pdf'),
            os.path.join(flow_dir, 'jain_fairness_cdf.pdf'),
            os.path.join(flow_dir, 'min_max_goodput_cdf.pdf'),
        ])
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            int(float(row.get('flow_count', 0))),
            int(float(row.get('run', 0))),
        ))
    for row in ordered_rows:
        flow_count = int(float(row['flow_count']))
        report_plot = _model_individual_path(
            report_dir, row, flow_count)
        run_plot = str(row.get('individual_plot', '')).strip()
        paths.append(report_plot if os.path.isfile(report_plot) else run_plot)
    return paths


def _write_model_report(output_root: str, approach: str, rows: list,
                        bench: dict) -> str:
    report_dir = _model_report_dir(output_root, approach)
    os.makedirs(report_dir, exist_ok=True)
    _plot_cdfs(rows, bench, report_dir)
    _plot_fairness_cdfs_all_flows(
        rows, bench,
        os.path.join(report_dir, 'fairness_all_flows_cdfs.pdf'))
    _plot_responsiveness_cdfs_all_flows(
        rows, bench,
        os.path.join(report_dir, 'responsiveness_all_flows_cdfs.pdf'))
    combined = os.path.join(report_dir, 'responsive_fairness_all.pdf')
    _combine_pdfs(
        _model_pdf_inputs(report_dir, rows, bench), combined)
    return combined


def _write_live_plots(output_root: str, rows: list, bench: dict) -> None:
    """Atomically refresh model-specific reports from completed rows."""
    completed = [
        row for row in rows if _row_complete(row, bench)
    ]
    if not completed:
        return
    for approach in sorted({row['approach'] for row in completed}):
        model_rows = [
            row for row in completed if row['approach'] == approach]
        combined = _write_model_report(
            output_root, approach, model_rows, bench)
        report_dir = _model_report_dir(output_root, approach)
        progress = {
            'approach': approach,
            'completed_rows': len(model_rows),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
            'combined_pdf': combined,
            'flow_counts': {
                str(flow_count): sum(
                    1 for row in model_rows
                    if int(float(row['flow_count'])) == flow_count)
                for flow_count in bench['flow_counts']
            },
        }
        progress_path = os.path.join(
            report_dir, 'live_plot_progress.json')
        tmp_path = f'{progress_path}.tmp.{os.getpid()}'
        with open(tmp_path, 'w') as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp_path, progress_path)


def _watch_plots(output_root: str, cfg: dict, bench: dict,
                 selected: set = None, interval_s: float = 15.0) -> None:
    """Refresh plots whenever new completed metrics rows appear."""
    configured = _configured_approaches(cfg, selected or None)
    expected_per_model = len(bench['flow_counts']) * int(bench['runs'])
    expected = expected_per_model * max(1, len(configured))
    seen = set()
    last_count = -1
    print(
        f'[responsive_fairness] plot watcher started interval={interval_s:g}s '
        f'expected_rows={expected}',
        flush=True)
    while True:
        all_rows = _collect_rows(output_root, cfg)
        rows = [
            row for row in all_rows
            if not selected or _slug(row.get('approach', '')) in selected
        ]
        completed = []
        for row in rows:
            if str(row.get('error', '')).strip():
                continue
            try:
                int(float(row.get('flow_count', '')))
                int(float(row.get('run', '')))
            except (TypeError, ValueError):
                # The metrics writer may be between bytes of its newest row.
                continue
            completed.append(row)
        new_rows = []
        for row in completed:
            key = (
                str(row.get('approach', '')),
                int(float(row.get('flow_count', 0))),
                int(float(row.get('run', 0))),
            )
            if key not in seen:
                seen.add(key)
                new_rows.append(row)
        if new_rows:
            plotted = skipped = failed = 0
            for approach in sorted({
                    row['approach'] for row in new_rows}):
                model_rows = [
                    row for row in new_rows
                    if row['approach'] == approach]
                result = _replot_individual_rows(
                    model_rows, bench,
                    report_dir=_model_report_dir(
                        output_root, approach))
                plotted += result[0]
                skipped += result[1]
                failed += result[2]
            _write_live_plots(output_root, completed, bench)
            print(
                f'[responsive_fairness] watcher rows={len(completed)} '
                f'new={len(new_rows)} individual_plots={plotted} '
                f'skipped={skipped} failed={failed}',
                flush=True)
            last_count = len(completed)
        elif len(completed) != last_count:
            _write_live_plots(output_root, completed, bench)
            last_count = len(completed)
        if len(completed) >= expected:
            print(
                '[responsive_fairness] plot watcher complete',
                flush=True)
            return
        time.sleep(max(1.0, float(interval_s)))


def _replot_all(output_root: str, cfg: dict, bench: dict,
                selected: set = None) -> None:
    all_rows = _collect_rows(output_root, cfg)
    rows = [
        row for row in all_rows
        if not selected or _slug(row.get('approach', '')) in selected
    ]
    if selected and not rows:
        raise SystemExit(
            '[responsive_fairness] no existing rows matched: '
            + ', '.join(sorted(selected)))
    plotted = skipped = failed = 0
    for approach in sorted({row['approach'] for row in rows}):
        model_rows = [
            row for row in rows if row['approach'] == approach]
        result = _replot_individual_rows(
            model_rows, bench,
            report_dir=_model_report_dir(output_root, approach))
        plotted += result[0]
        skipped += result[1]
        failed += result[2]
    for row in rows:
        _read_samples(row, bench)
    _write_rows(
        os.path.join(output_root, 'metrics.csv'), METRIC_FIELDS, all_rows)
    _write_live_plots(output_root, rows, bench)
    _restore_sudo_user_ownership(output_root)
    print(
        f'[responsive_fairness] replotted rows={len(rows)} '
        f'total_rows={len(all_rows)} individual_plots={plotted} '
        f'skipped={skipped} failed={failed}',
        flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Run the dynamic responsive fairness benchmark.')
    ap.add_argument('--config', default=_DEFAULT_CONFIG)
    ap.add_argument(
        '--approach', action='append', default=None,
        help='approach data_folder/name to run; repeatable')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = _load_benchmark_config(config_path)
    bench = _bench_cfg(cfg)
    config_dir = os.path.dirname(config_path)
    output_root = cfg.get('output_root', 'data')
    if not os.path.isabs(str(output_root)):
        output_root = os.path.abspath(
            os.path.join(config_dir, str(output_root)))
    os.makedirs(output_root, exist_ok=True)

    selected = {_slug(value) for value in (args.approach or [])}
    configured = _configured_approaches(cfg, selected or None)
    _validate_unique_data_folders(configured)
    if not configured:
        available = sorted({
            key for raw in (cfg.get('approaches') or [])
            for key in _approach_selection_keys(raw or {})
        })
        raise SystemExit(
            '[responsive_fairness] no matching approaches'
            + (f'; available: {", ".join(available)}' if available else ''))

    approaches = [
        approach for approach in configured
        if not _approach_complete(output_root, approach, bench)
    ]
    skipped = [
        approach for approach in configured
        if _approach_complete(output_root, approach, bench)
    ]
    total = len(bench['flow_counts']) * int(bench['runs'])
    print(f'[responsive_fairness] output_root={output_root}', flush=True)
    print(
        f'[responsive_fairness] duration={bench["duration_s"]:g}s '
        f'score_window=({bench["arrival_window_s"]:g},'
        f'{bench["duration_s"]:g}] cadence={bench["change_interval_s"]}s '
        f'flow_counts={bench["flow_counts"]} trials/approach={total}',
        flush=True)
    for flow_count in bench['flow_counts']:
        starts = [
            item['start'] for item in _flow_plan(bench, flow_count)]
        print(
            f'[responsive_fairness] {flow_count}_flow_starts='
            + ','.join(f'{value:g}' for value in starts),
            flush=True)
    if skipped:
        print(
            '[responsive_fairness] skipping complete data='
            + ', '.join(item['_label'] for item in skipped),
            flush=True)
    print(
        '[responsive_fairness] approaches_to_run='
        + (', '.join(item['_label'] for item in approaches)
           if approaches else '(none)'),
        flush=True)

    if args.dry_run:
        for approach in skipped:
            _, approach_config_path = _load_approach_runtime_config(
                approach, config_path)
            print(
                f'[responsive_fairness] dry-run would skip complete data: '
                f'{os.path.join(output_root, approach["_label"])}'
                + (f' config={approach_config_path}'
                   if approach_config_path else ''),
                flush=True)
        for approach in approaches:
            _validate_approach(approach)
            _, approach_config_path = _load_approach_runtime_config(
                approach, config_path)
            _sample_schedule(1, bench)
            missing_keys = _missing_trial_keys(
                output_root, approach, bench)
            print(
                f'[responsive_fairness] dry-run would run '
                f'{approach["_label"]}: {len(missing_keys)}/{total} '
                f'missing trials '
                f'by_flow={_missing_trials_summary(missing_keys)}'
                + (f' config={approach_config_path}'
                   if approach_config_path else ''),
                flush=True)
        return

    if not approaches:
        print(
            '[responsive_fairness] no approaches need running',
            flush=True)
        return

    failed_trials = 0
    try:
        for approach in approaches:
            base_cfg, approach_config_path = _load_approach_runtime_config(
                approach, config_path)
            paths = base_cfg.get('paths', {}) or {}
            kind = str(approach.get('kind', 'model')).lower()
            if kind not in ('kernel', 'astraea', 'orca') and (
                    'listener' not in paths or 'py' not in paths):
                raise SystemExit(
                    f'[responsive_fairness] approach config for '
                    f'{approach["_label"]} must define '
                    'paths.listener and paths.py')
            listener_bin = (
                _resolve_repo_path(paths['listener'])
                if 'listener' in paths else '')
            python_bin = (
                _resolve_repo_path(paths['py']) if 'py' in paths else '')
            if approach_config_path:
                print(
                    f'[responsive_fairness] {approach["_label"]} '
                    f'config={approach_config_path}',
                    flush=True)
            try:
                rows = _run_approach(
                    base_cfg, approach, bench, output_root, listener_bin,
                    python_bin, config_path)
                failed_trials += sum(
                    bool(str(row.get('error', '')).strip())
                    for row in rows)
            finally:
                learner_port = int(
                    (base_cfg.get('learner', {}) or {}).get('port', 6301))
                _final_runtime_cleanup(learner_port)
    finally:
        _restore_sudo_user_ownership(output_root)

    if failed_trials:
        raise SystemExit(
            f'[responsive_fairness] {failed_trials} trial(s) failed')

    print(
        '[responsive_fairness] done; data checked and missing trials '
        'attempted; metrics='
        + ','.join(
            os.path.join(output_root, approach['_label'], 'metrics.csv')
            for approach in approaches),
        flush=True)


if __name__ == '__main__':
    main()
