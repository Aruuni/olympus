#!/usr/bin/env python3
"""
Fixed-output single-flow responsiveness benchmark.

Runs each selected approach for the same 50 deterministic random BW/RTT traces:
10-100 Mbps and 10-100 ms, changing every 15 seconds for 300 seconds.

Example:
  sudo -E env PATH="$PATH" HOME="$HOME" \\
    ./venv_training/bin/python benchmarks/benchmark_responsiveness/responsiveness.py \\
      --config benchmarks/benchmark_responsiveness/config.yaml
"""

import argparse
import csv
import copy
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
import threading
import time
import traceback
import textwrap

import yaml

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
_DEFAULT_CONFIG = os.path.join(_HERE, 'config.yaml')
sys.path.insert(0, _ROOT)

from olympus.environments.mininet.env import MininetEnv
from olympus.common.checkpoint_config import (
    apply_model_config_from_checkpoint,
)
from olympus.common.state_log_metrics import (
    multiflow_state_log_metrics as _state_log_metrics,
)
from olympus.orchestrator import (
    _activate_runtime_blocks,
    _as_bool,
    _final_runtime_cleanup,
    _resolve_repo_path,
    _slug,
    run_episode,
)
from olympus.common.bench_utils import (
    SAGE_REWARD_LABELS,
    _ASTRAEA_DEFAULTS,
    _ORCA_DEFAULTS,
    _SS_RTT_RE,
    _append_csv,
    _approach_label,
    _approach_label_map,
    _approach_plot_label,
    _approach_selection_keys,
    _astraea_settings,
    _binned_series,
    _collect_approach_rows_with_labels,
    _configured_approaches,
    _copy_if_exists,
    _default_approach_plot_label,
    _deterministic_env,
    _ensure_kernel_cc_available,
    _finite_float,
    _kernel_flows,
    _load_approach_runtime_config,
    _load_benchmark_config,
    _orca_actor_id,
    _orca_paths,
    _orca_port,
    _orca_receiver_to_csvs,
    _orca_run_user,
    _orca_settings,
    _orca_ss_log_path,
    _orca_ss_rtt_flows,
    _orca_ss_rtt_samples,
    _orca_user_home,
    _parse_iperf_json,
    _parse_orca_client_output,
    _parse_orca_ss,
    _prepare_runtime_cfg,
    _read_metrics,
    _resolve_astraea_python,
    _resolve_from,
    normalize_bdp_mult,
    _restore_sudo_user_ownership,
    _resume_rows_by_key,
    _row_in_specific,
    _run_orca_on_env,
    _safe_overwrite_dir,
    _safe_unlink,
    _specific_plot_keys,
    _start_astraea_listener,
    _stop_astraea_listener,
    _validate_approach,
    _validate_unique_data_folders,
    _write_flow_plan,
    _write_measurement_csv,
    _write_rows,
    _write_schedule,
)


METRIC_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'kind', 'algorithm', 'kernel_cc',
    'state', 'reward', 'training_environment', 'benchmark_environment', 'run',
    'seed', 'duration_s', 'change_interval_s', 'initial_bw_mbps', 'initial_rtt_ms',
    'mean_scheduled_bw_mbps', 'mean_scheduled_rtt_ms', 'schedule_changes', 'return',
    'score_per_second', 'steps', 'mean_thr_mbps', 'p50_thr_mbps',
    'p05_thr_mbps', 'mean_rtt_ms', 'p95_rtt_ms', 'mean_srtt_ms',
    'p95_srtt_ms', 'mean_min_rtt_ms',
    'mean_kalman_rtt_ms', 'mean_cwnd', 'mean_cwnd_mult',
    'frac_mult_below_1', 'frac_mult_above_1', 'mean_loss_ratio',
    'mean_reward', 'switch_rate', 'dominant_option_share', 'option_entropy',
    'n_options_seen', 'state_log', 'schedule_csv', 'individual_plot',
    'run_dir', 'checkpoint', 'error',
]

APPROACH_COLOR_PALETTE = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#393b79',  # indigo
    '#637939',  # moss
    '#8c6d31',  # ochre
    '#843c39',  # brick
    '#7b4173',  # plum
    '#3182bd',  # steel
    '#e6550d',  # burnt orange
    '#31a354',  # forest
    '#756bb1',  # violet
    '#636363',  # dark gray
]

CDF_BINS = 50
_STATE_LOG_MEAN_CACHE = {}







def _benchmark_cfg(cfg: dict) -> dict:
    b = dict(cfg.get('benchmark') or {})
    b.setdefault('name', 'random_bw_rtt_15s_300s')
    b.setdefault('runs', 50)
    if b.get('n_parallel') is None:
        b['n_parallel'] = 1
    b.setdefault('duration_s', 300)
    b.setdefault('change_interval_s', 15)
    b.setdefault('bw_min_mbps', 10)
    b.setdefault('bw_max_mbps', 100)
    b.setdefault('rtt_min_ms', 10)
    b.setdefault('rtt_max_ms', 100)
    b.setdefault('seed_offset', 0)
    b.setdefault('flows', 1)
    b['bdp_mult'] = normalize_bdp_mult(b.get('bdp_mult', 4))[0]
    b.setdefault('measure_interval_s', 1.0)
    b.setdefault('kernel_cport_base', 33000)
    b.setdefault('plot_episodes', True)
    b.setdefault('deterministic_inference', True)
    return b
















def _approach_output_dir(output_root: str, approach: dict) -> str:
    return os.path.abspath(os.path.join(output_root, approach['_label']))


def _expected_run_keys(bench: dict) -> set:
    return set(range(1, int(bench['runs']) + 1))


def _run_key(row: dict):
    """Return the integer run number for a metrics row, or None if unparsable."""
    try:
        return int(float(row.get('run', 0)))
    except (TypeError, ValueError):
        return None


def _row_complete(row: dict) -> bool:
    """A run counts as done once it produced a row without an error."""
    if _run_key(row) is None:
        return False
    return not str(row.get('error', '')).strip()


def _completed_keys(rows_by_key: dict) -> set:
    return {key for key, row in rows_by_key.items() if _row_complete(row)}


def _approach_complete(output_root: str, approach: dict, bench: dict) -> bool:
    metrics_csv = os.path.join(
        _approach_output_dir(output_root, approach), 'metrics.csv')
    rows_by_key = _resume_rows_by_key(metrics_csv, METRIC_FIELDS, _run_key)
    return _expected_run_keys(bench).issubset(_completed_keys(rows_by_key))


def _split_approaches_by_existing_data(approaches: list, output_root: str,
                                       bench: dict) -> tuple:
    """Split approaches into (to_run, skipped) by whether every run is done.

    Approaches with only partial data go to ``to_run`` so they resume from the
    missing runs rather than being skipped wholesale.
    """
    to_run = []
    skipped = []
    for approach in approaches:
        if _approach_complete(output_root, approach, bench):
            skipped.append(approach)
        else:
            to_run.append(approach)
    return to_run, skipped










def _sample_schedule(run_number: int, bench: dict) -> dict:
    seed = int(bench.get('seed_offset', 0)) + int(run_number)
    rng = random.Random(seed)
    duration = int(bench['duration_s'])
    interval = int(bench['change_interval_s'])
    bw_min = int(bench['bw_min_mbps'])
    bw_max = int(bench['bw_max_mbps'])
    rtt_min = int(bench['rtt_min_ms'])
    rtt_max = int(bench['rtt_max_ms'])

    def draw():
        return {
            'bw': int(rng.randint(bw_min, bw_max)),
            'delay': int(rng.randint(rtt_min, rtt_max)),
        }

    initial = draw()
    changes = []
    for t_s in range(interval, duration, interval):
        value = draw()
        changes.append({'t': int(t_s), 'bw': value['bw'], 'delay': value['delay']})
    rows = [{'t': 0, 'bw': initial['bw'], 'delay': initial['delay']}] + changes
    return {
        'seed': seed,
        'initial': initial,
        'changes': changes,
        'rows': rows,
    }


def _time_weighted_mean(rows: list, key: str, duration_s: int) -> float:
    if not rows:
        return math.nan
    total = 0.0
    for idx, row in enumerate(rows):
        start = float(row['t'])
        end = float(rows[idx + 1]['t']) if idx + 1 < len(rows) else float(duration_s)
        total += float(row[key]) * max(0.0, end - start)
    return total / max(1.0, float(duration_s))




def _run_link_schedule(env, changes: list, episode_start: float, stop) -> None:
    for entry in changes:
        t_target = episode_start + float(entry['t'])
        while not stop.is_set():
            remaining = t_target - time.monotonic()
            if remaining <= 0:
                break
            stop.wait(timeout=min(remaining, 0.05))
        if stop.is_set():
            return
        env.change_link(bw=entry.get('bw'), delay=entry.get('delay'))
        print(f'[resp_bench] link change t={time.monotonic() - episode_start:.1f}s '
              f'bw={entry.get("bw")} delay={entry.get("delay")}', flush=True)








def _read_trace_csv(csv_path: str):
    times, bandwidths = [], []
    try:
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                t_s = _finite_float(row.get('time'))
                bandwidth = _finite_float(row.get('bandwidth'))
                if math.isfinite(t_s) and math.isfinite(bandwidth):
                    times.append(t_s)
                    bandwidths.append(bandwidth)
    except FileNotFoundError:
        pass
    return np.asarray(times, dtype=float), np.asarray(bandwidths, dtype=float)


def _step_xy(rows: list, duration_s: float, key: str):
    if not rows:
        return np.asarray([]), np.asarray([])
    points = sorted(rows, key=lambda item: float(item['t']))
    times = [float(item['t']) for item in points]
    values = [float(item[key]) for item in points]
    times.append(float(duration_s))
    values.append(values[-1])
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def plot_goodput_run(run_dir: str, output_path: str, label: str, run_number: int,
                     schedule_rows: list, duration_s: int) -> str:
    tx, bw = _step_xy(schedule_rows, duration_s, 'bw')
    _, rtt = _step_xy(schedule_rows, duration_s, 'delay')

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 4.8), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.35], 'hspace': 0.12},
    )
    fig.suptitle(f'{label} run {run_number}', fontsize=11, fontweight='bold')

    ax = axes[0]
    if len(tx):
        ax.step(tx, bw, where='post', color='black', linewidth=1.25, label='capacity')
    ax.set_ylabel('BW (Mbps)')
    ax.set_ylim(0, max(105.0, float(np.nanmax(bw)) * 1.08 if len(bw) else 105.0))
    ax.grid(True, alpha=0.28, linewidth=0.5)

    ax_rtt = ax.twinx()
    if len(tx):
        ax_rtt.step(tx, rtt, where='post', color='#7f7f7f',
                    linewidth=1.0, linestyle='--', label='RTT')
    ax_rtt.set_ylabel('RTT (ms)')
    ax_rtt.set_ylim(0, max(105.0, float(np.nanmax(rtt)) * 1.08 if len(rtt) else 105.0))
    lines, legend_labels = ax.get_legend_handles_labels()
    rlines, rlabels = ax_rtt.get_legend_handles_labels()
    if lines or rlines:
        ax.legend(lines + rlines, legend_labels + rlabels, loc='upper right',
                  frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    if len(tx):
        ax.step(tx, bw, where='post', color='black', linewidth=0.9,
                alpha=0.4, label='capacity')

    total_by_time = {}
    plotted = 0
    csv_dir = os.path.join(run_dir, 'csvs')
    for flow_idx in range(1, 1000):
        t, goodput = _read_trace_csv(os.path.join(csv_dir, f'x{flow_idx}.csv'))
        if not len(t):
            if flow_idx == 1:
                continue
            break
        ax.plot(t, goodput, linewidth=0.8, alpha=0.55, label=f'flow {flow_idx}')
        for tv, gv in zip(t, goodput):
            total_by_time[float(tv)] = total_by_time.get(float(tv), 0.0) + float(gv)
        plotted += 1

    if total_by_time:
        t_total = np.asarray(sorted(total_by_time), dtype=float)
        y_total = np.asarray([total_by_time[t] for t in t_total], dtype=float)
        ax.plot(t_total, y_total, color='#4878cf', linewidth=1.15,
                alpha=0.95, label=f'{label} total')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Goodput (Mbps)')
    ax.set_xlim(0, float(duration_s))
    ax.set_ylim(0, max(105.0, float(np.nanmax(bw)) * 1.12 if len(bw) else 105.0))
    ax.grid(True, alpha=0.28, linewidth=0.5)
    if plotted:
        ax.legend(loc='upper right', frameon=False, fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, 'No csvs/x*.csv trace found',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _read_schedule_rows(path: str, row: dict) -> list:
    rows = []
    try:
        with open(path, newline='') as f:
            for item in csv.DictReader(f):
                rows.append({
                    't': float(item.get('t', 0.0)),
                    'bw': float(item.get('bw', row.get('initial_bw_mbps', 100))),
                    'delay': float(item.get('delay', row.get('initial_rtt_ms', 20))),
                })
    except (FileNotFoundError, OSError, TypeError, ValueError):
        rows = []

    if not rows:
        return [{
            't': 0.0,
            'bw': float(row.get('initial_bw_mbps') or 100.0),
            'delay': float(row.get('initial_rtt_ms') or 20.0),
        }]
    rows.sort(key=lambda item: item['t'])
    return rows


def _plot_path_for_row(row: dict, bw: float, delay: float) -> str:
    existing = row.get('individual_plot')
    if existing:
        return existing
    run_dir = row.get('run_dir') or os.path.dirname(row.get('state_log', ''))
    try:
        run = int(float(row.get('run', 0)))
    except (TypeError, ValueError):
        run = 0
    if str(row.get('kind', '')).lower() in ('kernel', 'astraea'):
        return os.path.join(run_dir, 'run_plot.pdf')
    return os.path.join(run_dir, f'ep{run:06d}_bw{bw:.0f}_d{delay:.0f}.pdf')


def _orchestrator_plot_path(row: dict, bw: float, delay: float) -> str:
    run_dir = row.get('run_dir') or os.path.dirname(row.get('state_log', ''))
    try:
        run = int(float(row.get('run', 0)))
        flows = int(float(row.get('flows', 1) or 1))
    except (TypeError, ValueError):
        run = 0
        flows = 1
    return os.path.join(
        run_dir,
        f'ep{run:06d}_bw{bw:.0f}_d{delay:.0f}_n{flows}.pdf',
    )


def _plot_individual_runs(rows: list) -> tuple:
    """Generate individual plots only for non-orchestrator baselines."""
    plotted = 0
    skipped = 0
    failed = 0
    for row in rows:
        if row.get('error'):
            skipped += 1
            continue
        schedule_rows = _read_schedule_rows(row.get('schedule_csv', ''), row)
        initial = schedule_rows[0]
        bw = float(initial['bw'])
        delay = float(initial['delay'])
        kind = str(row.get('kind', 'model')).lower()
        if kind not in ('kernel', 'astraea', 'orca'):
            row['individual_plot'] = _orchestrator_plot_path(
                row, bw, delay)
            skipped += 1
            continue
        output = _plot_path_for_row(row, bw, delay)
        if not output:
            skipped += 1
            continue
        try:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            plot_goodput_run(
                row.get('run_dir', os.path.dirname(output)),
                output,
                row.get('plot_label') or row.get('approach', 'approach'),
                int(float(row.get('run', 0) or 0)),
                schedule_rows,
                int(float(row.get('duration_s', 300) or 300)),
            )
            row['individual_plot'] = output
            plotted += 1
        except Exception as exc:
            failed += 1
            print(f'[resp_bench] individual plot failed for {output}: {exc}', flush=True)
    return plotted, skipped, failed






def _metric_row(approach: dict, bench: dict, run_number: int, schedule: dict,
                checkpoint: str, run_dir: str, state_log: str, schedule_csv: str,
                ep_return, metrics: dict, error: str = '') -> dict:
    duration_s = int(bench['duration_s'])
    kind = str(approach.get('kind', 'model')).lower()
    if kind in ('kernel', 'astraea', 'orca'):
        individual_plot = os.path.join(run_dir, 'run_plot.pdf')
    else:
        individual_plot = os.path.join(
            run_dir,
            f'ep{run_number:06d}_bw{schedule["initial"]["bw"]:.0f}_'
            f'd{schedule["initial"]["delay"]:.0f}_'
            f'n{int(bench.get("flows", 1))}.pdf',
        )
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
        'run': int(run_number),
        'seed': int(schedule['seed']),
        'duration_s': duration_s,
        'change_interval_s': int(bench['change_interval_s']),
        'initial_bw_mbps': schedule['initial']['bw'],
        'initial_rtt_ms': schedule['initial']['delay'],
        'mean_scheduled_bw_mbps': _time_weighted_mean(schedule['rows'], 'bw', duration_s),
        'mean_scheduled_rtt_ms': _time_weighted_mean(schedule['rows'], 'delay', duration_s),
        'schedule_changes': len(schedule['changes']),
        'return': ep_return if ep_return is not None else '',
        'state_log': state_log,
        'schedule_csv': schedule_csv,
        'individual_plot': individual_plot,
        'run_dir': run_dir,
        'checkpoint': checkpoint,
        'error': error,
    }
    row.update(metrics or {})
    for key in METRIC_FIELDS:
        row.setdefault(key, '')
    return row








def _float_values(rows: list, key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ''))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _nanmean_or_none(values: np.ndarray):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _rtt_metrics_from_iperf_files(run_dir: str) -> dict:
    rtt_values = _rtt_samples_from_iperf_files(run_dir)
    arr = np.asarray(rtt_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {}
    return {
        'mean_rtt_ms': float(np.mean(arr)),
        'p95_rtt_ms': float(np.percentile(arr, 95)),
    }


def _rtt_samples_from_iperf_files(run_dir: str) -> list:
    rtt_values = []
    try:
        names = sorted(os.listdir(run_dir))
    except (FileNotFoundError, OSError):
        names = []
    for name in names:
        if not (name.startswith('iperf_flow') and name.endswith('.json')):
            continue
        parsed = _parse_iperf_json(os.path.join(run_dir, name))
        rtt_values.extend(v for _, v in parsed.get('rtt_samples_ms', []))
    return rtt_values


def _enrich_kernel_rtt_rows(rows: list) -> list:
    """Backfill RTT for kernel baselines from saved iperf3 client JSON files.

    Older benchmark runs wrote goodput metrics for BBR/CUBIC but left RTT fields
    empty. The per-run iperf JSON has interval stream RTT samples, so summary
    plotting can recover the RTT CDF without rerunning Mininet.
    """
    enriched = []
    for row in rows:
        out = dict(row)
        needs_rtt = (
            not math.isfinite(_finite_float(out.get('mean_rtt_ms')))
            or not math.isfinite(_finite_float(out.get('p95_rtt_ms')))
        )
        if needs_rtt and str(out.get('kind', '')).lower() in ('kernel', 'astraea'):
            rtt_metrics = _rtt_metrics_from_iperf_files(str(out.get('run_dir', '')))
            for key, value in rtt_metrics.items():
                out[key] = value
        enriched.append(out)
    return enriched


def _recover_kernel_row(row: dict, bench: dict) -> dict:
    """Recover a completed kernel trial whose iperf end summary was unusable."""
    if str(row.get('kind', '')).lower() not in ('kernel', 'astraea'):
        return row
    if not str(row.get('error', '')).strip():
        return row

    run_dir = str(row.get('run_dir', '')).strip()
    flows = int(bench.get('flows', 1))
    duration_s = float(bench['duration_s'])
    measure_interval_s = float(bench.get('measure_interval_s', 1.0))
    minimum_samples = max(
        1, int(math.floor(duration_s / measure_interval_s)) - 1)
    goodputs = []
    all_samples = []
    all_rtt_samples = []
    for flow in range(1, flows + 1):
        parsed = _parse_iperf_json(
            os.path.join(run_dir, f'iperf_flow{flow}.json'))
        samples = parsed.get('samples', [])
        goodput = _finite_float(parsed.get('goodput_mbps'))
        if len(samples) < minimum_samples or not (
                math.isfinite(goodput) and goodput > 0.0):
            return row
        goodputs.append(goodput)
        all_samples.extend(value for _, value in samples)
        all_rtt_samples.extend(
            value for _, value in parsed.get('rtt_samples_ms', []))

    samples_arr = np.asarray(all_samples, dtype=float)
    samples_arr = samples_arr[np.isfinite(samples_arr)]
    rtt_arr = np.asarray(all_rtt_samples, dtype=float)
    rtt_arr = rtt_arr[np.isfinite(rtt_arr)]
    recovered = dict(row)
    recovered.update({
        'steps': int(samples_arr.size),
        'mean_thr_mbps': float(sum(goodputs)),
        'p50_thr_mbps': (
            float(np.percentile(samples_arr, 50))
            if samples_arr.size else math.nan),
        'p05_thr_mbps': (
            float(np.percentile(samples_arr, 5))
            if samples_arr.size else math.nan),
        'mean_rtt_ms': (
            float(np.mean(rtt_arr)) if rtt_arr.size else math.nan),
        'p95_rtt_ms': (
            float(np.percentile(rtt_arr, 95))
            if rtt_arr.size else math.nan),
        'error': '',
    })
    print(
        f'[resp_bench] recovered {recovered.get("approach")} '
        f'run={recovered.get("run")} from saved iperf samples',
        flush=True)
    return recovered


def _plot_cdf(ax, values: np.ndarray, label: str, **plot_kwargs) -> bool:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    values = np.sort(values)
    pct = np.arange(1, values.size + 1, dtype=float) / values.size * 100.0
    kwargs = {'linewidth': 1.4, 'label': label}
    kwargs.update(plot_kwargs)
    ax.plot(values, pct, **kwargs)
    return True


def _plot_histogram_cdf(ax, values: np.ndarray, label: str,
                        bins: int = CDF_BINS, **plot_kwargs) -> bool:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    counts, edges = np.histogram(values, bins=bins)
    pct = np.cumsum(counts, dtype=float) / values.size * 100.0
    kwargs = {'linewidth': 1.4, 'label': label}
    kwargs.update(plot_kwargs)
    ax.plot(edges[:-1], pct, **kwargs)
    return True


def _generated_approach_color(index: int) -> str:
    # Golden-ratio hue spacing keeps extra generated colors from clustering.
    hue = (0.08 + (index * 0.618033988749895)) % 1.0
    saturation = 0.70 if index % 2 == 0 else 0.55
    value = 0.82 if (index // 2) % 2 == 0 else 0.68
    return mcolors.to_hex(mcolors.hsv_to_rgb((hue, saturation, value)))


def _approach_color_palette(size: int) -> list:
    colors = list(APPROACH_COLOR_PALETTE)
    seen = {color.lower() for color in colors}
    index = 0
    while len(colors) < size:
        color = _generated_approach_color(index)
        if color.lower() not in seen:
            colors.append(color)
            seen.add(color.lower())
        index += 1
    return colors[:size]


def _approach_color_map(rows: list) -> dict:
    approaches = sorted({
        str(r.get('approach') or '')
        for r in rows
        if r.get('approach')
    })
    palette = _approach_color_palette(len(approaches))
    return {
        approach: palette[idx]
        for idx, approach in enumerate(approaches)
    }


def _row_run_number(row: dict) -> int:
    try:
        return int(float(row.get('run') or 0))
    except (TypeError, ValueError):
        return 0


def _state_log_basename(row: dict) -> str:
    state_log = str(row.get('state_log') or '').strip()
    if state_log:
        return os.path.basename(state_log)
    run_number = _row_run_number(row)
    if run_number <= 0:
        return ''
    algorithm = str(row.get('algorithm') or '').strip() or _row_algorithm(row)
    if not algorithm:
        return ''
    return f'{algorithm}_state_ep{run_number:06d}.csv'


def _state_log_candidates(row: dict, output_root: str = None) -> list:
    candidates = []
    state_log = str(row.get('state_log') or '').strip()
    if state_log:
        candidates.append(state_log)

    basename = _state_log_basename(row)
    if basename:
        run_dir = str(row.get('run_dir') or '').strip()
        if run_dir:
            candidates.append(os.path.join(run_dir, basename))
        run_number = _row_run_number(row)
        data_folder = str(row.get('data_folder') or row.get('approach') or '').strip()
        if output_root and data_folder and run_number > 0:
            candidates.append(
                os.path.join(output_root, data_folder, f'run{run_number}', basename)
            )

    deduped = []
    seen = set()
    for path in candidates:
        if path and path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _positive_state_log_mean(row: dict, column: str,
                             output_root: str = None) -> float:
    for path in _state_log_candidates(row, output_root):
        cache_key = (os.path.abspath(path), column)
        if cache_key in _STATE_LOG_MEAN_CACHE:
            value = _STATE_LOG_MEAN_CACHE[cache_key]
            if math.isfinite(value):
                return value
            continue

        value = math.nan
        values = []
        times = []
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                if column not in (reader.fieldnames or []):
                    _STATE_LOG_MEAN_CACHE[cache_key] = math.nan
                    continue
                for item in reader:
                    t = _finite_float(item.get('t_s'))
                    sample = _finite_float(item.get(column))
                    if math.isfinite(t) and math.isfinite(sample):
                        times.append(t)
                        values.append(sample)
        except (FileNotFoundError, OSError):
            _STATE_LOG_MEAN_CACHE[cache_key] = math.nan
            continue

        if not values:
            _STATE_LOG_MEAN_CACHE[cache_key] = math.nan
            continue
        t_arr = np.asarray(times, dtype=float)
        v_arr = np.asarray(values, dtype=float)
        if t_arr.size and t_arr.size == v_arr.size:
            mask = t_arr <= t_arr[-1] - 5.0
            if mask.any():
                v_arr = v_arr[mask]
        v_arr = v_arr[np.isfinite(v_arr) & (v_arr > 0.0)]
        if v_arr.size:
            value = float(np.mean(v_arr))
            _STATE_LOG_MEAN_CACHE[cache_key] = value
            return value
        _STATE_LOG_MEAN_CACHE[cache_key] = math.nan
    return math.nan


def _rtt_values_for_plot(rows: list, output_root: str = None) -> np.ndarray:
    values = []
    for row in rows:
        if str(row.get('kind', '')).lower() in ('kernel', 'astraea'):
            value = _finite_float(row.get('mean_rtt_ms'))
        else:
            value = _positive_state_log_mean(row, 'srtt_ms', output_root)
            if not math.isfinite(value):
                value = _finite_float(row.get('mean_srtt_ms'))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _avg_rtt_values_for_plot(rows: list, output_root: str = None) -> np.ndarray:
    values = []
    for row in rows:
        if str(row.get('kind', '')).lower() in ('kernel', 'astraea'):
            value = _finite_float(row.get('mean_rtt_ms'))
        else:
            value = _positive_state_log_mean(row, 'avg_urtt_ms', output_root)
            if not math.isfinite(value):
                value = _finite_float(row.get('mean_rtt_ms'))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _row_algorithm(row: dict) -> str:
    alg = str(row.get('algorithm') or '').strip().lower()
    if alg:
        return alg
    approach = str(row.get('approach') or row.get('data_folder') or '').lower()
    if approach.startswith('mbpo_td3') or 'mbpo_td3' in approach:
        return 'mbpo_td3'
    if approach.startswith('orca_td3') or approach.startswith('td3') or 'orca_td3' in approach:
        return 'td3'
    return ''


def _row_environment(row: dict) -> str:
    return str(row.get('training_environment') or row.get('environment') or '').strip().lower()


def _is_sage_reward_variant(row: dict) -> bool:
    reward = str(row.get('reward') or '').strip().lower()
    if reward.startswith('sage'):
        return True
    label = str(row.get('plot_label') or row.get('approach') or '').lower()
    return 'sage' in label


def _is_multi_addition_sage_variant(row: dict) -> bool:
    reward = str(row.get('reward') or '').strip().lower()
    if reward in {
        'sage_drift_overshoot',
        'sage_drift_sparse',
        'sage_drift_sparse_overshoot',
    }:
        return True
    label = str(row.get('plot_label') or row.get('approach') or '')
    return 'sage' in label.lower() and label.count('+') >= 2


def _is_final_kalman_fit_product(row: dict) -> bool:
    approach = str(row.get('approach') or row.get('data_folder') or '').strip().lower()
    if approach == 'mbpo_td3_kalman_reference_all':
        return True
    return (
        _row_algorithm(row) == 'mbpo_td3'
        and _row_environment(row) == 'dynamic'
        and str(row.get('state') or '').strip().lower() == 'kalman'
        and str(row.get('reward') or '').strip().lower() == 'kalman_fit'
    )


def _schedule_reference_rows(rows: list) -> list:
    refs = []
    seen = set()
    for row in rows:
        bw = _finite_float(row.get('mean_scheduled_bw_mbps'))
        rtt = _finite_float(row.get('mean_scheduled_rtt_ms'))
        if not (math.isfinite(bw) and math.isfinite(rtt)):
            continue
        key = (
            str(row.get('benchmark_environment') or ''),
            str(row.get('seed') or row.get('run') or ''),
            round(bw, 6),
            round(rtt, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            'mean_scheduled_bw_mbps': bw,
            'mean_scheduled_rtt_ms': rtt,
        })
    return refs


def _plot_summary_page(pdf: PdfPages, rows: list, title: str,
                       color_map: dict = None,
                       output_root: str = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.2))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    plot_rows = list(rows)
    schedule_refs = _schedule_reference_rows(rows)
    labels = sorted({r['approach'] for r in plot_rows})
    goodput_plotted = False
    srtt_plotted = False
    avg_rtt_plotted = False
    for label in labels:
        subset = [r for r in plot_rows if r['approach'] == label]
        plot_label = next(
            (r.get('plot_label') for r in subset if r.get('plot_label')),
            label,
        )
        wrapped_label = '\n'.join(textwrap.wrap(plot_label, width=24)) or plot_label
        color = (color_map or {}).get(label)
        goodput_plotted |= _plot_cdf(
            axes[0], _float_values(subset, 'mean_thr_mbps'),
            wrapped_label, color=color)
        srtt_plotted |= _plot_histogram_cdf(
            axes[1], _rtt_values_for_plot(subset, output_root),
            wrapped_label, color=color)
        avg_rtt_plotted |= _plot_histogram_cdf(
            axes[2], _avg_rtt_values_for_plot(subset, output_root),
            wrapped_label, color=color)

    goodput_plotted |= _plot_cdf(
        axes[0],
        _float_values(schedule_refs, 'mean_scheduled_bw_mbps'),
        'Scheduled BW',
        color='black',
        linewidth=1.7,
        zorder=10,
    )
    srtt_plotted |= _plot_histogram_cdf(
        axes[1],
        _float_values(schedule_refs, 'mean_scheduled_rtt_ms'),
        'Base RTT',
        color='black',
        linewidth=1.7,
        zorder=10,
    )
    avg_rtt_plotted |= _plot_histogram_cdf(
        axes[2],
        _float_values(schedule_refs, 'mean_scheduled_rtt_ms'),
        'Base RTT',
        color='black',
        linewidth=1.7,
        zorder=10,
    )

    axes[0].set_xlabel('Average Goodput (Mbps)')
    axes[0].set_ylabel('% of Runs')
    axes[0].set_title('Goodput CDF')
    axes[1].set_xlabel('Average SRTT (ms)')
    axes[1].set_ylabel('Percent of Trials (%)')
    axes[1].set_title('SRTT CDF')
    axes[1].set_xlim(30, 175)
    axes[2].set_xlabel('Average avg_urtt RTT (ms)')
    axes[2].set_ylabel('Percent of Trials (%)')
    axes[2].set_title('Avg RTT CDF')
    axes[2].set_xlim(30, 175)
    if not goodput_plotted:
        axes[0].text(0.5, 0.5, 'No goodput data',
                     transform=axes[0].transAxes, ha='center', va='center')
    if not srtt_plotted:
        axes[1].text(0.5, 0.5, 'No SRTT data',
                     transform=axes[1].transAxes, ha='center', va='center')
    if not avg_rtt_plotted:
        axes[2].text(0.5, 0.5, 'No avg RTT data',
                     transform=axes[2].transAxes, ha='center', va='center')
    for ax in axes:
        ax.grid(True, alpha=0.25)
    # Collect legend entries once and place them below the axes so long approach
    # names never cover the CDF curves.
    unique = {}
    for ax in axes:
        handles, labels_for_ax = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels_for_ax):
            unique.setdefault(label, handle)
    if unique:
        n_items = len(unique)
        ncol = min(6, max(1, n_items))
        legend_rows = int(math.ceil(n_items / float(ncol)))
        bottom = min(0.56, 0.17 + 0.09 * legend_rows)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.88,
                            bottom=bottom, wspace=0.22)
        fig.legend(
            list(unique.values()), list(unique.keys()),
            title='Approach', loc='lower center',
            bbox_to_anchor=(0.5, 0.025), frameon=False,
            fontsize=8.4, title_fontsize=10, ncol=ncol,
            columnspacing=1.0, handlelength=2.0, handletextpad=0.55,
            borderaxespad=0.0,
        )
    else:
        fig.subplots_adjust(left=0.07, right=0.985, top=0.88,
                            bottom=0.12, wspace=0.22)
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_summary(metrics_csv: str, output_pdf: str,
                  specific_keys: set = None) -> None:
    rows = _enrich_kernel_rtt_rows(_read_metrics(metrics_csv))
    completed = [r for r in rows if not r.get('error')]
    if not completed:
        print('[resp_bench] no completed rows to plot', flush=True)
        return
    output_root = os.path.dirname(os.path.abspath(metrics_csv))

    mbpo_rows = [
        r for r in completed
        if _row_algorithm(r) == 'mbpo_td3' and _row_environment(r) == 'dynamic'
    ]
    td3_rows = [
        r for r in completed
        if _row_algorithm(r).endswith('td3') and _row_algorithm(r) != 'mbpo_td3'
    ]
    color_map = _approach_color_map(completed)
    dynamic_rows = [
        r for r in completed
        if _row_environment(r) == 'dynamic'
        and '+' not in str(r.get('plot_label') or r.get('approach') or '')
    ]
    static_rows = [r for r in completed if _row_environment(r) == 'static']
    sage_rows = [r for r in completed if _is_sage_reward_variant(r)]
    sage_additions_vs_final_rows = [
        r for r in completed
        if _is_multi_addition_sage_variant(r) or _is_final_kalman_fit_product(r)
    ]
    specific_rows = [
        r for r in completed if _row_in_specific(r, specific_keys or set())
    ]
    pages = [
        ('All Approaches', completed),
        ('Specific Subset', specific_rows),
        ('MBPO-TD3 Dynamic Approaches', mbpo_rows),
        ('SAGE Reward Variants', sage_rows),
        ('SAGE Additions vs Kalman-Fit', sage_additions_vs_final_rows),
        ('TD3 Approaches', td3_rows),
        ('Dynamic Environment Approaches', dynamic_rows),
        ('Static Environment Approaches', static_rows),
    ]

    tmp_path = f'{output_pdf}.tmp.{os.getpid()}.pdf'
    with PdfPages(tmp_path) as pdf:
        for title, page_rows in pages:
            if page_rows:
                _plot_summary_page(
                    pdf, page_rows, title,
                    color_map=color_map,
                    output_root=output_root,
                )
    os.replace(tmp_path, output_pdf)
    print(f'[resp_bench] summary plot -> {output_pdf}', flush=True)


def _write_summary(metrics_csv: str, output_json: str) -> None:
    rows = [r for r in _enrich_kernel_rtt_rows(_read_metrics(metrics_csv))
            if not r.get('error')]
    summary = {}
    for label in sorted({r['approach'] for r in rows}):
        subset = [r for r in rows if r['approach'] == label]
        summary[label] = {
            'completed_runs': len(subset),
            'mean_goodput_mbps': _nanmean_or_none(_float_values(subset, 'mean_thr_mbps')),
            'mean_p95_rtt_ms': _nanmean_or_none(_float_values(subset, 'p95_rtt_ms')),
            'mean_return': _nanmean_or_none(_float_values(subset, 'return')),
        }
    tmp_path = f'{output_json}.tmp.{os.getpid()}'
    with open(tmp_path, 'w') as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, output_json)


def _collect_approach_rows(output_root: str) -> list:
    return _collect_approach_rows_with_labels(output_root, {})




def _replot_all(output_root: str, bench_cfg: dict = None) -> None:
    label_map = _approach_label_map(bench_cfg or {})
    all_metrics_csv = os.path.join(output_root, 'metrics.csv')
    all_rows = _enrich_kernel_rtt_rows(
        _collect_approach_rows_with_labels(output_root, label_map))
    plotted, skipped, failed = _plot_individual_runs(all_rows)
    _write_rows(all_metrics_csv, METRIC_FIELDS, all_rows)
    _write_summary(all_metrics_csv, os.path.join(output_root, 'summary.json'))
    _plot_summary(all_metrics_csv, os.path.join(output_root, 'responsiveness_summary.pdf'),
                  specific_keys=_specific_plot_keys(bench_cfg or {}))
    _restore_sudo_user_ownership(output_root)
    print(f'[resp_bench] replotted approaches={len({r["approach"] for r in all_rows})} '
          f'rows={len(all_rows)}', flush=True)
    print(f'[resp_bench] individual_plots={plotted} skipped={skipped} failed={failed}',
          flush=True)


def _run_approach(base_cfg: dict, approach: dict, bench: dict, output_root: str,
                  listener_bin: str, python_bin: str, bench_config_path: str,
                  dry_run: bool = False) -> list:
    _validate_approach(approach)
    label = approach['_label']
    kind = str(approach.get('kind', 'model')).lower()
    _ensure_kernel_cc_available(approach)
    checkpoint = ''
    if kind not in ('kernel', 'astraea', 'orca'):
        checkpoint = _resolve_from(os.path.dirname(bench_config_path), str(approach['checkpoint']))
        if not os.path.exists(checkpoint):
            raise SystemExit(f'[resp_bench] checkpoint not found for {label}: {checkpoint}')

    approach_dir = os.path.abspath(os.path.join(output_root, label))
    if dry_run:
        print(f'[resp_bench] dry-run approach={label} output={approach_dir}', flush=True)
        return []

    os.makedirs(approach_dir, exist_ok=True)
    os.makedirs(os.path.join(approach_dir, 'telemetry'), exist_ok=True)
    os.makedirs(os.path.join(approach_dir, 'checkpoints'), exist_ok=True)

    metrics_csv = os.path.join(approach_dir, 'metrics.csv')
    runs = int(bench['runs'])

    # Resume support: keep finished runs, re-run only missing/errored ones.
    rows_by_key = _resume_rows_by_key(metrics_csv, METRIC_FIELDS, _run_key)
    rows_by_key = {
        key: _recover_kernel_row(row, bench)
        for key, row in rows_by_key.items()
    }
    expected = _expected_run_keys(bench)
    done = _completed_keys(rows_by_key) & expected
    rows_by_key = {key: rows_by_key[key] for key in done}
    missing = sorted(expected - done)

    if not missing:
        rows = [rows_by_key[key] for key in sorted(rows_by_key)]
        _write_rows(metrics_csv, METRIC_FIELDS, rows)
        _write_summary(metrics_csv, os.path.join(approach_dir, 'summary.json'))
        _restore_sudo_user_ownership(approach_dir)
        return rows

    approach_meta = {
        'approach': {k: v for k, v in approach.items() if not k.startswith('_')},
        'benchmark': bench,
        'benchmark_config': os.path.abspath(bench_config_path),
        'approach_config': approach.get('_config_path', ''),
        'checkpoint': checkpoint,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    with open(os.path.join(approach_dir, 'run_meta.json'), 'w') as f:
        json.dump(approach_meta, f, indent=2)

    if rows_by_key:
        print(f'[resp_bench] {label} resuming missing={len(missing)}/{runs} '
              f'(already done={len(rows_by_key)})', flush=True)

    total = len(missing)
    n_parallel = max(1, min(int(bench.get('n_parallel', 1)), total))
    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()

    for run_number in missing:
        run_dir = os.path.join(approach_dir, f'run{run_number}')
        _safe_overwrite_dir(run_dir, approach_dir)
        schedule = _sample_schedule(run_number, bench)
        schedule_csv, _ = _write_schedule(run_dir, schedule, bench)
        work_q.put(
            (run_number, schedule, run_dir, schedule_csv)
        )
    for _ in range(n_parallel):
        work_q.put(None)

    astraea_handle = None
    if kind == 'astraea':
        astraea_handle = _start_astraea_listener(approach, bench, approach_dir)

    procs = []
    for instance_id in range(n_parallel):
        proc = multiprocessing.Process(
            target=_approach_slot,
            args=(
                instance_id, work_q, result_q, base_cfg, approach, bench,
                checkpoint, approach_dir, listener_bin, python_bin,
            ),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    completed = 0
    try:
        while completed < total:
            row = result_q.get()
            completed += 1
            rows_by_key[_run_key(row)] = row
            _append_csv(metrics_csv, METRIC_FIELDS, row)
            status = 'failed' if row.get('error') else 'done'
            print(f'[resp_bench] {label} run={row["run"]}/{runs} {status} '
                  f'({completed}/{total})', flush=True)
    finally:
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
        _stop_astraea_listener(astraea_handle)

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    _write_rows(metrics_csv, METRIC_FIELDS, rows)
    _write_summary(metrics_csv, os.path.join(approach_dir, 'summary.json'))
    _restore_sudo_user_ownership(approach_dir)
    return rows
















# --- real Orca (SIGCOMM'20) baseline -----------------------------------------
#
# `kind: orca` runs the *original* Orca C++/TF1 implementation (the
# Aruuni/Orca repo, as installed for cctestbed) rather than the PyTorch
# re-creation. It does NOT use iperf3: the orca-server-mahimahi binary is the
# data *source* (runs on each c{i} sender, spawns a TF1 actor) and clientThr
# is the data *sink* (runs on each x{i} receiver). clientThr prints, between
# `----START----`/`----END----` markers, one CSV line per `interval`:
#
#     tick , <mbps>e+06 , total_bytes , avg_mbps
#
# We parse that into the same per-flow `csvs/x{f}.csv` (`time,bandwidth`
# Mbps) that the iperf3 path produces, so all downstream binning/plotting is
# unchanged. The launch mirrors cctestbed's start_orca_sender/receiver
# (sender.sh / receiver.sh wrapped in `sudo -u <user>`).































def _orca_has_ss(run_dir: str, flow_plan: list) -> bool:
    return any(os.path.exists(_orca_ss_log_path(run_dir, int(i['flow'])))
               for i in flow_plan)


def _run_kernel_baseline(instance_id: int, approach: dict, bench: dict,
                         checkpoint: str, run_number: int, schedule: dict,
                         run_dir: str, schedule_csv: str) -> dict:
    label = _approach_plot_label(approach)
    kind = str(approach.get('kind', 'kernel')).lower()
    # `kind: astraea` reuses this iperf-driven flow; the astraea_listener
    # running in the parent process auto-attaches by CC name and drives
    # inference until each flow goes idle.
    kernel_cc = 'astraea' if kind == 'astraea' else str(approach['kernel_cc'])
    cport = int(bench.get('kernel_cport_base', 33000)) + int(instance_id) * 100
    flows = int(bench.get('flows', 1))
    duration_s = int(bench['duration_s'])
    measure_interval_s = float(bench.get('measure_interval_s', 1.0))
    tmp_paths = [f'/tmp/iperf_{cport}_{idx}.json' for idx in range(1, flows + 1)]
    for path in tmp_paths:
        _safe_unlink(path)

    error = ''
    try:
        print(f'[resp_bench] slot={instance_id} {approach["_label"]} '
              f'run={run_number} seed={schedule["seed"]} -C {kernel_cc}', flush=True)
        env = MininetEnv(
            n=flows,
            bw=float(schedule['initial']['bw']),
            delay=float(schedule['initial']['delay']),
            bdp_mult=float(bench['bdp_mult']),
            duration=duration_s,
            cport=cport,
            cc_algo=kernel_cc,
            instance_id=instance_id,
        )
        try:
            env.start()
            env.setup_environment(link_schedule=schedule['changes'])
            env.start_episode(monitor_interval=measure_interval_s)
            env.wait()
        finally:
            env.stop()
            time.sleep(0.5)

        csv_dir = os.path.join(run_dir, 'csvs')
        goodputs = []
        all_samples = []
        all_rtt_samples = []
        for idx, tmp_path in enumerate(tmp_paths, start=1):
            dst = os.path.join(run_dir, f'iperf_flow{idx}.json')
            _copy_if_exists(tmp_path, dst)
            parsed = _parse_iperf_json(
                dst,
                os.path.join(csv_dir, f'x{idx}.csv'),
                measure_interval_s=measure_interval_s,
            )
            goodputs.append(parsed['goodput_mbps'])
            all_samples.extend(v for _, v in parsed['samples'])
            all_rtt_samples.extend(v for _, v in parsed.get('rtt_samples_ms', []))

        goodput = sum(v for v in goodputs if math.isfinite(v))
        samples_arr = np.asarray(all_samples, dtype=float)
        samples_arr = samples_arr[np.isfinite(samples_arr)]
        rtt_arr = np.asarray(all_rtt_samples, dtype=float)
        rtt_arr = rtt_arr[np.isfinite(rtt_arr)]
        if goodput <= 0:
            error = 'missing or invalid iperf goodput'

        individual_plot = os.path.join(run_dir, 'run_plot.pdf')
        if _as_bool(bench.get('plot_episodes'), default=True):
            try:
                plot_goodput_run(
                    run_dir, individual_plot, label, run_number,
                    schedule['rows'], duration_s)
            except Exception as plot_exc:
                print(f'[resp_bench] kernel run plot failed: {plot_exc}', flush=True)

        metrics = {
            'duration_s': duration_s,
            'steps': int(samples_arr.size),
            'mean_thr_mbps': goodput if goodput > 0 else math.nan,
            'p50_thr_mbps': float(np.percentile(samples_arr, 50)) if samples_arr.size else math.nan,
            'p05_thr_mbps': float(np.percentile(samples_arr, 5)) if samples_arr.size else math.nan,
            'mean_rtt_ms': float(np.mean(rtt_arr)) if rtt_arr.size else math.nan,
            'p95_rtt_ms': float(np.percentile(rtt_arr, 95)) if rtt_arr.size else math.nan,
            'individual_plot': individual_plot,
        }
        return _metric_row(
            approach, bench, run_number, schedule, checkpoint,
            run_dir, '', schedule_csv, None, metrics, error=error)
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'
        print(f'[resp_bench] slot={instance_id} {approach["_label"]} '
              f'run={run_number} failed: {exc}', flush=True)
        return _metric_row(
            approach, bench, run_number, schedule, checkpoint,
            run_dir, '', schedule_csv, None, {}, error=error)


def _run_orca_baseline(instance_id: int, approach: dict, bench: dict,
                       checkpoint: str, run_number: int, schedule: dict,
                       run_dir: str, schedule_csv: str) -> dict:
    """`kind: orca` responsiveness trial: same dumbbell + dynamic link
    schedule as the kernel path, but driven by the real Orca server/clientThr
    instead of iperf3. RTT is unavailable from clientThr (mean/p95 RTT are
    NaN for Orca)."""
    label = _approach_plot_label(approach)
    settings = _orca_settings(approach, bench)
    flows = int(bench.get('flows', 1))
    duration_s = int(bench['duration_s'])
    measure_interval_s = float(bench.get('measure_interval_s', 1.0))
    flow_plan = [{'flow': idx, 'start': 0.0, 'duration': float(duration_s),
                  'end': float(duration_s)} for idx in range(1, flows + 1)]

    error = ''
    try:
        print(f'[resp_bench] slot={instance_id} {approach["_label"]} '
              f'run={run_number} seed={schedule["seed"]} (real-orca)',
              flush=True)
        env = MininetEnv(
            n=flows,
            bw=float(schedule['initial']['bw']),
            delay=float(schedule['initial']['delay']),
            bdp_mult=float(bench['bdp_mult']),
            duration=duration_s,
            cport=int(bench.get('kernel_cport_base', 33000)) + int(instance_id) * 100,
            cc_algo='cubic',
            instance_id=instance_id,
        )
        sched_stop = threading.Event()
        sched_thread = None
        try:
            env.start()
            env.setup_environment()
            episode_start = time.monotonic()
            sched_thread = threading.Thread(
                target=_run_link_schedule,
                args=(env, schedule['changes'], episode_start, sched_stop),
                daemon=True,
            )
            sched_thread.start()
            _run_orca_on_env(env, settings, flow_plan, instance_id,
                             run_dir, duration_s)
        finally:
            sched_stop.set()
            if sched_thread:
                sched_thread.join(timeout=2)
            env.stop()
            time.sleep(0.5)

        csv_dir = os.path.join(run_dir, 'csvs')
        goodputs = []
        all_samples = []
        for idx in range(1, flows + 1):
            parsed = _parse_orca_client_output(
                os.path.join(run_dir, f'orca_x{idx}.log'),
                os.path.join(csv_dir, f'x{idx}.csv'),
                measure_interval_s=measure_interval_s, settings=settings)
            goodputs.append(parsed['goodput_mbps'])
            all_samples.extend(v for _, v in parsed['samples'])

        goodput = sum(v for v in goodputs if math.isfinite(v))
        samples_arr = np.asarray(all_samples, dtype=float)
        samples_arr = samples_arr[np.isfinite(samples_arr)]
        if goodput <= 0:
            error = 'missing or invalid Orca goodput'

        # RTT from the ss sidecar (kernel srtt of the Orca socket); NaN only
        # if ss was disabled or produced nothing.
        rtt_arr = np.asarray(
            [r for _, r in _orca_ss_rtt_samples(run_dir, flow_plan)],
            dtype=float)
        rtt_arr = rtt_arr[np.isfinite(rtt_arr)]

        individual_plot = os.path.join(run_dir, 'run_plot.pdf')
        if _as_bool(bench.get('plot_episodes'), default=True):
            try:
                plot_goodput_run(
                    run_dir, individual_plot, label, run_number,
                    schedule['rows'], duration_s)
            except Exception as plot_exc:
                print(f'[resp_bench] orca run plot failed: {plot_exc}', flush=True)

        metrics = {
            'duration_s': duration_s,
            'steps': int(samples_arr.size),
            'mean_thr_mbps': goodput if goodput > 0 else math.nan,
            'p50_thr_mbps': float(np.percentile(samples_arr, 50)) if samples_arr.size else math.nan,
            'p05_thr_mbps': float(np.percentile(samples_arr, 5)) if samples_arr.size else math.nan,
            'mean_rtt_ms': float(np.mean(rtt_arr)) if rtt_arr.size else math.nan,
            'p95_rtt_ms': float(np.percentile(rtt_arr, 95)) if rtt_arr.size else math.nan,
            'individual_plot': individual_plot,
        }
        return _metric_row(
            approach, bench, run_number, schedule, checkpoint,
            run_dir, '', schedule_csv, None, metrics, error=error)
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'
        print(f'[resp_bench] slot={instance_id} {approach["_label"]} '
              f'run={run_number} failed: {exc}', flush=True)
        return _metric_row(
            approach, bench, run_number, schedule, checkpoint,
            run_dir, '', schedule_csv, None, {}, error=error)


def _approach_slot(instance_id: int, work_q, result_q, base_cfg: dict,
                   approach: dict, bench: dict, checkpoint: str,
                   approach_dir: str, listener_bin: str, python_bin: str) -> None:
    while True:
        item = work_q.get()
        if item is None:
            break
        run_number, schedule, run_dir, schedule_csv = item
        kind = str(approach.get('kind', 'model')).lower()
        if kind in ('kernel', 'astraea'):
            row = _run_kernel_baseline(
                instance_id, approach, bench, checkpoint,
                run_number, schedule, run_dir, schedule_csv)
            result_q.put(row)
            continue
        if kind == 'orca':
            row = _run_orca_baseline(
                instance_id, approach, bench, checkpoint,
                run_number, schedule, run_dir, schedule_csv)
            result_q.put(row)
            continue

        cfg = _prepare_runtime_cfg(
            base_cfg, approach, checkpoint, approach_dir, run_dir,
            plot_episodes=_as_bool(bench.get('plot_episodes'), default=False),
        )
        if _as_bool(bench.get('deterministic_inference'), default=True):
            _deterministic_env(cfg)

        with open(os.path.join(run_dir, 'config.resolved.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        ecfg = {
            'bw': float(schedule['initial']['bw']),
            'delay': float(schedule['initial']['delay']),
            'bdp_mult': float(bench['bdp_mult']),
            'flows': int(bench['flows']),
            'duration': int(bench['duration_s']),
            'start_delays': [0.0] * int(bench['flows']),
            'flow_durations': [float(bench['duration_s'])] * int(bench['flows']),
            'link_schedule': schedule['changes'],
            'environment': str(bench['name']),
            'training_environment': str(approach['environment']),
        }
        alg_name = str(approach['algorithm'])
        state_log = os.path.join(run_dir, f'{alg_name}_state_ep{run_number:06d}.csv')
        try:
            print(f'[resp_bench] slot={instance_id} {approach["_label"]} '
                  f'run={run_number} seed={schedule["seed"]}', flush=True)
            ep_return, _, _ = run_episode(
                cfg, ecfg, run_number, listener_bin, python_bin,
                '', '', instance_id)
            metrics = _state_log_metrics(state_log, int(bench['flows']), ep_return)
            row = _metric_row(
                approach, bench, run_number, schedule, checkpoint,
                run_dir, state_log, schedule_csv, ep_return, metrics)
        except Exception as exc:
            error = f'{exc}\n{traceback.format_exc()}'
            print(f'[resp_bench] slot={instance_id} {approach["_label"]} '
                  f'run={run_number} failed: {exc}', flush=True)
            row = _metric_row(
                approach, bench, run_number, schedule, checkpoint,
                run_dir, state_log, schedule_csv, None, {}, error=error)
        result_q.put(row)


def main():
    ap = argparse.ArgumentParser(
        description='Run fixed-output random BW/RTT responsiveness benchmarks.')
    ap.add_argument('--config', default=_DEFAULT_CONFIG,
                    help='benchmark config yaml (defaults to the multi-flow variant)')
    ap.add_argument('--approach', action='append', default=None,
                    help='approach data_folder/name to run; may be passed more than once')
    ap.add_argument('--plot-only', action='store_true',
                    help='rebuild baseline run plots, metrics.csv, summary.json, and responsiveness_summary.pdf from existing approach folders')
    ap.add_argument('--dry-run', action='store_true',
                    help='resolve config and print selected approaches without running Mininet')
    args = ap.parse_args()

    bench_config_path = os.path.abspath(args.config)
    bench_cfg = _load_benchmark_config(bench_config_path)
    bench_dir = os.path.dirname(bench_config_path)

    output_root = bench_cfg.get('output_root', 'data')
    if not os.path.isabs(str(output_root)):
        output_root = os.path.abspath(os.path.join(bench_dir, str(output_root)))
    os.makedirs(output_root, exist_ok=True)

    if args.plot_only:
        _replot_all(output_root, bench_cfg)
        return

    bench = _benchmark_cfg(bench_cfg)
    selected = {_slug(value) for value in (args.approach or [])}
    configured_approaches = _configured_approaches(bench_cfg, selected or None)
    _validate_unique_data_folders(configured_approaches)
    if not configured_approaches:
        available_labels = sorted({
            key
            for raw in (bench_cfg.get('approaches') or [])
            for key in _approach_selection_keys(raw or {})
        })
        available = ', '.join(available_labels)
        if selected:
            missing = ', '.join(sorted(selected - set(available_labels)))
            raise SystemExit(
                '[resp_bench] requested approach is not listed in benchmark config'
                + (f': {missing}' if missing else '')
                + (f'; available: {available}' if available else '')
            )
        raise SystemExit(
            '[resp_bench] no approaches listed in benchmark config'
            + (f'; available: {available}' if available else '')
        )
    approaches, skipped = _split_approaches_by_existing_data(
        configured_approaches, output_root, bench)

    print(f'[resp_bench] output_root={output_root}', flush=True)
    print(f'[resp_bench] benchmark={bench["name"]} runs={bench["runs"]}', flush=True)
    print(f'[resp_bench] n_parallel={bench.get("n_parallel", 1)}', flush=True)
    if skipped:
        print('[resp_bench] skipping existing data='
              + ', '.join(a['_label'] for a in skipped), flush=True)
    print('[resp_bench] approaches_to_run='
          + (', '.join(a['_label'] for a in approaches) if approaches else '(none)'),
          flush=True)

    if args.dry_run:
        for approach in skipped:
            _, approach_config_path = _load_approach_runtime_config(
                approach, bench_config_path)
            print(f'[resp_bench] dry-run would skip existing data: '
                  f'{os.path.join(output_root, approach["_label"])}'
                  + (f' config={approach_config_path}'
                     if approach_config_path else ''),
                  flush=True)
        for approach in approaches:
            _validate_approach(approach)
            _, approach_config_path = _load_approach_runtime_config(
                approach, bench_config_path)
            _sample_schedule(1, bench)
            print(f'[resp_bench] dry-run would run into empty data folder: '
                  f'{os.path.join(output_root, approach["_label"])}'
                  + (f' config={approach_config_path}'
                     if approach_config_path else ''),
                  flush=True)
        return

    if not approaches:
        print('[resp_bench] no approaches need running; rebuilding plots from existing data',
              flush=True)
        _replot_all(output_root, bench_cfg)
        all_metrics_csv = os.path.join(output_root, 'metrics.csv')
        print(f'[resp_bench] done metrics={all_metrics_csv}', flush=True)
        return

    failed_trials = 0
    try:
        for approach in approaches:
            base_cfg, approach_config_path = _load_approach_runtime_config(
                approach, bench_config_path)
            paths = base_cfg.get('paths', {}) or {}
            kind = str(approach.get('kind', 'model')).lower()
            if kind not in ('kernel', 'astraea', 'orca') and (
                    'listener' not in paths or 'py' not in paths):
                raise SystemExit(
                    f'[resp_bench] approach config for '
                    f'{approach["_label"]} must define '
                    'paths.listener and paths.py')
            listener_bin = (
                _resolve_repo_path(paths['listener'])
                if 'listener' in paths else '')
            python_bin = (
                _resolve_repo_path(paths['py']) if 'py' in paths else '')
            if approach_config_path:
                print(
                    f'[resp_bench] {approach["_label"]} '
                    f'config={approach_config_path}',
                    flush=True)
            try:
                rows = _run_approach(
                    base_cfg, approach, bench, output_root,
                    listener_bin, python_bin, bench_config_path,
                    dry_run=False)
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
            f'[resp_bench] {failed_trials} trial(s) failed')

    _replot_all(output_root, bench_cfg)
    all_metrics_csv = os.path.join(output_root, 'metrics.csv')
    print(f'[resp_bench] done metrics={all_metrics_csv}', flush=True)


if __name__ == '__main__':
    main()
