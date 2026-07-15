#!/usr/bin/env python3
"""Four-flow staggered-arrival convergence benchmark (run-averaged).

4 flows share one bottleneck. Flow i joins at (i-1)*flow_join_interval_s and
runs for flow_duration_s seconds, so by default:

    flow1 0->100s   flow2 25->125s   flow3 50->150s   flow4 75->175s

`runs` independent runs are executed per approach and averaged. Goodput is the
iperf3 receiver goodput (same source for kernel, astraea and model kinds).

The summary PDF is a single page with one panel per approach (one approach per
data folder), each panel showing the 4 per-flow goodput traces averaged across
runs (mean line + std band) on a shared time axis and a log goodput axis --
i.e. the Fig.13-style convergence figure.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
import traceback
import warnings
import multiprocessing

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import numpy as np
import yaml


_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
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
    _orca_ss_rtt_samples,
    _parse_iperf_json,
    _prepare_runtime_cfg,
    _resolve_from,
    _restore_sudo_user_ownership,
    _resume_rows_by_key,
    _run_orca_on_env,
    _safe_overwrite_dir,
    _safe_unlink,
    _slug,
    _specific_plot_keys,
    _start_astraea_listener,
    _stop_astraea_listener,
    _validate_approach,
    _validate_unique_data_folders,
    _write_flow_plan,
    _write_rows,
    _write_schedule,
)


GOODPUT_SOURCE = 'iperf3_receiver'

CONV_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'kind', 'algorithm', 'kernel_cc',
    'state', 'reward', 'training_environment', 'benchmark_environment',
    'run', 'seed', 'duration_s', 'bw_mbps', 'rtt_ms', 'flows',
    'flow_join_interval_s', 'flow_duration_s',
    'mean_total_goodput_mbps', 'mean_jain_fairness',
    'mean_abs_fair_share_error_mbps', 'p95_abs_fair_share_error_mbps',
    'goodput_source', 'schedule_csv', 'flow_schedule_csv', 'state_logs',
    'individual_plot', 'run_dir', 'checkpoint', 'error',
]


def _conv_cfg(cfg: dict) -> dict:
    bench = dict(cfg.get('benchmark') or {})
    bench.setdefault('name', 'convergence_4flow_join25_stay100_175s')
    bench.setdefault('runs', 3)
    bench.setdefault('n_parallel', 5)
    bench.setdefault('flows', 4)
    bench.setdefault('flow_join_interval_s', 25)
    bench.setdefault('flow_duration_s', 100)
    flows = int(bench['flows'])
    auto_duration = (flows - 1) * int(bench['flow_join_interval_s']) + int(
        bench['flow_duration_s'])
    bench.setdefault('duration_s', auto_duration)
    bench.setdefault('bw_mbps', 100)
    bench.setdefault('rtt_ms', 50)
    # (bw, rtt) sweep: every flow in a given experiment sees the same
    # bottleneck bw/RTT; each combination is its own experiment, all plotted
    # together. Fall back to the scalar bw_mbps/rtt_ms when lists are absent.
    bench.setdefault('bw_mbps_values', [bench['bw_mbps']])
    bench.setdefault('rtt_ms_values', [bench['rtt_ms']])
    bench['bw_mbps_values'] = [float(v) for v in bench['bw_mbps_values']]
    bench['rtt_ms_values'] = [float(v) for v in bench['rtt_ms_values']]
    bench['bw_mbps'] = float(bench['bw_mbps_values'][0])
    bench['rtt_ms'] = float(bench['rtt_ms_values'][0])
    bench.setdefault('bdp_mult', 4)
    bench.setdefault('measure_interval_s', 1.0)
    bench.setdefault('kernel_cport_base', 44000)
    bench.setdefault('plot_episodes', True)
    bench.setdefault('deterministic_inference', True)
    bench.setdefault('seed_offset', 0)
    bench.setdefault('summary_log_yscale', False)
    bench.setdefault('summary_ymin_mbps', 1.0)
    bench['change_interval_s'] = 0
    return bench


def _flow_plan(bench: dict) -> list:
    flows = int(bench['flows'])
    join_every = int(bench['flow_join_interval_s'])
    flow_duration = float(bench['flow_duration_s'])
    plan = []
    for idx in range(flows):
        start = float(idx * join_every)
        plan.append({
            'flow': idx + 1,
            'start': start,
            'duration': flow_duration,
            'end': start + flow_duration,
        })
    return plan


def _grid_cells(bench: dict) -> list:
    """All (bw, rtt) experiment conditions; every flow in a cell sees this
    same bottleneck bw/RTT."""
    cells = []
    for rtt in bench['rtt_ms_values']:
        for bw in bench['bw_mbps_values']:
            cells.append({'bw': float(bw), 'rtt': float(rtt)})
    return cells


def _cell_name(bw: float, rtt: float) -> str:
    return f'bw{int(round(float(bw)))}_rtt{int(round(float(rtt)))}'


def _static_schedule(bench: dict, bw: float, rtt: float) -> dict:
    return {
        'seed': 1,
        'initial': {'bw': float(bw), 'delay': float(rtt)},
        'changes': [],
        'rows': [{'t': 0, 'bw': float(bw), 'delay': float(rtt)}],
    }


def _seed_for(bench: dict, cell_idx: int, run_number: int) -> int:
    return (int(bench.get('seed_offset', 0))
            + int(cell_idx) * 1000 + int(run_number))


# --- iperf3 receiver goodput plumbing (same source for every kind) -----------

def _iperf_client_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_{int(cport)}_{int(flow)}.json'


def _iperf_receiver_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_server_{int(cport)}_{int(flow)}.json'


def _clear_iperf_tmp_outputs(cport: int, n_flows: int) -> None:
    for flow in range(1, int(n_flows) + 1):
        _safe_unlink(_iperf_client_tmp(cport, flow))
        _safe_unlink(_iperf_receiver_tmp(cport, flow))


def _copy_receiver_iperf_outputs(cport: int, run_dir: str, flow_plan: list,
                                 bench: dict) -> str:
    csv_dir = os.path.join(run_dir, 'csvs')
    errors = []
    for item in flow_plan:
        flow = int(item['flow'])
        receiver_tmp = _iperf_receiver_tmp(cport, flow)
        receiver_dst = os.path.join(run_dir, f'iperf_receiver_flow{flow}.json')
        client_tmp = _iperf_client_tmp(cport, flow)
        client_dst = os.path.join(run_dir, f'iperf_client_flow{flow}.json')

        _copy_if_exists(client_tmp, client_dst)
        if not _copy_if_exists(receiver_tmp, receiver_dst):
            errors.append(f'missing receiver iperf JSON for flow {flow}: {receiver_tmp}')
            continue
        parsed = _parse_iperf_json(
            receiver_dst, os.path.join(csv_dir, f'x{flow}.csv'),
            measure_interval_s=float(bench.get('measure_interval_s', 1.0)),
        )
        if not parsed.get('samples'):
            errors.append(f'no receiver iperf interval samples for flow {flow}')
    return '; '.join(errors)


def _receiver_iperf_flows(run_dir: str, flow_plan: list) -> list:
    return _kernel_flows(run_dir, flow_plan)


def _flows_have_receiver_goodput(flows: list, n_flows: int) -> bool:
    if len(flows) < int(n_flows):
        return False
    for flow in flows[:int(n_flows)]:
        thr = np.asarray(flow.get('thr', []), dtype=float)
        if not np.isfinite(thr).any():
            return False
    return True


def _metrics_from_series(series: dict) -> dict:
    def mean_finite(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else math.nan

    def pct_finite(values, pct):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.percentile(arr, pct)) if arr.size else math.nan

    duration = len(series['time'])
    final_start = max(0, duration - 20)
    idx = np.arange(len(series['active_count']))
    active = (series['active_count'] > 0) & (idx >= final_start)
    return {
        'mean_total_goodput_mbps': mean_finite(series['total'][active]),
        'mean_jain_fairness': mean_finite(series['jain'][active]),
        'mean_abs_fair_share_error_mbps': mean_finite(series['fair_error'][active]),
        'p95_abs_fair_share_error_mbps': pct_finite(series['fair_error'][active], 95),
        'goodput_source': GOODPUT_SOURCE,
    }


def _prepare_runtime_cfg_slot(base_cfg: dict, approach: dict,
                              checkpoint: str, run_dir: str) -> dict:
    scratch_dir = os.path.join(
        '/tmp', f'olympus_conv4_runtime_{os.getpid()}', approach['_label'])
    cfg = _prepare_runtime_cfg(
        base_cfg, approach, checkpoint, scratch_dir, run_dir,
        plot_episodes=True)
    cfg['outputs']['run_dir'] = scratch_dir
    cfg['outputs']['checkpoints_dir'] = os.path.join(scratch_dir, 'checkpoints')
    cfg['outputs']['telemetry_dir'] = os.path.join(scratch_dir, 'telemetry')
    cfg['training']['log_path'] = os.path.join(
        scratch_dir, 'telemetry', 'benchmark_metrics.csv')
    return cfg


def _row(approach: dict, bench: dict, bw: float, rtt: float,
         run_number: int, seed: int,
         run_dir: str, schedule_csv: str, flow_schedule_csv: str,
         checkpoint: str, state_logs: list, individual_plot: str,
         metrics: dict, error: str = '') -> dict:
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
        'seed': int(seed),
        'duration_s': int(bench['duration_s']),
        'bw_mbps': float(bw),
        'rtt_ms': float(rtt),
        'flows': len(_flow_plan(bench)),
        'flow_join_interval_s': int(bench['flow_join_interval_s']),
        'flow_duration_s': int(bench['flow_duration_s']),
        'schedule_csv': schedule_csv,
        'flow_schedule_csv': flow_schedule_csv,
        'state_logs': ';'.join(state_logs),
        'individual_plot': individual_plot,
        'run_dir': run_dir,
        'checkpoint': checkpoint,
        'error': error,
    }
    row.update(metrics or {})
    for key in CONV_FIELDS:
        row.setdefault(key, '')
    return row


def _run_kernel_trial(instance_id: int, approach: dict, bench: dict,
                      bw: float, rtt: float, run_number: int, seed: int,
                      run_dir: str, schedule_csv: str,
                      flow_schedule_csv: str) -> dict:
    flow_plan = _flow_plan(bench)
    cport = int(bench.get('kernel_cport_base', 44000)) + int(instance_id) * 100
    _clear_iperf_tmp_outputs(cport, len(flow_plan))

    kind = str(approach.get('kind', 'kernel')).lower()
    cc_algo = 'astraea' if kind == 'astraea' else str(approach['kernel_cc'])

    error = ''
    flows = []
    metrics = {}
    try:
        print(f'[conv4] slot={instance_id} {approach["_label"]} '
              f'bw={bw:g} rtt={rtt:g} run={run_number} -C {cc_algo}', flush=True)
        env = MininetEnv(
            n=len(flow_plan),
            bw=float(bw),
            delay=float(rtt),
            bdp_mult=float(bench['bdp_mult']),
            duration=int(bench['duration_s']),
            cport=cport,
            cc_algo=cc_algo,
            instance_id=instance_id,
        )
        try:
            env.start()
            env.setup_environment()
            env.start_episode(
                monitor_interval=float(bench.get('measure_interval_s', 1.0)),
                start_delays=[p['start'] for p in flow_plan],
                flow_durations=[p['duration'] for p in flow_plan],
            )
            env.wait()
        finally:
            env.stop()
            time.sleep(0.5)

        parse_error = _copy_receiver_iperf_outputs(cport, run_dir, flow_plan, bench)
        flows = _receiver_iperf_flows(run_dir, flow_plan)
        if parse_error:
            error = parse_error
        elif not _flows_have_receiver_goodput(flows, len(flow_plan)):
            error = 'missing receiver iperf goodput samples'
        else:
            metrics = _metrics_from_series(_binned_series(flows, flow_plan, {
                'duration_s': int(bench['duration_s']),
                'bw_mbps': float(bw),
            }))
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'

    plot_path = os.path.join(run_dir, 'convergence_run.pdf')
    if flows and _as_bool(bench.get('plot_episodes'), default=True):
        try:
            _plot_run(flows, flow_plan, {**bench, 'bw_mbps': float(bw)},
                      f'{_approach_plot_label(approach)} '
                      f'bw={bw:g} rtt={rtt:g} run={run_number}', plot_path)
        except Exception as plot_exc:
            print(f'[conv4] run plot failed for {plot_path}: {plot_exc}', flush=True)

    return _row(approach, bench, bw, rtt, run_number, seed, run_dir,
                schedule_csv, flow_schedule_csv, '', [], plot_path,
                metrics, error)


def _run_orca_trial(instance_id: int, approach: dict, bench: dict,
                    bw: float, rtt: float, run_number: int, seed: int,
                    run_dir: str, schedule_csv: str,
                    flow_schedule_csv: str) -> dict:
    """Real Orca (SIGCOMM'20) trial: same dumbbell as the kernel path, but
    driven by orca-server-mahimahi/clientThr instead of iperf3. Receiver
    goodput is parsed into the same csvs/x{f}.csv, so binning/plotting and
    the completion check (GOODPUT_SOURCE) are unchanged."""
    flow_plan = _flow_plan(bench)
    settings = _orca_settings(approach, bench)
    cport = int(bench.get('kernel_cport_base', 44000)) + int(instance_id) * 100

    error = ''
    flows = []
    metrics = {}
    try:
        print(f'[conv4] slot={instance_id} {approach["_label"]} '
              f'bw={bw:g} rtt={rtt:g} run={run_number} (real-orca)', flush=True)
        env = MininetEnv(
            n=len(flow_plan),
            bw=float(bw),
            delay=float(rtt),
            bdp_mult=float(bench['bdp_mult']),
            duration=int(bench['duration_s']),
            cport=cport,
            cc_algo='cubic',
            instance_id=instance_id,
        )
        try:
            env.start()
            env.setup_environment()
            _run_orca_on_env(env, settings, flow_plan, instance_id,
                             run_dir, int(bench['duration_s']))
        finally:
            env.stop()
            time.sleep(0.5)

        parse_error = _orca_receiver_to_csvs(
            run_dir, flow_plan,
            measure_interval_s=float(bench.get('measure_interval_s', 1.0)),
            settings=settings)
        flows = _receiver_iperf_flows(run_dir, flow_plan)
        if parse_error:
            error = parse_error
        elif not _flows_have_receiver_goodput(flows, len(flow_plan)):
            error = 'missing Orca receiver goodput samples'
        else:
            metrics = _metrics_from_series(_binned_series(flows, flow_plan, {
                'duration_s': int(bench['duration_s']),
                'bw_mbps': float(bw),
            }))
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'

    plot_path = os.path.join(run_dir, 'convergence_run.pdf')
    if flows and _as_bool(bench.get('plot_episodes'), default=True):
        try:
            _plot_run(flows, flow_plan, {**bench, 'bw_mbps': float(bw)},
                      f'{_approach_plot_label(approach)} '
                      f'bw={bw:g} rtt={rtt:g} run={run_number}', plot_path)
        except Exception as plot_exc:
            print(f'[conv4] run plot failed for {plot_path}: {plot_exc}', flush=True)

    return _row(approach, bench, bw, rtt, run_number, seed, run_dir,
                schedule_csv, flow_schedule_csv, '', [], plot_path,
                metrics, error)


def _run_model_trial(instance_id: int, approach: dict, bench: dict,
                     bw: float, rtt: float, base_cfg: dict, checkpoint: str,
                     listener_bin: str, python_bin: str, run_number: int,
                     seed: int, run_dir: str, schedule_csv: str,
                     flow_schedule_csv: str) -> dict:
    flow_plan = _flow_plan(bench)
    cfg = _prepare_runtime_cfg_slot(base_cfg, approach, checkpoint, run_dir)
    cfg['listener_single_flow'] = True
    cport = int(cfg.get('cport_base', 21000)) + int(instance_id) * 100
    _clear_iperf_tmp_outputs(cport, len(flow_plan))
    if _as_bool(bench.get('deterministic_inference'), default=True):
        _deterministic_env(cfg)
    with open(os.path.join(run_dir, 'config.resolved.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    ecfg = {
        'bw': float(bw),
        'delay': float(rtt),
        'bdp_mult': float(bench['bdp_mult']),
        'flows': len(flow_plan),
        'duration': int(bench['duration_s']),
        'start_delays': [p['start'] for p in flow_plan],
        'flow_durations': [p['duration'] for p in flow_plan],
        'measure_interval_s': float(bench.get('measure_interval_s', 1.0)),
        'per_flow_state_logs': True,
        'unique_cports': False,
        'link_schedule': [],
        'environment': str(bench['name']),
        'training_environment': str(approach['environment']),
    }

    error = ''
    flows = []
    metrics = {}
    try:
        print(f'[conv4] slot={instance_id} {approach["_label"]} '
              f'bw={bw:g} rtt={rtt:g} run={run_number}', flush=True)
        run_episode_auto(cfg, ecfg, 1, listener_bin, python_bin, '', '', instance_id)
        parse_error = _copy_receiver_iperf_outputs(cport, run_dir, flow_plan, bench)
        flows = _receiver_iperf_flows(run_dir, flow_plan)
        if parse_error:
            error = parse_error
        elif not _flows_have_receiver_goodput(flows, len(flow_plan)):
            error = 'missing receiver iperf goodput samples'
        else:
            metrics = _metrics_from_series(_binned_series(flows, flow_plan, {
                'duration_s': int(bench['duration_s']),
                'bw_mbps': float(bw),
            }))
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'

    state_logs = sorted(glob.glob(os.path.join(
        run_dir, f'{approach["algorithm"]}_state_ep000001_a*.csv')))
    plot_path = os.path.join(run_dir, 'convergence_run.pdf')
    if flows and _as_bool(bench.get('plot_episodes'), default=True):
        try:
            _plot_run(flows, flow_plan, {**bench, 'bw_mbps': float(bw)},
                      f'{_approach_plot_label(approach)} '
                      f'bw={bw:g} rtt={rtt:g} run={run_number}', plot_path)
        except Exception as plot_exc:
            print(f'[conv4] run plot failed for {plot_path}: {plot_exc}', flush=True)

    return _row(approach, bench, bw, rtt, run_number, seed, run_dir,
                schedule_csv, flow_schedule_csv, checkpoint, state_logs,
                plot_path, metrics, error)


def _approach_slot(instance_id: int, work_q, result_q, approach: dict,
                   bench: dict, base_cfg: dict, checkpoint: str,
                   listener_bin: str, python_bin: str) -> None:
    while True:
        item = work_q.get()
        if item is None:
            break
        (bw, rtt, run_number, seed, run_dir,
         schedule_csv, flow_schedule_csv) = item
        kind = str(approach.get('kind', 'model')).lower()
        if kind in ('kernel', 'astraea'):
            row = _run_kernel_trial(
                instance_id, approach, bench, bw, rtt, run_number, seed,
                run_dir, schedule_csv, flow_schedule_csv)
        elif kind == 'orca':
            row = _run_orca_trial(
                instance_id, approach, bench, bw, rtt, run_number, seed,
                run_dir, schedule_csv, flow_schedule_csv)
        else:
            row = _run_model_trial(
                instance_id, approach, bench, bw, rtt, base_cfg, checkpoint,
                listener_bin, python_bin, run_number, seed, run_dir,
                schedule_csv, flow_schedule_csv)
        result_q.put(row)


def _expected_trial_keys(bench: dict) -> set:
    return {
        (float(c['bw']), float(c['rtt']), run)
        for c in _grid_cells(bench)
        for run in range(1, int(bench['runs']) + 1)
    }


def _trial_key(row: dict):
    """Return the ``(bw, rtt, run)`` key for a metrics row, or None if garbled."""
    bw = _finite_float(row.get('bw_mbps'))
    rtt = _finite_float(row.get('rtt_ms'))
    if not (math.isfinite(bw) and math.isfinite(rtt)):
        return None
    try:
        run = int(float(row.get('run', 0)))
    except (TypeError, ValueError):
        return None
    return (float(bw), float(rtt), run)


def _row_complete(row: dict) -> bool:
    if str(row.get('error', '')).strip():
        return False
    return str(row.get('goodput_source', '')).strip() == GOODPUT_SOURCE


def _completed_keys(rows_by_key: dict) -> set:
    return {key for key, row in rows_by_key.items() if _row_complete(row)}


def _approach_complete(output_root: str, approach: dict, bench: dict) -> bool:
    metrics_csv = os.path.join(output_root, approach['_label'], 'metrics.csv')
    rows_by_key = _resume_rows_by_key(metrics_csv, CONV_FIELDS, _trial_key)
    return _expected_trial_keys(bench).issubset(_completed_keys(rows_by_key))


def _split_by_completion(approaches: list, output_root: str, bench: dict) -> tuple:
    to_run, skipped = [], []
    for approach in approaches:
        if _approach_complete(output_root, approach, bench):
            skipped.append(approach)
        else:
            to_run.append(approach)
    return to_run, skipped


def _run_approach(base_cfg: dict, approach: dict, bench: dict,
                  output_root: str, listener_bin: str, python_bin: str,
                  bench_config_path: str) -> list:
    _validate_approach(approach)
    label = approach['_label']
    kind = str(approach.get('kind', 'model')).lower()
    _ensure_kernel_cc_available(approach)
    checkpoint = ''
    if kind not in ('kernel', 'astraea', 'orca'):
        checkpoint = _resolve_from(
            os.path.dirname(bench_config_path), str(approach['checkpoint']))
        if not os.path.exists(checkpoint):
            raise SystemExit(f'[conv4] checkpoint not found for {label}: {checkpoint}')

    approach_dir = os.path.abspath(os.path.join(output_root, label))
    os.makedirs(approach_dir, exist_ok=True)
    metrics_csv = os.path.join(approach_dir, 'metrics.csv')

    # Resume support: keep finished trials, re-run only missing/errored cells.
    rows_by_key = _resume_rows_by_key(metrics_csv, CONV_FIELDS, _trial_key)
    expected = _expected_trial_keys(bench)
    done = _completed_keys(rows_by_key) & expected
    rows_by_key = {key: rows_by_key[key] for key in done}
    missing = sorted(expected - done)

    if not missing:
        rows = [rows_by_key[key] for key in sorted(rows_by_key)]
        _write_rows(metrics_csv, CONV_FIELDS, rows)
        _restore_sudo_user_ownership(approach_dir)
        return rows

    with open(os.path.join(approach_dir, 'run_meta.json'), 'w') as f:
        json.dump({
            'approach': {k: v for k, v in approach.items() if not k.startswith('_')},
            'benchmark': bench,
            'benchmark_config': os.path.abspath(bench_config_path),
            'approach_config': approach.get('_config_path', ''),
            'checkpoint': checkpoint,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        }, f, indent=2)

    if rows_by_key:
        print(f'[conv4] {label} resuming missing={len(missing)}/{len(expected)} '
              f'(already done={len(rows_by_key)})', flush=True)

    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    flow_plan = _flow_plan(bench)

    cells = _grid_cells(bench)
    runs = int(bench['runs'])
    missing_set = set(missing)
    for cell_idx, cell in enumerate(cells, start=1):
        bw, rtt = float(cell['bw']), float(cell['rtt'])
        cell_dir = os.path.join(approach_dir, _cell_name(bw, rtt))
        for run_number in range(1, runs + 1):
            if (bw, rtt, run_number) not in missing_set:
                continue
            seed = _seed_for(bench, cell_idx, run_number)
            run_dir = os.path.join(cell_dir, f'run{run_number}')
            _safe_overwrite_dir(run_dir, approach_dir)
            schedule_csv, _ = _write_schedule(
                run_dir, _static_schedule(bench, bw, rtt), bench)
            flow_schedule_csv = _write_flow_plan(run_dir, flow_plan)
            work_q.put((bw, rtt, run_number, seed, run_dir,
                        schedule_csv, flow_schedule_csv))

    total = len(missing)
    n_parallel = max(1, min(int(bench.get('n_parallel', 1)), total))
    for _ in range(n_parallel):
        work_q.put(None)

    astraea_handle = None
    if kind == 'astraea':
        astraea_handle = _start_astraea_listener(approach, bench, approach_dir)

    procs = []
    for instance_id in range(n_parallel):
        proc = multiprocessing.Process(
            target=_approach_slot,
            args=(instance_id, work_q, result_q, approach, bench, base_cfg,
                  checkpoint, listener_bin, python_bin),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    completed = 0
    try:
        while completed < total:
            row = result_q.get()
            completed += 1
            rows_by_key[_trial_key(row)] = row
            _append_csv(metrics_csv, CONV_FIELDS, row)
            status = 'failed' if row.get('error') else 'done'
            print(f'[conv4] {label} bw={float(row["bw_mbps"]):g} '
                  f'rtt={float(row["rtt_ms"]):g} run={row["run"]}/{runs} '
                  f'{status} ({completed}/{total})', flush=True)
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
    _write_rows(metrics_csv, CONV_FIELDS, rows)
    _restore_sudo_user_ownership(approach_dir)
    return rows


def _collect_rows(output_root: str, cfg: dict) -> list:
    label_map = _approach_label_map(cfg)
    required = {'approach', 'run', 'run_dir'}
    rows = []
    for row in _collect_approach_rows_with_labels(output_root, label_map):
        if required.issubset(row.keys()):
            for key in CONV_FIELDS:
                row.setdefault(key, '')
            rows.append(row)
    return rows


# --- plotting ---------------------------------------------------------------

_FLOW_COLORS = ['#4878cf', '#59a14f', '#e1812c', '#e15759',
                '#b279a2', '#9c755f', '#76b7b2', '#bab0ac']


def _flow_plan_from_row(row: dict, bench: dict) -> list:
    plan = _flow_plan(bench)
    flows = int(_finite_float(row.get('flows'), len(plan)))
    return plan[:flows] if flows and flows <= len(plan) else plan


def _cell_key(row: dict) -> tuple:
    return (float(_finite_float(row.get('bw_mbps'))),
            float(_finite_float(row.get('rtt_ms'))))


def _runs_mean_per_flow(rows: list, bench: dict, cell_bw: float) -> tuple:
    """Average per-flow binned goodput across one (approach, cell) group's
    runs. Returns (times, mean[n_flows, T], std[n_flows, T], n_runs).
    """
    flow_plan = _flow_plan(bench)
    series_bench = {'duration_s': int(bench['duration_s']),
                    'bw_mbps': float(cell_bw)}
    stacks = []
    times = None
    for row in rows:
        if str(row.get('error', '')).strip():
            continue
        flows = _receiver_iperf_flows(str(row.get('run_dir', '')), flow_plan)
        if not _flows_have_receiver_goodput(flows, len(flow_plan)):
            continue
        series = _binned_series(flows, flow_plan, series_bench)
        times = series['time']
        stacks.append(np.asarray(series['per_flow'], dtype=float))
    if not stacks:
        return None, None, None, 0
    cube = np.stack(stacks, axis=0)  # (runs, n_flows, T)
    # Bins outside a flow's active window are all-NaN across runs; nanmean/
    # nanstd correctly return NaN there (masked out at plot time) but warn.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(cube, axis=0)
        std = (np.nanstd(cube, axis=0)
               if cube.shape[0] > 1 else np.zeros_like(mean))
    return times, mean, std, cube.shape[0]


def _draw_panel(ax, times, mean, std, flow_plan, bench, label) -> bool:
    ymin = float(bench.get('summary_ymin_mbps', 1.0))
    log_y = _as_bool(bench.get('summary_log_yscale'), default=False)
    plotted = False
    data_hi = 0.0
    for fidx in range(mean.shape[0]):
        m = np.asarray(mean[fidx], dtype=float)
        s = np.nan_to_num(np.asarray(std[fidx], dtype=float), nan=0.0)
        valid = np.isfinite(m)
        if log_y:
            valid &= m > 0
        if not valid.any():
            continue
        color = _FLOW_COLORS[fidx % len(_FLOW_COLORS)]
        lo = m - s
        hi = m + s
        if log_y:
            lo = np.maximum(lo, ymin * 0.5)
        else:
            lo = np.maximum(lo, 0.0)  # goodput can't be negative
        # Shaded ±1σ band plus thin envelope lines so the spread stays
        # visible even when run-to-run variance is sub-pixel small.
        ax.fill_between(times[valid], lo[valid], hi[valid],
                        color=color, alpha=0.35, linewidth=0.0, zorder=1)
        ax.plot(times[valid], hi[valid], color=color, linewidth=0.6,
                alpha=0.55, zorder=2)
        ax.plot(times[valid], lo[valid], color=color, linewidth=0.6,
                alpha=0.55, zorder=2)
        ax.plot(times[valid], m[valid], color=color, linewidth=1.7,
                zorder=3, label=f'flow {fidx + 1}')
        data_hi = max(data_hi, float(np.nanmax(hi[valid])))
        plotted = True
    for item in flow_plan:
        ax.axvline(float(item['start']), color='black', alpha=0.22,
                   linewidth=0.9)
        ax.axvline(float(item['end']), color='black', alpha=0.12,
                   linewidth=0.9, linestyle=':')
    ax.grid(True, which='both', alpha=0.20)
    ax.margins(x=0)
    if log_y:
        ax.set_yscale('log')
        ax.set_ylim(ymin, float(bench['bw_mbps']) * 1.6)
    else:
        top = data_hi * 1.10 if data_hi > 0 else float(bench['bw_mbps']) * 1.1
        ax.set_ylim(0, max(top, float(bench['bw_mbps']) * 0.25))
    ax.set_xlim(0, float(bench['duration_s']))
    if label:
        ax.set_title(label, loc='left', fontsize=10, fontweight='bold',
                     pad=3)
    return plotted


def _plot_run(flows: list, flow_plan: list, bench: dict,
              label: str, output: str) -> str:
    """Single-run per-flow goodput panel (same style as a summary panel)."""
    series = _binned_series(flows, flow_plan, {
        'duration_s': int(bench['duration_s']),
        'bw_mbps': float(bench['bw_mbps'])})
    mean = np.asarray(series['per_flow'], dtype=float)
    std = np.zeros_like(mean)
    # constrained_layout sizes the suptitle / title / axes; the legend gets
    # its own dedicated bottom row so nothing can ever overlap.
    fig, (ax, legax) = plt.subplots(
        2, 1, figsize=(11.0, 4.4),
        gridspec_kw={'height_ratios': [1.0, 0.12]},
        constrained_layout=True)
    fig.suptitle(label, fontsize=11, fontweight='bold')
    _draw_panel(ax, series['time'], mean, std, flow_plan, bench, '')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Goodput (Mbps)')
    legax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legax.legend(handles, labels, loc='center', ncol=len(flow_plan),
                     frameon=False, fontsize=9)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    try:
        fig.savefig(output, dpi=200)
    finally:
        plt.close(fig)
    return output


def _plot_summary(rows: list, bench: dict, output_pdf: str) -> None:
    """Vertically stacked summary: one full-width panel per
    (approach, bw/RTT condition), all conditions on the same figure.
    constrained_layout + a dedicated bottom legend row keep elements from
    ever overrunning."""
    rows = [r for r in rows if not str(r.get('error', '')).strip()]
    flow_plan = _flow_plan(bench)
    cells = _grid_cells(bench)
    cell_order = [(_cell_name(c['bw'], c['rtt']), float(c['bw']), float(c['rtt']))
                  for c in cells]

    by_approach = {}
    for row in rows:
        by_approach.setdefault(row.get('approach', ''), []).append(row)
    approaches = sorted(a for a in by_approach if a)

    # One stacked panel per (approach, condition), grouped by approach.
    panels = []
    for approach in approaches:
        a_rows = by_approach[approach]
        a_label = a_rows[0].get('plot_label') or approach
        for cname, bw, rtt in cell_order:
            crows = [r for r in a_rows if _cell_key(r) == (bw, rtt)]
            if not crows:
                continue
            times, mean, std, n_runs = _runs_mean_per_flow(crows, bench, bw)
            if times is None:
                continue
            label = (f'{a_label}  —  BW={bw:g} Mbps · RTT={rtt:g} ms  '
                     f'(n={n_runs})')
            panels.append((label, times, mean, std, bw))
    if not panels:
        print('[conv4] no summary panels to plot', flush=True)
        return

    n = len(panels)
    fig, axs = plt.subplots(
        n + 1, 1, figsize=(11.0, 1.3 + 2.4 * n),
        gridspec_kw={'height_ratios': [1.0] * n + [0.16]},
        sharex=True, constrained_layout=True)
    axs = np.atleast_1d(axs)
    data_axes = list(axs[:n])
    legax = axs[n]
    fig.suptitle(
        f'Convergence: {len(flow_plan)} flows, +1 every '
        f'{int(bench["flow_join_interval_s"])}s, '
        f'{int(bench["flow_duration_s"])}s each  '
        f'(mean ± 1σ over runs)',
        fontsize=12, fontweight='bold')
    for i, (ax, (label, times, mean, std, cbw)) in enumerate(
            zip(data_axes, panels)):
        _draw_panel(ax, times, mean, std, flow_plan,
                    {**bench, 'bw_mbps': cbw}, label)
        ax.set_ylabel('Goodput (Mbps)', fontsize=9)
        ax.tick_params(labelbottom=(i == n - 1))
    data_axes[-1].set_xlabel('Time (s)')

    legax.axis('off')
    handles, labels = data_axes[0].get_legend_handles_labels()
    if handles:
        legax.legend(handles, labels, loc='center', ncol=len(flow_plan),
                     frameon=False, fontsize=9, columnspacing=1.8,
                     handlelength=2.2)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    tmp_path = f'{output_pdf}.tmp.{os.getpid()}.pdf'
    fig.savefig(tmp_path, dpi=260)
    plt.close(fig)
    os.replace(tmp_path, output_pdf)
    print(f'[conv4] summary -> {output_pdf} ({n} panels: '
          f'{len(approaches)} approaches × conditions)', flush=True)


def _replot_individual_rows(rows: list, bench: dict) -> int:
    plotted = 0
    flow_plan = _flow_plan(bench)
    for row in rows:
        if str(row.get('error', '')).strip():
            continue
        flows = _receiver_iperf_flows(str(row.get('run_dir', '')), flow_plan)
        if not _flows_have_receiver_goodput(flows, len(flow_plan)):
            continue
        output = row.get('individual_plot') or os.path.join(
            str(row.get('run_dir', '')), 'convergence_run.pdf')
        bw = float(_finite_float(row.get('bw_mbps'), bench['bw_mbps']))
        rtt = float(_finite_float(row.get('rtt_ms'), bench['rtt_ms']))
        label = (f'{row.get("plot_label") or row.get("approach", "approach")} '
                 f'bw={bw:g} rtt={rtt:g} run={row.get("run", "")}')
        try:
            _plot_run(flows, flow_plan, {**bench, 'bw_mbps': bw}, label, output)
            row['individual_plot'] = output
            plotted += 1
        except OSError as exc:
            print(f'[conv4] run plot skipped for {output}: {exc}', flush=True)
    return plotted


def _filter_rows_by_approach(rows: list, selected: set = None) -> list:
    if not selected:
        return rows
    return [
        row for row in rows
        if _slug(row.get('approach') or row.get('data_folder') or '') in selected
    ]


_APPROACH_COLORS = [
    '#4878cf', '#e1812c', '#59a14f', '#e15759', '#b279a2', '#9c755f',
    '#76b7b2', '#ff9da7', '#5e7ce2', '#d37295', '#7f7f7f', '#bcbd22',
]
_APPROACH_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'p']


def _iperf_rtt_samples(run_dir: str, flow_plan: list) -> list:
    """All (episode_time_s, rtt_ms) TCP RTT samples, one entry per flow,
    shifted onto the episode clock. iperf3 client JSON for the iperf-driven
    kinds; the ss-sidecar srtt for `kind: orca` (no iperf JSON) so the
    efficiency scatter works for orca_real too."""
    out = []
    for item in flow_plan:
        flow = int(item['flow'])
        start = float(item['start'])
        parsed = _parse_iperf_json(
            os.path.join(run_dir, f'iperf_client_flow{flow}.json'))
        for t_end, rtt_ms in parsed.get('rtt_samples_ms', []) or []:
            out.append((float(t_end) + start, float(rtt_ms)))
    if not out:
        # No iperf RTT (Orca): fall back to the ss sidecar capture.
        out = _orca_ss_rtt_samples(run_dir, flow_plan)
    return out


def _efficiency_point(run_dir: str, flow_plan: list, bench: dict,
                      bw: float, rtt: float):
    """One (norm_delay, norm_goodput) point for a single run, measured over
    the window where all flows compete concurrently.

    norm_goodput = aggregate receiver goodput / link capacity (this cell's bw)
    norm_delay   = mean iperf3 RTT in that window / base RTT (this cell's rtt)
    """
    n = len(flow_plan)
    bw = float(bw)
    base_rtt = float(rtt)
    flows = _receiver_iperf_flows(run_dir, flow_plan)
    if not _flows_have_receiver_goodput(flows, n):
        return None
    series = _binned_series(flows, flow_plan, {
        'duration_s': int(bench['duration_s']), 'bw_mbps': bw})
    times = np.asarray(series['time'], dtype=float)
    active = np.asarray(series['active_count'], dtype=float)
    mask = active >= n                       # all flows competing
    if not mask.any():
        mask = active > 0                    # fallback: any flow active
    if not mask.any():
        return None
    agg = np.asarray(series['total'], dtype=float)[mask]
    agg = agg[np.isfinite(agg)]
    if not agg.size or bw <= 0:
        return None
    norm_goodput = float(np.mean(agg)) / bw

    wt = times[mask]
    lo, hi = float(wt.min()) - 0.5, float(wt.max()) + 0.5
    rtts = [r for (t, r) in _iperf_rtt_samples(run_dir, flow_plan)
            if lo <= t < hi and math.isfinite(r) and r > 0]
    if not rtts or base_rtt <= 0:
        return None
    norm_delay = float(np.mean(rtts)) / base_rtt
    return norm_delay, norm_goodput


def _science_style():
    """scienceplots 'science' look if available, else a serif/in-tick
    emulation. text.usetex is force-disabled (no LaTeX dependency)."""
    names = []
    try:
        import scienceplots  # noqa: F401  (registers the styles on import)
    except Exception:
        pass
    for s in ('science', 'no-latex'):
        if s in plt.style.available:
            names.append(s)
    rc = {
        'text.usetex': False,           # never require a LaTeX install
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.grid': True,
        'grid.alpha': 0.30,
        'grid.linewidth': 0.4,
        'axes.linewidth': 0.7,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': False,
    }
    return names + [rc]


def _confidence_ellipse(x, y, ax, n_std: float = 1.0, facecolor='none',
                        **kwargs):
    """Pearson covariance ellipse (the reference recipe: unit ellipse,
    rotate 45deg, scale by sigma*n_std, translate to the mean)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return None
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = float(np.clip(pearson, -0.999999, 0.999999))
    ell = Ellipse((0, 0), width=2 * np.sqrt(1 + pearson),
                  height=2 * np.sqrt(1 - pearson),
                  facecolor=facecolor, **kwargs)
    transf = (mtransforms.Affine2D()
              .rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
              .translate(float(np.mean(x)), float(np.mean(y))))
    ell.set_transform(transf + ax.transData)
    return ax.add_patch(ell)


def _plot_efficiency(rows: list, bench: dict, output_pdf: str) -> None:
    """Fig.9-style efficiency scatter (scienceplots look): per approach,
    mean (norm delay, norm goodput) over runs with a shaded Pearson
    covariance ellipse, hollow markers, inverted delay axis."""
    rows = [r for r in rows if not str(r.get('error', '')).strip()]
    flow_plan = _flow_plan(bench)
    cells = _grid_cells(bench)
    cell_list = [(_cell_name(c['bw'], c['rtt']), float(c['bw']), float(c['rtt']))
                 for c in cells]
    cell_marker = {cn: _APPROACH_MARKERS[i % len(_APPROACH_MARKERS)]
                   for i, (cn, _, _) in enumerate(cell_list)}

    by_approach = {}
    for row in rows:
        by_approach.setdefault(row.get('approach', ''), []).append(row)
    approaches = sorted(a for a in by_approach if a)

    # points[(approach_idx, cell_name)] -> (label, color, marker, pts ndarray)
    plotted = []
    used_cells = []
    all_y = []
    for aidx, approach in enumerate(approaches):
        a_rows = by_approach[approach]
        a_label = a_rows[0].get('plot_label') or approach
        color = _APPROACH_COLORS[aidx % len(_APPROACH_COLORS)]
        for cname, bw, rtt in cell_list:
            crows = [r for r in a_rows if _cell_key(r) == (bw, rtt)]
            pts = []
            for r in crows:
                p = _efficiency_point(str(r.get('run_dir', '')), flow_plan,
                                      bench, bw, rtt)
                if p is not None:
                    pts.append(p)
            if not pts:
                continue
            pts = np.asarray(pts, dtype=float)
            plotted.append((a_label, color, cell_marker[cname], pts))
            all_y.append(pts[:, 1])
            if cname not in used_cells:
                used_cells.append(cname)
    if not plotted:
        print('[conv4] no efficiency points to plot', flush=True)
        return

    all_y = np.concatenate(all_y)
    y_lo = min(0.5, float(np.nanmin(all_y)) - 0.03)
    y_hi = max(1.0, float(np.nanmax(all_y)) + 0.03)

    with plt.style.context(_science_style()):
        fig, (ax, legax) = plt.subplots(
            2, 1, figsize=(6.0, 4.4),
            gridspec_kw={'height_ratios': [1.0, 0.26]},
            constrained_layout=True)
        fig.suptitle(
            f'Efficiency: {len(flow_plan)} competing flows  ·  '
            f'buffer={bench["bdp_mult"]:g}x BDP  ·  mean +/- 1 sigma\n'
            f'colour = approach, marker = bw/RTT condition',
            fontsize=9, fontweight='bold')
        approach_seen = {}
        for a_label, color, marker, pts in plotted:
            mx, my = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            _confidence_ellipse(pts[:, 0], pts[:, 1], ax, n_std=1.0,
                                facecolor=color, edgecolor='none', alpha=0.25)
            ax.scatter([mx], [my], s=46, edgecolors=color,
                       facecolors='none', marker=marker, linewidths=1.3,
                       zorder=3)
            approach_seen.setdefault(a_label, color)
        ax.set_xlabel('Norm. Delay')
        ax.set_ylabel('Norm. Throughput')
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True)
        ax.invert_xaxis()   # better = upper-right (high tput, low delay)

        approach_handles = [
            Line2D([], [], color=c, marker='o', linestyle='None',
                   markersize=6, markeredgewidth=1.3, markerfacecolor='none',
                   label=lbl)
            for lbl, c in approach_seen.items()]
        cond_handles = [
            Line2D([], [], color='black', marker=cell_marker[cn],
                   linestyle='None', markersize=6, markeredgewidth=1.2,
                   markerfacecolor='none',
                   label=f'BW={bw:g} · RTT={rtt:g}')
            for cn, bw, rtt in cell_list if cn in used_cells]

        legax.axis('off')
        leg1 = legax.legend(
            handles=approach_handles, loc='center left',
            bbox_to_anchor=(0.0, 0.5), ncol=max(1, min(2, len(approach_handles))),
            frameon=False, fontsize=7, handletextpad=0.6, labelspacing=0.3,
            title='Approach', title_fontsize=7)
        legax.add_artist(leg1)
        legax.legend(
            handles=cond_handles, loc='center right',
            bbox_to_anchor=(1.0, 0.5), ncol=max(1, min(3, len(cond_handles))),
            frameon=False, fontsize=7, handletextpad=0.6, labelspacing=0.3,
            title='Condition', title_fontsize=7)
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        tmp_path = f'{output_pdf}.tmp.{os.getpid()}.pdf'
        fig.savefig(tmp_path, dpi=600)
        plt.close(fig)
    os.replace(tmp_path, output_pdf)
    print(f'[conv4] efficiency -> {output_pdf} '
          f'({len(approaches)} approaches × {len(used_cells)} conditions)',
          flush=True)


def _write_per_approach_outputs(output_root: str, bench: dict, rows: list) -> None:
    for approach in sorted({r.get('approach', '') for r in rows if r.get('approach')}):
        a_rows = [r for r in rows if r.get('approach') == approach]
        if not a_rows:
            continue
        a_dir = os.path.join(output_root, approach)
        os.makedirs(a_dir, exist_ok=True)
        try:
            _write_rows(os.path.join(a_dir, 'metrics.csv'), CONV_FIELDS, a_rows)
            _plot_summary(a_rows, bench, os.path.join(a_dir, 'convergence_summary.pdf'))
            _plot_efficiency(a_rows, bench, os.path.join(a_dir, 'efficiency.pdf'))
        except PermissionError as exc:
            print(f'[conv4] per-approach output skipped for {approach}: {exc}',
                  flush=True)


def _replot_all(output_root: str, cfg: dict, bench: dict,
                replot_individual: bool = True, selected: set = None) -> None:
    all_rows = _collect_rows(output_root, cfg)
    rows = _filter_rows_by_approach(all_rows, selected)
    if selected and not rows:
        raise SystemExit('[conv4] no existing rows matched requested approach: '
                         + ', '.join(sorted(selected)))
    plotted = _replot_individual_rows(rows, bench) if replot_individual else 0
    _write_per_approach_outputs(output_root, bench, rows)
    try:
        _write_rows(os.path.join(output_root, 'metrics.csv'), CONV_FIELDS, all_rows)
        _plot_summary(all_rows, bench,
                      os.path.join(output_root, 'convergence_summary.pdf'))
        _plot_efficiency(all_rows, bench,
                         os.path.join(output_root, 'efficiency.pdf'))
        specific = _specific_plot_keys(cfg)
        specific_rows = _filter_rows_by_approach(all_rows, specific) if specific else []
        if specific_rows:
            _plot_summary(specific_rows, bench,
                          os.path.join(output_root, 'convergence_summary_specific.pdf'))
            _plot_efficiency(specific_rows, bench,
                             os.path.join(output_root, 'efficiency_specific.pdf'))
    except PermissionError as exc:
        print(f'[conv4] root summary skipped: {exc}', flush=True)
    _restore_sudo_user_ownership(output_root)
    print(f'[conv4] replotted rows={len(rows)} total_rows={len(all_rows)} '
          f'individual_plots={plotted}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Run the four-flow staggered convergence benchmark.')
    ap.add_argument('--config', default=os.path.join(_HERE, 'config.yaml'))
    ap.add_argument('--approach', action='append', default=None,
                    help='approach data_folder/name to run; repeatable')
    ap.add_argument('--plot-only', action='store_true',
                    help='rebuild run plots and the summary from existing data')
    ap.add_argument('--heatmaps-only', action='store_true',
                    help='with --plot-only: skip per-run PDFs, summary only')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    bench_config_path = os.path.abspath(args.config)
    cfg = _load_benchmark_config(bench_config_path)
    bench_dir = os.path.dirname(bench_config_path)
    bench = _conv_cfg(cfg)

    output_root = cfg.get('output_root', 'data')
    if not os.path.isabs(str(output_root)):
        output_root = os.path.abspath(os.path.join(bench_dir, str(output_root)))
    os.makedirs(output_root, exist_ok=True)

    selected = {_slug(value) for value in (args.approach or [])}
    if args.plot_only:
        _replot_all(output_root, cfg, bench,
                    replot_individual=not args.heatmaps_only,
                    selected=selected or None)
        return

    configured_approaches = _configured_approaches(cfg, selected or None)
    _validate_unique_data_folders(configured_approaches)
    if not configured_approaches:
        available = sorted({
            key for raw in (cfg.get('approaches') or [])
            for key in _approach_selection_keys(raw or {})
        })
        if selected:
            missing = ', '.join(sorted(selected - set(available)))
            raise SystemExit(
                '[conv4] requested approach is not listed in benchmark config'
                + (f': {missing}' if missing else '')
                + (f'; available: {", ".join(available)}' if available else ''))
        raise SystemExit('[conv4] no approaches listed in benchmark config'
                         + (f'; available: {", ".join(available)}' if available else ''))

    approaches, skipped = _split_by_completion(configured_approaches, output_root, bench)
    flow_plan = _flow_plan(bench)
    cells = _grid_cells(bench)
    trials = len(cells) * int(bench['runs'])
    print(f'[conv4] output_root={output_root}', flush=True)
    print(f'[conv4] benchmark={bench["name"]} duration={bench["duration_s"]}s '
          f'flows={len(flow_plan)} runs={bench["runs"]} '
          f'n_parallel={bench.get("n_parallel", 1)}', flush=True)
    print('[conv4] flow_starts='
          + ','.join(str(int(p['start'])) for p in flow_plan), flush=True)
    print('[conv4] bw_grid='
          + ','.join(f'{v:g}' for v in bench['bw_mbps_values'])
          + '  rtt_grid='
          + ','.join(f'{v:g}' for v in bench['rtt_ms_values'])
          + f'  ({len(cells)} conditions, {trials} trials/approach)',
          flush=True)
    if skipped:
        print('[conv4] skipping complete data='
              + ', '.join(a['_label'] for a in skipped), flush=True)
    print('[conv4] approaches_to_run='
          + (', '.join(a['_label'] for a in approaches) if approaches else '(none)'),
          flush=True)

    if args.dry_run:
        for approach in skipped:
            _, approach_config_path = _load_approach_runtime_config(
                approach, bench_config_path)
            print(f'[conv4] dry-run would skip complete data: '
                  f'{os.path.join(output_root, approach["_label"])}'
                  + (f' config={approach_config_path}'
                     if approach_config_path else ''),
                  flush=True)
        for approach in approaches:
            _, approach_config_path = _load_approach_runtime_config(
                approach, bench_config_path)
            print(f'[conv4] dry-run would run: '
                  f'{os.path.join(output_root, approach["_label"])} '
                  f'trials={trials} ({len(cells)} conditions × '
                  f'{bench["runs"]} runs)'
                  + (f' config={approach_config_path}'
                     if approach_config_path else ''),
                  flush=True)
        return

    if not approaches:
        print('[conv4] no approaches need running; rebuilding plots', flush=True)
        _replot_all(output_root, cfg, bench)
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
                    f'[conv4] approach config for '
                    f'{approach["_label"]} must define '
                    'paths.listener and paths.py')
            listener_bin = (
                _resolve_repo_path(paths['listener'])
                if 'listener' in paths else '')
            python_bin = (
                _resolve_repo_path(paths['py']) if 'py' in paths else '')
            if approach_config_path:
                print(
                    f'[conv4] {approach["_label"]} '
                    f'config={approach_config_path}',
                    flush=True)
            try:
                rows = _run_approach(
                    base_cfg, approach, bench, output_root,
                    listener_bin, python_bin, bench_config_path)
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
            f'[conv4] {failed_trials} trial(s) failed')

    _replot_all(output_root, cfg, bench)
    print(f'[conv4] done metrics={os.path.join(output_root, "metrics.csv")}',
          flush=True)


if __name__ == '__main__':
    main()
