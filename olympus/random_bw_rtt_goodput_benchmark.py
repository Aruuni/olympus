"""
Random BW/RTT goodput benchmark for Olympus models and TCP baselines.

The benchmark creates 50 reproducible runs by default. Each run lasts 300s and
uses a piecewise-constant bottleneck schedule: a fresh bandwidth and RTT value
is sampled from [10, 100] every hardcoded 15 seconds. Goodput measurements are
written at 1 second granularity. Each run's schedule is generated from only the
run number, so run 1 is identical for selected/bbr3/cubic/bbr1/orca even if a
single protocol is rerun by itself.

Compared protocols:
  - selected Olympus model from the config/checkpoint
  - bbr3, using Linux congestion-control name "bbr"
  - cubic
  - bbr1
  - external Orca scripts

Example:
  sudo -E env PATH="$PATH" HOME="$HOME" \\
    ./venv_training/bin/python olympus/random_bw_rtt_goodput_benchmark.py \\
      --config olympus/config.yaml \\
      --checkpoint olympus/data/<run>/checkpoints/mat_cwnd_model.pt \\
      --n-parallel 5
"""

import argparse
import copy
import csv
import glob
import json
import math
import multiprocessing
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict

import yaml

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from arm_bandit_mutant.mininet_env import MininetEnv

from olympus.common.checkpoint_config import (
    apply_model_config_from_checkpoint,
)
import olympus.orchestrator as single_orch
import olympus.orchestrator as multi_orch


PROTOCOL_ALIASES = {
    'selected': 'selected',
    'model': 'selected',
    'olympus': 'selected',
    'bbr3': 'bbr3',
    'bbr': 'bbr3',
    'cubic': 'cubic',
    'bbr1': 'bbr1',
    'orca': 'orca',
}

KERNEL_CC = {
    'bbr3': 'bbr',
    'cubic': 'cubic',
    'bbr1': 'bbr1',
}

DEFAULT_COLORS = {
    'capacity': 'black',
    'selected': '#4878cf',
    'bbr3': '#e15759',
    'cubic': '#59a14f',
    'bbr1': '#f28e2c',
    'orca': '#b07aa1',
}

CHANGE_INTERVAL_S = 15.0


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return default


def _slug(text: str) -> str:
    return multi_orch._slug(text)


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_ROOT, path))


def _runner(kind: str):
    if kind == 'single':
        return {
            'activate': single_orch._activate_runtime_blocks,
            'run_episode': single_orch.run_episode,
            'final_cleanup': single_orch._final_runtime_cleanup,
            'default_config': os.path.join(_ROOT, 'olympus', 'config.yaml'),
            'default_alg': 'td3',
            'default_learner_port': 6301,
            'default_cport_base': 21000,
        }
    return {
        'activate': multi_orch._activate_runtime_blocks,
        'run_episode': multi_orch.run_episode,
        'final_cleanup': multi_orch._final_runtime_cleanup,
        'default_config': os.path.join(_ROOT, 'olympus', 'config.yaml'),
        'default_alg': 'mat',
        'default_learner_port': 6401,
        'default_cport_base': 24000,
    }


def _finite_float(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _mean(values, default=math.nan):
    vals = [_finite_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else default


def _weighted_mean_schedule(schedule, duration_s: float, key: str) -> float:
    total = 0.0
    points = sorted(schedule, key=lambda item: float(item['t']))
    for idx, entry in enumerate(points):
        start = float(entry['t'])
        end = float(points[idx + 1]['t']) if idx + 1 < len(points) else float(duration_s)
        span = max(0.0, min(end, float(duration_s)) - max(0.0, start))
        total += span * float(entry[key])
    return total / max(float(duration_s), 1e-9)


def _run_schedule_seed(run: int) -> int:
    return int(run)


def _generate_run_schedule(run: int, args):
    rng = random.Random(_run_schedule_seed(run))
    points = []
    t = 0.0
    while t < float(args.duration) - 1e-9:
        points.append({
            't': round(t, 3),
            'bw': rng.randint(int(args.bw_min), int(args.bw_max)),
            'delay': rng.randint(int(args.rtt_min), int(args.rtt_max)),
        })
        t += CHANGE_INTERVAL_S
    return points


def _generate_schedules(args):
    return {
        run: _generate_run_schedule(run, args)
        for run in range(1, int(args.runs) + 1)
    }


def _write_schedule_csv(path, schedule):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t', 'bw_mbps', 'rtt_ms'])
        w.writeheader()
        for entry in schedule:
            w.writerow({
                't': entry['t'],
                'bw_mbps': entry['bw'],
                'rtt_ms': entry['delay'],
            })


def _link_schedule(schedule):
    return [{'t': item['t'], 'bw': item['bw'], 'delay': item['delay']}
            for item in schedule[1:]]


def _run_link_schedule(env, schedule, episode_start, stop):
    for entry in _link_schedule(schedule):
        t_target = episode_start + float(entry['t'])
        while not stop.is_set():
            rem = t_target - time.monotonic()
            if rem <= 0:
                break
            stop.wait(timeout=min(rem, 0.05))
        if stop.is_set():
            return
        try:
            env.set_link(bw=entry['bw'], delay=entry['delay'])
            print(f'[sched] t={time.monotonic() - episode_start:.1f}s '
                  f'bw={entry["bw"]}Mbps rtt={entry["delay"]}ms', flush=True)
        except Exception as exc:
            print(f'[sched] link update failed: {exc}', flush=True)


def _safe_unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _copy_if_exists(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def _write_measurement_csv(csv_path, rows, measure_interval=1.0):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    interval = max(float(measure_interval or 1.0), 1e-6)
    buckets = defaultdict(list)
    for t_s, bandwidth_mbps in rows:
        if not (math.isfinite(t_s) and math.isfinite(bandwidth_mbps)):
            continue
        bucket = int(max(t_s - 1e-9, 0.0) // interval)
        buckets[bucket].append(float(bandwidth_mbps))

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['time', 'bandwidth'])
        w.writeheader()
        for bucket in sorted(buckets):
            t_end = (bucket + 1) * interval
            w.writerow({
                'time': f'{t_end:.3f}',
                'bandwidth': f'{_mean(buckets[bucket]):.6f}',
            })


def _parse_iperf_json(json_path, csv_path=None, measure_interval=1.0):
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return math.nan

    csv_rows = []
    for interval in data.get('intervals', []) or []:
        total = interval.get('sum') or {}
        t_end = _finite_float(total.get('end'))
        bps = _finite_float(total.get('bits_per_second'))
        if math.isfinite(t_end) and math.isfinite(bps):
            csv_rows.append((t_end, bps / 1e6))

    if csv_path:
        _write_measurement_csv(csv_path, csv_rows, measure_interval=measure_interval)

    end = data.get('end') or {}
    for key in ('sum_received', 'sum', 'sum_sent'):
        bps = _finite_float((end.get(key) or {}).get('bits_per_second'))
        if math.isfinite(bps):
            return bps / 1e6
    return math.nan


def _read_trace_csv(csv_path):
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


def _step_xy(schedule, duration_s, key):
    if not schedule:
        return np.asarray([]), np.asarray([])
    points = sorted(schedule, key=lambda item: float(item['t']))
    times = [float(item['t']) for item in points]
    values = [float(item[key]) for item in points]
    times.append(float(duration_s))
    values.append(values[-1])
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _plot_protocol_run(run_dir, protocol, label, run, schedule, row, args):
    output_path = os.path.join(run_dir, 'run_plot.pdf')
    duration = float(args.duration)
    tx, bw = _step_xy(schedule, duration, 'bw')
    _, rtt = _step_xy(schedule, duration, 'delay')

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 4.8), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.35], 'hspace': 0.12},
    )
    fig.suptitle(f'{label} run {run}', fontsize=11, fontweight='bold')

    ax = axes[0]
    if len(tx):
        ax.step(tx, bw, where='post', color=DEFAULT_COLORS['capacity'],
                linewidth=1.25, label='capacity')
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

    total_by_time = defaultdict(float)
    plotted = 0
    csv_dir = os.path.join(run_dir, 'csvs')
    for flow_idx in range(1, int(args.flows) + 1):
        t, goodput = _read_trace_csv(os.path.join(csv_dir, f'x{flow_idx}.csv'))
        if not len(t):
            continue
        if int(args.flows) == 1:
            avg = _finite_float(row.get('average_goodput_mbps'))
            trace_label = label
            if math.isfinite(avg):
                trace_label = f'{label} ({avg:.1f} Mbps)'
            ax.plot(t, goodput, color=DEFAULT_COLORS.get(protocol, '#4878cf'),
                    linewidth=1.0, alpha=0.9, label=trace_label)
        else:
            ax.plot(t, goodput, linewidth=0.7, alpha=0.35,
                    label=f'flow {flow_idx}')
            for tv, gv in zip(t, goodput):
                total_by_time[float(tv)] += float(gv)
        plotted += 1

    if total_by_time:
        t_total = np.asarray(sorted(total_by_time), dtype=float)
        y_total = np.asarray([total_by_time[t] for t in t_total], dtype=float)
        ax.plot(t_total, y_total, color=DEFAULT_COLORS.get(protocol, '#4878cf'),
                linewidth=1.1, alpha=0.95, label=f'{label} total')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Goodput (Mbps)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, max(105.0, float(np.nanmax(bw)) * 1.12 if len(bw) else 105.0))
    ax.grid(True, alpha=0.28, linewidth=0.5)
    if plotted:
        ax.legend(loc='upper right', frameon=False, fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, 'No csvs/x*.csv trace found',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)

    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _parse_orca_receiver_log(log_path, csv_path=None, measure_interval=1.0):
    rows = []
    in_block = False
    try:
        with open(log_path) as f:
            for raw in f:
                line = raw.strip()
                if line == '----START----':
                    in_block = True
                    continue
                if line == '----END----':
                    in_block = False
                    continue
                if not in_block or ',' not in line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 4:
                    continue
                t_s = _finite_float(parts[0])
                interval_bps = _finite_float(parts[1])
                avg_mbps = _finite_float(parts[-1])
                if not math.isfinite(t_s):
                    continue
                interval_mbps = interval_bps / 1e6 if math.isfinite(interval_bps) else math.nan
                rows.append((t_s, interval_mbps, avg_mbps))
    except FileNotFoundError:
        return math.nan

    if csv_path:
        csv_rows = [(t_s, interval_mbps) for t_s, interval_mbps, _ in rows]
        _write_measurement_csv(csv_path, csv_rows, measure_interval=measure_interval)

    for _, _, avg_mbps in reversed(rows):
        if math.isfinite(avg_mbps):
            return avg_mbps
    return math.nan


def _state_log_goodput(state_log_path, n_flows, csv_dir=None, measure_interval=1.0):
    paths = []
    if os.path.exists(state_log_path):
        paths.append(state_log_path)
    base, ext = os.path.splitext(state_log_path)
    for flow_id in range(int(n_flows)):
        flow_path = f'{base}_a{flow_id}{ext}'
        if os.path.exists(flow_path):
            paths.append(flow_path)

    seen = set()
    unique_paths = []
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    flow_means = []
    for idx, path in enumerate(unique_paths, start=1):
        samples = []
        csv_rows = []
        try:
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    t_s = _finite_float(row.get('t_s'))
                    bw = _finite_float(row.get('avg_thr_mbps'))
                    if math.isfinite(t_s) and math.isfinite(bw):
                        samples.append(bw)
                        csv_rows.append((t_s, bw))
        except FileNotFoundError:
            continue
        if samples:
            flow_means.append(_mean(samples))
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
                _write_measurement_csv(
                    os.path.join(csv_dir, f'x{idx}.csv'),
                    csv_rows,
                    measure_interval=measure_interval,
                )
    return sum(flow_means) if flow_means else math.nan


def _ensure_user_writable(path, user):
    os.makedirs(path, exist_ok=True)
    if not user or os.geteuid() != 0:
        return
    subprocess.run(['chown', '-R', f'{user}:{user}', path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=False)


def _shq(value):
    return shlex.quote(str(value))


def _wait_for_files(paths, timeout_s, poll_s=0.5):
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    pending = {path for path in paths if path}
    while pending and time.monotonic() < deadline:
        pending = {path for path in pending if not os.path.exists(path)}
        if not pending:
            return True
        time.sleep(float(poll_s))
    return not pending


def _file_contains(path, needles):
    try:
        with open(path, errors='replace') as f:
            text = f.read()
    except FileNotFoundError:
        return False
    lowered = text.lower()
    return any(str(needle).lower() in lowered for needle in needles)


def _wait_for_orca_connections(flow_logs, timeout_s, poll_s=0.25):
    """Return the 1-based flow indexes that did not print a connection marker."""
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    pending = set(flow_logs)
    needles = ('connected!', 'connected')
    while pending and time.monotonic() < deadline:
        for flow_idx in list(pending):
            if any(_file_contains(path, needles) for path in flow_logs[flow_idx]):
                pending.discard(flow_idx)
        if pending:
            time.sleep(float(poll_s))
    return sorted(pending)


def _base_row(protocol, label, run, schedule, args, run_dir):
    bw_values = [entry['bw'] for entry in schedule]
    rtt_values = [entry['delay'] for entry in schedule]
    mean_bw = _weighted_mean_schedule(schedule, args.duration, 'bw')
    mean_rtt = _weighted_mean_schedule(schedule, args.duration, 'delay')
    return {
        'protocol': protocol,
        'label': label,
        'run': run,
        'success': 0,
        'average_goodput_mbps': '',
        'optimal_goodput_mbps': f'{mean_bw:.6f}',
        'mean_bw_mbps': f'{mean_bw:.6f}',
        'mean_rtt_ms': f'{mean_rtt:.6f}',
        'initial_bw_mbps': schedule[0]['bw'],
        'initial_rtt_ms': schedule[0]['delay'],
        'min_bw_mbps': min(bw_values),
        'max_bw_mbps': max(bw_values),
        'min_rtt_ms': min(rtt_values),
        'max_rtt_ms': max(rtt_values),
        'flows': int(args.flows),
        'duration_s': int(args.duration),
        'change_interval_s': CHANGE_INTERVAL_S,
        'measure_interval_s': float(args.measure_interval),
        'schedule_seed': _run_schedule_seed(run),
        'run_dir': run_dir,
        'plot_path': '',
        'error': '',
    }


def _write_run_meta(run_dir, protocol, label, run, schedule, args):
    meta = {
        'protocol': protocol,
        'label': label,
        'run': int(run),
        'duration_s': int(args.duration),
        'change_interval_s': CHANGE_INTERVAL_S,
        'measure_interval_s': float(args.measure_interval),
        'schedule_seed': _run_schedule_seed(run),
        'flows': int(args.flows),
        'bdp_mult': float(args.bdp_mult),
        'schedule': schedule,
        'optimal_goodput_mbps': _weighted_mean_schedule(schedule, args.duration, 'bw'),
        'schedule_seed': _run_schedule_seed(run),
        'note': 'RTT values are passed to the repo MininetEnv delay field.',
    }
    with open(os.path.join(run_dir, 'emulation_info.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def _run_kernel(protocol, run, schedule, run_dir, args, instance_id):
    label = 'BBR3' if protocol == 'bbr3' else protocol.upper()
    row = _base_row(protocol, label, run, schedule, args, run_dir)
    _write_schedule_csv(os.path.join(run_dir, 'schedule.csv'), schedule)
    _write_run_meta(run_dir, protocol, label, run, schedule, args)

    cport = int(args.kernel_cport_base) + int(instance_id) * 100
    tmp_paths = [f'/tmp/iperf_{cport}_{i}.json' for i in range(1, int(args.flows) + 1)]
    for path in tmp_paths:
        _safe_unlink(path)

    env = MininetEnv(
        n=int(args.flows),
        bw=float(schedule[0]['bw']),
        delay=float(schedule[0]['delay']),
        bdp_mult=float(args.bdp_mult),
        duration=int(args.duration),
        cport=cport,
        cc_algo=KERNEL_CC[protocol],
        instance_id=instance_id,
    )
    sched_stop = threading.Event()
    sched_thread = None
    try:
        env.start()
        episode_start = time.monotonic()
        env.run_iperf(monitor_interval=float(args.measure_interval))
        sched_thread = threading.Thread(
            target=_run_link_schedule,
            args=(env, schedule, episode_start, sched_stop),
            daemon=True,
        )
        sched_thread.start()
        time.sleep(float(args.duration) + 3.0)
    finally:
        sched_stop.set()
        if sched_thread:
            sched_thread.join(timeout=2)
        env.stop()
        time.sleep(0.5)

    csv_dir = os.path.join(run_dir, 'csvs')
    goodputs = []
    for i, tmp_path in enumerate(tmp_paths, start=1):
        dst = os.path.join(run_dir, f'iperf_flow{i}.json')
        _copy_if_exists(tmp_path, dst)
        goodputs.append(_parse_iperf_json(
            dst,
            os.path.join(csv_dir, f'x{i}.csv'),
            measure_interval=float(args.measure_interval),
        ))
    goodput = sum(v for v in goodputs if math.isfinite(v))
    if goodput > 0:
        row['success'] = 1
        row['average_goodput_mbps'] = f'{goodput:.6f}'
    else:
        row['error'] = 'missing or invalid iperf goodput'
    return row


def _run_orca(run, schedule, run_dir, args, instance_id):
    protocol = 'orca'
    label = 'Orca'
    row = _base_row(protocol, label, run, schedule, args, run_dir)
    _write_schedule_csv(os.path.join(run_dir, 'schedule.csv'), schedule)
    _write_run_meta(run_dir, protocol, label, run, schedule, args)

    if not os.path.exists(os.path.join(args.orca_dir, 'sender.sh')):
        row['error'] = f'Orca sender.sh not found under {args.orca_dir}'
        return row
    if not os.path.exists(os.path.join(args.orca_dir, 'receiver.sh')):
        row['error'] = f'Orca receiver.sh not found under {args.orca_dir}'
        return row

    _ensure_user_writable(run_dir, args.orca_user)

    env = MininetEnv(
        n=int(args.flows),
        bw=float(schedule[0]['bw']),
        delay=float(schedule[0]['delay']),
        bdp_mult=float(args.bdp_mult),
        duration=int(args.duration),
        cport=int(args.orca_port_base) + int(instance_id) * 100,
        cc_algo='cubic',
        instance_id=instance_id,
    )
    sched_stop = threading.Event()
    sched_thread = None
    receiver_done_paths = []
    flow_logs = {}
    launch_info = []
    try:
        env.start()
        p = env.prefix
        base_port = int(args.orca_port_base) + int(instance_id) * 100
        for flow_idx in range(1, int(args.flows) + 1):
            c = env.net.get(f'{p}c{flow_idx}')
            x = env.net.get(f'{p}x{flow_idx}')
            port = base_port + flow_idx - 1
            flow_id = (
                int(args.orca_flow_id_base)
                + int(instance_id) * int(args.flows)
                + flow_idx - 1
            )
            flow_dir = os.path.join(run_dir, f'flow{flow_idx}')
            _ensure_user_writable(flow_dir, args.orca_user)
            sender_log = os.path.join(run_dir, f'orca_sender_flow{flow_idx}.log')
            receiver_log = os.path.join(run_dir, f'orca_receiver_flow{flow_idx}.log')
            receiver_done = os.path.join(run_dir, f'orca_receiver_flow{flow_idx}.done')
            _safe_unlink(receiver_done)
            receiver_done_paths.append(receiver_done)
            flow_logs[flow_idx] = (sender_log, receiver_log)
            sender = (
                f'cd {_shq(args.orca_dir)} && '
                f'sudo -u {_shq(args.orca_user)} env EXPERIMENT_PATH={_shq(flow_dir)} '
                f'stdbuf -oL -eL '
                f'{_shq(os.path.join(args.orca_dir, "sender.sh"))} '
                f'{port} {flow_id} {int(args.duration)} {_shq(args.orca_dir)} '
                f'> {_shq(sender_log)} 2>&1 &'
            )
            receiver = (
                f'(cd {_shq(args.orca_dir)} && '
                f'sudo -u {_shq(args.orca_user)} '
                f'stdbuf -oL -eL '
                f'{_shq(os.path.join(args.orca_dir, "receiver.sh"))} '
                f'{_shq(c.IP())} {port} {flow_id} {_shq(args.orca_dir)} '
                f'> {_shq(receiver_log)} 2>&1; '
                f'printf "%s\\n" "$?" > {_shq(receiver_done)}) &'
            )
            launch_info.append({
                'flow_idx': flow_idx,
                'flow_id': flow_id,
                'port': port,
                'client': c.name,
                'receiver': x.name,
                'client_ip': c.IP(),
                'sender_log': sender_log,
                'receiver_log': receiver_log,
                'sender_cmd': sender,
                'receiver_cmd': receiver,
            })
            c.cmd(sender)
            time.sleep(max(0.0, float(args.orca_sender_lead)))
            x.cmd(receiver)

        with open(os.path.join(run_dir, 'orca_launch.json'), 'w') as f:
            json.dump(launch_info, f, indent=2)

        missing_connected = _wait_for_orca_connections(
            flow_logs,
            timeout_s=float(args.orca_startup_grace),
        )
        if missing_connected:
            row['error'] = (
                'Orca did not print Connected before startup grace for '
                f'flows {missing_connected}; proceeding anyway'
            )
            print(f'[orca] {row["error"]}', flush=True)

        episode_start = time.monotonic()
        sched_thread = threading.Thread(
            target=_run_link_schedule,
            args=(env, schedule, episode_start, sched_stop),
            daemon=True,
        )
        sched_thread.start()
        time.sleep(float(args.duration) + 3.0)
        if not _wait_for_files(receiver_done_paths, timeout_s=float(args.orca_finish_grace)):
            missing = [p for p in receiver_done_paths if not os.path.exists(p)]
            print(f'[orca] receiver stdout not complete before timeout: {missing}', flush=True)
    finally:
        sched_stop.set()
        if sched_thread:
            sched_thread.join(timeout=2)
        env.stop()
        time.sleep(0.5)

    csv_dir = os.path.join(run_dir, 'csvs')
    goodputs = []
    for flow_idx in range(1, int(args.flows) + 1):
        receiver_log = os.path.join(run_dir, f'orca_receiver_flow{flow_idx}.log')
        goodputs.append(_parse_orca_receiver_log(
            receiver_log,
            os.path.join(csv_dir, f'x{flow_idx}.csv'),
            measure_interval=float(args.measure_interval),
        ))
    goodput = sum(v for v in goodputs if math.isfinite(v))
    if goodput > 0:
        row['success'] = 1
        row['average_goodput_mbps'] = f'{goodput:.6f}'
    else:
        row['error'] = 'missing or invalid Orca receiver goodput'
    return row


def _run_selected(run, schedule, run_dir, args, instance_id, selected_ctx):
    protocol = 'selected'
    label = selected_ctx['label']
    row = _base_row(protocol, label, run, schedule, args, run_dir)
    _write_schedule_csv(os.path.join(run_dir, 'schedule.csv'), schedule)
    _write_run_meta(run_dir, protocol, label, run, schedule, args)

    cfg = copy.deepcopy(selected_ctx['cfg'])
    kind = selected_ctx['kind']
    runner = _runner(kind)
    cport = int(cfg.get('cport_base', runner['default_cport_base'])) + int(instance_id) * 100
    tmp_paths = [f'/tmp/iperf_{cport}_{i}.json' for i in range(1, int(args.flows) + 1)]
    for path in tmp_paths:
        _safe_unlink(path)

    ecfg = {
        'bw': float(schedule[0]['bw']),
        'delay': float(schedule[0]['delay']),
        'duration': int(args.duration),
        'bdp_mult': float(args.bdp_mult),
        'flows': int(args.flows),
        'link_schedule': _link_schedule(schedule),
    }
    episode = int(run)
    try:
        runner['run_episode'](
            cfg,
            ecfg,
            episode,
            selected_ctx['listener_bin'],
            selected_ctx['python_bin'],
            '',
            '',
            int(instance_id),
        )
    finally:
        time.sleep(0.5)

    csv_dir = os.path.join(run_dir, 'csvs')
    goodputs = []
    for i, tmp_path in enumerate(tmp_paths, start=1):
        dst = os.path.join(run_dir, f'iperf_flow{i}.json')
        _copy_if_exists(tmp_path, dst)
        goodputs.append(_parse_iperf_json(
            dst,
            os.path.join(csv_dir, f'x{i}.csv'),
            measure_interval=float(args.measure_interval),
        ))
    goodput = sum(v for v in goodputs if math.isfinite(v))

    alg_name = selected_ctx['alg_name']
    state_base = os.path.join(
        cfg['outputs']['episodes_dir'],
        f'{alg_name}_state_ep{episode:06d}.csv',
    )
    if os.path.exists(state_base):
        _copy_if_exists(state_base, os.path.join(run_dir, os.path.basename(state_base)))
    base, ext = os.path.splitext(state_base)
    for flow_idx in range(int(args.flows)):
        src = f'{base}_a{flow_idx}{ext}'
        if os.path.exists(src):
            _copy_if_exists(src, os.path.join(run_dir, os.path.basename(src)))

    if goodput <= 0:
        fallback = _state_log_goodput(
            state_base,
            int(args.flows),
            csv_dir=csv_dir,
            measure_interval=float(args.measure_interval),
        )
        if math.isfinite(fallback):
            goodput = fallback
            row['error'] = 'used state-log throughput fallback because iperf JSON was missing'

    if goodput > 0:
        row['success'] = 1
        row['average_goodput_mbps'] = f'{goodput:.6f}'
    else:
        row['error'] = row['error'] or 'missing selected-model goodput'
    return row


def _slot_process(instance_id, work_q, result_q, args, schedules, selected_ctx):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        item = work_q.get()
        if item is None:
            break
        protocol, run = item
        protocol_dir = 'selected_' + _slug(selected_ctx['alg_name']) if protocol == 'selected' else protocol
        run_dir = os.path.join(args.run_dir, protocol_dir, f'run{run}')
        os.makedirs(run_dir, exist_ok=True)
        schedule = schedules[int(run)]
        try:
            if protocol == 'selected':
                row = _run_selected(run, schedule, run_dir, args, instance_id, selected_ctx)
            elif protocol in KERNEL_CC:
                row = _run_kernel(protocol, run, schedule, run_dir, args, instance_id)
            elif protocol == 'orca':
                row = _run_orca(run, schedule, run_dir, args, instance_id)
            else:
                raise ValueError(f'unknown protocol: {protocol}')
            try:
                row['plot_path'] = _plot_protocol_run(
                    run_dir,
                    protocol,
                    row.get('label') or protocol,
                    run,
                    schedule,
                    row,
                    args,
                )
            except Exception as plot_exc:
                print(f'[goodput_bench] plot failed for {protocol} run{run}: {plot_exc}',
                      flush=True)
            _ensure_user_writable(run_dir, args.orca_user)
            result_q.put((protocol, run, row, None))
        except Exception as exc:
            row = _base_row(protocol, protocol, run, schedule, args, run_dir)
            row['error'] = f'{exc}\n{traceback.format_exc()}'
            try:
                row['plot_path'] = _plot_protocol_run(
                    run_dir, protocol, protocol, run, schedule, row, args)
            except Exception:
                pass
            _ensure_user_writable(run_dir, args.orca_user)
            result_q.put((protocol, run, row, row['error']))


def _normalise_protocols(raw_protocols):
    out = []
    for raw in raw_protocols:
        key = raw.strip().lower()
        if key not in PROTOCOL_ALIASES:
            choices = ', '.join(sorted(PROTOCOL_ALIASES))
            raise SystemExit(f'[goodput_bench] unknown protocol {raw!r}; choices: {choices}')
        canonical = PROTOCOL_ALIASES[key]
        if canonical not in out:
            out.append(canonical)
    return out


def _find_latest_checkpoint(cfg, kind, alg_name):
    outputs = cfg.get('outputs', {}) or {}
    default_root = os.path.join(_ROOT, f'{kind}_agent_olympus', 'data')
    root = _resolve_repo_path(outputs.get('root', default_root))
    alg_slug = _slug(alg_name)
    pattern = os.path.join(root, f'{alg_slug}_*', 'checkpoints', f'{alg_slug}_cwnd_model.pt')
    candidates = [p for p in glob.glob(pattern) if os.path.exists(p)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _select_checkpoint(cfg, args, kind, alg_name):
    t_cfg = cfg.get('training', {}) or {}
    raw = args.checkpoint or t_cfg.get('resume_from') or t_cfg.get('checkpoint')
    if raw:
        path = _resolve_repo_path(str(raw))
    else:
        path = _find_latest_checkpoint(cfg, kind, alg_name)
        if path:
            print(f'[goodput_bench] using latest checkpoint: {path}', flush=True)
    if not path:
        raise SystemExit('[goodput_bench] no checkpoint found. Pass --checkpoint.')
    if not os.path.exists(path):
        raise SystemExit(f'[goodput_bench] checkpoint not found: {path}')
    return os.path.abspath(path)


def _prepare_selected_context(args, protocols):
    kind = args.olympus
    runner = _runner(kind)
    config_path = _resolve_repo_path(args.config or runner['default_config'])
    if 'selected' not in protocols:
        return {
            'kind': kind,
            'cfg': {},
            'alg_name': 'selected',
            'label': 'Selected model',
            'listener_bin': '',
            'python_bin': '',
            'learner_port': runner['default_learner_port'],
        }

    if kind == 'single' and int(args.flows) != 1:
        raise SystemExit('[goodput_bench] olympus selected model supports --flows 1 only')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    runner['activate'](cfg)
    runtime = cfg.get('runtime', {}) or {}
    alg_name = runtime.get('algorithm', runner['default_alg'])
    checkpoint = _select_checkpoint(cfg, args, kind, alg_name)
    if kind == 'single':
        model_config = apply_model_config_from_checkpoint(cfg, checkpoint)
        if model_config:
            runner['activate'](cfg)
            runtime = cfg.get('runtime', {}) or {}
            alg_name = runtime.get('algorithm', runner['default_alg'])
            print(f'[goodput_bench] using model config: {model_config}', flush=True)

    outputs = cfg.setdefault('outputs', {})
    selected_root = os.path.join(args.run_dir, 'selected_runtime')
    outputs.update({
        'run_dir': selected_root,
        'checkpoints_dir': os.path.join(selected_root, 'checkpoints'),
        'episodes_dir': os.path.join(selected_root, 'episodes'),
        'plots_dir': os.path.join(selected_root, 'plots'),
        'telemetry_dir': os.path.join(selected_root, 'telemetry'),
        'traces_dir': os.path.join(selected_root, 'episodes'),
        'plot_episodes': bool(args.model_plots),
    })
    for key in ('checkpoints_dir', 'episodes_dir', 'plots_dir', 'telemetry_dir'):
        os.makedirs(outputs[key], exist_ok=True)
    cfg.setdefault('training', {})['checkpoint'] = checkpoint
    cfg['training']['log_path'] = os.path.join(outputs['telemetry_dir'], 'goodput_benchmark_metrics.csv')
    if kind == 'single':
        # Inference must not pass scheduled BW/RTT to the selected model worker.
        # The real Mininet link still follows the schedule; only the worker-side
        # oracle metadata is hidden.
        cfg['hide_link_oracle_from_worker'] = True

    if args.selected_cport_base is not None:
        cfg['cport_base'] = int(args.selected_cport_base)

    paths = cfg.get('paths', {}) or {}
    if 'listener' not in paths or 'py' not in paths:
        raise SystemExit('[goodput_bench] selected config must define paths.listener and paths.py')
    listener_bin = _resolve_repo_path(paths['listener'])
    python_bin = _resolve_repo_path(paths['py'])

    if not args.stochastic:
        os.environ['SAO_DETERMINISTIC'] = '1'
        os.environ['OC_DETERMINISTIC'] = '1'
        os.environ['SAO_NOISE_STD'] = '0.0'
    os.environ['SAO_REQUIRE_CHECKPOINT'] = '1'

    with open(os.path.join(outputs['telemetry_dir'], 'config.resolved.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(outputs['telemetry_dir'], 'selected_model_meta.json'), 'w') as f:
        json.dump({
            'kind': kind,
            'algorithm': alg_name,
            'state': (runtime.get('state', 'default_orca') if kind == 'single' else ''),
            'checkpoint': checkpoint,
            'source_config': config_path,
            'hide_link_oracle_from_worker': cfg.get('hide_link_oracle_from_worker', False),
        }, f, indent=2)

    return {
        'kind': kind,
        'cfg': cfg,
        'alg_name': alg_name,
        'label': alg_name,
        'listener_bin': listener_bin,
        'python_bin': python_bin,
        'learner_port': int((cfg.get('learner', {}) or {}).get(
            'port', runner['default_learner_port'])),
    }


def _metrics_fields():
    return [
        'protocol', 'label', 'run', 'success', 'average_goodput_mbps',
        'optimal_goodput_mbps', 'mean_bw_mbps', 'mean_rtt_ms',
        'initial_bw_mbps', 'initial_rtt_ms', 'min_bw_mbps', 'max_bw_mbps',
        'min_rtt_ms', 'max_rtt_ms', 'flows', 'duration_s',
        'change_interval_s', 'measure_interval_s', 'schedule_seed',
        'run_dir', 'plot_path', 'error',
    ]


def _read_metrics(metrics_csv):
    rows = []
    try:
        with open(metrics_csv, newline='') as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def _empirical_cdf(values):
    vals = np.asarray([_finite_float(v) for v in values], dtype=float)
    vals = vals[np.isfinite(vals)]
    vals.sort()
    if vals.size == 0:
        return vals, vals
    y = np.arange(1, vals.size + 1, dtype=float) / float(vals.size) * 100.0
    return vals, y


def plot_goodput_cdf(metrics_csv, output_pdf):
    rows = _read_metrics(metrics_csv)
    success_rows = [r for r in rows if str(r.get('success', '')).strip() in ('1', '1.0', 'true', 'True')]
    if not success_rows:
        print(f'[goodput_plot] no successful rows in {metrics_csv}', flush=True)
        return False

    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.96)

    by_run = {}
    for row in rows:
        run = row.get('run')
        if run not in by_run:
            by_run[run] = row
    opt_x, opt_y = _empirical_cdf([r.get('optimal_goodput_mbps') for r in by_run.values()])
    if opt_x.size:
        ax.plot(opt_x, opt_y, color=DEFAULT_COLORS['capacity'], linewidth=1.0,
                linestyle='-', label='Mean capacity')

    grouped = defaultdict(list)
    labels = {}
    for row in success_rows:
        protocol = row.get('protocol', '')
        grouped[protocol].append(row.get('average_goodput_mbps'))
        labels[protocol] = row.get('label') or protocol

    for protocol in ('selected', 'bbr3', 'cubic', 'bbr1', 'orca'):
        if protocol not in grouped:
            continue
        x, y = _empirical_cdf(grouped[protocol])
        if not x.size:
            continue
        ax.plot(
            x, y,
            color=DEFAULT_COLORS.get(protocol),
            linewidth=1.1,
            label=labels.get(protocol, protocol),
        )

    ax.set_xlabel('Average Goodput (Mbps)')
    ax.set_ylabel('% of Trials')
    ax.set_ylim(0, 100)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(loc='lower right', frameon=False, fontsize=6)

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    fig.savefig(output_pdf, dpi=1080)
    plt.close(fig)
    print(f'[goodput_plot] saved -> {output_pdf}', flush=True)
    return True


def _write_summary(run_dir, metrics_csv, protocols, schedules, args):
    rows = _read_metrics(metrics_csv)
    summary = {
        'mode': 'random_bw_rtt_goodput_benchmark',
        'run_dir': os.path.abspath(run_dir),
        'protocols': protocols,
        'runs': int(args.runs),
        'duration_s': int(args.duration),
        'change_interval_s': CHANGE_INTERVAL_S,
        'measure_interval_s': float(args.measure_interval),
        'bw_range_mbps': [int(args.bw_min), int(args.bw_max)],
        'rtt_range_ms': [int(args.rtt_min), int(args.rtt_max)],
        'flows': int(args.flows),
        'bdp_mult': float(args.bdp_mult),
        'schedule_seed': 'run_number',
        'completed_rows': len(rows),
        'successful_rows': sum(1 for r in rows if str(r.get('success')) in ('1', '1.0', 'true', 'True')),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    by_protocol = defaultdict(list)
    for row in rows:
        if str(row.get('success')) in ('1', '1.0', 'true', 'True'):
            by_protocol[row.get('protocol')].append(_finite_float(row.get('average_goodput_mbps')))
    summary['mean_goodput_mbps'] = {
        proto: _mean(vals, default=None)
        for proto, vals in by_protocol.items()
    }
    schedules_json = {
        str(run): schedule
        for run, schedule in sorted(schedules.items())
    }
    with open(os.path.join(run_dir, 'schedules.json'), 'w') as f:
        json.dump(schedules_json, f, indent=2)
    out = os.path.join(run_dir, 'benchmark_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _cleanup_orca_processes():
    patterns = [
        'orca-server-mahimahi',
        '/Desktop/mininettestbed/CC/Orca/rl-module/d5.py',
        'clientThr',
    ]
    for pattern in patterns:
        subprocess.run(['pkill', '-KILL', '-f', pattern],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)


def _parse_args(default_olympus='multi'):
    ap = argparse.ArgumentParser(
        description='Run a 50-trial random BW/RTT goodput CDF benchmark.')
    ap.add_argument('--olympus', choices=('multi', 'single'), default=default_olympus,
                    help='which Olympus selected-model runner to use')
    ap.add_argument('--config', default=None,
                    help='selected-model config; defaults to the chosen Olympus config')
    ap.add_argument('--checkpoint', default=None,
                    help='selected-model checkpoint; defaults to latest matching checkpoint')
    ap.add_argument('--protocols', nargs='+',
                    default=['selected', 'bbr3', 'cubic', 'bbr1', 'orca'])
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--duration', type=int, default=300)
    ap.add_argument('--measure-interval', '--measurement-interval',
                    dest='measure_interval', type=float, default=1.0,
                    help='seconds per goodput measurement bucket')
    ap.add_argument('--bw-min', type=int, default=10)
    ap.add_argument('--bw-max', type=int, default=100)
    ap.add_argument('--rtt-min', type=int, default=10)
    ap.add_argument('--rtt-max', type=int, default=100)
    ap.add_argument('--flows', type=int, default=1)
    ap.add_argument('--bdp-mult', type=float, default=1.0)
    ap.add_argument('--n-parallel', type=int, default=1)
    ap.add_argument('--output-root', default=None)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--plot-only', default=None,
                    help='existing benchmark run_dir to replot without running experiments')
    ap.add_argument('--model-plots', action='store_true',
                    help='keep per-episode Olympus PDFs for the selected model')
    ap.add_argument('--stochastic', action='store_true',
                    help='do not force deterministic selected-model inference')
    ap.add_argument('--selected-cport-base', type=int, default=None)
    ap.add_argument('--kernel-cport-base', type=int, default=33000)
    ap.add_argument('--orca-dir', default='/its/home/mm2350/Desktop/mininettestbed/CC/Orca')
    ap.add_argument('--orca-user', default=os.environ.get('SUDO_USER') or os.environ.get('USER') or 'mm2350')
    ap.add_argument('--orca-port-base', type=int, default=4444)
    ap.add_argument('--orca-flow-id-base', type=int, default=0,
                    help='base Orca flow id; per parallel slot adds flows*slot')
    ap.add_argument('--orca-sender-lead', type=float, default=2.0,
                    help='seconds to wait after starting sender.sh before receiver.sh')
    ap.add_argument('--orca-startup-grace', type=float, default=30.0,
                    help='seconds to wait for Orca to print Connected before timing the run')
    ap.add_argument('--orca-finish-grace', type=float, default=30.0,
                    help='extra seconds to wait for receiver.sh stdout to flush after the run duration')
    return ap.parse_args()


def main(default_olympus='multi'):
    args = _parse_args(default_olympus=default_olympus)

    if args.plot_only:
        run_dir = os.path.abspath(args.plot_only)
        metrics_csv = os.path.join(run_dir, 'goodput_metrics.csv')
        plot_goodput_cdf(metrics_csv, os.path.join(run_dir, 'goodput_cdf.pdf'))
        return

    if args.runs <= 0:
        raise SystemExit('[goodput_bench] --runs must be > 0')
    if args.duration <= 0:
        raise SystemExit('[goodput_bench] --duration must be > 0')
    if args.measure_interval <= 0:
        raise SystemExit('[goodput_bench] --measure-interval must be > 0')
    if args.orca_finish_grace < 0:
        raise SystemExit('[goodput_bench] --orca-finish-grace must be >= 0')
    if args.orca_startup_grace < 0:
        raise SystemExit('[goodput_bench] --orca-startup-grace must be >= 0')
    if args.orca_sender_lead < 0:
        raise SystemExit('[goodput_bench] --orca-sender-lead must be >= 0')
    if args.bw_min > args.bw_max or args.rtt_min > args.rtt_max:
        raise SystemExit('[goodput_bench] min range values must be <= max values')
    args.flows = max(1, min(int(args.flows), 4))

    protocols = _normalise_protocols(args.protocols)
    runner = _runner(args.olympus)

    if args.output_root:
        output_root = _resolve_repo_path(args.output_root)
    else:
        cfg_path = _resolve_repo_path(args.config or runner['default_config'])
        try:
            with open(cfg_path) as f:
                tmp_cfg = yaml.safe_load(f)
            output_root = _resolve_repo_path(
                (tmp_cfg.get('outputs', {}) or {}).get(
                    'root',
                    os.path.join(_ROOT, f'{args.olympus}_agent_olympus', 'data'),
                )
            )
        except Exception:
            output_root = os.path.join(_ROOT, f'{args.olympus}_agent_olympus', 'data')

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    args.run_dir = os.path.abspath(os.path.join(
        output_root,
        args.run_name or f'random_bw_rtt_goodput_{args.olympus}_{timestamp}',
    ))
    os.makedirs(args.run_dir, exist_ok=True)

    schedules = _generate_schedules(args)
    selected_ctx = _prepare_selected_context(args, protocols)

    metrics_csv = os.path.join(args.run_dir, 'goodput_metrics.csv')
    with open(metrics_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=_metrics_fields()).writeheader()

    with open(os.path.join(args.run_dir, 'benchmark_config.json'), 'w') as f:
        json.dump({
            'olympus': args.olympus,
            'protocols': protocols,
            'runs': args.runs,
            'duration_s': args.duration,
            'change_interval_s': CHANGE_INTERVAL_S,
            'measure_interval_s': args.measure_interval,
            'bw_range_mbps': [args.bw_min, args.bw_max],
            'rtt_range_ms': [args.rtt_min, args.rtt_max],
            'flows': args.flows,
            'bdp_mult': args.bdp_mult,
            'schedule_seed': 'run_number',
            'per_run_plot': 'protocol/runN/run_plot.pdf',
            'hide_selected_link_oracle': (
                'selected' in protocols and args.olympus == 'single'
            ),
            'orca_dir': args.orca_dir,
            'orca_user': args.orca_user,
            'orca_flow_id_base': args.orca_flow_id_base,
            'orca_sender_lead_s': args.orca_sender_lead,
            'orca_startup_grace_s': args.orca_startup_grace,
            'orca_finish_grace_s': args.orca_finish_grace,
        }, f, indent=2)

    tasks = [(protocol, run) for protocol in protocols for run in range(1, int(args.runs) + 1)]
    n_parallel = max(1, min(int(args.n_parallel), len(tasks)))

    print(f'[goodput_bench] run_dir={args.run_dir}', flush=True)
    print(f'[goodput_bench] protocols={protocols}', flush=True)
    print(f'[goodput_bench] runs={args.runs} duration={args.duration}s '
          f'change_interval={CHANGE_INTERVAL_S}s n_parallel={n_parallel}', flush=True)
    if 'selected' in protocols:
        print(f'[goodput_bench] selected={selected_ctx["alg_name"]}', flush=True)

    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    procs = []
    for instance_id in range(n_parallel):
        proc = multiprocessing.Process(
            target=_slot_process,
            args=(instance_id, work_q, result_q, args, schedules, selected_ctx),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    for task in tasks:
        work_q.put(task)
    for _ in procs:
        work_q.put(None)

    completed = 0
    failures = 0
    try:
        while completed < len(tasks):
            protocol, run, row, error = result_q.get()
            completed += 1
            if error or not _as_bool(row.get('success')):
                failures += 1
                err_line = str(row.get('error', '')).splitlines()[0]
                print(f'[goodput_bench] {protocol} run{run} failed: {err_line}', flush=True)
            else:
                print(f'[goodput_bench] {protocol} run{run} '
                      f'goodput={row["average_goodput_mbps"]} Mbps '
                      f'({completed}/{len(tasks)})', flush=True)
            with open(metrics_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=_metrics_fields())
                for key in _metrics_fields():
                    row.setdefault(key, '')
                writer.writerow(row)
    except KeyboardInterrupt:
        print('\n[goodput_bench] interrupted - stopping workers', flush=True)
    finally:
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
        try:
            runner['final_cleanup'](selected_ctx.get('learner_port', runner['default_learner_port']))
        except Exception:
            pass
        if 'orca' in protocols:
            _cleanup_orca_processes()

    summary = _write_summary(args.run_dir, metrics_csv, protocols, schedules, args)
    plot_goodput_cdf(metrics_csv, os.path.join(args.run_dir, 'goodput_cdf.pdf'))
    print(f'[goodput_bench] done failures={failures}', flush=True)
    print(f'[goodput_bench] summary={os.path.join(args.run_dir, "benchmark_summary.json")}', flush=True)
    print(f'[goodput_bench] cdf={os.path.join(args.run_dir, "goodput_cdf.pdf")}', flush=True)
    if summary.get('mean_goodput_mbps'):
        print(f'[goodput_bench] mean_goodput_mbps={summary["mean_goodput_mbps"]}', flush=True)


if __name__ == '__main__':
    main()
