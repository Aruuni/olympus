#!/usr/bin/env python3
"""Fairness heatmap benchmark for two-flow sharing.

Grid:
  BW  = 20, 50, 100, 150, 200 Mbps
  RTT = 10, 20, 30, 40, 50 ms

Each cell runs 5 trials by default. Flow 1 runs for the full episode;
flow 2 joins at the configured absolute start time. Metrics are scored from
the configured final window.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
import traceback
import multiprocessing

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    _parse_iperf_json,
    _prepare_runtime_cfg,
    _resolve_from,
    _restore_sudo_user_ownership,
    _resume_rows_by_key,
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


GOODPUT_SOURCE = 'iperf3_receiver_final20s'

FAIRNESS_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'kind', 'algorithm', 'kernel_cc',
    'state', 'reward', 'training_environment', 'benchmark_environment',
    'cell', 'run', 'seed', 'duration_s', 'bw_mbps', 'rtt_ms', 'flows',
    'first_flow_start_s', 'second_flow_start_s', 'score_window_s',
    'flow1_duration_s', 'flow2_duration_s',
    'mean_total_goodput_mbps', 'mean_jain_fairness',
    'mean_abs_fair_share_error_mbps', 'p95_abs_fair_share_error_mbps',
    'goodput_ratio_total_mean', 'goodput_ratio_total_std',
    'goodput_ratio_total_samples', 'goodput_ratio_total_sum',
    'goodput_ratio_total_sumsq', 'goodput_source',
    'state_logs', 'schedule_csv', 'flow_schedule_csv', 'individual_plot',
    'run_dir', 'checkpoint', 'error',
]

SUMMARY_FIELDS = [
    'approach', 'data_folder', 'plot_label', 'kind', 'algorithm', 'kernel_cc',
    'state', 'reward', 'training_environment', 'benchmark_environment',
    'bw_mbps', 'rtt_ms', 'runs_completed', 'runs_expected',
    'goodput_ratio_total_mean', 'goodput_ratio_total_std',
    'goodput_ratio_total_samples', 'goodput_source',
    'mean_jain_fairness', 'std_jain_fairness',
    'mean_total_goodput_mbps', 'std_total_goodput_mbps',
    'mean_abs_fair_share_error_mbps', 'std_abs_fair_share_error_mbps',
]


def _fairness_cfg(cfg: dict) -> dict:
    bench = dict(cfg.get('benchmark') or {})
    bench.setdefault('name', 'fairness_two_flow_join_halfway')
    bench.setdefault('runs_per_cell', 5)
    bench.setdefault('duration_s', 60)
    bench.setdefault('first_flow_start_s', 0)
    bench.setdefault('second_flow_start_s', 20)
    bench.setdefault('score_window_s', 20)
    bench.setdefault('bw_mbps_values', [20, 50, 100, 150, 200])
    bench.setdefault('rtt_ms_values', [10, 20, 30, 40, 50])
    bench.setdefault('bdp_mult', 4)
    bench.setdefault('measure_interval_s', 1.0)
    bench.setdefault('kernel_cport_base', 53000)
    bench.setdefault('n_parallel', 5)
    bench.setdefault('plot_episodes', True)
    bench.setdefault('deterministic_inference', True)
    bench.setdefault('seed_offset', 0)
    bench.setdefault('cwnd_ylim_percentile', 99.0)
    bench.setdefault('cwnd_ylim_headroom', 1.15)
    bench['flows'] = 2
    bench['change_interval_s'] = 0
    return bench


def _prepare_fairness_runtime_cfg(base_cfg: dict, approach: dict,
                                  checkpoint: str, run_dir: str) -> dict:
    scratch_dir = os.path.join(
        '/tmp', f'olympus_fairness_runtime_{os.getpid()}', approach['_label'])
    cfg = _prepare_runtime_cfg(
        base_cfg, approach, checkpoint, scratch_dir, run_dir,
        plot_episodes=True)
    cfg['outputs']['run_dir'] = scratch_dir
    cfg['outputs']['checkpoints_dir'] = os.path.join(scratch_dir, 'checkpoints')
    cfg['outputs']['telemetry_dir'] = os.path.join(scratch_dir, 'telemetry')
    cfg['training']['log_path'] = os.path.join(
        scratch_dir, 'telemetry', 'benchmark_metrics.csv')
    return cfg


def _grid_cells(bench: dict) -> list:
    cells = []
    for rtt in bench['rtt_ms_values']:
        for bw in bench['bw_mbps_values']:
            cells.append({'bw': float(bw), 'rtt': float(rtt)})
    return cells


def _flow_plan(bench: dict) -> list:
    duration = float(bench['duration_s'])
    first_start = float(bench.get('first_flow_start_s', 0.0))
    second_start = float(bench['second_flow_start_s'])
    return [
        {'flow': 1, 'start': first_start,
         'duration': max(1.0, duration - first_start), 'end': duration},
        {'flow': 2, 'start': second_start,
         'duration': max(1.0, duration - second_start), 'end': duration},
    ]


def _static_schedule(bench: dict, bw_mbps: float, rtt_ms: float, seed: int) -> dict:
    return {
        'seed': int(seed),
        'initial': {'bw': float(bw_mbps), 'delay': float(rtt_ms)},
        'changes': [],
        'rows': [{'t': 0, 'bw': float(bw_mbps), 'delay': float(rtt_ms)}],
    }


def _cell_name(bw_mbps: float, rtt_ms: float) -> str:
    return f'bw{int(round(float(bw_mbps)))}_rtt{int(round(float(rtt_ms)))}'


def _seed_for(bench: dict, cell_idx: int, run_number: int) -> int:
    return int(bench.get('seed_offset', 0)) + cell_idx * 1000 + int(run_number)


def _iperf_client_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_{int(cport)}_{int(flow)}.json'


def _iperf_receiver_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_server_{int(cport)}_{int(flow)}.json'


def _clear_iperf_tmp_outputs(cport: int, n_flows: int) -> None:
    for flow in range(1, int(n_flows) + 1):
        _safe_unlink(_iperf_client_tmp(cport, flow))
        _safe_unlink(_iperf_receiver_tmp(cport, flow))


def _receiver_iperf_flows(run_dir: str, flow_plan: list) -> list:
    return _kernel_flows(run_dir, flow_plan)


def _iperf_rtt_flows(run_dir: str, flow_plan: list) -> list:
    """Per-flow TCP RTT samples from the saved iperf3 client JSON."""
    out = []
    for item in flow_plan:
        flow = int(item['flow'])
        start = float(item['start'])
        parsed = _parse_iperf_json(
            os.path.join(run_dir, f'iperf_client_flow{flow}.json'))
        samples = parsed.get('rtt_samples_ms', []) or []
        out.append({
            'flow': flow,
            'start': start,
            'end': float(item['end']),
            't': np.asarray(
                [float(time_s) + start for time_s, _ in samples],
                dtype=float),
            'iperf_rtt': np.asarray(
                [float(rtt_ms) for _, rtt_ms in samples], dtype=float),
        })
    return out


def _flows_have_receiver_goodput(flows: list, n_flows: int) -> bool:
    if len(flows) < int(n_flows):
        return False
    for flow in flows[:int(n_flows)]:
        thr = np.asarray(flow.get('thr', []), dtype=float)
        if not np.isfinite(thr).any():
            return False
    return True


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


def _flow_id_from_suffix(raw_id: int, min_suffix: int) -> int:
    return int(raw_id) + 1 if int(min_suffix) == 0 else int(raw_id)


def _read_rl_state_flow(path: str, flow: int, start: float, end: float) -> dict:
    data = {
        'flow': flow, 'start': start, 'end': end,
        't': [], 'thr': [], 'rtt': [], 'srtt': [], 'cwnd': [],
        'cwnd_mult': [], 'reward': [], 'loss': [],
        'min_rtt': [], 'kalman_rtt': [], 'act_mu': [], 'act_sig': [],
    }
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                t_s = _finite_float(row.get('t_s'))
                if not math.isfinite(t_s):
                    continue
                data['t'].append(t_s)
                for key, col in (
                    ('thr', 'avg_thr_mbps'),
                    ('rtt', 'avg_urtt_ms'),
                    ('srtt', 'srtt_ms'),
                    ('cwnd', 'cwnd'),
                    ('cwnd_mult', 'cwnd_mult_applied'),
                    ('reward', 'reward'),
                    ('loss', 'loss_ratio'),
                    ('min_rtt', 'min_rtt_ms'),
                    ('kalman_rtt', 'kalman_rtt_ms'),
                    ('act_mu', 'act_mu'),
                    ('act_sig', 'act_sig'),
                ):
                    value = _finite_float(row.get(col))
                    if key == 'srtt' and not math.isfinite(value):
                        value = _finite_float(row.get('avg_urtt_ms'))
                    if key == 'cwnd_mult' and not math.isfinite(value):
                        value = _finite_float(row.get('cwnd_mult'))
                    data[key].append(value if math.isfinite(value) else math.nan)
    except (FileNotFoundError, OSError):
        pass
    return {
        k: np.asarray(v, dtype=float) if isinstance(v, list) else v
        for k, v in data.items()
    }


def _state_flows(run_dir: str, alg_name: str, flow_plan: list) -> list:
    pattern = os.path.join(run_dir, f'{alg_name}_state_ep000001_a*.csv')
    return _state_flows_from_paths(sorted(glob.glob(pattern)), flow_plan)


def _state_flows_from_paths(paths: list, flow_plan: list) -> list:
    out = []
    suffixes = []
    for path in paths:
        m = re.search(r'_flow(\d+)\.csv$', path)
        if m:
            suffixes.append(int(m.group(1)))
    min_suffix = min(suffixes) if suffixes else 1
    for path in paths:
        agent_match = re.search(r'_a(\d+)\.csv$', path)
        flow_match = re.search(r'_flow(\d+)\.csv$', path)
        if agent_match:
            flow = int(agent_match.group(1)) + 1
        elif flow_match:
            flow = _flow_id_from_suffix(int(flow_match.group(1)), min_suffix)
        else:
            flow = len(out) + 1
        plan = flow_plan[flow - 1] if flow - 1 < len(flow_plan) else {
            'start': 0.0, 'end': float('inf')}
        out.append(_read_rl_state_flow(path, flow, plan['start'], plan['end']))
    return out


def _rl_flow_traces_for_row(row: dict, flow_plan: list) -> list:
    logs = [p for p in str(row.get('state_logs', '')).split(';') if p]
    if logs:
        return _state_flows_from_paths(sorted(logs), flow_plan)
    alg_name = str(row.get('algorithm') or '').strip()
    run_dir = str(row.get('run_dir') or '').strip()
    if alg_name and run_dir:
        return _state_flows(run_dir, alg_name, flow_plan)
    return []


def _flow_plan_from_row(row: dict) -> list:
    first_start = _finite_float(row.get('first_flow_start_s'), 0.0)
    second_start = _finite_float(row.get('second_flow_start_s'), 20.0)
    duration = _finite_float(row.get('duration_s'), 60.0)
    return [
        {'flow': 1, 'start': first_start,
         'duration': max(1.0, duration - first_start), 'end': duration},
        {'flow': 2, 'start': second_start,
         'duration': max(1.0, duration - second_start), 'end': duration},
    ]


def _score_window_mask(series: dict, bench: dict) -> np.ndarray:
    times = np.asarray(series.get('time', []), dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    duration = float(bench.get('duration_s', times[-1] + 0.5))
    score_window = float(bench.get('score_window_s', 20.0))
    window_start = max(0.0, duration - max(0.0, score_window))
    active_count = np.asarray(series.get('active_count', []), dtype=float)
    if active_count.size != times.size:
        active_count = np.ones(times.size, dtype=float)
    return (times >= window_start) & (times < duration) & (active_count > 0)


def _metrics_from_score_window(series: dict, bench: dict) -> dict:
    def mean_finite(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else math.nan

    def pct_finite(values, pct):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.percentile(arr, pct)) if arr.size else math.nan

    active = _score_window_mask(series, bench)
    return {
        'mean_total_goodput_mbps': mean_finite(series['total'][active]),
        'mean_jain_fairness': mean_finite(series['jain'][active]),
        'mean_abs_fair_share_error_mbps': mean_finite(series['fair_error'][active]),
        'p95_abs_fair_share_error_mbps': pct_finite(series['fair_error'][active], 95),
    }


def _goodput_ratio_stats(series: dict, flow_plan: list, bench: dict) -> dict:
    per_flow = np.asarray(series.get('per_flow', []), dtype=float)
    times = np.asarray(series.get('time', []), dtype=float)
    score_mask = _score_window_mask(series, bench)
    ratios = []
    for sec, t_mid in enumerate(times):
        if sec >= score_mask.size or not bool(score_mask[sec]):
            continue
        active = [
            int(p['flow']) - 1 for p in flow_plan
            if float(p['start']) <= float(t_mid) < float(p['end'])
        ]
        if len(active) < 2:
            continue
        vals = []
        for idx in active:
            val = per_flow[idx, sec] if idx < per_flow.shape[0] else math.nan
            vals.append(float(val) if math.isfinite(val) else 0.0)
        vals = np.asarray(vals, dtype=float)
        hi = float(np.max(vals)) if vals.size else 0.0
        if hi <= 0.0:
            continue
        ratios.append(float(np.min(vals) / hi))

    arr = np.asarray(ratios, dtype=float)
    arr = arr[np.isfinite(arr)]
    samples = int(arr.size)
    total = float(np.sum(arr)) if samples else 0.0
    sumsq = float(np.sum(arr * arr)) if samples else 0.0
    return {
        'goodput_ratio_total_mean': float(np.mean(arr)) if samples else math.nan,
        'goodput_ratio_total_std': (
            float(np.std(arr, ddof=1)) if samples > 1
            else 0.0 if samples == 1 else math.nan
        ),
        'goodput_ratio_total_samples': samples,
        'goodput_ratio_total_sum': total,
        'goodput_ratio_total_sumsq': sumsq,
    }


def _fairness_metrics_from_flows(flows: list, flow_plan: list, bench: dict) -> dict:
    series = _binned_series(flows, flow_plan, bench)
    metrics = _metrics_from_score_window(series, bench)
    metrics.update(_goodput_ratio_stats(series, flow_plan, bench))
    metrics['goodput_source'] = GOODPUT_SOURCE
    return metrics


def _plot_flow_lines(ax, flows: list, key: str, colors: list,
                     label_prefix: str = 'flow', linewidth: float = 1.0,
                     positive_only: bool = True, linestyle: str = '-') -> bool:
    plotted = False
    for flow in flows:
        vals = np.asarray(flow.get(key, []), dtype=float)
        t_vals = np.asarray(flow.get('t', []), dtype=float)
        mask = np.isfinite(t_vals) & np.isfinite(vals)
        if positive_only:
            mask &= vals > 0
        if not mask.any():
            continue
        color = colors[(int(flow['flow']) - 1) % len(colors)]
        ax.plot(t_vals[mask], vals[mask], color=color, linewidth=linewidth,
                alpha=0.9, linestyle=linestyle,
                label=f'{label_prefix} {int(flow["flow"])}')
        plotted = True
    return plotted


def _flow_values(flows: list, key: str, positive_only: bool = True) -> np.ndarray:
    out = []
    for flow in flows:
        vals = np.asarray(flow.get(key, []), dtype=float)
        mask = np.isfinite(vals)
        if positive_only:
            mask &= vals > 0
        if mask.any():
            out.append(vals[mask])
    if not out:
        return np.asarray([], dtype=float)
    return np.concatenate(out)


def _set_cwnd_ylim(ax, flows: list, bench: dict) -> None:
    vals = _flow_values(flows, 'cwnd')
    if vals.size == 0:
        ax.set_ylim(bottom=0)
        return

    pct = float(bench.get('cwnd_ylim_percentile', 99.0))
    pct = min(100.0, max(50.0, pct))
    headroom = max(1.0, float(bench.get('cwnd_ylim_headroom', 1.15)))
    robust_hi = float(np.nanpercentile(vals, pct))
    max_hi = float(np.nanmax(vals))
    upper = max(10.0, robust_hi * headroom)
    ax.set_ylim(0, upper)

    if max_hi > upper:
        ax.text(
            0.01, 0.92, f'zoomed to p{pct:g}; max cwnd {max_hi:.0f}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=7, color='#555555',
        )


def _plot_fairness_run(receiver_flows: list, flow_plan: list, bench: dict,
                       label: str, output: str, rl_flows: list = None,
                       iperf_rtt_flows: list = None) -> str:
    rl_flows = rl_flows or []
    iperf_rtt_flows = iperf_rtt_flows or []
    series = _binned_series(receiver_flows, flow_plan, bench)
    t = series['time']
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [f'C{i}' for i in range(10)])

    has_rl_state = bool(rl_flows)
    nrows = 7 if has_rl_state else 5
    fig, axes = plt.subplots(
        nrows, 1, figsize=(11.5, 16.0 if has_rl_state else 11.5),
        sharex=True)
    fig.suptitle(label, fontsize=13, fontweight='bold')

    score_start = max(
        0.0,
        float(bench['duration_s']) - float(bench.get('score_window_s', 20.0)),
    )
    for ax in axes:
        for item in flow_plan:
            ax.axvline(float(item['start']), color='black', alpha=0.18, linewidth=0.8)
            ax.axvline(float(item['end']), color='black', alpha=0.12,
                       linewidth=0.8, linestyle=':')
        ax.axvspan(score_start, float(bench['duration_s']),
                   color='#d9ead3', alpha=0.22, linewidth=0)
        ax.grid(True, alpha=0.25)

    axes[0].step(t, series['active_count'], where='mid', color='black', linewidth=1.4)
    axes[0].set_ylabel('Active flows')
    axes[0].set_ylim(bottom=0)
    ax_share = axes[0].twinx()
    ax_share.step(t, series['fair_share'], where='mid', color='#4878cf',
                  linewidth=1.2, label='fair share')
    ax_share.set_ylabel('Fair share Mbps')
    axes[0].set_title('Flow schedule and final scoring window', fontsize=9, loc='left')

    for flow in receiver_flows:
        color = colors[(int(flow['flow']) - 1) % len(colors)]
        axes[1].plot(flow['t'], flow['thr'], color=color, linewidth=1.0,
                     label=f'flow {int(flow["flow"])}')
    axes[1].step(t, series['fair_share'], where='mid', color='black',
                 linestyle='--', linewidth=1.2, label='fair share')
    axes[1].set_ylabel('Receiver goodput Mbps')
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='upper right', ncol=3, fontsize=7, frameon=False)

    srtt_plotted = _plot_flow_lines(
        axes[2], rl_flows, 'srtt', colors,
        label_prefix='srtt flow', linewidth=0.85)
    kalman_plotted = _plot_flow_lines(
        axes[2], rl_flows, 'kalman_rtt', colors,
        label_prefix='kalman flow', linewidth=1.0, linestyle='-.')
    iperf_plotted = _plot_flow_lines(
        axes[2], iperf_rtt_flows, 'iperf_rtt', colors,
        label_prefix='iperf RTT flow', linewidth=1.2)
    axes[2].axhline(float(bench['rtt_ms']), color='black', linestyle='--',
                    linewidth=1.0, label='base RTT')
    axes[2].set_ylabel('RTT ms')
    axes[2].set_ylim(bottom=0)
    axes[2].set_title('iperf3 TCP RTT and RL estimates when available',
                      fontsize=9, loc='left')
    if srtt_plotted or kalman_plotted or iperf_plotted:
        axes[2].legend(loc='upper right', ncol=3, fontsize=7, frameon=False)
    else:
        axes[2].text(0.5, 0.5, 'No iperf3 RTT samples for this run',
                     transform=axes[2].transAxes, ha='center', va='center',
                     fontsize=9, color='#555555')

    if has_rl_state:
        cwnd_plotted = _plot_flow_lines(
            axes[3], rl_flows, 'cwnd', colors, linewidth=0.85)
        axes[3].set_ylabel('RL CWND pkts')
        _set_cwnd_ylim(axes[3], rl_flows, bench)
        axes[3].set_title('Per-flow CWND from RL clients', fontsize=9, loc='left')
        if cwnd_plotted:
            axes[3].legend(loc='upper right', ncol=3, fontsize=7, frameon=False)

        mult_plotted = _plot_flow_lines(
            axes[4], rl_flows, 'cwnd_mult', colors,
            label_prefix='mult flow', linewidth=0.8, positive_only=False)
        axes[4].axhline(
            1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        axes[4].set_ylabel('RL CWND mult')
        axes[4].set_title('RL action multiplier and reward', fontsize=9, loc='left')
        ax_reward = axes[4].twinx()
        reward_plotted = _plot_flow_lines(
            ax_reward, rl_flows, 'reward', colors,
            label_prefix='reward flow', linewidth=0.7,
            positive_only=False, linestyle=':')
        ax_reward.set_ylabel('Reward')
        if mult_plotted or reward_plotted:
            handles, labels = axes[4].get_legend_handles_labels()
            r_handles, r_labels = ax_reward.get_legend_handles_labels()
            axes[4].legend(
                handles + r_handles, labels + r_labels,
                loc='upper right', ncol=2, fontsize=7, frameon=False)

    total_ax = axes[5] if has_rl_state else axes[3]
    fairness_ax = axes[6] if has_rl_state else axes[4]
    total_ax.plot(t, series['total'], color='#59a14f', linewidth=1.5,
                  label='aggregate goodput')
    total_ax.axhline(float(bench['bw_mbps']), color='black', linestyle='--',
                     linewidth=1.0, label='link capacity')
    total_ax.set_ylabel('Total Mbps')
    total_ax.set_ylim(bottom=0)
    total_ax.legend(loc='upper right', fontsize=8, frameon=False)

    fairness_ax.plot(t, series['jain'], color='#e15759', linewidth=1.4,
                     label='Jain fairness')
    fairness_ax.set_ylabel('Jain fairness')
    fairness_ax.set_xlabel('Time (s)')
    fairness_ax.set_ylim(0, 1.05)
    fairness_ax.legend(loc='lower right', fontsize=8, frameon=False)
    fairness_ax.set_xlim(0, float(bench['duration_s']))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    try:
        fig.savefig(output, dpi=220, bbox_inches='tight')
    finally:
        plt.close(fig)
    return output


def _row(approach: dict, bench: dict, cell_idx: int, run_number: int,
         bw_mbps: float, rtt_ms: float, seed: int, run_dir: str,
         schedule_csv: str, flow_schedule_csv: str, checkpoint: str,
         state_logs: list, individual_plot: str, metrics: dict,
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
        'cell': cell_idx,
        'run': int(run_number),
        'seed': int(seed),
        'duration_s': int(bench['duration_s']),
        'bw_mbps': float(bw_mbps),
        'rtt_ms': float(rtt_ms),
        'flows': 2,
        'first_flow_start_s': float(bench.get('first_flow_start_s', 0.0)),
        'second_flow_start_s': float(bench['second_flow_start_s']),
        'score_window_s': float(bench.get('score_window_s', 20.0)),
        'flow1_duration_s': max(
            1.0, float(bench['duration_s']) - float(bench.get('first_flow_start_s', 0.0))),
        'flow2_duration_s': max(
            1.0, float(bench['duration_s']) - float(bench['second_flow_start_s'])),
        'state_logs': ';'.join(state_logs),
        'schedule_csv': schedule_csv,
        'flow_schedule_csv': flow_schedule_csv,
        'individual_plot': individual_plot,
        'run_dir': run_dir,
        'checkpoint': checkpoint,
        'error': error,
    }
    row.update(metrics or {})
    for key in FAIRNESS_FIELDS:
        row.setdefault(key, '')
    return row


def _run_kernel_trial(instance_id: int, approach: dict, bench: dict,
                      cell_idx: int, run_number: int, bw_mbps: float,
                      rtt_ms: float, seed: int, run_dir: str,
                      schedule_csv: str, flow_schedule_csv: str) -> dict:
    flow_plan = _flow_plan(bench)
    cport = int(bench.get('kernel_cport_base', 53000)) + int(instance_id) * 100
    _clear_iperf_tmp_outputs(cport, len(flow_plan))

    kind = str(approach.get('kind', 'kernel')).lower()
    cc_algo = 'astraea' if kind == 'astraea' else str(approach['kernel_cc'])

    error = ''
    flows = []
    metrics = {}
    try:
        print(f'[fair_bench] slot={instance_id} {approach["_label"]} '
              f'bw={bw_mbps:g} rtt={rtt_ms:g} run={run_number} '
              f'-C {cc_algo}', flush=True)
        env = MininetEnv(
            n=2,
            bw=float(bw_mbps),
            delay=float(rtt_ms),
            bdp_mult=float(bench['bdp_mult']),
            duration=int(bench['duration_s']),
            cport=cport,
            cc_algo=cc_algo,
            instance_id=instance_id,
        )
        try:
            env.start()
            env.run_iperf(
                monitor_interval=float(bench.get('measure_interval_s', 1.0)),
                start_delays=[p['start'] for p in flow_plan],
                flow_durations=[p['duration'] for p in flow_plan],
            )
            time.sleep(float(bench['duration_s']) + 3.0)
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
            metrics = _fairness_metrics_from_flows(flows, flow_plan, {
                'duration_s': int(bench['duration_s']),
                'bw_mbps': float(bw_mbps),
                'score_window_s': float(bench.get('score_window_s', 20.0)),
            })
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'

    plot_path = os.path.join(run_dir, 'fairness_run.pdf')
    if flows and _as_bool(bench.get('plot_episodes'), default=True):
        try:
            _plot_fairness_run(
                flows, flow_plan,
                {'duration_s': int(bench['duration_s']), 'bw_mbps': float(bw_mbps),
                 'rtt_ms': float(rtt_ms),
                 'score_window_s': float(bench.get('score_window_s', 20.0))},
                f'{_approach_plot_label(approach)} BW={bw_mbps:g} RTT={rtt_ms:g} run={run_number}',
                plot_path,
                iperf_rtt_flows=_iperf_rtt_flows(run_dir, flow_plan),
            )
        except Exception as plot_exc:
            print(f'[fair_bench] individual plot failed for {plot_path}: {plot_exc}',
                  flush=True)

    return _row(
        approach, bench, cell_idx, run_number, bw_mbps, rtt_ms, seed,
        run_dir, schedule_csv, flow_schedule_csv, '', [], plot_path,
        metrics, error)


def _run_orca_trial(instance_id: int, approach: dict, bench: dict,
                    cell_idx: int, run_number: int, bw_mbps: float,
                    rtt_ms: float, seed: int, run_dir: str,
                    schedule_csv: str, flow_schedule_csv: str) -> dict:
    """Real Orca (SIGCOMM'20) fairness trial: same 2-flow dumbbell as the
    kernel path, driven by orca-server-mahimahi/clientThr; receiver goodput
    is parsed into csvs/x{f}.csv so metrics/plots are unchanged."""
    flow_plan = _flow_plan(bench)
    settings = _orca_settings(approach, bench)
    cport = int(bench.get('kernel_cport_base', 53000)) + int(instance_id) * 100

    error = ''
    flows = []
    metrics = {}
    try:
        print(f'[fair_bench] slot={instance_id} {approach["_label"]} '
              f'bw={bw_mbps:g} rtt={rtt_ms:g} run={run_number} (real-orca)',
              flush=True)
        env = MininetEnv(
            n=2,
            bw=float(bw_mbps),
            delay=float(rtt_ms),
            bdp_mult=float(bench['bdp_mult']),
            duration=int(bench['duration_s']),
            cport=cport,
            cc_algo='cubic',
            instance_id=instance_id,
        )
        try:
            env.start()
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
            metrics = _fairness_metrics_from_flows(flows, flow_plan, {
                'duration_s': int(bench['duration_s']),
                'bw_mbps': float(bw_mbps),
                'score_window_s': float(bench.get('score_window_s', 20.0)),
            })
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'

    plot_path = os.path.join(run_dir, 'fairness_run.pdf')
    if flows and _as_bool(bench.get('plot_episodes'), default=True):
        try:
            _plot_fairness_run(
                flows, flow_plan,
                {'duration_s': int(bench['duration_s']), 'bw_mbps': float(bw_mbps),
                 'rtt_ms': float(rtt_ms),
                 'score_window_s': float(bench.get('score_window_s', 20.0))},
                f'{_approach_plot_label(approach)} BW={bw_mbps:g} RTT={rtt_ms:g} run={run_number}',
                plot_path,
            )
        except Exception as plot_exc:
            print(f'[fair_bench] individual plot failed for {plot_path}: {plot_exc}',
                  flush=True)

    return _row(
        approach, bench, cell_idx, run_number, bw_mbps, rtt_ms, seed,
        run_dir, schedule_csv, flow_schedule_csv, '', [], plot_path,
        metrics, error)


def _run_model_trial(instance_id: int, approach: dict, bench: dict,
                     base_cfg: dict, checkpoint: str, listener_bin: str,
                     python_bin: str, cell_idx: int, run_number: int,
                     bw_mbps: float, rtt_ms: float, seed: int,
                     run_dir: str, schedule_csv: str,
                     flow_schedule_csv: str) -> dict:
    flow_plan = _flow_plan(bench)
    cfg = _prepare_fairness_runtime_cfg(base_cfg, approach, checkpoint, run_dir)
    cfg['listener_single_flow'] = True
    cport = int(cfg.get('cport_base', 21000)) + int(instance_id) * 100
    _clear_iperf_tmp_outputs(cport, len(flow_plan))
    if _as_bool(bench.get('deterministic_inference'), default=True):
        _deterministic_env(cfg)
    with open(os.path.join(run_dir, 'config.resolved.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    ecfg = {
        'bw': float(bw_mbps),
        'delay': float(rtt_ms),
        'bdp_mult': float(bench['bdp_mult']),
        'flows': 2,
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
    rl_flows = []
    metrics = {}
    try:
        print(f'[fair_bench] slot={instance_id} {approach["_label"]} '
              f'bw={bw_mbps:g} rtt={rtt_ms:g} run={run_number}', flush=True)
        run_episode_auto(cfg, ecfg, 1, listener_bin, python_bin, '', '', instance_id)
        parse_error = _copy_receiver_iperf_outputs(cport, run_dir, flow_plan, bench)
        flows = _receiver_iperf_flows(run_dir, flow_plan)
        rl_flows = _state_flows(run_dir, str(approach['algorithm']), flow_plan)
        if parse_error:
            error = parse_error
        elif not _flows_have_receiver_goodput(flows, len(flow_plan)):
            error = 'missing receiver iperf goodput samples'
        else:
            metrics = _fairness_metrics_from_flows(flows, flow_plan, {
                'duration_s': int(bench['duration_s']),
                'bw_mbps': float(bw_mbps),
                'score_window_s': float(bench.get('score_window_s', 20.0)),
            })
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'

    state_logs = sorted(glob.glob(os.path.join(
        run_dir, f'{approach["algorithm"]}_state_ep000001_a*.csv')))
    plot_path = os.path.join(run_dir, 'fairness_run.pdf')
    if flows and _as_bool(bench.get('plot_episodes'), default=True):
        try:
            _plot_fairness_run(
                flows, flow_plan,
                {'duration_s': int(bench['duration_s']), 'bw_mbps': float(bw_mbps),
                 'rtt_ms': float(rtt_ms),
                 'score_window_s': float(bench.get('score_window_s', 20.0))},
                f'{_approach_plot_label(approach)} BW={bw_mbps:g} RTT={rtt_ms:g} run={run_number}',
                plot_path,
                rl_flows=rl_flows,
                iperf_rtt_flows=_iperf_rtt_flows(run_dir, flow_plan),
            )
        except Exception as plot_exc:
            print(f'[fair_bench] individual plot failed for {plot_path}: {plot_exc}',
                  flush=True)

    return _row(
        approach, bench, cell_idx, run_number, bw_mbps, rtt_ms, seed,
        run_dir, schedule_csv, flow_schedule_csv, checkpoint, state_logs,
        plot_path, metrics, error)


def _approach_slot(instance_id: int, work_q, result_q, approach: dict,
                   bench: dict, base_cfg: dict, checkpoint: str,
                   listener_bin: str, python_bin: str) -> None:
    while True:
        item = work_q.get()
        if item is None:
            break
        (cell_idx, run_number, bw_mbps, rtt_ms, seed, run_dir,
         schedule_csv, flow_schedule_csv) = item
        kind = str(approach.get('kind', 'model')).lower()
        if kind in ('kernel', 'astraea'):
            row = _run_kernel_trial(
                instance_id, approach, bench, cell_idx, run_number,
                bw_mbps, rtt_ms, seed, run_dir, schedule_csv,
                flow_schedule_csv)
        elif kind == 'orca':
            row = _run_orca_trial(
                instance_id, approach, bench, cell_idx, run_number,
                bw_mbps, rtt_ms, seed, run_dir, schedule_csv,
                flow_schedule_csv)
        else:
            row = _run_model_trial(
                instance_id, approach, bench, base_cfg, checkpoint,
                listener_bin, python_bin, cell_idx, run_number,
                bw_mbps, rtt_ms, seed, run_dir, schedule_csv,
                flow_schedule_csv)
        result_q.put(row)


def _expected_trial_count(bench: dict) -> int:
    return len(_grid_cells(bench)) * int(bench['runs_per_cell'])


def _expected_trial_keys(bench: dict) -> set:
    return {
        (float(cell['bw']), float(cell['rtt']), run)
        for cell in _grid_cells(bench)
        for run in range(1, int(bench['runs_per_cell']) + 1)
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
    """A trial counts as done only if it finished without error and scored."""
    if str(row.get('error', '')).strip():
        return False
    return str(row.get('goodput_source', '')).strip() == GOODPUT_SOURCE


def _completed_keys(rows_by_key: dict) -> set:
    return {key for key, row in rows_by_key.items() if _row_complete(row)}


def _approach_complete(output_root: str, approach: dict, bench: dict) -> bool:
    metrics_csv = os.path.join(output_root, approach['_label'], 'metrics.csv')
    rows_by_key = _resume_rows_by_key(metrics_csv, FAIRNESS_FIELDS, _trial_key)
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
            raise SystemExit(f'[fair_bench] checkpoint not found for {label}: {checkpoint}')

    approach_dir = os.path.abspath(os.path.join(output_root, label))
    os.makedirs(approach_dir, exist_ok=True)
    metrics_csv = os.path.join(approach_dir, 'metrics.csv')

    # Resume support: keep trials already finished successfully and only
    # re-run the missing (or previously errored) cells. Errored/foreign rows
    # are dropped here so the rebuilt metrics.csv stays clean.
    rows_by_key = _resume_rows_by_key(metrics_csv, FAIRNESS_FIELDS, _trial_key)
    expected = _expected_trial_keys(bench)
    done = _completed_keys(rows_by_key) & expected
    rows_by_key = {key: rows_by_key[key] for key in done}
    missing = sorted(expected - done)

    if not missing:
        rows = [rows_by_key[key] for key in sorted(rows_by_key)]
        _write_approach_outputs(output_root, bench, rows)
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
        print(f'[fair_bench] {label} resuming '
              f'missing={len(missing)}/{len(expected)} '
              f'(already done={len(rows_by_key)})', flush=True)

    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    flow_plan = _flow_plan(bench)
    missing_set = set(missing)

    for cell_idx, cell in enumerate(_grid_cells(bench), start=1):
        bw_mbps = float(cell['bw'])
        rtt_ms = float(cell['rtt'])
        cell_dir = os.path.join(approach_dir, _cell_name(bw_mbps, rtt_ms))
        for run_number in range(1, int(bench['runs_per_cell']) + 1):
            if (bw_mbps, rtt_ms, run_number) not in missing_set:
                continue
            seed = _seed_for(bench, cell_idx, run_number)
            run_dir = os.path.join(cell_dir, f'run{run_number}')
            _safe_overwrite_dir(run_dir, approach_dir)
            schedule = _static_schedule(bench, bw_mbps, rtt_ms, seed)
            schedule_csv, _ = _write_schedule(run_dir, schedule, bench)
            flow_schedule_csv = _write_flow_plan(run_dir, flow_plan)
            work_q.put((
                cell_idx, run_number, bw_mbps, rtt_ms, seed, run_dir,
                schedule_csv, flow_schedule_csv,
            ))

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
            args=(
                instance_id, work_q, result_q, approach, bench, base_cfg,
                checkpoint, listener_bin, python_bin,
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
            rows_by_key[_trial_key(row)] = row
            _append_csv(metrics_csv, FAIRNESS_FIELDS, row)
            status = 'failed' if row.get('error') else 'done'
            print(f'[fair_bench] {label} cell={row["cell"]} '
                  f'bw={float(row["bw_mbps"]):g} rtt={float(row["rtt_ms"]):g} '
                  f'run={row["run"]}/{bench["runs_per_cell"]} {status} '
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
    _write_approach_outputs(output_root, bench, rows)
    _restore_sudo_user_ownership(approach_dir)
    return rows


def _collect_rows(output_root: str, cfg: dict) -> list:
    label_map = _approach_label_map(cfg)
    required = {'approach', 'bw_mbps', 'rtt_ms', 'run_dir'}
    rows = []
    for row in _collect_approach_rows_with_labels(output_root, label_map):
        if required.issubset(row.keys()):
            for key in FAIRNESS_FIELDS:
                row.setdefault(key, '')
            rows.append(row)
    return rows


def _finite_values(rows: list, key: str) -> np.ndarray:
    vals = []
    for row in rows:
        val = _finite_float(row.get(key))
        if math.isfinite(val):
            vals.append(val)
    return np.asarray(vals, dtype=float)


def _ensure_goodput_ratio_metrics(row: dict) -> None:
    samples = _finite_float(row.get('goodput_ratio_total_samples'))
    if (str(row.get('goodput_source', '')).strip() == GOODPUT_SOURCE
            and math.isfinite(samples) and samples > 0):
        return
    if str(row.get('error', '')).strip():
        return
    flows = _flow_traces_for_row(row)
    flow_plan = _flow_plan_from_row(row)
    if not _flows_have_receiver_goodput(flows, len(flow_plan)):
        for key in (
            'goodput_ratio_total_mean', 'goodput_ratio_total_std',
            'goodput_ratio_total_samples', 'goodput_ratio_total_sum',
            'goodput_ratio_total_sumsq', 'goodput_source',
        ):
            row[key] = ''
        return
    metrics = _fairness_metrics_from_flows(flows, flow_plan, {
        'duration_s': int(_finite_float(row.get('duration_s'), 60.0)),
        'bw_mbps': float(_finite_float(row.get('bw_mbps'), 100.0)),
        'score_window_s': float(_finite_float(row.get('score_window_s'), 20.0)),
    })
    row.update(metrics)


def _ratio_summary_values(rows: list) -> tuple:
    samples = 0
    total = 0.0
    sumsq = 0.0
    for row in rows:
        n = _finite_float(row.get('goodput_ratio_total_samples'))
        s = _finite_float(row.get('goodput_ratio_total_sum'))
        q = _finite_float(row.get('goodput_ratio_total_sumsq'))
        if not (math.isfinite(n) and math.isfinite(s) and math.isfinite(q)):
            continue
        if n <= 0:
            continue
        samples += int(n)
        total += float(s)
        sumsq += float(q)
    if samples <= 0:
        return math.nan, math.nan, 0
    mean = total / samples
    if samples > 1:
        var = max(0.0, (sumsq - (total * total / samples)) / (samples - 1))
        std = math.sqrt(var)
    else:
        std = 0.0
    return float(mean), float(std), int(samples)


def _summary_rows(rows: list, bench: dict) -> list:
    completed = [r for r in rows if not str(r.get('error', '')).strip()]
    for row in completed:
        _ensure_goodput_ratio_metrics(row)
    out = []
    for approach in sorted({r.get('approach', '') for r in completed if r.get('approach')}):
        a_rows = [r for r in completed if r.get('approach') == approach]
        for rtt in bench['rtt_ms_values']:
            for bw in bench['bw_mbps_values']:
                subset = [
                    r for r in a_rows
                    if float(_finite_float(r.get('bw_mbps'))) == float(bw)
                    and float(_finite_float(r.get('rtt_ms'))) == float(rtt)
                ]
                first = subset[0] if subset else (a_rows[0] if a_rows else {})
                jain = _finite_values(subset, 'mean_jain_fairness')
                total = _finite_values(subset, 'mean_total_goodput_mbps')
                fair_err = _finite_values(subset, 'mean_abs_fair_share_error_mbps')
                ratio_mean, ratio_std, ratio_samples = _ratio_summary_values(subset)
                out.append({
                    'approach': approach,
                    'data_folder': approach,
                    'plot_label': first.get('plot_label') or approach,
                    'kind': first.get('kind', ''),
                    'algorithm': first.get('algorithm', ''),
                    'kernel_cc': first.get('kernel_cc', ''),
                    'state': first.get('state', ''),
                    'reward': first.get('reward', ''),
                    'training_environment': first.get('training_environment', ''),
                    'benchmark_environment': bench['name'],
                    'bw_mbps': float(bw),
                    'rtt_ms': float(rtt),
                    'runs_completed': int(len(subset)),
                    'runs_expected': int(bench['runs_per_cell']),
                    'goodput_ratio_total_mean': ratio_mean,
                    'goodput_ratio_total_std': ratio_std,
                    'goodput_ratio_total_samples': ratio_samples,
                    'goodput_source': GOODPUT_SOURCE if ratio_samples else '',
                    'mean_jain_fairness': float(np.mean(jain)) if jain.size else math.nan,
                    'std_jain_fairness': float(np.std(jain, ddof=1)) if jain.size > 1 else 0.0 if jain.size == 1 else math.nan,
                    'mean_total_goodput_mbps': float(np.mean(total)) if total.size else math.nan,
                    'std_total_goodput_mbps': float(np.std(total, ddof=1)) if total.size > 1 else 0.0 if total.size == 1 else math.nan,
                    'mean_abs_fair_share_error_mbps': float(np.mean(fair_err)) if fair_err.size else math.nan,
                    'std_abs_fair_share_error_mbps': float(np.std(fair_err, ddof=1)) if fair_err.size > 1 else 0.0 if fair_err.size == 1 else math.nan,
                })
    return out


def _write_summary_json(summary_rows: list, output_json: str) -> None:
    payload = {}
    for row in summary_rows:
        app = row['approach']
        payload.setdefault(app, {})
        cell = _cell_name(row['bw_mbps'], row['rtt_ms'])
        payload[app][cell] = {
            'bw_mbps': row['bw_mbps'],
            'rtt_ms': row['rtt_ms'],
            'runs_completed': row['runs_completed'],
            'goodput_ratio_total_mean': row['goodput_ratio_total_mean'],
            'goodput_ratio_total_std': row['goodput_ratio_total_std'],
            'goodput_ratio_total_samples': row['goodput_ratio_total_samples'],
            'goodput_source': row.get('goodput_source', ''),
            'mean_jain_fairness': row['mean_jain_fairness'],
            'std_jain_fairness': row['std_jain_fairness'],
            'mean_total_goodput_mbps': row['mean_total_goodput_mbps'],
            'std_total_goodput_mbps': row['std_total_goodput_mbps'],
            'mean_abs_fair_share_error_mbps': row['mean_abs_fair_share_error_mbps'],
            'std_abs_fair_share_error_mbps': row['std_abs_fair_share_error_mbps'],
        }
    with open(output_json, 'w') as f:
        json.dump(payload, f, indent=2)


def _matrix(summary_rows: list, approach: str, bench: dict, key: str) -> np.ndarray:
    rtts = [float(v) for v in bench['rtt_ms_values']]
    bws = [float(v) for v in bench['bw_mbps_values']]
    mat = np.full((len(rtts), len(bws)), np.nan, dtype=float)
    for row in summary_rows:
        if row.get('approach') != approach:
            continue
        try:
            r_idx = rtts.index(float(row['rtt_ms']))
            b_idx = bws.index(float(row['bw_mbps']))
        except ValueError:
            continue
        mat[r_idx, b_idx] = _finite_float(row.get(key))
    return mat


def _plot_heatmaps(summary_rows: list, bench: dict, output_pdf: str) -> None:
    if not summary_rows:
        print('[fair_bench] no summary rows to plot', flush=True)
        return
    approaches = sorted({r['approach'] for r in summary_rows})
    bws = [float(v) for v in bench['bw_mbps_values']]
    rtts = [float(v) for v in bench['rtt_ms_values']]
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    tmp_path = f'{output_pdf}.tmp.{os.getpid()}.pdf'
    with PdfPages(tmp_path) as pdf:
        for approach in approaches:
            subset = [r for r in summary_rows if r['approach'] == approach]
            label = subset[0].get('plot_label') or approach
            mean_mat = _matrix(summary_rows, approach, bench, 'goodput_ratio_total_mean')
            std_mat = _matrix(summary_rows, approach, bench, 'goodput_ratio_total_std')
            fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8), constrained_layout=True)
            fig.suptitle(label, fontsize=12, fontweight='bold')
            im = ax.imshow(mean_mat, origin='lower', aspect='auto',
                           vmin=0.0, vmax=1.0, cmap='RdYlGn')
            ax.set_xticks(range(len(bws)))
            ax.set_xticklabels([f'{v:g}' for v in bws])
            ax.set_yticks(range(len(rtts)))
            ax.set_yticklabels([f'{v:g}' for v in rtts])
            ax.set_xlabel('Bandwidth (Mbps)')
            ax.set_ylabel('RTT (ms)')
            ax.set_title('Goodput Ratio')
            for r in range(mean_mat.shape[0]):
                for c in range(mean_mat.shape[1]):
                    mean_val = mean_mat[r, c]
                    std_val = std_mat[r, c]
                    if not math.isfinite(mean_val):
                        continue
                    label_text = f'{mean_val:.3f}'
                    if math.isfinite(std_val):
                        label_text += f'\n+/- {std_val:.3f}'
                    ax.text(c, r, label_text, ha='center', va='center',
                            fontsize=7, color='black')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Mean Goodput Ratio')
            pdf.savefig(fig, dpi=260)
            plt.close(fig)
    os.replace(tmp_path, output_pdf)
    print(f'[fair_bench] heatmap -> {output_pdf}', flush=True)


def _write_approach_outputs(output_root: str, bench: dict, rows: list) -> int:
    written = 0
    for approach in sorted({r.get('approach', '') for r in rows if r.get('approach')}):
        approach_rows = [r for r in rows if r.get('approach') == approach]
        if not approach_rows:
            continue
        approach_dir = os.path.join(output_root, approach)
        os.makedirs(approach_dir, exist_ok=True)
        summary_rows = _summary_rows(approach_rows, bench)
        try:
            _write_rows(os.path.join(approach_dir, 'metrics.csv'),
                        FAIRNESS_FIELDS, approach_rows)
            _write_rows(os.path.join(approach_dir, 'fairness_summary.csv'),
                        SUMMARY_FIELDS, summary_rows)
            _write_summary_json(summary_rows, os.path.join(approach_dir, 'summary.json'))
            _plot_heatmaps(summary_rows, bench,
                           os.path.join(approach_dir, 'fairness_heatmap.pdf'))
            written += 1
        except PermissionError as exc:
            print(f'[fair_bench] approach summary skipped for {approach}: {exc}',
                  flush=True)
    return written


def _flow_traces_for_row(row: dict) -> list:
    flow_plan = _flow_plan_from_row(row)
    return _receiver_iperf_flows(str(row.get('run_dir', '')), flow_plan)


def _replot_individual_rows(rows: list, bench: dict = None) -> int:
    plotted = 0
    bench = bench or {}
    for row in rows:
        if str(row.get('error', '')).strip():
            continue
        flows = _flow_traces_for_row(row)
        if not flows:
            continue
        duration = int(float(row.get('duration_s', 60) or 60))
        bw = float(row.get('bw_mbps', 100) or 100)
        rtt = float(row.get('rtt_ms', 50) or 50)
        score_window = float(row.get('score_window_s', 20) or 20)
        flow_plan = _flow_plan_from_row(row)
        rl_flows = _rl_flow_traces_for_row(row, flow_plan)
        output = row.get('individual_plot') or os.path.join(
            str(row.get('run_dir', '')), 'fairness_run.pdf')
        label = (
            f'{row.get("plot_label") or row.get("approach", "approach")} '
            f'BW={bw:g} RTT={rtt:g} run={row.get("run", "")}'
        )
        plot_bench = dict(bench)
        plot_bench.update({
            'duration_s': duration,
            'bw_mbps': bw,
            'rtt_ms': rtt,
            'score_window_s': score_window,
        })
        try:
            _plot_fairness_run(
                flows, flow_plan, plot_bench, label, output,
                rl_flows=rl_flows,
                iperf_rtt_flows=_iperf_rtt_flows(
                    str(row.get('run_dir', '')), flow_plan))
            row['individual_plot'] = output
            plotted += 1
        except OSError as exc:
            print(f'[fair_bench] individual plot skipped for {output}: {exc}',
                  flush=True)
    return plotted


def _filter_rows_by_approach(rows: list, selected: set = None) -> list:
    if not selected:
        return rows
    return [
        row for row in rows
        if _slug(row.get('approach') or row.get('data_folder') or '') in selected
    ]


def _replot_all(output_root: str, cfg: dict, bench: dict,
                replot_individual: bool = True, selected: set = None) -> None:
    all_rows = _collect_rows(output_root, cfg)
    rows = _filter_rows_by_approach(all_rows, selected)
    if selected and not rows:
        raise SystemExit('[fair_bench] no existing rows matched requested approach: '
                         + ', '.join(sorted(selected)))

    plotted = _replot_individual_rows(rows, bench) if replot_individual else 0
    summary_rows = _summary_rows(all_rows, bench)
    approach_heatmaps = _write_approach_outputs(output_root, bench, rows)
    metrics_csv = os.path.join(output_root, 'metrics.csv')
    summary_csv = os.path.join(output_root, 'fairness_summary.csv')
    summary_json = os.path.join(output_root, 'summary.json')
    try:
        _write_rows(metrics_csv, FAIRNESS_FIELDS, all_rows)
        _write_rows(summary_csv, SUMMARY_FIELDS, summary_rows)
        _write_summary_json(summary_rows, summary_json)
        _plot_heatmaps(summary_rows, bench, os.path.join(output_root, 'fairness_heatmap.pdf'))
    except PermissionError as exc:
        print(f'[fair_bench] root summary skipped: {exc}', flush=True)
    _restore_sudo_user_ownership(output_root)
    print(f'[fair_bench] replotted rows={len(rows)} total_rows={len(all_rows)} '
          f'individual_plots={plotted} approach_heatmaps={approach_heatmaps}',
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run the two-flow fairness heatmap benchmark.')
    ap.add_argument('--config', default=os.path.join(_HERE, 'config.yaml'))
    ap.add_argument('--approach', action='append', default=None,
                    help='approach data_folder/name to run; may be passed more than once')
    ap.add_argument('--plot-only', action='store_true',
                    help='rebuild individual plots and heatmaps from existing data')
    ap.add_argument('--heatmaps-only', action='store_true',
                    help='rebuild summaries and heatmaps without redrawing each run PDF')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    bench_config_path = os.path.abspath(args.config)
    cfg = _load_benchmark_config(bench_config_path)
    bench_dir = os.path.dirname(bench_config_path)
    bench = _fairness_cfg(cfg)

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
                '[fair_bench] requested approach is not listed in benchmark config'
                + (f': {missing}' if missing else '')
                + (f'; available: {", ".join(available)}' if available else '')
            )
        raise SystemExit('[fair_bench] no approaches listed in benchmark config'
                         + (f'; available: {", ".join(available)}' if available else ''))

    approaches, skipped = _split_by_completion(configured_approaches, output_root, bench)
    print(f'[fair_bench] output_root={output_root}', flush=True)
    print(f'[fair_bench] benchmark={bench["name"]} duration={bench["duration_s"]}s '
          f'flow1_start={bench.get("first_flow_start_s", 0)}s '
          f'flow2_start={bench["second_flow_start_s"]}s '
          f'score_window={bench.get("score_window_s", 20)}s', flush=True)
    print('[fair_bench] bw_grid='
          + ','.join(str(v) for v in bench['bw_mbps_values']), flush=True)
    print('[fair_bench] rtt_grid='
          + ','.join(str(v) for v in bench['rtt_ms_values']), flush=True)
    print(f'[fair_bench] runs_per_cell={bench["runs_per_cell"]} '
          f'n_parallel={bench.get("n_parallel", 1)}', flush=True)
    if skipped:
        print('[fair_bench] skipping complete data='
              + ', '.join(a['_label'] for a in skipped), flush=True)
    print('[fair_bench] approaches_to_run='
          + (', '.join(a['_label'] for a in approaches) if approaches else '(none)'),
          flush=True)

    if args.dry_run:
        for approach in skipped:
            _, approach_config_path = _load_approach_runtime_config(
                approach, bench_config_path)
            print(f'[fair_bench] dry-run would skip complete data: '
                  f'{os.path.join(output_root, approach["_label"])}'
                  + (f' config={approach_config_path}'
                     if approach_config_path else ''),
                  flush=True)
        for approach in approaches:
            _, approach_config_path = _load_approach_runtime_config(
                approach, bench_config_path)
            print(f'[fair_bench] dry-run would run: '
                  f'{os.path.join(output_root, approach["_label"])} '
                  f'trials={_expected_trial_count(bench)}'
                  + (f' config={approach_config_path}'
                     if approach_config_path else ''),
                  flush=True)
        return

    if not approaches:
        print('[fair_bench] no approaches need running; rebuilding plots from existing data',
              flush=True)
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
                    f'[fair_bench] approach config for '
                    f'{approach["_label"]} must define '
                    'paths.listener and paths.py')
            listener_bin = (
                _resolve_repo_path(paths['listener'])
                if 'listener' in paths else '')
            python_bin = (
                _resolve_repo_path(paths['py']) if 'py' in paths else '')
            if approach_config_path:
                print(
                    f'[fair_bench] {approach["_label"]} '
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
            f'[fair_bench] {failed_trials} trial(s) failed')

    _replot_all(output_root, cfg, bench)
    print(f'[fair_bench] done metrics={os.path.join(output_root, "metrics.csv")}',
          flush=True)


if __name__ == '__main__':
    main()
