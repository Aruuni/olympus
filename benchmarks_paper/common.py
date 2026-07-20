#!/usr/bin/env python3
"""Shared driver for the paper-faithful inter-/intra-RTT fairness benchmarks.

The two suites (``benchmarks_paper/inter``, ``benchmarks_paper/intra``) are
thin wrappers over this module: each supplies its own ``config.yaml`` and plot
callback and calls :func:`run` with its suite name.

The benchmarks are self-contained: every trial runs through
:mod:`benchmarks_paper.runtime` (no orchestrator), and **both flows always run
the same protocol** — two kernel/Astraea/Orca flows, or two RL workers sharing
one checkpoint — exactly like the paper's setup.
"""

import argparse
import copy
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper import runtime
from benchmarks_paper.runtime import _as_bool
from olympus.common.bench_utils import (
    _append_csv,
    _approach_plot_label,
    _approach_selection_keys,
    _configured_approaches,
    _ensure_kernel_cc_available,
    _finite_float,
    _load_approach_runtime_config,
    _resolve_from,
    _restore_sudo_user_ownership,
    _safe_overwrite_dir,
    _slug,
    _start_astraea_listener,
    _stop_astraea_listener,
    _validate_approach,
    _validate_unique_data_folders,
    _write_flow_plan,
    _write_rows,
)


FIELDS = [
    'suite', 'approach', 'data_folder', 'plot_label', 'kind', 'algorithm',
    'state', 'reward', 'training_environment', 'run', 'seed',
    'bandwidth_mbps', 'rtt_ms', 'flow1_rtt_ms', 'flow2_rtt_ms',
    'bdp_multiplier', 'queue_bytes', 'duration_s', 'first_flow_start_s',
    'second_flow_start_s', 'score_window_s',
    'mean_total_goodput_mbps', 'mean_jain_fairness',
    'mean_abs_fair_share_error_mbps', 'p95_abs_fair_share_error_mbps',
    'goodput_ratio_total_mean', 'goodput_ratio_total_std',
    'goodput_ratio_total_samples', 'goodput_ratio_total_sum',
    'goodput_ratio_total_sumsq',
    'delay_ratio_mean', 'delay_ratio_std', 'delay_ratio_samples',
    'goodput_source', 'flow_schedule_csv', 'state_logs', 'run_dir',
    'checkpoint', 'error',
]

SUMMARY_FIELDS = [
    'suite', 'approach', 'plot_label', 'rtt_ms', 'bdp_multiplier',
    'runs_completed', 'runs_expected', 'goodput_ratio_total_mean',
    'goodput_ratio_total_std', 'goodput_ratio_total_samples',
    'delay_ratio_mean', 'delay_ratio_std',
    'mean_jain_fairness', 'mean_total_goodput_mbps',
]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_config(path: str) -> dict:
    """Load a suite config, deep-merged onto its master config.

    A sub-config points at the master with ``approaches_config`` (or the alias
    ``extends``); the master supplies the shared ``approaches`` list and the
    sub-config owns its full ``benchmark`` block.  Merging is recursive so a
    partial override keeps inherited keys.
    """
    path = os.path.abspath(path)
    with open(path) as handle:
        cfg = yaml.safe_load(handle) or {}
    parent_ref = cfg.pop('approaches_config', None) or cfg.pop('extends', None)
    if not parent_ref:
        return cfg
    parent_path = (parent_ref if os.path.isabs(parent_ref)
                   else os.path.join(os.path.dirname(path), parent_ref))
    return _deep_merge(_load_config(parent_path), cfg)


def _bench_cfg(cfg: dict) -> dict:
    bench = dict(cfg.get('benchmark') or {})
    bench.setdefault('bandwidth_mbps', 100)
    bench.setdefault('rtt_ms_values', list(range(20, 201, 20)))
    bench.setdefault('runs_per_cell', 5)
    bench.setdefault('duration_s', 200)
    bench.setdefault('first_flow_start_s', 0)
    bench.setdefault('second_flow_start_s', 50)
    bench.setdefault('score_window_s', 100)
    bench.setdefault('measure_interval_s', 1.0)
    bench.setdefault('n_parallel', 10)
    bench.setdefault('kernel_cport_base', 55000)
    bench.setdefault('deterministic_inference', True)
    bench.setdefault('inter_rtt', {})
    bench.setdefault('intra_rtt', {})
    inter = dict(bench['inter_rtt'] or {})
    intra = dict(bench['intra_rtt'] or {})
    inter.setdefault('fixed_flow_rtt_ms', 20)
    inter.setdefault('bdp_multipliers', [0.2, 1, 4])
    intra.setdefault('bdp_multipliers', [1])
    bench['inter_rtt'] = inter
    bench['intra_rtt'] = intra
    return bench


def _flow_plan(bench: dict) -> list:
    duration = float(bench['duration_s'])
    starts = [float(bench['first_flow_start_s']),
              float(bench['second_flow_start_s'])]
    return [
        {'flow': index + 1, 'start': start,
         'duration': max(1.0, duration - start), 'end': duration}
        for index, start in enumerate(starts)
    ]


def _rtt_pair(suite: str, rtt_ms: float, bench: dict) -> tuple:
    if suite == 'inter_rtt':
        return float(bench['inter_rtt']['fixed_flow_rtt_ms']), float(rtt_ms)
    return float(rtt_ms), float(rtt_ms)


def _queue_bytes(bandwidth_mbps: float, joining_rtt_ms: float,
                 bdp_multiplier: float) -> int:
    # The paper sizes the queue from the joining flow's BDP.  Mbps follows the
    # authors' implementation (Mi bits/s), keeping comparisons byte-identical.
    bdp = bandwidth_mbps * (2 ** 20) * joining_rtt_ms * 1e-3 / 8
    return max(int(float(bdp_multiplier) * bdp), 1500)


def _seed(suite: str, qmult: float, rtt_ms: float, run: int) -> int:
    suite_offset = 1_000_000 if suite == 'intra_rtt' else 0
    return suite_offset + int(round(qmult * 1000)) * 10_000 + int(round(rtt_ms)) * 10 + int(run)


def _trial_key(row: dict) -> tuple:
    try:
        return (
            str(row.get('suite')),
            float(row.get('bdp_multiplier')),
            float(row.get('rtt_ms')),
            int(float(row.get('run'))),
        )
    except (TypeError, ValueError):
        return ()


def _complete(row: dict) -> bool:
    if str(row.get('error', '')).strip():
        return False
    return _finite_float(row.get('goodput_ratio_total_samples'), 0.0) > 0


def _read_rows(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _cases(bench: dict, suite: str) -> list:
    cases = []
    for qmult in bench[suite]['bdp_multipliers']:
        for rtt in bench['rtt_ms_values']:
            for run in range(1, int(bench['runs_per_cell']) + 1):
                cases.append({
                    'suite': suite,
                    'qmult': float(qmult),
                    'rtt': float(rtt),
                    'run': run,
                    'seed': _seed(suite, float(qmult), float(rtt), run),
                })
    return cases


def _run_trial(instance_id: int, approach: dict, bench: dict,
               base_cfg: dict, checkpoint: str, listener_bin: str,
               python_bin: str, case: dict, run_dir: str,
               flow_schedule_csv: str) -> dict:
    suite = case['suite']
    rtt = float(case['rtt'])
    qmult = float(case['qmult'])
    run = int(case['run'])
    seed = int(case['seed'])
    bw = float(bench['bandwidth_mbps'])
    f1_rtt, f2_rtt = _rtt_pair(suite, rtt, bench)
    per_flow_delays = [f1_rtt, f2_rtt]
    qsize = _queue_bytes(bw, f2_rtt, qmult)
    flow_plan = _flow_plan(bench)
    kind = str(approach.get('kind', 'model')).lower()
    kernel_cport = int(bench.get('kernel_cport_base', 55000)) + int(instance_id) * 100
    metrics = {}
    error = ''
    state_logs = []
    flows = []

    try:
        if kind in ('kernel', 'astraea'):
            cc_algo = ('astraea' if kind == 'astraea'
                       else str(approach['kernel_cc']))
            flows, metrics, error = runtime.run_kernel_trial(
                cc_algo, bench, flow_plan, kernel_cport, instance_id,
                run_dir, qsize, qmult, per_flow_delays)
        elif kind == 'orca':
            flows, metrics, error = runtime.run_orca_paper_trial(
                approach, bench, flow_plan, kernel_cport, instance_id,
                run_dir, qsize, qmult, per_flow_delays)
        else:
            cfg = runtime.prepare_model_cfg(
                base_cfg, approach, checkpoint, run_dir)
            flows, metrics, error, state_logs = runtime.run_model_trial(
                cfg, approach, checkpoint, bench, flow_plan, instance_id,
                run_dir, qsize, qmult, per_flow_delays, listener_bin,
                python_bin, env_name=f'paper_{suite}')
    except Exception as exc:
        error = f'{exc}\n{traceback.format_exc()}'

    if not error and suite == 'intra_rtt':
        metrics.update(runtime.delay_ratio_stats(flows, rtt, bench))

    if flows:
        try:
            from benchmarks_paper.plot_run import plot_run
            plot_run(run_dir)
        except Exception as exc:
            print(f'[paper_bench] per-run plot failed: {exc}', flush=True)

    row = {
        'suite': suite,
        'approach': approach['_label'],
        'data_folder': approach['_label'],
        'plot_label': _approach_plot_label(approach),
        'kind': kind,
        'algorithm': approach.get('algorithm', ''),
        'state': approach.get('state', ''),
        'reward': approach.get('reward', ''),
        'training_environment': approach.get('environment', ''),
        'run': run,
        'seed': seed,
        'bandwidth_mbps': bw,
        'rtt_ms': rtt,
        'flow1_rtt_ms': f1_rtt,
        'flow2_rtt_ms': f2_rtt,
        'bdp_multiplier': qmult,
        'queue_bytes': qsize,
        'duration_s': int(bench['duration_s']),
        'first_flow_start_s': float(bench['first_flow_start_s']),
        'second_flow_start_s': float(bench['second_flow_start_s']),
        'score_window_s': float(bench['score_window_s']),
        'flow_schedule_csv': flow_schedule_csv,
        'state_logs': ';'.join(state_logs),
        'run_dir': run_dir,
        'checkpoint': checkpoint,
        'error': error,
    }
    row.update(metrics)
    for field in FIELDS:
        row.setdefault(field, '')
    return row


def _worker(instance_id: int, work_q, result_q, approach: dict, bench: dict,
            base_cfg: dict, checkpoint: str, listener_bin: str,
            python_bin: str) -> None:
    while True:
        item = work_q.get()
        if item is None:
            return
        case, run_dir, flow_schedule_csv = item
        result_q.put(_run_trial(
            instance_id, approach, bench, base_cfg, checkpoint, listener_bin,
            python_bin, case, run_dir, flow_schedule_csv))


def _approach_rows(approach: dict, bench: dict, cases: list,
                   output_root: str, base_cfg: dict, checkpoint: str,
                   listener_bin: str, python_bin: str) -> list:
    approach_dir = os.path.join(output_root, approach['_label'])
    os.makedirs(approach_dir, exist_ok=True)
    metrics_path = os.path.join(approach_dir, 'metrics.csv')
    previous_list = _read_rows(metrics_path)
    previous = {_trial_key(row): row for row in previous_list if _trial_key(row)}
    wanted = {_trial_key({
        'suite': case['suite'], 'bdp_multiplier': case['qmult'],
        'rtt_ms': case['rtt'], 'run': case['run'],
    }) for case in cases}
    rows = {key: row for key, row in previous.items()
            if key in wanted and _complete(row)}
    preserved = [row for row in previous_list if _trial_key(row) not in wanted]
    missing = [case for case in cases if (
        case['suite'], case['qmult'], case['rtt'], case['run']) not in rows]
    if not missing:
        return [rows[key] for key in sorted(rows)]

    with open(os.path.join(approach_dir, 'run_meta.json'), 'w') as handle:
        json.dump({
            'paper': 'An Experimental Study of Congestion Control in LEO Satellite Networks',
            'approach': {k: v for k, v in approach.items() if not k.startswith('_')},
            'benchmark': bench,
            'checkpoint': checkpoint,
        }, handle, indent=2)

    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    flow_plan = _flow_plan(bench)
    for case in missing:
        run_dir = os.path.join(
            approach_dir, case['suite'], f'q{case["qmult"]:g}',
            f'rtt{case["rtt"]:g}', f'run{case["run"]}')
        _safe_overwrite_dir(run_dir, approach_dir)
        flow_schedule_csv = _write_flow_plan(run_dir, flow_plan)
        work_q.put((case, run_dir, flow_schedule_csv))

    n_parallel = max(1, min(int(bench['n_parallel']), len(missing)))
    for _ in range(n_parallel):
        work_q.put(None)
    astraea_handle = None
    if str(approach.get('kind', '')).lower() == 'astraea':
        astraea_handle = _start_astraea_listener(
            approach, bench, approach_dir)
    processes = [
        multiprocessing.Process(
            target=_worker,
            args=(slot, work_q, result_q, approach, bench, base_cfg,
                  checkpoint, listener_bin, python_bin),
            daemon=True)
        for slot in range(n_parallel)
    ]
    for process in processes:
        process.start()
    try:
        for completed in range(1, len(missing) + 1):
            row = result_q.get()
            rows[_trial_key(row)] = row
            _append_csv(metrics_path, FIELDS, row)
            status = 'failed' if row['error'] else 'done'
            print(
                f'[paper_bench] {approach["_label"]} {row["suite"]} '
                f'q={float(row["bdp_multiplier"]):g} RTT={float(row["rtt_ms"]):g} '
                f'run={row["run"]} {status} ({completed}/{len(missing)})',
                flush=True)
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        _stop_astraea_listener(astraea_handle)
    ordered = [rows[key] for key in sorted(rows)]
    _write_rows(metrics_path, FIELDS, preserved + ordered)
    _restore_sudo_user_ownership(approach_dir)
    return ordered


def _pooled_ratio(rows: list) -> tuple:
    count = 0
    total = 0.0
    sumsq = 0.0
    for row in rows:
        n = _finite_float(row.get('goodput_ratio_total_samples'))
        s = _finite_float(row.get('goodput_ratio_total_sum'))
        q = _finite_float(row.get('goodput_ratio_total_sumsq'))
        if not all(math.isfinite(v) for v in (n, s, q)) or n <= 0:
            continue
        count += int(n)
        total += s
        sumsq += q
    if count <= 0:
        return math.nan, math.nan, 0
    mean = total / count
    variance = max(0.0, (sumsq - total * total / count) / (count - 1)) if count > 1 else 0.0
    return mean, math.sqrt(variance), count


def _mean_std(rows: list, key: str) -> tuple:
    values = np.asarray([_finite_float(row.get(key)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    return float(np.mean(values)), float(np.std(values, ddof=1) if values.size > 1 else 0.0)


def _summaries(rows: list, bench: dict) -> list:
    complete = [row for row in rows if _complete(row)]
    output = []
    groups = sorted({
        (row['suite'], row['approach'], float(row['bdp_multiplier']),
         float(row['rtt_ms'])) for row in complete
    })
    for suite, approach, qmult, rtt in groups:
        subset = [row for row in complete
                  if row['suite'] == suite and row['approach'] == approach
                  and float(row['bdp_multiplier']) == qmult
                  and float(row['rtt_ms']) == rtt]
        mean, std, samples = _pooled_ratio(subset)
        jain = np.asarray([_finite_float(row.get('mean_jain_fairness'))
                           for row in subset], dtype=float)
        total = np.asarray([_finite_float(row.get('mean_total_goodput_mbps'))
                            for row in subset], dtype=float)
        jain = jain[np.isfinite(jain)]
        total = total[np.isfinite(total)]
        delay_mean, delay_std = _mean_std(subset, 'delay_ratio_mean')
        output.append({
            'suite': suite,
            'approach': approach,
            'plot_label': subset[0].get('plot_label') or approach,
            'rtt_ms': rtt,
            'bdp_multiplier': qmult,
            'runs_completed': len(subset),
            'runs_expected': int(bench['runs_per_cell']),
            'goodput_ratio_total_mean': mean,
            'goodput_ratio_total_std': std,
            'goodput_ratio_total_samples': samples,
            'delay_ratio_mean': delay_mean,
            'delay_ratio_std': delay_std,
            'mean_jain_fairness': float(np.mean(jain)) if jain.size else math.nan,
            'mean_total_goodput_mbps': float(np.mean(total)) if total.size else math.nan,
        })
    return output


def _collect_output_rows(output_root: str) -> list:
    collected = []
    root = Path(output_root)
    if not root.exists():
        return collected
    for child in root.iterdir():
        if child.is_dir():
            collected.extend(_read_rows(str(child / 'metrics.csv')))
    return collected


def _write_root_outputs(output_root: str, bench: dict) -> list:
    collected = _collect_output_rows(output_root)
    _write_rows(os.path.join(output_root, 'metrics.csv'), FIELDS, collected)
    _write_rows(os.path.join(output_root, 'summary.csv'), SUMMARY_FIELDS,
                _summaries(collected, bench))
    return collected


def _read_sysctl(name: str) -> str:
    result = subprocess.run(
        ['sysctl', '-n', name], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _set_sysctl(name: str, value: str) -> None:
    subprocess.run(
        ['sysctl', '-w', f'{name}={value}'], check=True,
        stdout=subprocess.DEVNULL)


def _configure_tcp_buffers(bench: dict, suite: str) -> dict:
    """Raise TCP autotuning maxima using the authors' 22x multiplier."""
    bw = float(bench['bandwidth_mbps'])
    max_rtt = max(float(value) for value in bench['rtt_ms_values'])
    max_qmult = max(float(value) for value in bench[suite]['bdp_multipliers'])
    bdp = bw * (2 ** 20) * max_rtt * 1e-3 / 8
    qsize = _queue_bytes(bw, max_rtt, max_qmult)
    maximum = int(float(bench.get('tcp_buffer_multiplier', 22)) * (bdp + qsize))
    previous = {
        'net.ipv4.tcp_rmem': _read_sysctl('net.ipv4.tcp_rmem'),
        'net.ipv4.tcp_wmem': _read_sysctl('net.ipv4.tcp_wmem'),
    }
    try:
        _set_sysctl('net.ipv4.tcp_rmem', f'10240 87380 {maximum}')
        _set_sysctl('net.ipv4.tcp_wmem', f'10240 87380 {maximum}')
    except Exception:
        _restore_tcp_buffers(previous)
        raise
    print(f'[paper_bench] TCP buffer autotuning max={maximum} bytes', flush=True)
    return previous


def _restore_tcp_buffers(previous: dict) -> None:
    for name, value in previous.items():
        try:
            _set_sysctl(name, value)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f'[paper_bench] failed to restore {name}: {exc}', flush=True)


def _active_approaches(cfg: dict, selected: set, include_disabled: bool) -> list:
    approaches = _configured_approaches(cfg, selected or None)
    return [approach for approach in approaches
            if include_disabled or _as_bool(approach.get('enabled', True), True)]


def run(config_path: str, suite: str, plot_fn, argv=None) -> int:
    """Drive one fairness suite end to end.

    ``suite`` is ``'inter_rtt'`` or ``'intra_rtt'``.  ``plot_fn(config_path)``
    renders the paper figures from the aggregated ``summary.csv``.
    """
    parser = argparse.ArgumentParser(
        description=f'Run the IFIP 2026 {suite} fairness benchmark.')
    parser.add_argument('--config', default=config_path)
    parser.add_argument('--approach', action='append', default=[])
    parser.add_argument('--include-disabled', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)

    config_path = os.path.abspath(args.config)
    cfg = _load_config(config_path)
    bench = _bench_cfg(cfg)
    selected = {_slug(value) for value in args.approach}
    approaches = _active_approaches(cfg, selected, args.include_disabled)
    _validate_unique_data_folders(approaches)
    if selected and not approaches:
        available = sorted({key for raw in cfg.get('approaches', [])
                            for key in _approach_selection_keys(raw)})
        raise SystemExit('no enabled requested approaches; available: '
                         + ', '.join(available))

    output_root = str(cfg.get('output_root', 'data'))
    if not os.path.isabs(output_root):
        output_root = os.path.abspath(os.path.join(
            os.path.dirname(config_path), output_root))
    os.makedirs(output_root, exist_ok=True)
    cases = _cases(bench, suite)
    print(f'[paper_bench] suite={suite} output={output_root}', flush=True)
    print(f'[paper_bench] cases_per_approach={len(cases)} '
          f'approaches={[a["_label"] for a in approaches]}', flush=True)
    disabled = [raw.get('data_folder') or raw.get('name')
                for raw in cfg.get('approaches', [])
                if not _as_bool(raw.get('enabled', True), True)]
    if disabled and not args.include_disabled:
        print('[paper_bench] disabled (awaiting checkpoint): '
              + ', '.join(str(value) for value in disabled), flush=True)

    if args.plot_only:
        _write_root_outputs(output_root, bench)
        return plot_fn(config_path)

    if args.dry_run:
        for approach in approaches:
            _validate_approach(approach)
            base_cfg, config = _load_approach_runtime_config(approach, config_path)
            kind = str(approach.get('kind', 'model')).lower()
            checkpoint = approach.get('checkpoint', '')
            if kind not in ('kernel', 'astraea', 'orca'):
                resolved = _resolve_from(os.path.dirname(config_path), checkpoint)
                if not os.path.isfile(resolved):
                    raise SystemExit(f'checkpoint missing for {approach["_label"]}: {resolved}')
            print(f'[paper_bench] dry-run {approach["_label"]}: '
                  f'{len(cases)} trials' + (f' config={config}' if config else ''),
                  flush=True)
        return 0

    failures = 0
    previous_tcp_buffers = {}
    try:
        previous_tcp_buffers = _configure_tcp_buffers(bench, suite)
        for approach in approaches:
            _validate_approach(approach)
            _ensure_kernel_cc_available(approach)
            kind = str(approach.get('kind', 'model')).lower()
            base_cfg, _ = _load_approach_runtime_config(approach, config_path)
            checkpoint = ''
            if kind not in ('kernel', 'astraea', 'orca'):
                checkpoint = _resolve_from(
                    os.path.dirname(config_path), str(approach['checkpoint']))
                if not os.path.isfile(checkpoint):
                    raise SystemExit(
                        f'checkpoint missing for {approach["_label"]}: {checkpoint}')
            paths = base_cfg.get('paths', {}) or {}
            listener_bin = (runtime.resolve_repo_path(paths['listener'])
                            if 'listener' in paths else '')
            python_bin = (runtime.resolve_repo_path(paths['py'])
                          if 'py' in paths else '')
            rows = _approach_rows(
                approach, bench, cases, output_root, base_cfg, checkpoint,
                listener_bin, python_bin)
            failures += sum(not _complete(row) for row in rows)
            _write_root_outputs(output_root, bench)
            runtime.final_cleanup()
    finally:
        if previous_tcp_buffers:
            _restore_tcp_buffers(previous_tcp_buffers)
        _restore_sudo_user_ownership(output_root)

    _write_root_outputs(output_root, bench)
    plot_fn(config_path)
    if failures:
        print(f'[paper_bench] {failures} trial(s) failed; rerun to resume', flush=True)
        return 1
    return 0
