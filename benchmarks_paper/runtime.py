#!/usr/bin/env python3
"""Self-contained trial engine for the paper fairness benchmarks.

This module owns everything needed to run one two-flow dumbbell trial without
the Olympus orchestrator:

- kernel CC / released-Astraea trials (iperf3 driven by MininetEnv),
- original-Orca trials (sender.sh / receiver.sh per flow),
- Olympus model trials where **each flow gets its own listener + RL worker**
  running the same checkpoint — two identical agents competing, exactly like
  the paper's intra/inter-RTT setup.  This replaces the orchestrator's
  single-agent episode path, which drove only flow 1 and left the other flow
  running the plain kernel CC.

Imports are restricted to the sanctioned shared layers: ``olympus.common``
(helpers) and ``olympus.environments`` (the emulator).  No
``olympus.orchestrator``, no ``benchmarks/`` siblings.
"""

import csv
import glob
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olympus.common.action_plugins import action_env
from olympus.common.bench_utils import (
    _binned_series,
    _copy_if_exists,
    _finite_float,
    _kernel_flows,
    _orca_receiver_to_csvs,
    _orca_settings,
    _orca_ss_rtt_flows,
    _parse_iperf_json,
    _prepare_runtime_cfg,
    _run_orca_on_env,
    _safe_unlink,
)
from olympus.common.episode_plotting import render_episode_plots
from olympus.common.link_context import write_link_context
from olympus.common.registry import (
    is_multi_agent,
    worker_script as resolve_worker_script,
)
from olympus.environments.mininet.env import MininetEnv


GOODPUT_SOURCE = 'iperf3_receiver'


# ── Small local helpers (formerly imported from the orchestrator) ─────────────

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


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(str(ROOT), path))


def _terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=3)
            break
        except subprocess.TimeoutExpired:
            pass


def _pkill(pattern: str) -> None:
    try:
        subprocess.run(['pkill', '-KILL', '-f', pattern],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10, check=False)
    except Exception:
        pass


def final_cleanup() -> None:
    """Sweep stray per-flow workers/listeners after an approach finishes."""
    _pkill('olympus/algorithms/.*/worker.py')
    _pkill('astraea_listener')


# ── iperf capture + fairness metrics (self-contained copies) ──────────────────

def _iperf_client_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_{int(cport)}_{int(flow)}.json'


def _iperf_receiver_tmp(cport: int, flow: int) -> str:
    return f'/tmp/iperf_server_{int(cport)}_{int(flow)}.json'


def clear_iperf_tmp_outputs(cport: int, n_flows: int) -> None:
    for flow in range(1, int(n_flows) + 1):
        _safe_unlink(_iperf_client_tmp(cport, flow))
        _safe_unlink(_iperf_receiver_tmp(cport, flow))


def receiver_iperf_flows(run_dir: str, flow_plan: list) -> list:
    return _kernel_flows(run_dir, flow_plan)


def iperf_rtt_flows(run_dir: str, flow_plan: list) -> list:
    """Per-flow TCP RTT from the iperf3 client JSON (ss-sidecar fallback)."""
    out = []
    for item in flow_plan:
        flow = int(item['flow'])
        start = float(item['start'])
        client_json = os.path.join(run_dir, f'iperf_client_flow{flow}.json')
        parsed = _parse_iperf_json(client_json)
        samples = parsed.get('rtt_samples_ms', []) or []
        t = np.asarray([s[0] for s in samples], dtype=float) + start
        rtt = np.asarray([s[1] for s in samples], dtype=float)
        out.append({
            'flow': flow, 'start': start, 'end': float(item['end']),
            't': t, 'iperf_rtt': rtt,
        })
    if not any(f['t'].size for f in out):
        return _orca_ss_rtt_flows(run_dir, flow_plan)
    return out


def flows_have_receiver_goodput(flows: list, n_flows: int) -> bool:
    if len(flows) < int(n_flows):
        return False
    for flow in flows[:int(n_flows)]:
        thr = np.asarray(flow.get('thr', []), dtype=float)
        if not np.isfinite(thr).any():
            return False
    return True


def copy_receiver_iperf_outputs(cport: int, run_dir: str, flow_plan: list,
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
            errors.append(
                f'missing receiver iperf JSON for flow {flow}: {receiver_tmp}')
            continue

        parsed = _parse_iperf_json(
            receiver_dst, os.path.join(csv_dir, f'x{flow}.csv'),
            measure_interval_s=float(bench.get('measure_interval_s', 1.0)),
        )
        if not parsed.get('samples'):
            errors.append(f'no receiver iperf interval samples for flow {flow}')
    return '; '.join(errors)


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
    return {
        'goodput_ratio_total_mean': float(np.mean(arr)) if samples else math.nan,
        'goodput_ratio_total_std': (
            float(np.std(arr, ddof=1)) if samples > 1
            else 0.0 if samples == 1 else math.nan
        ),
        'goodput_ratio_total_samples': samples,
        'goodput_ratio_total_sum': float(np.sum(arr)) if samples else 0.0,
        'goodput_ratio_total_sumsq': float(np.sum(arr * arr)) if samples else 0.0,
    }


def fairness_metrics_from_flows(flows: list, flow_plan: list, bench: dict) -> dict:
    series = _binned_series(flows, flow_plan, bench)
    metrics = _metrics_from_score_window(series, bench)
    metrics.update(_goodput_ratio_stats(series, flow_plan, bench))
    metrics['goodput_source'] = GOODPUT_SOURCE
    return metrics


def delay_ratio_stats(flows: list, rtt_ref_ms: float, bench: dict) -> dict:
    """Pool per-flow SRTT normalised by the reference RTT over the score window."""
    if not (rtt_ref_ms and math.isfinite(float(rtt_ref_ms)) and rtt_ref_ms > 0):
        return {}
    duration = float(bench['duration_s'])
    window = float(bench.get('score_window_s', 100.0))
    lo = max(0.0, duration - max(0.0, window))
    samples = []
    for flow in flows:
        t = np.asarray(flow.get('t', []), dtype=float)
        srtt = np.asarray(flow.get('srtt', []), dtype=float)
        if t.size == 0 or srtt.size == 0 or t.size != srtt.size:
            continue
        mask = np.isfinite(t) & np.isfinite(srtt) & (srtt > 0) & (t >= lo)
        if mask.any():
            samples.append(srtt[mask] / float(rtt_ref_ms))
    if not samples:
        return {}
    pooled = np.concatenate(samples)
    return {
        'delay_ratio_mean': float(np.mean(pooled)),
        'delay_ratio_std': float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0,
        'delay_ratio_samples': int(pooled.size),
    }


def copy_and_measure(cport: int, run_dir: str, flow_plan: list,
                     bench: dict) -> tuple:
    error = copy_receiver_iperf_outputs(cport, run_dir, flow_plan, bench)
    flows = receiver_iperf_flows(run_dir, flow_plan)
    if error:
        return flows, {}, error
    if not flows_have_receiver_goodput(flows, len(flow_plan)):
        return flows, {}, 'missing receiver goodput samples'
    metrics = fairness_metrics_from_flows(flows, flow_plan, {
        'duration_s': int(bench['duration_s']),
        'bw_mbps': float(bench['bandwidth_mbps']),
        'score_window_s': float(bench['score_window_s']),
    })
    return flows, metrics, ''


# ── Trial runners ─────────────────────────────────────────────────────────────

def _mininet_env(bench: dict, cc_algo: str, cport: int, instance_id: int,
                 qsize: int, qmult: float, per_flow_delays: list,
                 unique_cports: bool = False) -> MininetEnv:
    return MininetEnv(
        n=2,
        bw=float(bench['bandwidth_mbps']),
        delay=min(per_flow_delays),
        qsize=qsize,
        bdp_mult=qmult,
        duration=int(bench['duration_s']),
        cport=cport,
        cc_algo=cc_algo,
        instance_id=instance_id,
        unique_cports=unique_cports,
        per_flow_delays=list(per_flow_delays),
        disable_offload=_as_bool(bench.get('disable_offload'), True),
    )


def run_kernel_trial(cc_algo: str, bench: dict, flow_plan: list, cport: int,
                     instance_id: int, run_dir: str, qsize: int,
                     qmult: float, per_flow_delays: list) -> tuple:
    """Two identical kernel-CC flows (also the released-Astraea path)."""
    clear_iperf_tmp_outputs(cport, 2)
    env = _mininet_env(bench, cc_algo, cport, instance_id, qsize, qmult,
                       per_flow_delays)
    try:
        env.start()
        env.setup_environment()
        env.start_episode(
            monitor_interval=float(bench['measure_interval_s']),
            start_delays=[p['start'] for p in flow_plan],
            flow_durations=[p['duration'] for p in flow_plan],
            episode_start=time.monotonic(),
        )
        env.wait()
    finally:
        env.stop()
        time.sleep(0.5)
    return copy_and_measure(cport, run_dir, flow_plan, bench)


def run_orca_paper_trial(approach: dict, bench: dict, flow_plan: list,
                         cport: int, instance_id: int, run_dir: str,
                         qsize: int, qmult: float,
                         per_flow_delays: list) -> tuple:
    """Two original-Orca flows (sender.sh/receiver.sh per flow)."""
    settings = _orca_settings(approach, bench)
    env = _mininet_env(bench, 'cubic', cport, instance_id, qsize, qmult,
                       per_flow_delays)
    try:
        env.start()
        env.setup_environment()
        _run_orca_on_env(env, settings, flow_plan, instance_id, run_dir,
                         int(bench['duration_s']))
    finally:
        env.stop()
        time.sleep(0.5)
    error = _orca_receiver_to_csvs(
        run_dir, flow_plan,
        measure_interval_s=float(bench['measure_interval_s']),
        settings=settings)
    flows = receiver_iperf_flows(run_dir, flow_plan)
    if not error and flows_have_receiver_goodput(flows, 2):
        metrics = fairness_metrics_from_flows(flows, flow_plan, {
            'duration_s': int(bench['duration_s']),
            'bw_mbps': float(bench['bandwidth_mbps']),
            'score_window_s': float(bench['score_window_s']),
        })
        return flows, metrics, ''
    return flows, {}, error or 'missing Orca receiver goodput samples'


def prepare_model_cfg(base_cfg: dict, approach: dict, checkpoint: str,
                      run_dir: str) -> dict:
    """Resolve the approach's training config for deterministic inference."""
    scratch_dir = os.path.join(
        '/tmp', f'olympus_paper_bench_{os.getpid()}', approach['_label'])
    cfg = _prepare_runtime_cfg(
        base_cfg, approach, checkpoint, scratch_dir, run_dir,
        plot_episodes=False)
    cfg['outputs']['run_dir'] = scratch_dir
    cfg['outputs']['checkpoints_dir'] = os.path.join(scratch_dir, 'checkpoints')
    cfg['outputs']['telemetry_dir'] = os.path.join(scratch_dir, 'telemetry')
    cfg['training']['log_path'] = os.path.join(
        scratch_dir, 'telemetry', 'benchmark_metrics.csv')
    resolved_path = os.path.join(run_dir, 'config.resolved.yaml')
    with open(resolved_path, 'w') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    cfg['_resolved_config_path'] = resolved_path
    return cfg


def _model_worker_env(cfg: dict, approach: dict, checkpoint: str,
                      bench: dict, run_dir: str, flow_plan: list,
                      state_log: str, python_bin: str) -> dict:
    """The SAO_*/OC_* contract the RL worker reads, in eval mode.

    Mirrors the orchestrator's worker environment for one episode, minus
    training-only machinery (learner manager, lagged self-play, raynet,
    oracle state).  Both flows share this base; per-flow overrides are added
    at listener launch.
    """
    t_cfg = cfg.get('training', cfg) or {}
    a_cfg = cfg.get('agent', {}) or {}
    r_cfg = cfg.get('reward', {}) or {}
    s_cfg = cfg.get('state_options', {}) or {}
    runtime = cfg.get('runtime', {}) or {}
    alg_name = str(runtime.get('algorithm', approach.get('algorithm', 'td3')))
    bw = float(bench['bandwidth_mbps'])
    resolved_config = cfg.get('_resolved_config_path', '')

    env = dict(os.environ)
    env.update({
        'OC_PYTHON': python_bin,
        'SAO_PYTHON': python_bin,
        'SAO_LISTENER_CC': str(cfg.get('listener_cc', 'cubic')),
        'OC_LISTENER_CC': str(cfg.get('listener_cc', 'cubic')),
        'SAO_CONFIG': resolved_config,
        'OC_CONFIG': resolved_config,
        'SAO_ALGORITHM': alg_name,
        'SAO_REWARD': str(runtime.get('reward', approach.get('reward', ''))),
        'SAO_STATE': str(runtime.get('state', approach.get('state', ''))),
        'OC_STATE': str(runtime.get('state', approach.get('state', ''))),
        **action_env(cfg),
        'SAO_ORCA_TARGET_MS': str(float(s_cfg.get('orca_target_ms', 50.0))),
        **_tempest_kalman_env(s_cfg),
        'SAO_CHECKPOINT': os.path.abspath(checkpoint),
        # Eval only: no learner manager — workers run pure inference.
        'SAO_MANAGER_ADDR': '',
        'SAO_MANAGER_KEY': '',
        'SAO_LINK_BW': str(bw),
        'SAO_INTERVAL_MS': str(float(a_cfg.get('interval_ms', 20))),
        'SAO_CWND_MIN': str(int(a_cfg.get('cwnd_min', 10))),
        'SAO_CWND_MAX': str(int(a_cfg.get('cwnd_max', 10000))),
        'SAO_HIDDEN': str(int(a_cfg.get('hidden', 128))),
        'SAO_HEAD_HIDDEN': ('' if a_cfg.get('head_hidden') is None
                            else str(int(a_cfg.get('head_hidden')))),
        'SAO_REC_DIM': str(int(a_cfg.get('rec_dim', 10))),
        'SAO_ORCA_REC_DIM': str(int(a_cfg.get('rec_dim', 10))),
        'SAO_ORCA_USE_NORMALIZER': (
            '1' if a_cfg.get('use_normalizer', False) else '0'),
        'SAO_ORCA_SLOW_START_GATE': (
            '1' if a_cfg.get('slow_start_gate', alg_name == 'orca') else '0'),
        **_bbr_probe_env(cfg, a_cfg),
        'SAO_DREAMER_HIDDEN': str(int(a_cfg.get('hidden', 256))),
        'SAO_DREAMER_EMBED': str(int(a_cfg.get('embed_dim', 128))),
        'SAO_DREAMER_H_DIM': str(int(a_cfg.get('h_dim', 256))),
        'SAO_DREAMER_GROUPS': str(int(a_cfg.get('latent_groups', 8))),
        'SAO_DREAMER_CLASSES': str(int(a_cfg.get('latent_classes', 8))),
        'SAO_DREAMER_REWARD_BINS': str(int(a_cfg.get('reward_bins', 255))),
        'SAO_DREAMER_ACTOR_LOG_STD_MIN': str(
            float(t_cfg.get('actor_log_std_min', -2.0))),
        'SAO_DREAMER_ACTOR_LOG_STD_MAX': str(
            float(t_cfg.get('actor_log_std_max', 0.0))),
        'SAO_DEAD_FLOW_MS': str(int(a_cfg.get('dead_flow_ms', 1000))),
        # Deterministic checkpoint inference on every flow.
        'SAO_NOISE_STD': '0.0',
        'SAO_DETERMINISTIC': '1',
        'SAO_REQUIRE_CHECKPOINT': '1',
        'OC_DETERMINISTIC': '1',
        'OC_REQUIRE_CHECKPOINT': '1',
        'SAO_TRACE_LOG': state_log,
        'SAO_TRACE_LOG_SUFFIX_BY_FLOW': '1',
        'SAO_RAW_STATE_LOG_ENABLED': '0',
        'SAO_EPISODE': '1',
        'SAO_HIDE_LINK_ORACLE': '0',
        'OC_ORACLE_RTT': '0',
        'OC_LINK_BW': str(bw),
        'OC_INTERVAL_MS': str(float(a_cfg.get('interval_ms', 20))),
        'OC_LINK_SCHEDULE': json.dumps([]),
        'OC_W_SRTT_DRIFT': str(float(r_cfg.get('w_srtt_drift', 2.0))),
        'OC_SRTT_DRIFT_CAP': str(float(r_cfg.get('srtt_drift_cap', 4.0))),
        'OC_ORCA_DELAY_MARGIN_COEF': str(
            float(r_cfg.get('delay_margin_coef', 1.25))),
        'OC_FAIRNESS_METRIC': str(r_cfg.get('fairness_metric', 'r_fair')),
        'OC_W_R_FAIR': str(float(r_cfg.get('w_r_fair', 25.0))),
        'OC_R_FAIR_CAP': str(float(r_cfg.get('r_fair_cap', 1.0))),
        'OC_W_JAIN': str(float(r_cfg.get('w_jain', 25.0))),
        'OC_FAIR_FLOW_START_DELAYS': json.dumps(
            [float(p['start']) for p in flow_plan]),
        'OC_FAIR_FLOW_DURATIONS': json.dumps(
            [float(p['duration']) for p in flow_plan]),
        'OC_PUSH_EVERY': str(int(t_cfg.get('worker_push_every', 16))),
        # Multi-agent extras (ignored by single-agent workers).
        'SAO_N_AGENTS': str(len(flow_plan)),
        'SAO_MAT_LAYERS': str(int(a_cfg.get('n_layers', 2))),
        'SAO_MAT_HEADS': str(int(a_cfg.get('n_heads', 4))),
    })
    return env


def _bbr_probe_env(cfg, agent_cfg):
    top = (cfg.get('bbr_probe') or {}) if isinstance(cfg, dict) else {}
    per_alg = (agent_cfg or {}).get('bbr_probe') or {}

    def _get(key, default):
        if key in top:
            return top[key]
        if key in per_alg:
            return per_alg[key]
        return default

    return {
        'SAO_BBR_PROBE_ENABLED': '1' if bool(_get('enabled', False)) else '0',
        'SAO_BBR_PROBE_INTERVAL_S': str(float(_get('interval_s', 5.0))),
        'SAO_BBR_PROBE_DURATION_S': str(float(_get('duration_s', 0.2))),
        'SAO_BBR_PROBE_FACTOR': str(float(_get('factor', 0.5))),
        'SAO_BBR_MIN_RTT_WINDOW_S': str(float(_get('min_rtt_window_s', 10.0))),
    }


def _tempest_kalman_env(state_cfg):
    state_cfg = state_cfg or {}
    tk_cfg = state_cfg.get('tempest_kalman', {}) or {}

    def _get(key, default):
        return tk_cfg.get(key, state_cfg.get(f'tempest_kalman_{key}', default))

    return {
        'SAO_TEMPEST_KALMAN_INIT_US': str(float(_get('init_us', 20_000.0))),
        'SAO_TEMPEST_KALMAN_Q': str(float(_get('q', 5.0e-4))),
        'SAO_TEMPEST_KALMAN_R_DOWN': str(float(_get('r_down', 1.0e-3))),
        'SAO_TEMPEST_KALMAN_R_UP': str(float(_get('r_up', 5.0))),
        'SAO_TEMPEST_KALMAN_JUMP_THRESH': str(float(_get('jump_thresh', 1.5))),
    }


def run_model_trial(cfg: dict, approach: dict, checkpoint: str, bench: dict,
                    flow_plan: list, instance_id: int, run_dir: str,
                    qsize: int, qmult: float, per_flow_delays: list,
                    listener_bin: str, python_bin: str,
                    env_name: str = 'paper') -> tuple:
    """Two flows, each driven by its own listener + worker on the SAME policy.

    Every flow gets a dedicated listener at ``cport + flow_id``; MininetEnv is
    started with ``unique_cports=True`` so flow *i*'s iperf client binds source
    port ``cport + i - 1`` and is caught by exactly its own listener.  Both
    workers load the same checkpoint deterministically — a genuine
    same-protocol pair, like the paper.
    """
    alg_name = str((cfg.get('runtime', {}) or {}).get(
        'algorithm', approach.get('algorithm', 'td3')))
    cport = int(cfg.get('cport_base', 21000)) + int(instance_id) * 100
    bw = float(bench['bandwidth_mbps'])
    clear_iperf_tmp_outputs(cport, len(flow_plan))

    state_log = os.path.join(run_dir, f'{alg_name}_state_ep000001.csv')
    worker_env = _model_worker_env(
        cfg, approach, checkpoint, bench, run_dir, flow_plan, state_log,
        python_bin)
    worker = resolve_worker_script(alg_name)

    env = _mininet_env(
        bench, str(cfg.get('listener_cc', 'cubic')), cport, instance_id,
        qsize, qmult, per_flow_delays, unique_cports=True)
    listeners = []
    try:
        env.start()
        env.setup_environment()

        episode_start = time.monotonic()
        worker_env['SAO_EPISODE_START'] = str(episode_start)
        worker_env['OC_EPISODE_START'] = str(episode_start)

        for flow_id in range(len(flow_plan)):
            listener_env = dict(worker_env)
            listener_env['SAO_AGENT_ID'] = str(flow_id)
            listener_env['OC_FLOW_ID'] = str(flow_id)
            flow_rtt_us = str(float(per_flow_delays[flow_id]) * 1000.0)
            context_path = write_link_context(
                os.path.join(
                    run_dir,
                    f'link_context_ep000001_slot{instance_id}_flow{flow_id}.json'),
                bw_mbps=bw,
                base_rtt_us=flow_rtt_us,
                link_schedule=[],
                episode=1,
                slot=instance_id,
                flow_id=flow_id,
            )
            listener_env['OC_BASE_RTT_US'] = flow_rtt_us
            listener_env['SAO_BASE_RTT_US'] = flow_rtt_us
            listener_env['OC_LINK_CONTEXT_PATH'] = context_path
            listener_env['SAO_LINK_CONTEXT_PATH'] = context_path
            listeners.append(subprocess.Popen(
                [listener_bin, '--cport', str(cport + flow_id),
                 '--worker', worker, '--mode', 'mininet',
                 '--scan-ms', str(cfg.get('scan_ms', 20)),
                 '--flow-id', str(flow_id),
                 '--single-flow', '1', '--no-state-pipe', '1'],
                env=listener_env, start_new_session=True))

        env.start_episode(
            monitor_interval=float(bench['measure_interval_s']),
            start_delays=[p['start'] for p in flow_plan],
            flow_durations=[p['duration'] for p in flow_plan],
            episode_start=episode_start,
        )
        env.wait()
    finally:
        for listener in listeners:
            _terminate(listener)
        env.stop()
        time.sleep(0.5)

    flows, metrics, error = copy_and_measure(cport, run_dir, flow_plan, bench)
    state_logs = sorted(
        glob.glob(os.path.join(run_dir, f'{alg_name}_state_ep000001_flow*.csv'))
        + glob.glob(os.path.join(run_dir, f'{alg_name}_state_ep000001_a*.csv')))
    if not error and len(state_logs) < len(flow_plan):
        error = (f'expected {len(flow_plan)} per-flow state logs, found '
                 f'{len(state_logs)} — an RL worker did not attach to every flow')

    if state_logs:
        render_state_plots(
            run_dir, alg_name, bench, per_flow_delays,
            n_flows=len(flow_plan), env_name=env_name,
            instance_id=instance_id)
    return flows, metrics, error, state_logs


def render_state_plots(run_dir: str, alg_name: str, bench: dict,
                       per_flow_delays: list, n_flows: int,
                       env_name: str = 'paper', instance_id: int = 0) -> None:
    """Per-flow episode PDFs, identical to the orchestrator's plots."""
    try:
        render_episode_plots(
            outputs={'plots_dir': run_dir, 'plot_episodes': True},
            episode=1,
            alg_name=alg_name,
            state_log=os.path.join(run_dir, f'{alg_name}_state_ep000001.csv'),
            ecfg={
                'bw': float(bench['bandwidth_mbps']),
                'delay': min(float(v) for v in per_flow_delays),
                'per_flow_state_logs': True,
            },
            backend_type='mininet',
            env_name=env_name,
            link_schedule=[],
            n_flows=int(n_flows),
            trim_tail_s=5.0,
            mode='multi' if is_multi_agent(alg_name) else 'single',
            slot_id=instance_id,
        )
    except Exception as exc:
        print(f'[paper_bench] episode plot failed: {exc}', flush=True)
