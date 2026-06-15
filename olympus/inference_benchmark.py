"""
Inference benchmark for olympus.

Runs the active MAT policy over the same sweep used for training, without
starting a learner. Each episode writes per-agent traces and a multi-flow PDF,
then this script writes aggregate CSV/PDF summaries.
"""

import argparse
import csv
import glob
import itertools
import json
import math
import multiprocessing
import os
import random
import signal
import sys
import time
import traceback

import yaml

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from olympus.plots.multi_flow_episode_plot import (
    fairness_series,
    load_agent_traces,
)
from olympus.plots.plot_returns_watcher import generate_plot as _plot_returns
from olympus.orchestrator import (
    _activate_runtime_blocks,
    _as_bool,
    _build_pool,
    _final_runtime_cleanup,
    _resolve_repo_path,
    _slug,
    materialize_episode_config,
    run_episode,
)


def _episode_type_label(change_count: int) -> str:
    return f'{int(change_count)}_change' if int(change_count) == 1 else f'{int(change_count)}_changes'


def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    w = min(window, len(arr))
    return np.concatenate([np.full(w - 1, np.nan),
                           np.convolve(arr, np.ones(w) / w, mode='valid')])


def _nanmean(values, default=math.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def _nanpercentile(values, pct, default=math.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.percentile(arr, pct))


def _aligned_matrix(traces, key):
    if not traces:
        return np.asarray([]), np.asarray([[]])
    min_len = min(len(t['data'][key]) for t in traces)
    if min_len <= 0:
        return np.asarray([]), np.asarray([[]])
    t = traces[0]['data']['t_s'][:min_len]
    mat = np.vstack([item['data'][key][:min_len] for item in traces])
    return t, mat


def _state_log_metrics(state_log_path: str, n_agents: int, ep_return):
    traces = load_agent_traces(state_log_path, n_agents=n_agents)
    if not traces:
        return {}

    t, thr = _aligned_matrix(traces, 'avg_thr_mbps')
    _, rtt = _aligned_matrix(traces, 'avg_urtt_ms')
    _, srtt = _aligned_matrix(traces, 'srtt_ms')
    _, min_rtt = _aligned_matrix(traces, 'min_rtt_ms')
    _, kalman_rtt = _aligned_matrix(traces, 'kalman_rtt_ms')
    _, cwnd = _aligned_matrix(traces, 'cwnd')
    _, mult = _aligned_matrix(traces, 'cwnd_mult')
    _, loss = _aligned_matrix(traces, 'loss_ratio')
    _, reward = _aligned_matrix(traces, 'reward')
    _, fair = fairness_series(state_log_path, n_agents=n_agents)

    if thr.size == 0:
        return {}
    if not np.isfinite(srtt).any():
        srtt = rtt

    sum_thr = thr.sum(axis=0)
    mean_reward_step = reward.mean(axis=0) if reward.size else np.asarray([])
    ret = float(ep_return) if ep_return is not None else float(np.nansum(reward))
    duration_s = float(max(t[-1] - t[0], 0.0)) if len(t) > 1 else 0.0
    total_steps = int(sum(len(item['data']['t_s']) for item in traces))

    return {
        'return': ret,
        'duration_s': duration_s,
        'score_per_second': ret / duration_s if duration_s > 0 else math.nan,
        'steps': total_steps,
        'n_agents_seen': int(len(traces)),
        'mean_thr_mbps': _nanmean(sum_thr),
        'p50_thr_mbps': _nanpercentile(sum_thr, 50),
        'p05_thr_mbps': _nanpercentile(sum_thr, 5),
        'mean_agent_thr_mbps': _nanmean(thr),
        'mean_rtt_ms': _nanmean(rtt),
        'p95_rtt_ms': _nanpercentile(rtt, 95),
        'mean_srtt_ms': _nanmean(srtt),
        'p95_srtt_ms': _nanpercentile(srtt, 95),
        'mean_min_rtt_ms': _nanmean(min_rtt),
        'mean_kalman_rtt_ms': _nanmean(kalman_rtt),
        'mean_cwnd': _nanmean(cwnd),
        'mean_cwnd_mult': _nanmean(mult),
        'frac_mult_below_1': float(np.nanmean(mult < 1.0)),
        'frac_mult_above_1': float(np.nanmean(mult > 1.0)),
        'mean_loss_ratio': _nanmean(loss),
        'mean_reward': _nanmean(mean_reward_step),
        'fairness_mean': _nanmean(fair),
        'fairness_p05': _nanpercentile(fair, 5),
        'switch_rate': 0.0,
        'dominant_option_share': 1.0,
        'option_entropy': 0.0,
        'n_options_seen': 1,
    }


def _find_latest_checkpoint(cfg: dict, alg_name: str):
    outputs = cfg.get('outputs', {}) or {}
    root = _resolve_repo_path(outputs.get('root', os.path.join(_HERE, 'data')))
    alg_slug = _slug(alg_name)
    pattern = os.path.join(root, f'{alg_slug}_*', 'checkpoints',
                           f'{alg_slug}_cwnd_model.pt')
    candidates = [p for p in glob.glob(pattern) if os.path.exists(p)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _select_checkpoint(cfg: dict, args, alg_name: str) -> str:
    t_cfg = cfg.get('training', {}) or {}
    raw = args.checkpoint or t_cfg.get('resume_from') or t_cfg.get('checkpoint')
    if raw:
        path = _resolve_repo_path(str(raw))
    else:
        path = _find_latest_checkpoint(cfg, alg_name)
        if path:
            print(f'[bench] using latest checkpoint: {path}', flush=True)
    if not path:
        raise SystemExit('[bench] no checkpoint provided. Pass --checkpoint or set resume_from.')
    if not os.path.exists(path):
        raise SystemExit(f'[bench] checkpoint not found: {path}')
    return os.path.abspath(path)


def _prepare_benchmark_run(cfg: dict, config_path: str, checkpoint: str,
                           run_name: str = None, output_root: str = None) -> None:
    runtime = cfg.setdefault('runtime', {})
    outputs = cfg.setdefault('outputs', {})
    training = cfg.setdefault('training', {})

    alg_name = runtime.get('algorithm', 'mat')
    reward_name = runtime.get('reward', 'tempest_fairness_ma')
    root = _resolve_repo_path(output_root or outputs.get('root', os.path.join(_HERE, 'data')))
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = run_name or f'benchmark_{_slug(alg_name)}_{timestamp}'
    run_dir = os.path.abspath(os.path.join(root, run_name))

    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    episodes_dir = os.path.join(run_dir, 'episodes')
    plots_dir = os.path.join(run_dir, 'plots')
    telemetry_dir = os.path.join(run_dir, 'telemetry')
    for path in (checkpoints_dir, episodes_dir, plots_dir, telemetry_dir):
        os.makedirs(path, exist_ok=True)

    outputs.update({
        'run_dir': run_dir,
        'checkpoints_dir': checkpoints_dir,
        'episodes_dir': episodes_dir,
        'plots_dir': plots_dir,
        'telemetry_dir': telemetry_dir,
        'traces_dir': episodes_dir,
    })
    training['checkpoint'] = os.path.abspath(checkpoint)
    training['log_path'] = os.path.join(telemetry_dir, 'benchmark_metrics.csv')

    meta = {
        'mode': 'multi_agent_inference_benchmark',
        'run_name': run_name,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'algorithm': alg_name,
        'reward': reward_name,
        'source_config': os.path.abspath(config_path),
        'checkpoint': os.path.abspath(checkpoint),
        'run_dir': run_dir,
    }
    with open(os.path.join(telemetry_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    resolved_config = os.path.join(telemetry_dir, 'config.resolved.yaml')
    with open(resolved_config, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    outputs['resolved_config'] = resolved_config


def _metric_rows(metrics_csv: str):
    rows = []
    try:
        with open(metrics_csv, newline='') as f:
            for row in csv.DictReader(f):
                parsed = {}
                for k, v in row.items():
                    if k in ('episode_type', 'state_log', 'plot_path', 'error'):
                        parsed[k] = v
                    else:
                        try:
                            parsed[k] = float(v) if v != '' else math.nan
                        except (TypeError, ValueError):
                            parsed[k] = v
                rows.append(parsed)
    except FileNotFoundError:
        pass
    rows.sort(key=lambda r: r.get('episode', 0))
    return rows


def _array(rows, key):
    return np.asarray([r.get(key, math.nan) for r in rows], dtype=float)


def _plot_benchmark_trends(metrics_csv: str, output: str,
                           alg_name: str, checkpoint: str) -> None:
    rows = [r for r in _metric_rows(metrics_csv)
            if isinstance(r.get('return'), float) and np.isfinite(r.get('return'))]
    if not rows:
        print('[bench_plot] no metrics yet - skipping', flush=True)
        return

    ep = _array(rows, 'episode')
    ret = _array(rows, 'return')
    bws = _array(rows, 'bw')
    delays = _array(rows, 'delay')
    thr = _array(rows, 'mean_thr_mbps')
    rtt = _array(rows, 'mean_rtt_ms')
    p95_rtt = _array(rows, 'p95_rtt_ms')
    srtt = _array(rows, 'mean_srtt_ms')
    p95_srtt = _array(rows, 'p95_srtt_ms')
    cwnd_mult = _array(rows, 'mean_cwnd_mult')
    fairness = _array(rows, 'fairness_mean')
    fairness_p05 = _array(rows, 'fairness_p05')
    changes = _array(rows, 'schedule_changes')

    out_abs = os.path.abspath(output)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    tmp_path = f'{out_abs}.tmp.{os.getpid()}'
    pdf = PdfPages(tmp_path)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                                 gridspec_kw={'hspace': 0.32, 'wspace': 0.24})
        fig.suptitle(
            f'{alg_name} multi-agent inference - score={np.nanmean(ret):.2f} '
            f'({len(rows)} examples)\n{os.path.basename(checkpoint)}',
            fontsize=12, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(ep, ret, color='#9fbbe6', linewidth=0.7, alpha=0.65, label='return')
        ax.plot(ep, _rolling(ret, 20), color='black', linewidth=1.5, label='20-ep mean')
        ax.axhline(np.nanmean(ret), color='#e15759', linestyle='--', linewidth=1.1,
                   label=f'mean={np.nanmean(ret):.1f}')
        ax.set_title('Score Trend')
        ax.set_xlabel('Benchmark episode')
        ax.set_ylabel('Sum return')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.8)

        ax = axes[0, 1]
        ax.hist(ret[np.isfinite(ret)], bins=min(50, max(10, len(ret) // 6)),
                color='#4878cf', alpha=0.8)
        ax.axvline(np.nanmedian(ret), color='black', linewidth=1.2,
                   label=f'median={np.nanmedian(ret):.1f}')
        ax.axvline(np.nanmean(ret), color='#e15759', linewidth=1.2,
                   label=f'mean={np.nanmean(ret):.1f}')
        ax.set_title('Score Distribution')
        ax.set_xlabel('Sum return')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.8)

        ax = axes[1, 0]
        for bw in sorted(np.unique(bws[np.isfinite(bws)])):
            m = bws == bw
            ax.scatter(ep[m], ret[m], s=18, alpha=0.55, label=f'{int(bw)} Mbps')
        ax.set_title('Score by BW')
        ax.set_xlabel('Benchmark episode')
        ax.set_ylabel('Sum return')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.8, ncol=2)

        ax = axes[1, 1]
        for delay in sorted(np.unique(delays[np.isfinite(delays)])):
            m = delays == delay
            ax.scatter(ep[m], ret[m], s=18, alpha=0.55, label=f'{int(delay)} ms')
        ax.set_title('Score by Base RTT')
        ax.set_xlabel('Benchmark episode')
        ax.set_ylabel('Sum return')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.8, ncol=2)
        pdf.savefig(fig, dpi=120, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                                 gridspec_kw={'hspace': 0.32, 'wspace': 0.24})
        fig.suptitle(f'{alg_name} multi-agent inference - network/control trends',
                     fontsize=12, fontweight='bold')
        ax = axes[0, 0]
        ax.plot(ep, _rolling(thr, 20), color='#59a14f', linewidth=1.5,
                label='sum throughput')
        ax.set_title('Throughput')
        ax.set_xlabel('Benchmark episode')
        ax.set_ylabel('Mbps')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.8)

        ax = axes[0, 1]
        ax.plot(ep, _rolling(rtt, 20), color='#f28e2c', linewidth=1.5,
                label='mean RTT')
        ax.plot(ep, _rolling(p95_rtt, 20), color='#e15759', linewidth=1.2,
                label='p95 RTT')
        if np.isfinite(srtt).any():
            ax.plot(ep, _rolling(srtt, 20), color='#4c78a8', linewidth=1.3,
                    label='mean SRTT')
        if np.isfinite(p95_srtt).any():
            ax.plot(ep, _rolling(p95_srtt, 20), color='#b07aa1', linewidth=1.1,
                    linestyle='--', label='p95 SRTT')
        ax.set_title('RTT')
        ax.set_xlabel('Benchmark episode')
        ax.set_ylabel('ms')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.8)

        ax = axes[1, 0]
        ax.plot(ep, _rolling(cwnd_mult, 20), color='#4878cf', linewidth=1.5,
                label='mean cwnd multiplier')
        ax.axhline(1.0, color='#666', linestyle='--', linewidth=1.0)
        ax.set_title('CWND Multiplier')
        ax.set_xlabel('Benchmark episode')
        ax.set_ylabel('Multiplier')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.8)

        ax = axes[1, 1]
        ax.plot(ep, _rolling(fairness, 20), color='black', linewidth=1.5,
                label='mean Jain fairness')
        ax.plot(ep, _rolling(fairness_p05, 20), color='#b07aa1', linewidth=1.2,
                label='p05 Jain fairness')
        ax.set_ylim(0, 1.05)
        ax.set_title('Fairness')
        ax.set_xlabel('Benchmark episode')
        ax.set_ylabel('Jain index')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.8)
        pdf.savefig(fig, dpi=120, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                                 gridspec_kw={'wspace': 0.28})
        fig.suptitle(f'{alg_name} multi-agent inference - schedules and fairness',
                     fontsize=12, fontweight='bold')
        labels = sorted(np.unique(changes[np.isfinite(changes)]).astype(int))
        data = [ret[changes == n] for n in labels]
        ax = axes[0]
        if data:
            ax.boxplot(data, showmeans=True)
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels([_episode_type_label(n).replace('_', ' ')
                                for n in labels], rotation=20)
        ax.set_title('Score by Schedule Type')
        ax.set_xlabel('Schedule changes')
        ax.set_ylabel('Sum return')
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[1]
        ax.scatter(fairness, ret, s=22, alpha=0.6, color='#4878cf')
        ax.set_title('Return vs Fairness')
        ax.set_xlabel('Mean Jain fairness')
        ax.set_ylabel('Sum return')
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, dpi=120, bbox_inches='tight')
        plt.close(fig)
    finally:
        pdf.close()

    os.replace(tmp_path, out_abs)
    print(f'[bench_plot] saved -> {out_abs}', flush=True)


def _write_summary(metrics_csv: str, output: str, cfg: dict,
                   checkpoint: str, total_examples: int, failures: int) -> dict:
    rows = [r for r in _metric_rows(metrics_csv)
            if isinstance(r.get('return'), float) and np.isfinite(r.get('return'))]
    returns = np.asarray([r['return'] for r in rows], dtype=float)
    runtime = cfg.get('runtime', {}) or {}
    summary = {
        'mode': 'multi_agent_inference_benchmark',
        'algorithm': runtime.get('algorithm', ''),
        'reward': runtime.get('reward', ''),
        'checkpoint': os.path.abspath(checkpoint),
        'total_examples': int(total_examples),
        'completed': int(len(rows)),
        'failed': int(failures),
        'score': float(np.nanmean(returns)) if returns.size else None,
        'score_name': 'mean_sum_episode_return',
        'median_return': float(np.nanmedian(returns)) if returns.size else None,
        'mean_throughput_mbps': _nanmean([r.get('mean_thr_mbps', math.nan) for r in rows], default=None),
        'mean_rtt_ms': _nanmean([r.get('mean_rtt_ms', math.nan) for r in rows], default=None),
        'mean_srtt_ms': _nanmean([r.get('mean_srtt_ms', math.nan) for r in rows], default=None),
        'mean_fairness': _nanmean([r.get('fairness_mean', math.nan) for r in rows], default=None),
        'p05_fairness_mean': _nanmean([r.get('fairness_p05', math.nan) for r in rows], default=None),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    with open(output, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _slot_process(instance_id, work_q, result_q, cfg, listener_bin, python_bin):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    alg_name = (cfg.get('runtime', {}) or {}).get('algorithm', 'mat')
    episodes_dir = os.path.abspath((cfg.get('outputs', {}) or {})['episodes_dir'])

    while True:
        item = work_q.get()
        if item is None:
            break
        ep, ecfg = item
        try:
            ep_return, ecfg_out, link_sched = run_episode(
                cfg, ecfg, ep, listener_bin, python_bin,
                '', '', instance_id)
            n_agents = int(ecfg_out.get('flows', (cfg.get('sweep', {}) or {}).get('flows', 2)))
            state_log = os.path.join(episodes_dir, f'{alg_name}_state_ep{ep:06d}.csv')
            metrics = _state_log_metrics(state_log, n_agents, ep_return)
            result_q.put((ep, ecfg_out, ep_return, link_sched, metrics, None))
        except Exception as e:
            result_q.put((ep, ecfg, None, ecfg.get('link_schedule', []), {},
                          f'{e}\n{traceback.format_exc()}'))


def _example_list(pool, n_examples, shuffle=False, seed=None):
    templates = list(pool)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(templates)
    source = templates if n_examples is None else itertools.islice(itertools.cycle(templates), n_examples)
    out = []
    for ep, ecfg in enumerate(source):
        out.append((ep, materialize_episode_config(ecfg, rng=rng)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Run deterministic inference over the olympus training sweep.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--episodes', type=int, default=None,
                    help='number of examples to run; default is exactly one pass over the pool')
    ap.add_argument('--n-parallel', type=int, default=None)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--output-root', default=None)
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _activate_runtime_blocks(cfg)

    runtime = cfg.get('runtime', {}) or {}
    alg_name = runtime.get('algorithm', 'mat')
    checkpoint = _select_checkpoint(cfg, args, alg_name)

    if args.no_plots:
        cfg.setdefault('outputs', {})['plot_episodes'] = False

    if not args.stochastic:
        os.environ['SAO_DETERMINISTIC'] = '1'
        os.environ['OC_DETERMINISTIC'] = '1'
        os.environ['SAO_NOISE_STD'] = '0.0'
    os.environ['SAO_REQUIRE_CHECKPOINT'] = '1'

    _prepare_benchmark_run(
        cfg, args.config, checkpoint,
        run_name=args.run_name,
        output_root=args.output_root,
    )

    paths = cfg.get('paths', {}) or {}
    if 'listener' not in paths or 'py' not in paths:
        raise SystemExit('[bench] config.yaml must define paths.listener and paths.py')
    listener_bin = _resolve_repo_path(paths['listener'])
    python_bin = _resolve_repo_path(paths['py'])

    pool, do_shuffle = _build_pool(cfg)
    examples = _example_list(
        pool,
        args.episodes if args.episodes is not None else len(pool),
        shuffle=args.shuffle or do_shuffle,
        seed=args.seed,
    )
    total = len(examples)
    n_parallel = args.n_parallel or int(cfg.get('n_parallel', 1))
    n_parallel = max(1, min(n_parallel, total))

    outputs = cfg.get('outputs', {}) or {}
    episodes_dir = os.path.abspath(outputs['episodes_dir'])
    plots_dir = os.path.abspath(outputs['plots_dir'])
    telemetry_dir = os.path.abspath(outputs['telemetry_dir'])
    run_dir = os.path.abspath(outputs['run_dir'])
    returns_csv = os.path.join(episodes_dir, 'episode_returns.csv')
    metrics_csv = os.path.join(telemetry_dir, 'benchmark_metrics.csv')
    trends_pdf = os.path.join(telemetry_dir, 'benchmark_trends.pdf')
    returns_pdf = os.path.join(telemetry_dir, 'episode_returns.pdf')
    summary_json = os.path.join(telemetry_dir, 'benchmark_summary.json')

    print(f'[bench] algorithm={alg_name}', flush=True)
    print(f'[bench] checkpoint={checkpoint}', flush=True)
    print(f'[bench] examples={total}  n_parallel={n_parallel}', flush=True)
    print(f'[bench] run_dir={run_dir}', flush=True)

    returns_fields = [
        'episode', 'bw', 'delay', 'scheduled',
        'schedule_changes', 'episode_type', 'flows', 'return',
    ]
    metrics_fields = returns_fields + [
        'duration', 'duration_s', 'score_per_second', 'steps',
        'n_agents_seen', 'mean_thr_mbps', 'p50_thr_mbps', 'p05_thr_mbps',
        'mean_agent_thr_mbps', 'mean_rtt_ms', 'p95_rtt_ms',
        'mean_srtt_ms', 'p95_srtt_ms',
        'mean_min_rtt_ms', 'mean_kalman_rtt_ms', 'mean_cwnd',
        'mean_cwnd_mult', 'frac_mult_below_1', 'frac_mult_above_1',
        'mean_loss_ratio', 'mean_reward', 'fairness_mean', 'fairness_p05',
        'switch_rate', 'dominant_option_share', 'option_entropy',
        'n_options_seen', 'state_log', 'plot_path', 'error',
    ]
    with open(returns_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=returns_fields).writeheader()
    with open(metrics_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    slot_procs = []
    for i in range(n_parallel):
        p = multiprocessing.Process(
            target=_slot_process,
            args=(i, work_q, result_q, cfg, listener_bin, python_bin),
            daemon=True,
        )
        p.start()
        slot_procs.append(p)

    for item in examples:
        work_q.put(item)
    for _ in slot_procs:
        work_q.put(None)

    completed = 0
    failures = 0
    try:
        while completed < total:
            ep, ecfg, ep_return, link_sched, metrics, error = result_q.get()
            completed += 1
            schedule_changes = len(link_sched or [])
            episode_type = _episode_type_label(schedule_changes)
            flows = ecfg.get('flows', (cfg.get('sweep', {}) or {}).get('flows', 2))
            base_row = {
                'episode': ep,
                'bw': ecfg.get('bw', 100),
                'delay': ecfg.get('delay', 20),
                'scheduled': int(bool(link_sched)),
                'schedule_changes': schedule_changes,
                'episode_type': episode_type,
                'flows': flows,
                'return': f'{ep_return:.4f}' if ep_return is not None else '',
            }
            if error:
                failures += 1
                print(f'[bench] ep={ep} failed: {error.splitlines()[0]}', flush=True)
            else:
                fair = metrics.get('fairness_mean', math.nan)
                print(f'[bench] ep={ep} return={ep_return} fairness={fair:.3f} '
                      f'({completed}/{total})', flush=True)

            with open(returns_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=returns_fields).writerow(base_row)

            bw_str = f"{ecfg.get('bw', 100):.0f}"
            delay_str = f"{ecfg.get('delay', 20):.0f}"
            metric_row = dict(base_row)
            metric_row.update({
                'duration': ecfg.get('duration', cfg.get('duration', '')),
                'state_log': os.path.join(
                    episodes_dir, f'{alg_name}_state_ep{ep:06d}.csv'),
                'plot_path': os.path.join(
                    plots_dir, f'ep{ep:06d}_bw{bw_str}_d{delay_str}_n{flows}.pdf')
                    if _as_bool(outputs.get('plot_episodes'), default=True) else '',
                'error': error or '',
            })
            metric_row.update(metrics)
            for key in metrics_fields:
                metric_row.setdefault(key, '')
            with open(metrics_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=metrics_fields).writerow(metric_row)

    except KeyboardInterrupt:
        print('\n[bench] interrupted - stopping slots', flush=True)
    finally:
        for p in slot_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
            if p.is_alive():
                p.kill()
        learner_port = int((cfg.get('learner', {}) or {}).get('port', 6401))
        _final_runtime_cleanup(learner_port)

    try:
        _plot_returns(
            csv_path=returns_csv,
            output=returns_pdf,
            learner_log=os.path.join(telemetry_dir, 'no_learner_metrics.csv'),
        )
    except Exception as e:
        print(f'[bench] training-style returns plot failed: {e}', flush=True)

    try:
        _plot_benchmark_trends(metrics_csv, trends_pdf, alg_name, checkpoint)
    except Exception as e:
        print(f'[bench] benchmark trend plot failed: {e}', flush=True)

    summary = _write_summary(
        metrics_csv, summary_json, cfg, checkpoint,
        total_examples=total,
        failures=failures,
    )
    print('\n[bench] done', flush=True)
    print(f'[bench] score(mean sum episode return)={summary.get("score")}', flush=True)
    print(f'[bench] summary={summary_json}', flush=True)
    print(f'[bench] trends={trends_pdf}', flush=True)
    print(f'[bench] training-style returns={returns_pdf}', flush=True)


if __name__ == '__main__':
    main()
