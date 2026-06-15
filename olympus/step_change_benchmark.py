"""
Step-change inference benchmark for olympus.

Runs simple one-step bandwidth and RTT changes with deterministic MAT inference.
Metrics are computed from all per-agent traces: goodput is summed across flows,
RTT is averaged across flow samples, and Jain fairness is reported before and
after the change.
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import random
import signal
import sys
import time
import traceback
from collections import defaultdict

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

from olympus.common.multi_flow_episode_plot import (
    fairness_series,
    load_agent_traces,
)
from olympus.inference_benchmark import (
    _nanmean,
    _select_checkpoint,
)
from olympus.orchestrator import (
    _activate_runtime_blocks,
    _as_bool,
    _final_runtime_cleanup,
    _flow_value,
    _resolve_repo_path,
    _slug,
    run_episode,
)


DEFAULT_FACTORS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]


def _fmt(value, digits=2):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 'nan'
    if not np.isfinite(v):
        return 'nan'
    return f'{v:.{digits}f}'


def _parse_factors(raw: str):
    vals = [float(part) for part in str(raw).replace(',', ' ').split()]
    if not vals:
        raise SystemExit('[step_bench] at least one factor is required')
    return vals


def _prepare_step_run(cfg: dict, config_path: str, checkpoint: str,
                      run_name: str = None, output_root: str = None) -> None:
    runtime = cfg.setdefault('runtime', {})
    outputs = cfg.setdefault('outputs', {})
    training = cfg.setdefault('training', {})

    alg_name = runtime.get('algorithm', 'mat')
    reward_name = runtime.get('reward', 'tempest_fairness_ma')
    root = _resolve_repo_path(output_root or outputs.get('root', os.path.join(_HERE, 'data')))
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = run_name or f'step_change_{_slug(alg_name)}_{timestamp}'
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
        'mode': 'multi_agent_step_change_benchmark',
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


def _build_step_pool(base_bw: float, base_delay: float, factors: list,
                     repeats: int, change_at: float, duration: int,
                     bdp_mult: float, flows: int):
    pool = []
    for change_kind in ('bandwidth', 'rtt'):
        for direction in ('up', 'down'):
            for factor in factors:
                for rep in range(repeats):
                    if change_kind == 'bandwidth':
                        delay_before = base_delay
                        delay_after = base_delay
                        if direction == 'up':
                            bw_before = base_bw
                            bw_after = base_bw * factor
                        else:
                            bw_before = base_bw * factor
                            bw_after = base_bw
                        schedule = [{'t': change_at, 'bw': round(bw_after, 3)}]
                    else:
                        bw_before = base_bw
                        bw_after = base_bw
                        if direction == 'up':
                            delay_before = base_delay
                            delay_after = base_delay * factor
                        else:
                            delay_before = base_delay * factor
                            delay_after = base_delay
                        schedule = [{'t': change_at, 'delay': round(delay_after, 3)}]

                    pool.append({
                        'bw': round(float(bw_before), 3),
                        'delay': round(float(delay_before), 3),
                        'bdp_mult': bdp_mult,
                        'flows': flows,
                        'duration': duration,
                        'link_schedule': schedule,
                        'change_kind': change_kind,
                        'direction': direction,
                        'factor': float(factor),
                        'repeat': int(rep),
                        'change_at': float(change_at),
                        'bw_before': round(float(bw_before), 3),
                        'bw_after': round(float(bw_after), 3),
                        'delay_before': round(float(delay_before), 3),
                        'delay_after': round(float(delay_after), 3),
                    })
    return pool


def _aligned_matrix(traces, key):
    if not traces:
        return np.asarray([]), np.asarray([[]])
    min_len = min(len(t['data'][key]) for t in traces)
    if min_len <= 0:
        return np.asarray([]), np.asarray([[]])
    t = traces[0]['data']['t_s'][:min_len]
    mat = np.vstack([item['data'][key][:min_len] for item in traces])
    return t, mat


def _load_trace_windows(state_log_path: str, n_agents: int, change_at: float,
                        warmup: float, guard: float, tail_trim: float):
    traces = load_agent_traces(state_log_path, n_agents=n_agents, trim_tail_s=0.0)
    if not traces:
        return {}

    t, thr = _aligned_matrix(traces, 'avg_thr_mbps')
    _, rtt = _aligned_matrix(traces, 'avg_urtt_ms')
    _, reward = _aligned_matrix(traces, 'reward')
    _, mult = _aligned_matrix(traces, 'cwnd_mult')
    fair_t, fair = fairness_series(state_log_path, n_agents=n_agents)
    if thr.size == 0 or len(t) == 0:
        return {}

    end_t = float(t[-1])
    pre = (t >= warmup) & (t < max(warmup, change_at - guard))
    post = (t >= change_at + guard) & (t <= max(change_at + guard, end_t - tail_trim))
    if not pre.any():
        pre = t < change_at
    if not post.any():
        post = t >= change_at

    sum_thr = thr.sum(axis=0)
    mean_rtt = rtt.mean(axis=0)
    mean_reward = reward.mean(axis=0)
    mean_mult = mult.mean(axis=0)

    if len(fair_t) == len(t):
        fair_pre = fair[pre]
        fair_post = fair[post]
    else:
        fair_pre = fair[fair_t < change_at] if len(fair) else np.asarray([])
        fair_post = fair[fair_t >= change_at] if len(fair) else np.asarray([])

    pre_goodput = _nanmean(sum_thr[pre])
    post_goodput = _nanmean(sum_thr[post])
    pre_rtt = _nanmean(mean_rtt[pre])
    post_rtt = _nanmean(mean_rtt[post])
    pre_reward = _nanmean(mean_reward[pre])
    post_reward = _nanmean(mean_reward[post])
    pre_mult = _nanmean(mean_mult[pre])
    post_mult = _nanmean(mean_mult[post])
    pre_fair = _nanmean(fair_pre)
    post_fair = _nanmean(fair_post)

    return {
        'pre_samples': int(pre.sum()),
        'post_samples': int(post.sum()),
        'pre_goodput_mbps': pre_goodput,
        'post_goodput_mbps': post_goodput,
        'goodput_delta_mbps': post_goodput - pre_goodput,
        'goodput_ratio': post_goodput / pre_goodput if pre_goodput > 0 else math.nan,
        'pre_rtt_ms': pre_rtt,
        'post_rtt_ms': post_rtt,
        'rtt_delta_ms': post_rtt - pre_rtt,
        'rtt_ratio': post_rtt / pre_rtt if pre_rtt > 0 else math.nan,
        'pre_reward': pre_reward,
        'post_reward': post_reward,
        'reward_delta': post_reward - pre_reward,
        'pre_cwnd_mult': pre_mult,
        'post_cwnd_mult': post_mult,
        'cwnd_mult_delta': post_mult - pre_mult,
        'pre_fairness': pre_fair,
        'post_fairness': post_fair,
        'fairness_delta': post_fair - pre_fair,
    }


def _vals(items, name):
    out = []
    for item in items:
        try:
            out.append(float(item.get(name, math.nan)))
        except (TypeError, ValueError):
            out.append(math.nan)
    return np.asarray(out, dtype=float)


def _summarize_runs(run_csv: str, summary_csv: str, summary_json: str):
    rows = []
    try:
        with open(run_csv, newline='') as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        pass

    groups = defaultdict(list)
    for row in rows:
        if row.get('error'):
            continue
        groups[(row.get('change_kind'), row.get('direction'),
                float(row.get('factor', 0)))].append(row)

    fields = [
        'change_kind', 'direction', 'factor', 'runs',
        'bw_before', 'bw_after', 'delay_before', 'delay_after',
        'expected_goodput_ratio', 'expected_rtt_ratio',
        'goodput_score', 'rtt_score',
        'pre_goodput_mbps_mean', 'post_goodput_mbps_mean',
        'goodput_delta_mbps_mean', 'goodput_ratio_mean', 'goodput_ratio_std',
        'pre_rtt_ms_mean', 'post_rtt_ms_mean',
        'rtt_delta_ms_mean', 'rtt_ratio_mean', 'rtt_ratio_std',
        'pre_reward_mean', 'post_reward_mean', 'reward_delta_mean',
        'pre_fairness_mean', 'post_fairness_mean', 'fairness_delta_mean',
    ]
    summary_rows = []
    for (kind, direction, factor), items in sorted(groups.items()):
        row0 = items[0]
        expected_goodput_ratio = 1.0
        if kind == 'bandwidth':
            expected_goodput_ratio = factor if direction == 'up' else (1.0 / factor)
        expected_rtt_ratio = 1.0
        if kind == 'rtt':
            expected_rtt_ratio = factor if direction == 'up' else (1.0 / factor)

        goodput_ratio = _nanmean(_vals(items, 'goodput_ratio'))
        rtt_ratio = _nanmean(_vals(items, 'rtt_ratio'))
        goodput_score = math.nan
        if np.isfinite(goodput_ratio) and goodput_ratio > 0 and expected_goodput_ratio > 0:
            goodput_score = float(math.exp(-abs(math.log(goodput_ratio / expected_goodput_ratio))))
        rtt_score = math.nan
        if np.isfinite(rtt_ratio) and rtt_ratio > 0 and expected_rtt_ratio > 0:
            rtt_score = float(min(1.0, expected_rtt_ratio / rtt_ratio))

        summary_rows.append({
            'change_kind': kind,
            'direction': direction,
            'factor': factor,
            'runs': len(items),
            'bw_before': row0.get('bw_before', ''),
            'bw_after': row0.get('bw_after', ''),
            'delay_before': row0.get('delay_before', ''),
            'delay_after': row0.get('delay_after', ''),
            'expected_goodput_ratio': expected_goodput_ratio,
            'expected_rtt_ratio': expected_rtt_ratio,
            'goodput_score': goodput_score,
            'rtt_score': rtt_score,
            'pre_goodput_mbps_mean': _nanmean(_vals(items, 'pre_goodput_mbps')),
            'post_goodput_mbps_mean': _nanmean(_vals(items, 'post_goodput_mbps')),
            'goodput_delta_mbps_mean': _nanmean(_vals(items, 'goodput_delta_mbps')),
            'goodput_ratio_mean': goodput_ratio,
            'goodput_ratio_std': float(np.nanstd(_vals(items, 'goodput_ratio'))),
            'pre_rtt_ms_mean': _nanmean(_vals(items, 'pre_rtt_ms')),
            'post_rtt_ms_mean': _nanmean(_vals(items, 'post_rtt_ms')),
            'rtt_delta_ms_mean': _nanmean(_vals(items, 'rtt_delta_ms')),
            'rtt_ratio_mean': rtt_ratio,
            'rtt_ratio_std': float(np.nanstd(_vals(items, 'rtt_ratio'))),
            'pre_reward_mean': _nanmean(_vals(items, 'pre_reward')),
            'post_reward_mean': _nanmean(_vals(items, 'post_reward')),
            'reward_delta_mean': _nanmean(_vals(items, 'reward_delta')),
            'pre_fairness_mean': _nanmean(_vals(items, 'pre_fairness')),
            'post_fairness_mean': _nanmean(_vals(items, 'post_fairness')),
            'fairness_delta_mean': _nanmean(_vals(items, 'fairness_delta')),
        })

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    tracking_errors = []
    for row in summary_rows:
        if row['change_kind'] != 'bandwidth':
            continue
        ratio = float(row['goodput_ratio_mean'])
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        factor = float(row['factor'])
        expected = factor if row['direction'] == 'up' else (1.0 / factor)
        tracking_errors.append(abs(math.log(ratio / expected)))
    bandwidth_tracking_score = float(math.exp(-np.mean(tracking_errors))) if tracking_errors else None

    payload = {
        'mode': 'multi_agent_step_change_benchmark',
        'completed_runs': sum(len(v) for v in groups.values()),
        'groups': len(summary_rows),
        'bandwidth_tracking_score': bandwidth_tracking_score,
        'mean_post_fairness': _nanmean([r['post_fairness_mean'] for r in summary_rows], default=None),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    with open(summary_json, 'w') as f:
        json.dump(payload, f, indent=2)
    return summary_rows, payload


def _load_summary(summary_csv: str):
    rows = []
    try:
        with open(summary_csv, newline='') as f:
            for row in csv.DictReader(f):
                parsed = {}
                for k, v in row.items():
                    if k in ('change_kind', 'direction'):
                        parsed[k] = v
                    else:
                        try:
                            parsed[k] = float(v) if v != '' else math.nan
                        except (TypeError, ValueError):
                            parsed[k] = v
                rows.append(parsed)
    except FileNotFoundError:
        pass
    return rows


def _summary_heatmap(rows, factors, kind, metric):
    directions = ['up', 'down']
    grid = np.full((len(directions), len(factors)), np.nan)
    for i, direction in enumerate(directions):
        for j, factor in enumerate(factors):
            for row in rows:
                if (row['change_kind'] == kind
                        and row['direction'] == direction
                        and abs(float(row['factor']) - float(factor)) < 1e-9):
                    grid[i, j] = float(row.get(metric, math.nan))
                    break
    return grid


def _draw_heatmap(ax, grid, factors, title, cmap='viridis', fmt='.2f',
                  vmin=None, vmax=None, cbar_label=None, display_grid=None):
    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    labels = grid if display_grid is None else display_grid
    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['up', 'down'])
    ax.set_xticks(np.arange(len(factors)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v)
                        for v in factors], rotation=25)
    ax.set_xlabel('Step factor')
    ax.set_ylabel('Direction')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(labels[i, j]):
                ax.text(j, i, format(labels[i, j], fmt),
                        ha='center', va='center', color='black',
                        fontsize=8, fontweight='bold',
                        bbox={'facecolor': 'white', 'edgecolor': 'none',
                              'alpha': 0.72, 'pad': 1.2})
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.82)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)


def _plot_step_summary(summary_csv: str, output: str, alg_name: str) -> None:
    rows = _load_summary(summary_csv)
    if not rows:
        print('[step_plot] no summary rows - skipping', flush=True)
        return
    factors = sorted({float(r['factor']) for r in rows if np.isfinite(float(r['factor']))})

    out_abs = os.path.abspath(output)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    tmp_path = f'{out_abs}.tmp.{os.getpid()}'
    pdf = PdfPages(tmp_path)
    try:
        for metric, title in [
            ('goodput_score', 'Goodput Score'),
            ('rtt_score', 'RTT Control Score'),
            ('post_fairness_mean', 'Post-change Jain Fairness'),
        ]:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                                     gridspec_kw={'wspace': 0.38})
            fig.suptitle(f'{alg_name} multi-agent step-change - {title}',
                         fontsize=12, fontweight='bold')
            for ax, kind in zip(axes, ('bandwidth', 'rtt')):
                grid = _summary_heatmap(rows, factors, kind, metric)
                _draw_heatmap(ax, grid, factors, f'{kind} steps',
                              cmap='RdYlGn', fmt='.2f',
                              vmin=0.0, vmax=1.0,
                              cbar_label='higher is better')
            pdf.savefig(fig, dpi=120, bbox_inches='tight')
            plt.close(fig)

        for before_key, after_key, ylabel, title, cmap in [
            ('pre_goodput_mbps_mean', 'post_goodput_mbps_mean',
             'Sum goodput (Mbps)', 'Before/After Goodput', 'RdYlGn'),
            ('pre_rtt_ms_mean', 'post_rtt_ms_mean',
             'Mean RTT (ms)', 'Before/After RTT', 'RdYlGn_r'),
            ('pre_fairness_mean', 'post_fairness_mean',
             'Jain fairness', 'Before/After Fairness', 'RdYlGn'),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                                     gridspec_kw={'hspace': 0.35, 'wspace': 0.38})
            fig.suptitle(f'{alg_name} multi-agent step-change - {title}',
                         fontsize=12, fontweight='bold')
            for i, kind in enumerate(('bandwidth', 'rtt')):
                before_grid = _summary_heatmap(rows, factors, kind, before_key)
                after_grid = _summary_heatmap(rows, factors, kind, after_key)
                finite = np.concatenate([
                    before_grid[np.isfinite(before_grid)],
                    after_grid[np.isfinite(after_grid)],
                ])
                vmin = float(np.min(finite)) if finite.size else None
                vmax = float(np.max(finite)) if finite.size else None
                if 'Fairness' in title:
                    vmin, vmax = 0.0, 1.0
                _draw_heatmap(axes[i, 0], before_grid, factors,
                              f'{kind} before - {ylabel}',
                              cmap=cmap, fmt='.2f' if 'Fairness' in title else '.1f',
                              vmin=vmin, vmax=vmax, cbar_label=ylabel)
                _draw_heatmap(axes[i, 1], after_grid, factors,
                              f'{kind} after - {ylabel}',
                              cmap=cmap, fmt='.2f' if 'Fairness' in title else '.1f',
                              vmin=vmin, vmax=vmax, cbar_label=ylabel)
            pdf.savefig(fig, dpi=120, bbox_inches='tight')
            plt.close(fig)
    finally:
        pdf.close()

    os.replace(tmp_path, out_abs)
    print(f'[step_plot] saved -> {out_abs}', flush=True)


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
            windows = _load_trace_windows(
                state_log,
                n_agents=n_agents,
                change_at=float(ecfg.get('change_at', 30.0)),
                warmup=float(cfg.get('_step_bench_warmup', 5.0)),
                guard=float(cfg.get('_step_bench_guard', 2.0)),
                tail_trim=float(cfg.get('_step_bench_tail_trim', 5.0)),
            )
            result_q.put((ep, ecfg_out, ep_return, link_sched, windows, None))
        except Exception as e:
            result_q.put((ep, ecfg, None, ecfg.get('link_schedule', []), {},
                          f'{e}\n{traceback.format_exc()}'))


def main():
    ap = argparse.ArgumentParser(
        description='Run bandwidth/RTT step-change inference benchmark.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--base-bw', type=float, default=50.0)
    ap.add_argument('--base-delay', type=float, default=20.0)
    ap.add_argument('--duration', type=int, default=30)
    ap.add_argument('--change-at', type=float, default=15.0)
    ap.add_argument('--factors', default='1 1.25 1.5 2 3 4 5')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--bdp-mult', type=float, default=None)
    ap.add_argument('--flows', type=int, default=None,
                    help='Number of concurrent flows per slot (1..4).')
    ap.add_argument('--agent-join-pattern',
                    choices=['simul', 'half_late', 'random'], default=None,
                    help='Override sweep.agent_join_pattern for this benchmark run.')
    ap.add_argument('--agent-join-max-frac', type=float, default=None,
                    help='Latest-join fraction for --agent-join-pattern=random '
                         '(default 0.2 => every late agent has at least the final 80%%).')
    ap.add_argument('--n-parallel', type=int, default=None)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--output-root', default=None)
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--warmup', type=float, default=5.0)
    ap.add_argument('--guard', type=float, default=2.0)
    ap.add_argument('--tail-trim', type=float, default=5.0)
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

    _prepare_step_run(
        cfg, args.config, checkpoint,
        run_name=args.run_name,
        output_root=args.output_root,
    )

    cfg['_step_bench_warmup'] = float(args.warmup)
    cfg['_step_bench_guard'] = float(args.guard)
    cfg['_step_bench_tail_trim'] = float(args.tail_trim)

    paths = cfg.get('paths', {}) or {}
    if 'listener' not in paths or 'py' not in paths:
        raise SystemExit('[step_bench] config.yaml must define paths.listener and paths.py')
    listener_bin = _resolve_repo_path(paths['listener'])
    python_bin = _resolve_repo_path(paths['py'])

    factors = _parse_factors(args.factors)
    bdp_mult = args.bdp_mult
    if bdp_mult is None:
        bdp_mult = float((cfg.get('sweep', {}) or {}).get('bdp_mult', 4.0))
    flows = args.flows
    if flows is None:
        flows = _flow_value((cfg.get('sweep', {}) or {}).get('flows', 2))
    flows = _flow_value(flows)
    # Make the chosen flows count and join pattern visible to run_episode().
    sweep_cfg = cfg.setdefault('sweep', {})
    sweep_cfg['flows'] = flows
    if args.agent_join_pattern is not None:
        sweep_cfg['agent_join_pattern'] = args.agent_join_pattern
    if args.agent_join_max_frac is not None:
        sweep_cfg['agent_join_max_frac'] = float(args.agent_join_max_frac)
    pool = _build_step_pool(
        base_bw=args.base_bw,
        base_delay=args.base_delay,
        factors=factors,
        repeats=max(1, int(args.repeats)),
        change_at=args.change_at,
        duration=args.duration,
        bdp_mult=bdp_mult,
        flows=flows,
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(pool)

    total = len(pool)
    n_parallel = args.n_parallel or int(cfg.get('n_parallel', 1))
    n_parallel = max(1, min(n_parallel, total))

    outputs = cfg.get('outputs', {}) or {}
    episodes_dir = os.path.abspath(outputs['episodes_dir'])
    plots_dir = os.path.abspath(outputs['plots_dir'])
    telemetry_dir = os.path.abspath(outputs['telemetry_dir'])
    run_dir = os.path.abspath(outputs['run_dir'])
    run_csv = os.path.join(telemetry_dir, 'benchmark_metrics.csv')
    summary_csv = os.path.join(telemetry_dir, 'step_change_summary.csv')
    summary_json = os.path.join(telemetry_dir, 'benchmark_summary.json')
    summary_pdf = os.path.join(telemetry_dir, 'benchmark_trends.pdf')

    print(f'[step_bench] algorithm={alg_name}', flush=True)
    print(f'[step_bench] checkpoint={checkpoint}', flush=True)
    print(f'[step_bench] examples={total}  repeats={args.repeats}  n_parallel={n_parallel}', flush=True)
    print(f'[step_bench] base_bw={args.base_bw}Mbps base_delay={args.base_delay}ms '
          f'flows={flows} change_at={args.change_at}s duration={args.duration}s', flush=True)
    print(f'[step_bench] run_dir={run_dir}', flush=True)

    fields = [
        'episode', 'change_kind', 'direction', 'factor', 'repeat',
        'bw_before', 'bw_after', 'delay_before', 'delay_after',
        'flows', 'return', 'pre_samples', 'post_samples',
        'pre_goodput_mbps', 'post_goodput_mbps',
        'goodput_delta_mbps', 'goodput_ratio',
        'pre_rtt_ms', 'post_rtt_ms', 'rtt_delta_ms', 'rtt_ratio',
        'pre_reward', 'post_reward', 'reward_delta',
        'pre_cwnd_mult', 'post_cwnd_mult', 'cwnd_mult_delta',
        'pre_fairness', 'post_fairness', 'fairness_delta',
        'state_log', 'plot_path', 'error',
    ]
    with open(run_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

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

    for ep, ecfg in enumerate(pool):
        work_q.put((ep, ecfg))
    for _ in slot_procs:
        work_q.put(None)

    completed = 0
    try:
        while completed < total:
            ep, ecfg, ep_return, _link_sched, windows, error = result_q.get()
            completed += 1

            bw_str = f"{ecfg.get('bw', 100):.0f}"
            delay_str = f"{ecfg.get('delay', 20):.0f}"
            row = {
                'episode': ep,
                'change_kind': ecfg.get('change_kind'),
                'direction': ecfg.get('direction'),
                'factor': ecfg.get('factor'),
                'repeat': ecfg.get('repeat'),
                'bw_before': ecfg.get('bw_before'),
                'bw_after': ecfg.get('bw_after'),
                'delay_before': ecfg.get('delay_before'),
                'delay_after': ecfg.get('delay_after'),
                'flows': ecfg.get('flows', flows),
                'return': f'{ep_return:.4f}' if ep_return is not None else '',
                'state_log': os.path.join(
                    episodes_dir, f'{alg_name}_state_ep{ep:06d}.csv'),
                'plot_path': os.path.join(
                    plots_dir, f'ep{ep:06d}_bw{bw_str}_d{delay_str}_n{ecfg.get("flows", flows)}.pdf')
                    if _as_bool(outputs.get('plot_episodes'), default=True) else '',
                'error': error or '',
            }
            row.update(windows)
            for key in fields:
                row.setdefault(key, '')

            with open(run_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            if error:
                print(f'[step_bench] ep={ep} failed: {error.splitlines()[0]}', flush=True)
            else:
                print(f'[step_bench] ep={ep} {ecfg.get("change_kind")} '
                      f'{ecfg.get("direction")} x{ecfg.get("factor")} '
                      f'goodput={_fmt(windows.get("pre_goodput_mbps"))}->'
                      f'{_fmt(windows.get("post_goodput_mbps"))} Mbps  '
                      f'rtt={_fmt(windows.get("pre_rtt_ms"))}->'
                      f'{_fmt(windows.get("post_rtt_ms"))} ms  '
                      f'fair={_fmt(windows.get("pre_fairness"), 3)}->'
                      f'{_fmt(windows.get("post_fairness"), 3)} '
                      f'({completed}/{total})',
                      flush=True)
    except KeyboardInterrupt:
        print('\n[step_bench] interrupted - stopping slots', flush=True)
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

    summary_rows, summary_payload = _summarize_runs(run_csv, summary_csv, summary_json)
    try:
        _plot_step_summary(summary_csv, summary_pdf, alg_name)
    except Exception as e:
        print(f'[step_bench] plot failed: {e}', flush=True)

    print('\n[step_bench] done', flush=True)
    print(f'[step_bench] completed_groups={len(summary_rows)}', flush=True)
    print(f'[step_bench] bandwidth_tracking_score={summary_payload.get("bandwidth_tracking_score")}', flush=True)
    print(f'[step_bench] runs={run_csv}', flush=True)
    print(f'[step_bench] summary={summary_csv}', flush=True)
    print(f'[step_bench] plot={summary_pdf}', flush=True)


if __name__ == '__main__':
    main()
