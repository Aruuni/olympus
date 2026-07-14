#!/usr/bin/env python3
"""Shared eval launcher and state-log plotter for benchmarks_new."""

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Checkpoints available to every benchmark. Benchmark YAMLs only select these
# names; inference configuration is discovered beside the checkpoint by eval.py.
CHECKPOINTS = {
    'test-protocol': ROOT / 'olympus/models/test-model/dreamer_v3_20260714-134022/checkpoints/dreamer_v3_cwnd_model.pt',
}


def _manifest(path):
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _canonical_manifest(config: Path) -> dict:
    """Expand the concise benchmark format into an Olympus eval manifest."""
    source = _manifest(config)
    matrix = dict(source.get('matrix') or {})
    checkpoint_names = list(matrix.get('checkpoints') or [])
    scenario_paths = list(matrix.get('scenarios') or [])
    environment_types = list(matrix.get('environments') or [])
    if not checkpoint_names or not scenario_paths or not environment_types:
        raise ValueError('matrix checkpoints, scenarios, and environments must be non-empty')

    unknown = [name for name in checkpoint_names if name not in CHECKPOINTS]
    if unknown:
        raise ValueError(
            f'unknown benchmark checkpoints {unknown}; add them to benchmarks_new.common.CHECKPOINTS')

    scenarios = {}
    scenario_names = []
    for index, value in enumerate(scenario_paths):
        path = Path(value)
        path = path if path.is_absolute() else (config.parent / path)
        name = path.stem
        if name in scenarios:
            name = f'{name}_{index + 1}'
        scenarios[name] = {'path': str(path.resolve())}
        scenario_names.append(name)

    environments = {}
    environment_names = []
    for environment_type in environment_types:
        name = str(environment_type)
        if name in environments:
            raise ValueError(f'duplicate environment type in matrix: {name}')
        environments[name] = {'type': name}
        environment_names.append(name)

    defaults = dict(source.get('defaults') or {})
    output_root = defaults.get('output_root')
    if output_root:
        output_path = Path(output_root)
        if not output_path.is_absolute():
            defaults['output_root'] = str((config.parent / output_path).resolve())

    canonical_matrix = dict(matrix)
    canonical_matrix.update({
        'checkpoints': checkpoint_names,
        'scenarios': scenario_names,
        'environments': environment_names,
    })
    return {
        'kind': 'olympus-eval',
        'version': 1,
        'defaults': defaults,
        'checkpoints': {
            name: {'path': str(Path(CHECKPOINTS[name]).resolve())}
            for name in checkpoint_names
        },
        'scenarios': scenarios,
        'environments': environments,
        'matrix': canonical_matrix,
        'runs': source.get('runs') or [],
    }


def _requires_mininet_privileges(manifest: dict) -> bool:
    """Return whether any selected manifest entry uses a privileged backend."""
    selected = (manifest.get('matrix') or {}).get('environments') or []
    return any(str(environment).lower() != 'raynet' for environment in selected)


def run_eval(config: Path, extra=None):
    try:
        manifest = _canonical_manifest(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f'[benchmarks_new] invalid benchmark config: {exc}', file=sys.stderr)
        return 2

    handle = tempfile.NamedTemporaryFile(
        mode='w', prefix='olympus_benchmark_', suffix='.yaml', delete=False)
    try:
        with handle:
            yaml.safe_dump(manifest, handle, sort_keys=False)
        manifest_path = Path(handle.name)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise

    command = [sys.executable, str(ROOT / 'olympus' / 'eval.py'),
               '--config', str(manifest_path)] + list(extra or [])
    if (hasattr(os, 'geteuid') and os.geteuid() != 0
            and _requires_mininet_privileges(manifest)):
        sudo = shutil.which('sudo')
        if not sudo:
            print('[benchmarks_new] Mininet evaluation requires sudo -E, but sudo was not found',
                  file=sys.stderr)
            manifest_path.unlink(missing_ok=True)
            return 2
        print('[benchmarks_new] Mininet selected; launching evaluation with sudo -E')
        command = [sudo, '-E', *command]
    try:
        return subprocess.run(command, cwd=str(ROOT)).returncode
    finally:
        manifest_path.unlink(missing_ok=True)


def _float(row, key):
    try:
        return float(row.get(key, ''))
    except (TypeError, ValueError):
        return np.nan


def _state_rows(output_root: Path):
    records = []
    for path in output_root.glob('*/episodes/*_state_ep*.csv'):
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        throughput = np.asarray([_float(r, 'avg_thr_mbps') for r in rows])
        rtt = np.asarray([_float(r, 'avg_urtt_ms') for r in rows])
        reward = np.asarray([_float(r, 'reward') for r in rows])
        records.append({
            'run': path.parents[1].name,
            'flow': path.stem,
            'throughput': float(np.nanmean(throughput)),
            'rtt': float(np.nanmean(rtt)),
            'return': float(np.nansum(reward)),
        })
    return records


def _returns_rows(output_root: Path):
    rows = []
    for path in output_root.glob('*/episodes/episode_returns.csv'):
        with path.open(newline='') as handle:
            rows.extend(dict(row, run_dir=path.parents[1].name)
                        for row in csv.DictReader(handle))
    return rows


def plot_results(config: Path, output=None):
    manifest = _manifest(config)
    root_value = (manifest.get('defaults') or {}).get('output_root', 'data')
    output_root = Path(root_value)
    if not output_root.is_absolute():
        output_root = (config.parent / output_root).resolve()
    output = Path(output) if output else output_root / 'benchmark_summary.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    states = _state_rows(output_root)
    returns = _returns_rows(output_root)
    if not states and not returns:
        print(f'[benchmarks_new] no completed results beneath {output_root}')
        return 1

    groups = sorted({r['run'] for r in states} | {r['run_dir'] for r in returns})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in groups:
        data = [r for r in states if r['run'] == name]
        if data:
            axes[0, 0].scatter([name] * len(data), [r['throughput'] for r in data], s=15)
            axes[0, 1].scatter([name] * len(data), [r['rtt'] for r in data], s=15)
            # Jain fairness across simultaneously exported flow traces.
            vals = np.asarray([r['throughput'] for r in data], dtype=float)
            finite = vals[np.isfinite(vals) & (vals >= 0)]
            if finite.size:
                jain = float(finite.sum() ** 2 /
                             max(finite.size * np.square(finite).sum(), 1e-9))
                axes[1, 0].scatter([name], [jain], s=28)
        ret = [_float(r, 'return') for r in returns if r['run_dir'] == name]
        if ret:
            axes[1, 1].scatter([name] * len(ret), ret, s=15)
    axes[0, 0].set_title('Mean per-flow throughput'); axes[0, 0].set_ylabel('Mbps')
    axes[0, 1].set_title('Mean per-flow RTT'); axes[0, 1].set_ylabel('ms')
    axes[1, 0].set_title('Jain fairness across exported flows'); axes[1, 0].set_ylim(0, 1.02)
    axes[1, 1].set_title('Episode return')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.tick_params(axis='x', rotation=25)
    fig.suptitle(config.parent.name.replace('_', ' ').title())
    fig.savefig(output)
    plt.close(fig)
    print(f'[benchmarks_new] wrote {output}')
    return 0


def suite_main(suite_file, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(suite_file).with_name('config.yaml')))
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)
    config = Path(args.config).resolve()
    if not args.plot_only:
        code = run_eval(config, ['--verbose'] if args.verbose else [])
        if code:
            return code
    return 0 if args.no_plot else plot_results(config)


def plot_main(suite_file, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(suite_file).with_name('config.yaml')))
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    return plot_results(Path(args.config).resolve(), args.output)
