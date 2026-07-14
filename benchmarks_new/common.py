#!/usr/bin/env python3
"""Shared eval launcher and state-log plotter for benchmarks_new."""

import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
APPROACHES_CONFIG = ROOT / 'benchmarks_new' / 'config.yaml'


def load_approaches(path=APPROACHES_CONFIG):
    """Load the shared model registry using the legacy benchmark schema."""
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    raw = config.get('approaches') or []
    if not isinstance(raw, list):
        raise ValueError(f'{path}: approaches must be a list')
    approaches = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f'{path}: approaches[{index}] must be a mapping')
        name = str(item.get('name') or item.get('data_folder') or '').strip()
        if not name:
            raise ValueError(f'{path}: approaches[{index}] needs name or data_folder')
        if name in approaches:
            raise ValueError(f'{path}: duplicate approach name {name!r}')
        if item.get('kind', 'model') != 'model':
            raise ValueError(f'{path}: {name!r} is not an Olympus model approach')
        if not item.get('checkpoint'):
            raise ValueError(f'{path}: {name!r} is missing checkpoint')
        resolved = copy.deepcopy(item)
        checkpoint = Path(str(resolved['checkpoint'])).expanduser()
        resolved['checkpoint'] = str(
            checkpoint.resolve() if checkpoint.is_absolute() else (ROOT / checkpoint).resolve())
        approaches[name] = resolved
    return approaches


def deep_merge(base, overlay):
    """Recursively merge mappings; lists and scalar values replace wholesale."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return copy.deepcopy(overlay)
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = deep_merge(result[key], value) if key in result else copy.deepcopy(value)
    return result


def load_benchmark(config: Path, debug=False):
    config = Path(config).resolve()
    with config.open() as handle:
        manifest = yaml.safe_load(handle) or {}
    overlay = {}
    if debug:
        debug_path = config.with_name('debug.yaml')
        if not debug_path.exists():
            raise ValueError(f'--debug requested but {debug_path} does not exist')
        with debug_path.open() as handle:
            overlay = yaml.safe_load(handle) or {}
        manifest = deep_merge(manifest, overlay.get('config') or {})
    return manifest, overlay


def load_scenario(path: Path, overlay: dict):
    path = Path(path).resolve()
    with path.open() as handle:
        value = yaml.safe_load(handle) or {}
    return deep_merge(value, overlay.get('scenario') or {})


def _manifest(path):
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _canonical_manifest(config: Path, debug=False):
    """Expand the concise benchmark format into an Olympus eval manifest."""
    source, overlay = load_benchmark(config, debug=debug)
    matrix = dict(source.get('matrix') or {})
    checkpoint_names = list(matrix.get('checkpoints') or [])
    scenario_paths = list(matrix.get('scenarios') or [])
    environment_types = list(matrix.get('environments') or [])
    if not checkpoint_names or not scenario_paths or not environment_types:
        raise ValueError('matrix checkpoints, scenarios, and environments must be non-empty')

    approaches = load_approaches()
    unknown = [name for name in checkpoint_names if name not in approaches]
    if unknown:
        raise ValueError(
            f'unknown benchmark approaches {unknown}; add them to {APPROACHES_CONFIG}')

    scenarios = {}
    scenario_names = []
    temporary_paths = []
    for index, value in enumerate(scenario_paths):
        path = Path(value)
        path = path if path.is_absolute() else (config.parent / path)
        name = path.stem
        if name in scenarios:
            name = f'{name}_{index + 1}'
        resolved_path = path.resolve()
        if debug:
            handle = tempfile.NamedTemporaryFile(
                mode='w', prefix=f'olympus_{name}_debug_', suffix='.yaml', delete=False)
            with handle:
                yaml.safe_dump(load_scenario(resolved_path, overlay), handle, sort_keys=False)
            resolved_path = Path(handle.name)
            temporary_paths.append(resolved_path)
        scenarios[name] = {'path': str(resolved_path)}
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
            name: {
                'path': approaches[name]['checkpoint'],
                'label': approaches[name].get('plot_label') or name,
                'metadata': {
                    key: copy.deepcopy(value)
                    for key, value in approaches[name].items()
                    if key not in {'checkpoint', 'config', 'plot_label'}
                },
            }
            for name in checkpoint_names
        },
        'scenarios': scenarios,
        'environments': environments,
        'matrix': canonical_matrix,
        'runs': source.get('runs') or [],
    }, temporary_paths


def _requires_mininet_privileges(manifest: dict) -> bool:
    """Return whether any selected manifest entry uses a privileged backend."""
    selected = (manifest.get('matrix') or {}).get('environments') or []
    return any(str(environment).lower() != 'raynet' for environment in selected)


def run_eval(config: Path, extra=None, debug=False):
    temporary_paths = []
    try:
        manifest, temporary_paths = _canonical_manifest(config, debug=debug)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f'[benchmarks_new] invalid benchmark config: {exc}', file=sys.stderr)
        return 2

    # Create the shared root before sudo so aggregate plots remain writable by
    # the invoking user. The elevated eval process only creates run children.
    output_root = (manifest.get('defaults') or {}).get('output_root')
    if output_root:
        try:
            output_path = Path(output_root)
            output_path.mkdir(parents=True, exist_ok=True)
            if not os.access(output_path, os.W_OK):
                raise PermissionError(
                    f'{output_path} is not writable; restore it with '
                    f'`sudo chown -R {os.getuid()}:{os.getgid()} {output_path}`')
        except OSError as exc:
            print(f'[benchmarks_new] cannot create output root {output_root}: {exc}',
                  file=sys.stderr)
            for path in temporary_paths:
                path.unlink(missing_ok=True)
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
            for path in temporary_paths:
                path.unlink(missing_ok=True)
            return 2
        print('[benchmarks_new] Mininet selected; launching evaluation with sudo -E')
        command = [sudo, '-E', *command]
    try:
        return subprocess.run(command, cwd=str(ROOT)).returncode
    finally:
        manifest_path.unlink(missing_ok=True)
        for path in temporary_paths:
            path.unlink(missing_ok=True)




def suite_main(suite_file, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(suite_file).with_name('config.yaml')))
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    config = Path(args.config).resolve()
    if not args.plot_only:
        code = run_eval(config, ['--verbose'] if args.verbose else [], debug=args.debug)
        if code:
            return code
    if args.no_plot:
        return 0
    plotter = Path(suite_file).with_name('plot.py')
    return subprocess.run(
        [sys.executable, str(plotter), '--config', str(config)]
        + (['--debug'] if args.debug else []),
        cwd=str(ROOT),
    ).returncode
