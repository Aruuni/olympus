"""Validation and expansion for versioned Olympus evaluation manifests."""

from __future__ import annotations

import copy
import itertools
import os
from pathlib import Path

import yaml


KIND = 'olympus-eval'
LOGGING_PROFILES = {'standard', 'minimal', 'none'}


def is_eval_manifest(value) -> bool:
    return isinstance(value, dict) and value.get('kind') == KIND


def _mapping(value, field):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{field} must be a mapping')
    return value


def _names(value, available, field):
    if value in (None, 'all'):
        return list(available)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f'{field} must be a non-empty list or "all"')
    unknown = [name for name in value if name not in available]
    if unknown:
        raise ValueError(f'{field} references unknown names: {unknown}')
    return value


def _path(base: Path, value, field):
    if not value:
        raise ValueError(f'{field} is required')
    path = Path(os.path.expanduser(str(value)))
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load_manifest(path: str):
    source = Path(path).resolve()
    with source.open() as handle:
        manifest = yaml.safe_load(handle) or {}
    if not is_eval_manifest(manifest):
        raise ValueError(f'{source} is not an {KIND} manifest')
    if int(manifest.get('version', 1)) != 1:
        raise ValueError(f'unsupported eval manifest version: {manifest.get("version")}')
    return expand_manifest(manifest, source.parent), manifest


def expand_manifest(manifest: dict, base_dir: Path):
    defaults = copy.deepcopy(_mapping(manifest.get('defaults'), 'defaults'))
    checkpoints = _mapping(manifest.get('checkpoints'), 'checkpoints')
    scenarios = _mapping(manifest.get('scenarios'), 'scenarios')
    environments = _mapping(manifest.get('environments'), 'environments')
    if not checkpoints or not scenarios or not environments:
        raise ValueError('checkpoints, scenarios, and environments must be non-empty')

    resolved_checkpoints = {}
    for name, raw in checkpoints.items():
        item = {'path': raw} if isinstance(raw, str) else copy.deepcopy(_mapping(raw, f'checkpoints.{name}'))
        item['path'] = _path(base_dir, item.get('path'), f'checkpoints.{name}.path')
        if item.get('model_config'):
            item['model_config'] = _path(base_dir, item['model_config'], f'checkpoints.{name}.model_config')
        item.setdefault('label', name)
        resolved_checkpoints[name] = item

    resolved_scenarios = {}
    for name, raw in scenarios.items():
        item = {'path': raw} if isinstance(raw, str) else copy.deepcopy(_mapping(raw, f'scenarios.{name}'))
        item['path'] = _path(base_dir, item.get('path'), f'scenarios.{name}.path')
        resolved_scenarios[name] = item

    resolved_environments = {}
    for name, raw in environments.items():
        item = {'type': raw} if isinstance(raw, str) else copy.deepcopy(_mapping(raw, f'environments.{name}'))
        if not item.get('type'):
            raise ValueError(f'environments.{name}.type is required')
        item.setdefault('options', {})
        _mapping(item['options'], f'environments.{name}.options')
        resolved_environments[name] = item

    specs = []
    matrix = _mapping(manifest.get('matrix'), 'matrix')
    if matrix:
        cp_names = _names(matrix.get('checkpoints'), resolved_checkpoints, 'matrix.checkpoints')
        sc_names = _names(matrix.get('scenarios'), resolved_scenarios, 'matrix.scenarios')
        env_names = _names(matrix.get('environments'), resolved_environments, 'matrix.environments')
        for cp, scenario, environment in itertools.product(cp_names, sc_names, env_names):
            specs.append({'checkpoint': cp, 'scenario': scenario, 'environment': environment,
                          **copy.deepcopy(_mapping(matrix.get('overrides'), 'matrix.overrides'))})

    runs = manifest.get('runs') or []
    if not isinstance(runs, list):
        raise ValueError('runs must be a list')
    for index, raw in enumerate(runs):
        run = copy.deepcopy(_mapping(raw, f'runs[{index}]'))
        for field, available in (('checkpoint', resolved_checkpoints),
                                 ('scenario', resolved_scenarios),
                                 ('environment', resolved_environments)):
            if run.get(field) not in available:
                raise ValueError(f'runs[{index}].{field} references unknown name {run.get(field)!r}')
        specs.append(run)
    if not specs:
        raise ValueError('manifest must define matrix and/or runs')

    expanded = []
    used_names = set()
    for index, spec in enumerate(specs):
        run = copy.deepcopy(defaults)
        run.update(spec)
        profile = str(run.get('logging', 'standard')).lower()
        if profile not in LOGGING_PROFILES:
            raise ValueError(f'run {index}: logging must be one of {sorted(LOGGING_PROFILES)}')
        run['logging'] = profile
        run['n_parallel'] = max(1, int(run.get('n_parallel', 1)))
        run['repetitions'] = max(1, int(run.get('repetitions', 1)))
        run['seed'] = int(run.get('seed', 0))
        run['checkpoint_spec'] = copy.deepcopy(resolved_checkpoints[run['checkpoint']])
        run['scenario_spec'] = copy.deepcopy(resolved_scenarios[run['scenario']])
        run['environment_spec'] = copy.deepcopy(resolved_environments[run['environment']])
        run.setdefault('name', f'{run["checkpoint"]}__{run["scenario"]}__{run["environment"]}')
        if run['name'] in used_names:
            raise ValueError(f'duplicate eval run name: {run["name"]!r}')
        used_names.add(run['name'])
        run['matrix_index'] = index
        expanded.append(run)
    return expanded


def scenario_episode_count(path: str) -> int:
    """Return the number of episode points in a backend-neutral scenario."""
    with open(path) as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f'scenario must be a mapping: {path}')
    if 'experiments' in cfg:
        return max(1, len(cfg.get('experiments') or []))
    sweep = cfg.get('sweep')
    if not isinstance(sweep, dict):
        raise ValueError(f'scenario must define sweep or experiments: {path}')
    sizes = []
    for plural, singular in (('bws', 'bw'), ('delays', 'delay'),
                             ('flows', 'flow'), ('link_schedules', 'link_schedule')):
        value = sweep.get(plural, sweep.get(singular))
        sizes.append(len(value) if isinstance(value, list) else 1)
    total = 1
    for size in sizes:
        total *= max(1, size)
    return total
