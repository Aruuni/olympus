"""Helpers for reusing the config saved beside a model checkpoint."""

import copy
import os

import yaml


def _candidate_config_paths(checkpoint_path: str):
    if not checkpoint_path:
        return []

    path = os.path.abspath(checkpoint_path)
    start = path if os.path.isdir(path) else os.path.dirname(path)
    out = []
    seen = set()
    cur = start
    for _ in range(8):
        for candidate in (
            os.path.join(cur, 'telemetry', 'config.resolved.yaml'),
            os.path.join(cur, 'config.resolved.yaml'),
        ):
            if candidate not in seen:
                out.append(candidate)
                seen.add(candidate)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


def find_checkpoint_config(checkpoint_path: str) -> str:
    """Return the saved config path associated with a checkpoint, if any."""
    for candidate in _candidate_config_paths(checkpoint_path):
        if os.path.exists(candidate):
            return candidate
    return ''


def load_checkpoint_config(checkpoint_path: str):
    """Return ``(config, path)`` for the checkpoint's saved run config."""
    config_path = find_checkpoint_config(checkpoint_path)
    if not config_path:
        return None, ''
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return None, ''
    return cfg, config_path


def apply_model_config_from_checkpoint(cfg: dict, checkpoint_path: str) -> str:
    """
    Apply model-facing settings from a checkpoint's saved run config.

    This intentionally avoids copying environment, outputs, paths, parallelism, and
    other run-control keys from the old training folder. Callers can still choose
    where/how to benchmark the model while using the architecture, reward, and
    state settings that the checkpoint was trained with.
    """
    saved, config_path = load_checkpoint_config(checkpoint_path)
    if not saved:
        return ''

    runtime = cfg.setdefault('runtime', {})
    saved_runtime = saved.get('runtime') or {}
    for key in ('algorithm', 'reward', 'state'):
        if key in saved_runtime:
            runtime[key] = copy.deepcopy(saved_runtime[key])

    alg_name = runtime.get('algorithm')
    saved_algorithms = saved.get('algorithms') or {}
    if alg_name and alg_name in saved_algorithms:
        cfg.setdefault('algorithms', {})[alg_name] = copy.deepcopy(
            saved_algorithms[alg_name])
    elif alg_name:
        block = {}
        if isinstance(saved.get('training'), dict):
            block['training'] = copy.deepcopy(saved['training'])
        if isinstance(saved.get('agent'), dict):
            block['agent'] = copy.deepcopy(saved['agent'])
        if block:
            cfg.setdefault('algorithms', {})[alg_name] = block

    for key in ('state_options', 'reward', 'reward_options'):
        if isinstance(saved.get(key), dict):
            cfg[key] = copy.deepcopy(saved[key])

    return config_path
