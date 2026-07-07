"""Load resolved Olympus runtime config inside workers.

Workers still receive dynamic episode context through environment variables:
flow ids, fds, checkpoint paths, link state, and trace paths. Stable run
configuration should come from the same resolved YAML the learner reads,
pointed to by ``SAO_CONFIG`` (or ``OC_CONFIG`` for legacy option-critic names).
"""

import os
from functools import lru_cache

import yaml


_MISSING = object()


def load_config(path=None):
    path = path or os.environ.get('SAO_CONFIG') or os.environ.get('OC_CONFIG')
    if not path:
        return {}
    return _load_config_path(os.path.abspath(path))


@lru_cache(maxsize=8)
def _load_config_path(path):
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def get(cfg, *path, default=None):
    cur = cfg or {}
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def value(cfg, *path, env=None, default=None):
    found = get(cfg, *path, default=_MISSING)
    if found is not _MISSING and found is not None:
        return found
    if env:
        raw = os.environ.get(env)
        if raw not in (None, ''):
            return raw
    return default


def agent_value(cfg, key, env=None, default=None):
    return value(cfg, 'agent', key, env=env, default=default)


def training_value(cfg, key, env=None, default=None):
    return value(cfg, 'training', key, env=env, default=default)


def runtime_value(cfg, key, env=None, default=None):
    return value(cfg, 'runtime', key, env=env, default=default)


def reward_value(cfg, key, env=None, default=None):
    return value(cfg, 'reward', key, env=env, default=default)


def state_option_value(cfg, key, env=None, default=None):
    return value(cfg, 'state_options', key, env=env, default=default)


def bool_value(raw, default=False):
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return default
