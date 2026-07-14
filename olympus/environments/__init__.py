"""Olympus network-emulation environments.

Backends live under ``olympus/environments/<type>/env.py`` and each exposes an
``ENV_CLASS`` implementing :class:`NetworkEnv`. ``make_env`` resolves a backend
by ``type`` and instantiates it; the default is the Mininet backend, so callers
that pass no type keep the existing behavior.
"""

import importlib

from olympus.environments.base import NetworkEnv

_DEFAULT_ENV_TYPE = 'mininet'

__all__ = ['NetworkEnv', 'make_env', 'validate_scenario']


def _backend_module(env_type):
    env_type = (str(env_type).strip() if env_type else '') or _DEFAULT_ENV_TYPE
    try:
        return importlib.import_module(f'olympus.environments.{env_type}.env')
    except ImportError as exc:
        raise ValueError(
            f'unknown environment backend {env_type!r}: '
            f'expected olympus/environments/{env_type}/env.py') from exc


def validate_scenario(env_type, scenario):
    """Ask a backend to preflight a backend-neutral scenario, when supported."""
    if not isinstance(scenario, dict):
        raise ValueError('environment scenario must be a mapping')
    if not isinstance(scenario.get('sweep'), dict) and not isinstance(
            scenario.get('experiments'), list):
        raise ValueError('environment scenario must define sweep or experiments')
    module = _backend_module(env_type)
    validator = getattr(module, 'validate_scenario', None)
    if validator is not None:
        validator(scenario)


def make_env(env_type=None, **kwargs):
    """Instantiate the network backend selected by ``env_type``.

    Imports ``olympus.environments.<env_type>.env`` and constructs its
    ``ENV_CLASS`` with ``kwargs``. An empty/None ``env_type`` falls back to the
    Mininet backend so existing single-backend callers are unaffected.
    """
    env_type = (str(env_type).strip() if env_type else '') or _DEFAULT_ENV_TYPE
    module = _backend_module(env_type)
    try:
        env_cls = module.ENV_CLASS
    except AttributeError as exc:
        raise ValueError(
            f'environment backend {env_type!r} ({module.__name__}) does not '
            f'expose ENV_CLASS') from exc
    return env_cls(**kwargs)
