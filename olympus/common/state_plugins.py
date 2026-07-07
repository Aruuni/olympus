"""State-representation plugin helpers.

Each algorithm can override states in:
  olympus.algorithms.<algorithm>.states.<state_name>

Otherwise the loader falls back to shared states in:
  olympus.states.<state_name>

The selected state is read at model import time from SAO_STATE. This keeps
state representations hot-swappable within an algorithm while avoiding an
accidental cross-algorithm shared state contract.
"""

import importlib
import importlib.util
import os

from olympus.common.action_plugins import (
    assert_action_compatible,
    current_action_meta,
)


_STATE_ALIASES = {
    'default': 'default_orca',
    'no_bdp': 'kalman',
    'shared_no_bdp': 'kalman',
    'clean_slate_astraea': 'astraea_deepcc',
}

_FEATURE_VERSION_ALIASES = {
    'shared_no_bdp_observation_v1_srtt_unshifted':
        'kalman_observation_v1_srtt_unshifted',
    'shared_no_bdp_no_kalman_v1': 'no_kalman_observation_v1',
    'orca_observation_v1_no_bdp': 'orca_observation_v1',
}


def normalize_state_name(name: str, default='default_orca') -> str:
    name = (name or default).strip() or default
    return _STATE_ALIASES.get(name, name)


def normalize_feature_version(version: str) -> str:
    version = str(version)
    return _FEATURE_VERSION_ALIASES.get(version, version)


def current_state_name(default='default_orca') -> str:
    name = os.environ.get('SAO_STATE') or os.environ.get('OC_STATE') or default
    return normalize_state_name(name, default)


def _import_if_exists(module_name: str):
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None
    if spec is None:
        return None
    return importlib.import_module(module_name)


def load_state_module(algorithm: str, state_name: str = None):
    state_name = normalize_state_name(state_name) if state_name else current_state_name()
    if '.' in state_name:
        return importlib.import_module(state_name)

    module_names = [
        f'olympus.algorithms.{algorithm}.states.{state_name}',
        f'olympus.states.{state_name}',
    ]
    for module_name in module_names:
        module = _import_if_exists(module_name)
        if module is not None:
            return module

    raise ModuleNotFoundError(
        f'unknown state representation {state_name!r} for algorithm '
        f'{algorithm!r}; tried {", ".join(module_names)}')


def state_meta(module, state_name=None) -> dict:
    return {
        'state_name': state_name or current_state_name(),
        'state_dim': int(module.STATE_DIM),
        'state_feature_version': str(module.STATE_FEATURE_VERSION),
        'state_features': list(getattr(module, 'STATE_FEATURES', [])),
        'action_meta': current_action_meta(),
    }


def checkpoint_state_meta(ckpt: dict) -> dict:
    ckpt = ckpt or {}
    meta = dict(ckpt.get('state_meta') or {})
    model_meta = dict(ckpt.get('model_meta') or {})
    if not meta and model_meta:
        if 'state_name' in model_meta:
            meta['state_name'] = model_meta['state_name']
        if 'state_dim' in model_meta:
            meta['state_dim'] = model_meta['state_dim']
        if 'state_feature_version' in model_meta:
            meta['state_feature_version'] = model_meta['state_feature_version']
        elif 'state_features' in model_meta:
            # Legacy option-critic checkpoints used state_features for the
            # version string.
            meta['state_feature_version'] = model_meta['state_features']
    return meta


def assert_state_compatible(current: dict, ckpt: dict, source='checkpoint') -> None:
    action = dict(current.get('action_meta') or current_action_meta())
    assert_action_compatible(action, ckpt, source=source)
    meta = checkpoint_state_meta(ckpt)
    if not meta:
        current_state = str(current.get('state_name', 'default_orca'))
        current_state = normalize_state_name(current_state.rsplit('.', 1)[-1])
        if current_state != 'kalman':
            raise ValueError(
                f'{source} has no state metadata, but configured state is '
                f'{current.get("state_name")!r}; refusing to guess compatibility')
        return

    mismatches = []
    if 'state_dim' in meta and int(meta['state_dim']) != int(current['state_dim']):
        mismatches.append(f'state_dim={meta["state_dim"]} vs {current["state_dim"]}')
    if ('state_feature_version' in meta and
            normalize_feature_version(meta['state_feature_version']) !=
            normalize_feature_version(current['state_feature_version'])):
        mismatches.append(
            'state_feature_version='
            f'{meta["state_feature_version"]} vs {current["state_feature_version"]}')
    if mismatches:
        raise ValueError(f'{source} state mismatch: ' + '; '.join(mismatches))
