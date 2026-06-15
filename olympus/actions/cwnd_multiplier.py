"""Default continuous CWND multiplier.

A normalized action in ``[-1, 1]`` is mapped exponentially onto
``[multiplier_min, multiplier_max]``. With the default options this is the
existing Olympus mapping: ``-1 -> 0.5``, ``0 -> 1.0``, ``1 -> 2.0``.
"""

import math

import numpy as np

from olympus.common.action_plugins import (
    current_action_name,
    current_action_options,
)


ACTION_NAME = 'cwnd_multiplier'
ACTION_DIM = 1
ACTION_VERSION = 'cwnd_multiplier_v1'

_DEFAULTS = {
    'action_min': -1.0,
    'action_max': 1.0,
    'multiplier_min': 0.5,
    'multiplier_max': 2.0,
    'min_cwnd_change': 1,
}


def validate_options(value: dict) -> dict:
    value = dict(value or {})
    unknown = sorted(set(value) - set(_DEFAULTS))
    if unknown:
        raise ValueError(
            f'unknown {ACTION_NAME} action options: {", ".join(unknown)}')
    out = dict(_DEFAULTS)
    out.update(value)
    out['action_min'] = float(out['action_min'])
    out['action_max'] = float(out['action_max'])
    out['multiplier_min'] = float(out['multiplier_min'])
    out['multiplier_max'] = float(out['multiplier_max'])
    out['min_cwnd_change'] = int(out['min_cwnd_change'])
    if (out['action_min'], out['action_max']) != (-1.0, 1.0):
        raise ValueError(
            'cwnd_multiplier currently requires action_min=-1 and action_max=1')
    if out['multiplier_min'] <= 0.0:
        raise ValueError('multiplier_min must be positive')
    if out['multiplier_max'] <= out['multiplier_min']:
        raise ValueError('multiplier_max must be greater than multiplier_min')
    if out['min_cwnd_change'] < 0:
        raise ValueError('min_cwnd_change must be non-negative')
    return out


_OPTIONS = validate_options(
    current_action_options()
    if current_action_name() == ACTION_NAME else {}
)
ACTION_MIN = _OPTIONS['action_min']
ACTION_MAX = _OPTIONS['action_max']
MULTIPLIER_MIN = _OPTIONS['multiplier_min']
MULTIPLIER_MAX = _OPTIONS['multiplier_max']
_LOG_MIN = math.log(MULTIPLIER_MIN)
_LOG_MAX = math.log(MULTIPLIER_MAX)
_LOG_MID = 0.5 * (_LOG_MIN + _LOG_MAX)
_LOG_HALF = 0.5 * (_LOG_MAX - _LOG_MIN)


def options() -> dict:
    return dict(_OPTIONS)


def is_legacy_default() -> bool:
    return _OPTIONS == _DEFAULTS


def to_multiplier(action):
    try:
        import torch
        if isinstance(action, torch.Tensor):
            return torch.exp(_LOG_MID + _LOG_HALF * action.clamp(
                ACTION_MIN, ACTION_MAX))
    except ImportError:
        pass
    if isinstance(action, np.ndarray):
        return np.exp(
            _LOG_MID + _LOG_HALF * np.clip(action, ACTION_MIN, ACTION_MAX))
    bounded = min(max(float(action), ACTION_MIN), ACTION_MAX)
    return math.exp(_LOG_MID + _LOG_HALF * bounded)


def from_multiplier(multiplier):
    try:
        import torch
        if isinstance(multiplier, torch.Tensor):
            value = multiplier.clamp(MULTIPLIER_MIN, MULTIPLIER_MAX).log()
            return ((value - _LOG_MID) / _LOG_HALF).clamp(
                ACTION_MIN, ACTION_MAX)
    except ImportError:
        pass
    if isinstance(multiplier, np.ndarray):
        value = np.log(np.clip(
            multiplier, MULTIPLIER_MIN, MULTIPLIER_MAX))
        return np.clip(
            (value - _LOG_MID) / _LOG_HALF, ACTION_MIN, ACTION_MAX)
    value = math.log(min(max(
        float(multiplier), MULTIPLIER_MIN), MULTIPLIER_MAX))
    return min(max((value - _LOG_MID) / _LOG_HALF, ACTION_MIN), ACTION_MAX)


def apply_cwnd(current_cwnd: int, action, cwnd_min: int,
               cwnd_max: int) -> int:
    current = max(int(current_cwnd), 1)
    multiplier = float(to_multiplier(action))
    desired = current * multiplier
    rounded = int(round(desired))
    min_change = int(_OPTIONS['min_cwnd_change'])
    if min_change > 0:
        if desired > current:
            rounded = max(rounded, current + min_change)
        elif desired < current:
            rounded = min(rounded, current - min_change)
    return int(np.clip(rounded, int(cwnd_min), int(cwnd_max)))
