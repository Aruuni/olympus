"""Raw RayNet exponent action for CleanSlate.

CleanSlate's simulator action is not an absolute congestion window. It is an
exponent ``x`` applied as ``cwnd *= 2 ** x`` inside RayNet, so workers must pass
the policy scalar through instead of converting it to a target cwnd.
"""

import numpy as np

from olympus.common.action_plugins import (
    current_action_name,
    current_action_options,
)


ACTION_NAME = 'raynet_exponent'
ACTION_DIM = 1
ACTION_VERSION = 'raynet_exponent_v1'
ACTION_OUTPUT = 'raynet_action'

_DEFAULTS = {
    'action_min': -1.0,
    'action_max': 1.0,
    'scale': 2.0,
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
    out['scale'] = float(out['scale'])
    if out['action_max'] <= out['action_min']:
        raise ValueError('action_max must be greater than action_min')
    if out['scale'] <= 0.0:
        raise ValueError('scale must be positive')
    return out


_OPTIONS = validate_options(
    current_action_options()
    if current_action_name() == ACTION_NAME else {}
)
ACTION_MIN = _OPTIONS['action_min']
ACTION_MAX = _OPTIONS['action_max']
SCALE = _OPTIONS['scale']


def options() -> dict:
    return dict(_OPTIONS)


def to_multiplier(action):
    """Return the simulator-visible cwnd multiplier for logging/model helpers."""
    try:
        import torch
        if isinstance(action, torch.Tensor):
            raw = (SCALE * action.clamp(ACTION_MIN, ACTION_MAX))
            return torch.pow(torch.as_tensor(2.0, device=action.device), raw)
    except ImportError:
        pass
    if isinstance(action, np.ndarray):
        return np.power(2.0, SCALE * np.clip(action, ACTION_MIN, ACTION_MAX))
    bounded = min(max(float(action), ACTION_MIN), ACTION_MAX)
    return 2.0 ** (SCALE * bounded)


def from_multiplier(multiplier):
    try:
        import torch
        if isinstance(multiplier, torch.Tensor):
            value = torch.log2(multiplier.clamp(min=1e-12)) / SCALE
            return value.clamp(ACTION_MIN, ACTION_MAX)
    except ImportError:
        pass
    if isinstance(multiplier, np.ndarray):
        value = np.log2(np.clip(multiplier, 1e-12, None)) / SCALE
        return np.clip(value, ACTION_MIN, ACTION_MAX)
    value = np.log2(max(float(multiplier), 1e-12)) / SCALE
    return min(max(value, ACTION_MIN), ACTION_MAX)


def to_raynet_action(action):
    try:
        import torch
        if isinstance(action, torch.Tensor):
            return SCALE * action.clamp(ACTION_MIN, ACTION_MAX)
    except ImportError:
        pass
    if isinstance(action, np.ndarray):
        return SCALE * np.clip(action, ACTION_MIN, ACTION_MAX)
    bounded = min(max(float(action), ACTION_MIN), ACTION_MAX)
    return SCALE * bounded


def apply_cwnd(current_cwnd: int, action, cwnd_min: int,
               cwnd_max: int):
    return to_raynet_action(action)
