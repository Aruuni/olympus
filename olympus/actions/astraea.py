"""Original Astraea-style small relative CWND action.

The normalized action remains in ``[-1, 1]``. Positive actions increase CWND
by ``step * action`` and negative actions use the reciprocal decrease from the
original Astraea implementation. Directional ceil/floor guarantees a non-zero
integer CWND movement for any non-zero action.
"""

import math

import numpy as np

from olympus.common.action_plugins import (
    current_action_name,
    current_action_options,
)


ACTION_NAME = 'astraea'
ACTION_DIM = 1
ACTION_VERSION = 'astraea_relative_cwnd_v1'

_DEFAULTS = {
    'action_min': -1.0,
    'action_max': 1.0,
    'step': 0.025,
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
    out['step'] = float(out['step'])
    if (out['action_min'], out['action_max']) != (-1.0, 1.0):
        raise ValueError(
            'astraea currently requires action_min=-1 and action_max=1')
    if not 0.0 < out['step'] < 1.0:
        raise ValueError('astraea step must be between 0 and 1')
    return out


_OPTIONS = validate_options(
    current_action_options()
    if current_action_name() == ACTION_NAME else {}
)
ACTION_MIN = _OPTIONS['action_min']
ACTION_MAX = _OPTIONS['action_max']
STEP = _OPTIONS['step']
MULTIPLIER_MIN = 1.0 / (1.0 + STEP)
MULTIPLIER_MAX = 1.0 + STEP


def options() -> dict:
    return dict(_OPTIONS)


def is_legacy_default() -> bool:
    return False


def to_multiplier(action):
    try:
        import torch
        if isinstance(action, torch.Tensor):
            bounded = action.clamp(ACTION_MIN, ACTION_MAX)
            return torch.where(
                bounded >= 0.0,
                1.0 + STEP * bounded,
                1.0 / (1.0 - STEP * bounded),
            )
    except ImportError:
        pass
    if isinstance(action, np.ndarray):
        bounded = np.clip(action, ACTION_MIN, ACTION_MAX)
        return np.where(
            bounded >= 0.0,
            1.0 + STEP * bounded,
            1.0 / (1.0 - STEP * bounded),
        )
    bounded = min(max(float(action), ACTION_MIN), ACTION_MAX)
    if bounded >= 0.0:
        return 1.0 + STEP * bounded
    return 1.0 / (1.0 - STEP * bounded)


def from_multiplier(multiplier):
    try:
        import torch
        if isinstance(multiplier, torch.Tensor):
            bounded = multiplier.clamp(MULTIPLIER_MIN, MULTIPLIER_MAX)
            action = torch.where(
                bounded >= 1.0,
                (bounded - 1.0) / STEP,
                (1.0 - bounded.reciprocal()) / STEP,
            )
            return action.clamp(ACTION_MIN, ACTION_MAX)
    except ImportError:
        pass
    if isinstance(multiplier, np.ndarray):
        bounded = np.clip(multiplier, MULTIPLIER_MIN, MULTIPLIER_MAX)
        action = np.where(
            bounded >= 1.0,
            (bounded - 1.0) / STEP,
            (1.0 - 1.0 / bounded) / STEP,
        )
        return np.clip(action, ACTION_MIN, ACTION_MAX)
    bounded = min(max(
        float(multiplier), MULTIPLIER_MIN), MULTIPLIER_MAX)
    if bounded >= 1.0:
        action = (bounded - 1.0) / STEP
    else:
        action = (1.0 - 1.0 / bounded) / STEP
    return min(max(action, ACTION_MIN), ACTION_MAX)


def apply_cwnd(current_cwnd: int, action, cwnd_min: int,
               cwnd_max: int) -> int:
    current = max(int(current_cwnd), 1)
    bounded = min(max(float(action), ACTION_MIN), ACTION_MAX)
    desired = current * float(to_multiplier(bounded))
    if bounded > 0.0:
        rounded = math.ceil(desired)
    elif bounded < 0.0:
        rounded = math.floor(desired)
    else:
        rounded = current
    return int(np.clip(rounded, int(cwnd_min), int(cwnd_max)))
