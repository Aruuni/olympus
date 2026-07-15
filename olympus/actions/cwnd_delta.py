"""Additive CWND step action.

Instead of scaling the window (``cwnd_multiplier``), this adds a flat number of
packets: a normalized action in ``[-1, 1]`` maps to ``delta = step * action``,
so with the default ``step=20`` a full-positive action is ``+20`` and
full-negative is ``-20``. Unlike the multiplier action the step is independent
of the current window, so ``+20`` is a big move at small cwnd and a small nudge
at large cwnd.

Modes (via YAML ``actions.cwnd_delta`` options):
  * continuous (default): ``delta = round(step * action)`` — smooth, keeps a
    gradient for continuous actors (TD3 / dreamer / PPO).
  * ``discrete: true``: bang-bang — ``+step`` / ``-step`` by the sign of the
    action, holding when ``|action| <= deadzone``.

``to_multiplier`` / ``from_multiplier`` are a linear, invertible proxy used only
for logging (the ``cwnd_mult`` column) and previous-action featurisation; the
real window change goes through ``apply_cwnd``.
"""

import numpy as np

from olympus.common.action_plugins import (
    current_action_name,
    current_action_options,
)


ACTION_NAME = 'cwnd_delta'
ACTION_DIM = 1
ACTION_VERSION = 'cwnd_delta_v1'

_DEFAULTS = {
    'action_min': -1.0,
    'action_max': 1.0,
    'step': 20,            # cwnd packets moved at |action| = 1
    'discrete': False,     # bang-bang +/- step instead of a proportional delta
    'deadzone': 0.0,       # discrete only: |action| <= deadzone holds the window
    # Logging/normalisation proxy range (does not affect the applied window).
    'multiplier_min': 0.5,
    'multiplier_max': 1.5,
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
    out['step'] = int(out['step'])
    out['discrete'] = bool(out['discrete'])
    out['deadzone'] = float(out['deadzone'])
    out['multiplier_min'] = float(out['multiplier_min'])
    out['multiplier_max'] = float(out['multiplier_max'])
    if (out['action_min'], out['action_max']) != (-1.0, 1.0):
        raise ValueError(
            'cwnd_delta currently requires action_min=-1 and action_max=1')
    if out['step'] <= 0:
        raise ValueError('step must be a positive integer')
    if not 0.0 <= out['deadzone'] < 1.0:
        raise ValueError('deadzone must be in [0, 1)')
    if out['multiplier_min'] <= 0.0 or out['multiplier_max'] <= out['multiplier_min']:
        raise ValueError('need 0 < multiplier_min < multiplier_max')
    return out


_OPTIONS = validate_options(
    current_action_options()
    if current_action_name() == ACTION_NAME else {}
)
ACTION_MIN = _OPTIONS['action_min']
ACTION_MAX = _OPTIONS['action_max']
STEP = _OPTIONS['step']
DISCRETE = _OPTIONS['discrete']
DEADZONE = _OPTIONS['deadzone']
MULTIPLIER_MIN = _OPTIONS['multiplier_min']
MULTIPLIER_MAX = _OPTIONS['multiplier_max']
_MID = 0.5 * (MULTIPLIER_MAX + MULTIPLIER_MIN)
_HALF = 0.5 * (MULTIPLIER_MAX - MULTIPLIER_MIN)


def options() -> dict:
    return dict(_OPTIONS)


def to_multiplier(action):
    """Linear proxy multiplier for logging/featurisation (not the applied move)."""
    try:
        import torch
        if isinstance(action, torch.Tensor):
            return _MID + _HALF * action.clamp(ACTION_MIN, ACTION_MAX)
    except ImportError:
        pass
    if isinstance(action, np.ndarray):
        return _MID + _HALF * np.clip(action, ACTION_MIN, ACTION_MAX)
    bounded = min(max(float(action), ACTION_MIN), ACTION_MAX)
    return _MID + _HALF * bounded


def from_multiplier(multiplier):
    try:
        import torch
        if isinstance(multiplier, torch.Tensor):
            value = multiplier.clamp(MULTIPLIER_MIN, MULTIPLIER_MAX)
            return ((value - _MID) / _HALF).clamp(ACTION_MIN, ACTION_MAX)
    except ImportError:
        pass
    if isinstance(multiplier, np.ndarray):
        value = np.clip(multiplier, MULTIPLIER_MIN, MULTIPLIER_MAX)
        return np.clip((value - _MID) / _HALF, ACTION_MIN, ACTION_MAX)
    value = min(max(float(multiplier), MULTIPLIER_MIN), MULTIPLIER_MAX)
    return min(max((value - _MID) / _HALF, ACTION_MIN), ACTION_MAX)


def apply_cwnd(current_cwnd: int, action, cwnd_min: int,
               cwnd_max: int) -> int:
    current = max(int(current_cwnd), 1)
    a = min(max(float(np.asarray(action).reshape(-1)[0]), ACTION_MIN), ACTION_MAX)
    if DISCRETE:
        if abs(a) <= DEADZONE:
            delta = 0
        else:
            delta = STEP if a > 0 else -STEP
    else:
        delta = int(round(STEP * a))
    return int(np.clip(current + delta, int(cwnd_min), int(cwnd_max)))
