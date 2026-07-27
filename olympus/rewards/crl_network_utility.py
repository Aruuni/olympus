"""Backward-compatible alias for the reward now named :mod:`proteus`."""

from olympus.rewards.proteus import (  # noqa: F401
    DEFAULTS,
    RewardCalc,
    make_reward_calc,
    reward_options,
)


__all__ = [
    'DEFAULTS',
    'RewardCalc',
    'make_reward_calc',
    'reward_options',
]
