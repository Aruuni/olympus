"""Proteus observation state: an exact alias of the Astraea state.

Proteus intentionally preserves Astraea's feature order, normalization,
bounds, dimensionality, and feature-version identifier. Keeping one canonical
implementation prevents the two state spaces from drifting while allowing
``runtime.state: proteus`` to be selected independently in configuration.
"""

from olympus.states.astraea import (  # noqa: F401
    STATE_DIM,
    STATE_FEATURES,
    STATE_FEATURE_VERSION,
    STATE_HIGH,
    STATE_HIGH_T,
    STATE_LOW,
    STATE_LOW_T,
    normalize_state,
)


__all__ = [
    'STATE_DIM',
    'STATE_FEATURES',
    'STATE_FEATURE_VERSION',
    'STATE_HIGH',
    'STATE_HIGH_T',
    'STATE_LOW',
    'STATE_LOW_T',
    'normalize_state',
]
