"""Astraea local observation state (EuroSys '24, Liao et al., section 3.3).

The per-flow local state is the original Astraea 10-feature transform of the
DeepCC/TCP_INFO statistics collected each Monitoring Time Period (MTP):
throughput ratio, latency ratios, cwnd/BDP ratio, scaled max-throughput and
min-RTT, loss ratio, packets-in-flight ratio, pacing ratio, and retransmit
ratio. This is byte-for-byte the transform in ``astraea.agent.definitions``
(the released inference bundle) and already lives in
:mod:`olympus.states.astraea_deepcc` — re-exported here under the canonical
``astraea`` name so the Astraea recreation selects it as ``runtime.state:
astraea``.

The Astraea policy consumes a fixed-length history (``w`` MTPs, ``rec_dim`` in
the original code) of this state. History stacking is intentionally NOT done
here: the plugin stays a per-step transform (the single-agent contract) and
the astraea algorithm's worker/learner stack the last ``agent.history_len``
frames themselves, exactly like the original recurrent buffer.

``STATE_FEATURE_VERSION`` is kept identical to ``astraea_deepcc`` so
checkpoints trained under either name stay interchangeable.
"""

from olympus.states.astraea_deepcc import (  # noqa: F401
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
