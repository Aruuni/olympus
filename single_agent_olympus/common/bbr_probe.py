"""BBR3-style independent cwnd probe — algorithm- and reward-agnostic.

Any worker that drives ``set_cwnd`` can opt into this probe via the
``SAO_BBR_PROBE_*`` env vars (or ``agent.bbr_probe`` config block, which
the orchestrator translates to those env vars). The probe is unaware of
which RL algorithm or reward is active; it sits between the agent's
intended cwnd and the kernel.

Every ``probe_interval_s`` seconds the controller drops cwnd to
``probe_factor * saved_cwnd`` for ``probe_duration_s`` seconds, then
restores the saved cwnd. During the probe window the agent's cwnd write
is overridden; callers are expected to gate experience pushes on the
``locked`` return value so the replay buffer never sees a transition
caused by the probe rather than the agent.

The controller maintains a BBR3-style windowed min-RTT filter: the
current minimum is replaced when (a) a lower RTT is observed, or
(b) the running minimum is older than ``min_rtt_window_s``. The
minimum is exposed via ``.min_rtt_us`` for logging only; it must NOT
be added to the agent's observation.
"""

import math
import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ('0', 'false', 'no', 'off', '')


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class BbrProbe:
    def __init__(self,
                 enabled: bool = False,
                 probe_interval_s: float = 5.0,
                 probe_duration_s: float = 0.2,
                 probe_factor: float = 0.5,
                 min_rtt_window_s: float = 10.0):
        self.enabled = bool(enabled)
        self.probe_interval_s = float(probe_interval_s)
        self.probe_duration_s = float(probe_duration_s)
        self.probe_factor = float(probe_factor)
        self.min_rtt_window_s = float(min_rtt_window_s)

        self._in_probe = False
        self._probe_started_t = 0.0
        self._last_probe_start_t = -math.inf
        self._saved_cwnd = 0

        self._min_rtt_us = math.inf
        self._min_rtt_set_t = 0.0
        self._probe_count = 0

    @classmethod
    def from_env(cls) -> 'BbrProbe':
        return cls(
            enabled        = _bool_env('SAO_BBR_PROBE_ENABLED', False),
            probe_interval_s = _float_env('SAO_BBR_PROBE_INTERVAL_S', 5.0),
            probe_duration_s = _float_env('SAO_BBR_PROBE_DURATION_S', 0.2),
            probe_factor     = _float_env('SAO_BBR_PROBE_FACTOR', 0.5),
            min_rtt_window_s = _float_env('SAO_BBR_MIN_RTT_WINDOW_S', 10.0),
        )

    def observe_rtt(self, now: float, rtt_us: float) -> None:
        """Feed an RTT sample into the BBR-style windowed min filter."""
        if rtt_us <= 0.0 or not math.isfinite(rtt_us):
            return
        # Force-replace when the current minimum is older than the window.
        if (not math.isfinite(self._min_rtt_us)
                or now - self._min_rtt_set_t > self.min_rtt_window_s):
            self._min_rtt_us = float(rtt_us)
            self._min_rtt_set_t = now
            return
        # Update on a new low; this also resets the window timer.
        if rtt_us < self._min_rtt_us:
            self._min_rtt_us = float(rtt_us)
            self._min_rtt_set_t = now

    def decide(self, now: float, current_cwnd: int, agent_cwnd: int):
        """Return (cwnd_to_set, locked, transitioned).

        * cwnd_to_set : int   — what the caller should pass to set_cwnd.
        * locked      : bool  — True when the agent's choice was overridden.
        * transitioned: bool  — True on the step that starts OR ends a probe.
        """
        if not self.enabled:
            return int(agent_cwnd), False, False

        if self._in_probe:
            elapsed = now - self._probe_started_t
            if elapsed < self.probe_duration_s:
                target = max(1, int(round(self._saved_cwnd * self.probe_factor)))
                return target, True, False
            # Probe complete — restore saved cwnd.
            self._in_probe = False
            return int(self._saved_cwnd), True, True

        # Idle. Start a fresh probe if the interval has elapsed since the last
        # probe-start (measured start-to-start, like BBR's MinRtt cycle).
        if now - self._last_probe_start_t >= self.probe_interval_s:
            self._in_probe = True
            self._probe_started_t = now
            self._last_probe_start_t = now
            self._saved_cwnd = int(current_cwnd)
            self._probe_count += 1
            target = max(1, int(round(current_cwnd * self.probe_factor)))
            return target, True, True

        # Agent in control.
        return int(agent_cwnd), False, False

    @property
    def in_probe(self) -> bool:
        return self._in_probe

    @property
    def min_rtt_us(self) -> float:
        return self._min_rtt_us if math.isfinite(self._min_rtt_us) else 0.0

    @property
    def probe_count(self) -> int:
        return self._probe_count
