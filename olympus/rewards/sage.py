"""SAGE base reward, but the RTT reference is the connection's observed
minimum RTT instead of the scheduled propagation RTT.

sage.py base:
    base = 25 * thr_ratio * rtt_rate^2
        thr_ratio = clip(avg_thr / sched_bw, 0, 1)
        rtt_rate  = clip(sched_rtt / avg_urtt, 0, 1)

This reward keeps the throughput term on the *scheduled* bandwidth but
swaps the RTT reference:
        rtt_rate  = clip(min_rtt / avg_urtt, 0, 1)

``min_rtt`` is the kernel's running connection-wide minimum RTT
(tp->deepcc_api.min_urtt from tcp_deepcc.c, in µs, never reset on
getsockopt). The reward therefore needs no scheduled-RTT oracle: the
agent is rewarded for keeping the instantaneous RTT close to the lowest
RTT this connection has ever observed (i.e. the empirical propagation
floor), and penalised as queueing inflates avg_urtt above that floor.

No drift cost, no sparse coin — just the base.
"""

import os
import time

from olympus.common import link_context


class RewardCalc:
    def __init__(self,
                 initial_bw_bytes_s: float = 100e6 / 8,
                 link_schedule:      list  = None,
                 episode_start:      float = 0.0):

        self._initial_bw    = initial_bw_bytes_s
        self._max_tput      = 1.0
        self._episode_start = episode_start
        self._last_min_rtt_us = 0.0
        self.last_components = {}

        self._bw_trace = link_context.build_step_trace(
            initial_bw_bytes_s, link_schedule, episode_start, 'bw',
            transform=lambda value: float(value) * 1e6 / 8.0,
            warn_prefix='reward',
        )

    # ── Schedule lookup (bandwidth only) ──────────────────────────────────────

    def _schedule_time(self, info: dict = None) -> float:
        if isinstance(info, dict) and 'time_s' in info:
            try:
                return self._episode_start + float(info.get('time_s') or 0.0)
            except (TypeError, ValueError):
                pass
        return time.monotonic()

    def _current_link_bw(self, info: dict = None) -> float:
        return self._bw_trace.at(self._schedule_time(info))

    # ── Per-step reward ───────────────────────────────────────────────────────

    def step(self, info: dict) -> float:
        avg_thr  = float(info.get('avg_thr',  0))          # bytes/s
        avg_urtt = float(info.get('avg_urtt', 0))          # µs

        # Connection-wide running min RTT from the kernel (µs).
        raw = info.get('min_rtt', info.get('min_rtt_us', 0)) or 0
        try:
            min_rtt_us = float(raw)
        except (TypeError, ValueError):
            min_rtt_us = 0.0
        self._last_min_rtt_us = min_rtt_us

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        bw_ref = max(self._current_link_bw(info), 1.0)

        # ── Base reward — RTT reference is min_rtt, not scheduled RTT ─────────
        thr_ratio = min(avg_thr / bw_ref, 1.0)
        if avg_urtt > 0 and min_rtt_us > 0:
            rtt_rate = min(min_rtt_us / avg_urtt, 1.0)
        else:
            # No fresh RTT sample or no observed minimum yet: no RTT penalty,
            # same convention as sage.py's avg_urtt == 0 path.
            rtt_rate = 1.0
        base_reward = 25.0 * thr_ratio * (rtt_rate ** 2)

        reward = max(0.0, base_reward)
        self.last_components = {
            'base': base_reward,
            'unclipped': base_reward,
        }
        return reward

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def max_tput(self) -> float:
        return self._max_tput

    @property
    def min_rtt_us(self) -> float:
        return self._last_min_rtt_us

    # Alias so td3 workers that log reward_calc.kalman_min_rtt_us
    # keep working unchanged.
    @property
    def kalman_min_rtt_us(self) -> float:
        return self._last_min_rtt_us


def make_reward_calc() -> RewardCalc:
    """Build from environment variables set by orchestrator."""
    bw_mbps, _base_rtt_us, link_schedule = link_context.context_values()
    episode_start = float(os.environ.get('OC_EPISODE_START', '0')) or time.monotonic()
    return RewardCalc(
        initial_bw_bytes_s = bw_mbps * 1e6 / 8.0,
        link_schedule      = link_schedule,
        episode_start      = episode_start,
    )
