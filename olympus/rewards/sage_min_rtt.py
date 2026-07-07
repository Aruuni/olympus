"""SAGE-style reward referenced against the connection's running min RTT.

Formula
-------
    base = 25 * thr_ratio * rtt_rate^2
        thr_ratio = clip(avg_thr / sched_bw, 0, 1)
        rtt_rate  = clip(min_rtt / avg_urtt, 0, 1)

No drift term, no sparse coin. The bandwidth term still
references the scheduled link bandwidth (the orchestrator supplies it via the
per-episode link-context file), but the RTT reference is the
connection's observed minimum — tp->deepcc_api.min_urtt from tcp_deepcc.c
in microseconds, never reset on getsockopt. This removes the scheduled-RTT
oracle from the reward path: the agent is rewarded for keeping
instantaneous avg_urtt close to the lowest RTT the connection has seen so
far, and penalised whenever queueing inflates it above that floor.

The `min_rtt` field is read from info['min_rtt'] (preferred) or, as a
fallback, info['min_rtt_us'] — both µs. If neither is available or both
are zero (no RTT samples yet), the RTT term is treated as 1.0, matching
the original sage.py convention when avg_urtt == 0.
"""

import os
import time

from olympus.common import link_context


class RewardCalc:
    def __init__(self,
                 initial_bw_bytes_s: float = 100e6 / 8,
                 link_schedule:      list  = None,
                 episode_start:      float = 0.0):

        self._initial_bw      = initial_bw_bytes_s
        self._max_tput        = 1.0
        self._episode_start   = episode_start
        self._last_min_rtt_us = 0.0
        self.last_components  = {}

        self._bw_trace = link_context.build_step_trace(
            initial_bw_bytes_s, link_schedule, episode_start, 'bw',
            transform=lambda value: float(value) * 1e6 / 8.0,
            warn_prefix='reward',
        )

    def _schedule_time(self, info: dict = None) -> float:
        if isinstance(info, dict) and 'time_s' in info:
            try:
                return self._episode_start + float(info.get('time_s') or 0.0)
            except (TypeError, ValueError):
                pass
        return time.monotonic()

    def _current_link_bw(self, info: dict = None) -> float:
        return self._bw_trace.at(self._schedule_time(info))

    @staticmethod
    def _read_min_rtt_us(info: dict) -> float:
        raw = info.get('min_rtt', info.get('min_rtt_us', 0)) or 0
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return val if val > 0.0 else 0.0

    def step(self, info: dict) -> float:
        avg_thr    = float(info.get('avg_thr',  0))   # bytes/s
        avg_urtt   = float(info.get('avg_urtt', 0))   # µs
        min_rtt_us = self._read_min_rtt_us(info)      # µs
        self._last_min_rtt_us = min_rtt_us

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        bw_ref = max(self._current_link_bw(info), 1.0)

        thr_ratio = min(avg_thr / bw_ref, 1.0)
        if avg_urtt > 0.0 and min_rtt_us > 0.0:
            rtt_rate = min(min_rtt_us / avg_urtt, 1.0)
        else:
            rtt_rate = 1.0
        base_reward = 25.0 * thr_ratio * (rtt_rate ** 2)

        reward = max(0.0, base_reward)
        self.last_components = {
            'base': base_reward,
            'thr_ratio': thr_ratio,
            'rtt_rate': rtt_rate,
            'min_rtt_us': min_rtt_us,
            'unclipped': base_reward,
        }
        return reward

    @property
    def max_tput(self) -> float:
        return self._max_tput

    @property
    def min_rtt_us(self) -> float:
        return self._last_min_rtt_us

    # Alias for workers that log reward_calc.kalman_min_rtt_us — this reward
    # has no Kalman estimator, so we expose the kernel min_rtt under that
    # same name. Keeps the logging contract identical for downstream tools.
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
