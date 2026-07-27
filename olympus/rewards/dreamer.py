"""Dreamer reward: Tempest with a raw-min-RTT proximity bonus.

The dense term is identical to Tempest: link utilization multiplied by the
squared ratio between the scheduled base RTT and observed average RTT. Every
three seconds, Dreamer adds a bonus based on how closely the connection's raw
``min_rtt`` matches the currently scheduled base RTT. The bonus tapers linearly
according to the ratio between the two RTT values, with no tolerance or cutoff.

This reward never estimates RTT and never exposes ``min_rtt`` to the Dreamer
policy state. It uses the measurement only as a training reward signal.
"""

import math
import os
import time

from olympus.common import link_context, runtime_config


PROXIMITY_BONUS_INTERVAL_MS = 3000.0


class RewardCalc:
    def __init__(self,
                 initial_bw_bytes_s: float = 100e6 / 8,
                 initial_rtt_us: float = 20_000.0,
                 link_schedule: list = None,
                 episode_start: float = 0.0,
                 interval_ms: float = 20.0,
                 min_rtt_bonus: float = 20.0):
        self._initial_bw = initial_bw_bytes_s
        self._initial_rtt = initial_rtt_us
        self._episode_start = episode_start
        self._max_tput = 1.0
        self._last_srtt_us = 0.0
        self._last_min_rtt_us = 0.0
        self.last_components = {}

        self._min_rtt_bonus = max(float(min_rtt_bonus), 0.0)
        self._step_count = 0
        self._steps_per_bonus = max(
            1, int(round(PROXIMITY_BONUS_INTERVAL_MS / interval_ms)))

        self._bw_trace = link_context.build_step_trace(
            initial_bw_bytes_s, link_schedule, episode_start, 'bw',
            transform=lambda value: float(value) * 1e6 / 8.0,
            warn_prefix='reward',
        )
        self._rtt_trace = link_context.build_step_trace(
            initial_rtt_us, link_schedule, episode_start, 'delay',
            transform=lambda value: float(value) * 1_000.0,
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

    def _current_link_rtt_us(self, info: dict = None) -> float:
        return self._rtt_trace.at(self._schedule_time(info))

    @staticmethod
    def _read_min_rtt_us(info: dict) -> float:
        raw = info.get('min_rtt', info.get('min_rtt_us', 0.0)) or 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value > 0.0 else 0.0

    @staticmethod
    def _srtt_us(info: dict, fallback_us: float = 0.0) -> float:
        try:
            srtt_raw = float(info.get('srtt_us', 0.0) or 0.0)
        except (TypeError, ValueError):
            srtt_raw = 0.0
        srtt_us = (
            srtt_raw / 8.0 if srtt_raw > 0.0 else float(fallback_us or 0.0))
        return srtt_us if math.isfinite(srtt_us) and srtt_us > 0.0 else 0.0

    def step(self, info: dict) -> float:
        avg_thr = float(info.get('avg_thr', 0.0) or 0.0)
        avg_urtt = float(info.get('avg_urtt', 0.0) or 0.0)
        srtt_us = self._srtt_us(info, fallback_us=avg_urtt)
        min_rtt_us = self._read_min_rtt_us(info)

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        bw_ref = max(self._current_link_bw(info), 1.0)
        rtt_ref = max(self._current_link_rtt_us(info), 1.0)

        thr_ratio = min(max(avg_thr / bw_ref, 0.0), 1.0)
        rtt_rate = min(rtt_ref / avg_urtt, 1.0) if avg_urtt > 0.0 else 1.0
        rtt_penalty = 1.0 - rtt_rate ** 2
        base_reward = 25.0 * thr_ratio * (1.0 - rtt_penalty)

        self._last_srtt_us = srtt_us
        self._last_min_rtt_us = min_rtt_us

        min_rtt_error_us = (
            abs(min_rtt_us - rtt_ref) if min_rtt_us > 0.0 else math.inf)
        min_rtt_closeness = (
            min(min_rtt_us / rtt_ref, rtt_ref / min_rtt_us)
            if min_rtt_us > 0.0 else 0.0)

        self._step_count += 1
        proximity_bonus = 0.0
        if self._step_count % self._steps_per_bonus == 0:
            rtt_scale = max(rtt_ref / max(self._initial_rtt, 1.0), 1.0)
            proximity_bonus = (
                self._min_rtt_bonus * min_rtt_closeness * rtt_scale)

        unclipped = base_reward + proximity_bonus
        reward = max(0.0, unclipped)
        self.last_components = {
            'base': base_reward,
            'min_rtt_proximity': proximity_bonus,
            'min_rtt_closeness': min_rtt_closeness,
            'min_rtt_error_us': min_rtt_error_us,
            'min_rtt_us': min_rtt_us,
            'rtt_ref_us': rtt_ref,
            'unclipped': unclipped,
        }
        return reward

    @property
    def max_tput(self) -> float:
        return self._max_tput

    @property
    def srtt_us(self) -> float:
        return self._last_srtt_us

    @property
    def min_rtt_us(self) -> float:
        return self._last_min_rtt_us

    @property
    def kalman_min_rtt_us(self) -> float:
        """Raw min RTT under the legacy logging property; no filter is used."""
        return self._last_min_rtt_us


def make_reward_calc() -> RewardCalc:
    """Build from the resolved config and episode link context."""
    cfg = runtime_config.load_config()
    bw_mbps, base_rtt_us, link_schedule = link_context.context_values()
    episode_start = (
        float(os.environ.get('OC_EPISODE_START', '0')) or time.monotonic())
    interval_ms = float(os.environ.get('OC_INTERVAL_MS', '20'))

    return RewardCalc(
        initial_bw_bytes_s=bw_mbps * 1e6 / 8.0,
        initial_rtt_us=base_rtt_us,
        link_schedule=link_schedule,
        episode_start=episode_start,
        interval_ms=interval_ms,
        min_rtt_bonus=float(runtime_config.reward_value(
            cfg, 'min_rtt_bonus', default=20.0)),
    )
