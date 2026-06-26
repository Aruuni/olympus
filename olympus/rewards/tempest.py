"""Tempest reward plugin: throughput-utilization base + sparse RTT-match bonus."""

import json
import math
import os
import time


SPARSE_BONUS_INTERVAL_MS = 3000.0


class RewardCalc:
    def __init__(self,
                 initial_bw_bytes_s: float = 100e6 / 8,
                 initial_rtt_us: float = 20_000.0,
                 link_schedule: list = None,
                 episode_start: float = 0.0,
                 interval_ms: float = 20.0,
                 bonus_tol_ms: float = 5.0,
                 bonus_val: float = 20.0):

        self._initial_bw = initial_bw_bytes_s
        self._initial_rtt = initial_rtt_us
        self._episode_start = episode_start
        self._max_tput = 1.0
        self.last_components = {}

        self._last_srtt_us = 0.0

        self._bonus_tol_us = bonus_tol_ms * 1_000.0
        self._bonus_val = bonus_val
        self._step_count = 0
        self._steps_per_bonus = max(1, int(round(SPARSE_BONUS_INTERVAL_MS / interval_ms)))

        self._bw_schedule = []
        self._rtt_schedule = []
        for entry in (link_schedule or []):
            if 't' not in entry:
                print(f'[reward] WARNING: schedule entry missing "t" key, skipped: {entry}',
                      flush=True)
                continue
            t_abs = episode_start + float(entry['t'])
            if 'bw' in entry:
                self._bw_schedule.append((t_abs, float(entry['bw']) * 1e6 / 8.0))
            if 'delay' in entry:
                self._rtt_schedule.append((t_abs, float(entry['delay']) * 1_000.0))
        self._bw_schedule.sort()
        self._rtt_schedule.sort()

    def _schedule_time(self, info: dict = None) -> float:
        if isinstance(info, dict) and 'time_s' in info:
            try:
                return self._episode_start + float(info.get('time_s') or 0.0)
            except (TypeError, ValueError):
                pass
        return time.monotonic()

    def _current_link_bw(self, info: dict = None) -> float:
        now = self._schedule_time(info)
        val = self._initial_bw
        for t_abs, bw_bs in self._bw_schedule:
            if now >= t_abs:
                val = bw_bs
            else:
                break
        return val

    def _current_link_rtt_us(self, info: dict = None) -> float:
        now = self._schedule_time(info)
        val = self._initial_rtt
        for t_abs, rtt_us in self._rtt_schedule:
            if now >= t_abs:
                val = rtt_us
            else:
                break
        return val

    def _srtt_us(self, info: dict, fallback_us: float = 0.0) -> float:
        try:
            srtt_raw = float(info['srtt_us'] or 0.0)
        except (TypeError, ValueError):
            srtt_raw = 0.0
        srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else float(fallback_us or 0.0)
        return srtt_us if math.isfinite(srtt_us) and srtt_us > 0.0 else 0.0

    def step(self, info: dict) -> float:
        avg_thr = float(info['avg_thr'])
        avg_urtt = float(info['avg_urtt'])
        srtt_us = self._srtt_us(info, fallback_us=avg_urtt)

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        bw_ref = max(self._current_link_bw(info), 1.0)
        rtt_ref = max(self._current_link_rtt_us(info), 1.0)

        thr_ratio = min(avg_thr / bw_ref, 1.0)
        rtt_rate = min(rtt_ref / avg_urtt, 1.0) if avg_urtt > 0 else 1.0
        rtt_penalty = 1.0 - rtt_rate ** 2
        base_reward = 25.0 * thr_ratio * (1.0 - rtt_penalty)

        self._last_srtt_us = srtt_us

        self._step_count += 1
        sparse_bonus = 0.0
        if self._step_count % self._steps_per_bonus == 0:
            if srtt_us > 0 and abs(srtt_us - rtt_ref) <= self._bonus_tol_us:
                rtt_scale = max(rtt_ref / max(self._initial_rtt, 1.0), 1.0)
                sparse_bonus = self._bonus_val * rtt_scale

        unclipped = base_reward + sparse_bonus
        reward = max(0.0, unclipped)
        self.last_components = {
            'base': base_reward,
            'sparse': sparse_bonus,
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
    def kalman_min_rtt_us(self) -> float:
        return 0.0


def make_reward_calc() -> RewardCalc:
    """Build from environment variables set by orchestrator."""
    bw_mbps = float(os.environ.get('OC_LINK_BW', '100'))
    base_rtt_us = float(os.environ.get('OC_BASE_RTT_US', '20000'))
    episode_start = float(os.environ.get('OC_EPISODE_START', '0')) or time.monotonic()
    link_schedule = json.loads(os.environ.get('OC_LINK_SCHEDULE', '[]'))
    interval_ms = float(os.environ.get('OC_INTERVAL_MS', '20'))

    return RewardCalc(
        initial_bw_bytes_s = bw_mbps * 1e6 / 8.0,
        initial_rtt_us = base_rtt_us,
        link_schedule = link_schedule,
        episode_start = episode_start,
        interval_ms = interval_ms,
    )
