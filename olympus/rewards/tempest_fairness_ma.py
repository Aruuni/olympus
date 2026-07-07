"""Fair-share Tempest reward plus learner-side average-throughput fairness.

Each worker normalizes throughput by the current ground-truth bottleneck
bandwidth divided by the number of flows present at that instant. MA-Dreamer
then combines simultaneous per-flow rewards and subtracts the exact attached
R_fair metric in the learner, where all active throughputs are available.
"""

import json
import os
import time

from olympus.common import link_context
from olympus.rewards.tempest import (
    RewardCalc as TempestRewardCalc,
)
from olympus.common.marl_team_reward import (
    r_fair_from_avg_throughput,
    shared_team_reward,
)


def _parse_json_list(raw, default=None):
    if raw in ('', None):
        return list(default or [])
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return list(default or [])
    return value if isinstance(value, list) else list(default or [])


class RewardCalc(TempestRewardCalc):
    def __init__(self, *args, flow_start_delays=None, flow_durations=None,
                 agent_id=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent_id = max(0, int(agent_id))
        self._flow_start_delays = [
            max(0.0, float(value))
            for value in (flow_start_delays or [0.0])
        ]
        durations = list(flow_durations or [])
        self._flow_durations = []
        for index in range(len(self._flow_start_delays)):
            value = durations[index] if index < len(durations) else None
            self._flow_durations.append(
                None if value in ('', None) else max(0.0, float(value)))

    def _flow_is_active(self, index, elapsed):
        if index < 0 or index >= len(self._flow_start_delays):
            return False
        start = self._flow_start_delays[index]
        duration = self._flow_durations[index]
        if elapsed + 1e-6 < start:
            return False
        return duration is None or elapsed <= start + duration

    def _active_flow_count(self, elapsed):
        active = sum(
            self._flow_is_active(index, elapsed)
            for index in range(len(self._flow_start_delays))
        )
        return max(1, active)

    def _elapsed(self, info: dict) -> float:
        if isinstance(info, dict) and 'time_s' in info:
            try:
                return max(0.0, float(info.get('time_s') or 0.0))
            except (TypeError, ValueError):
                pass
        return max(0.0, time.monotonic() - self._episode_start)

    def _thr_term(self, avg_thr: float, fair_bw: float) -> float:
        """Throughput factor of the base reward: rises to 1 at the fair share
        and stays flat beyond it (over-grabbing earns nothing extra)."""
        return min(avg_thr / fair_bw, 1.0)

    def step(self, info):
        avg_thr = float(info.get('avg_thr', 0.0) or 0.0)
        avg_urtt = float(info.get('avg_urtt', 0.0) or 0.0)
        srtt_us = self._srtt_us(info, fallback_us=avg_urtt)

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        elapsed = self._elapsed(info)
        link_bw = max(self._current_link_bw(info), 1.0)
        active_flows = self._active_flow_count(elapsed)
        flow_present = self._flow_is_active(self._agent_id, elapsed)
        fair_bw = max(link_bw / float(active_flows), 1.0)
        rtt_ref = max(self._current_link_rtt_us(info), 1.0)

        thr_ratio = self._thr_term(avg_thr, fair_bw)
        rtt_rate = min(rtt_ref / avg_urtt, 1.0) if avg_urtt > 0 else 1.0
        base_reward = 25.0 * thr_ratio * rtt_rate ** 2
        self._last_srtt_us = srtt_us

        self._step_count += 1
        sparse_bonus = 0.0
        if self._step_count % self._steps_per_bonus == 0:
            if srtt_us > 0 and abs(srtt_us - rtt_ref) <= self._bonus_tol_us:
                rtt_scale = max(
                    rtt_ref / max(self._initial_rtt, 1.0), 1.0)
                sparse_bonus = self._bonus_val * rtt_scale

        reward = max(0.0, base_reward + sparse_bonus)
        self.last_components = {
            'base': base_reward,
            'sparse': sparse_bonus,
            'unclipped': base_reward + sparse_bonus,
            'fair_bw_mbps': fair_bw * 8.0 / 1e6,
            'link_bw_mbps': link_bw * 8.0 / 1e6,
            'active_flows': float(active_flows),
            'flow_present': float(flow_present),
            'throughput_share_ratio': avg_thr / fair_bw,
        }
        return reward


def calc_kwargs_from_env() -> dict:
    """Worker-side construction arguments shared by the ma reward variants."""
    bw_mbps, base_rtt_us, link_schedule = link_context.context_values()
    episode_start = (
        float(os.environ.get('OC_EPISODE_START', '0'))
        or time.monotonic()
    )
    interval_ms = float(os.environ.get('OC_INTERVAL_MS', '20'))
    flow_start_delays = _parse_json_list(
        os.environ.get('OC_FAIR_FLOW_START_DELAYS'), default=[0.0])
    flow_durations = _parse_json_list(
        os.environ.get('OC_FAIR_FLOW_DURATIONS'), default=[])
    agent_id = int(os.environ.get('SAO_AGENT_ID', '0'))

    return dict(
        initial_bw_bytes_s=bw_mbps * 1e6 / 8.0,
        initial_rtt_us=base_rtt_us,
        link_schedule=link_schedule,
        episode_start=episode_start,
        interval_ms=interval_ms,
        flow_start_delays=flow_start_delays,
        flow_durations=flow_durations,
        agent_id=agent_id,
    )


def make_reward_calc():
    return RewardCalc(**calc_kwargs_from_env())


__all__ = [
    'RewardCalc',
    'make_reward_calc',
    'r_fair_from_avg_throughput',
    'shared_team_reward',
]
