"""Tempest reward with a fair-share bandwidth reference for joining flows.

Reward = base throughput-utilization * RTT-ratio term (+ sparse RTT-match
bonus) minus a fairness penalty. The fairness penalty is selectable between
two single-agent metrics via ``fairness_metric``:

  * ``r_fair`` (default): Astraea's coefficient-of-variation metric (Eq. 6).
  * ``jain``: the classic Jain fairness index, applied as a (1 - Jain) penalty.

Both are computed every step (so both are logged), but only the selected one
is subtracted from the returned reward.
"""

import json
import math
import os
import time

from olympus.common import link_context
from olympus.rewards.tempest import RewardCalc as TempestRewardCalc

_FAIRNESS_METRICS = ('r_fair', 'jain')


def _parse_json_list(raw, default=None):
    if raw in ('', None):
        return list(default or [])
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return list(default or [])
    if isinstance(value, list):
        return value
    return list(default or [])


class RewardCalc(TempestRewardCalc):
    def __init__(self,
                 *args,
                 flow_start_delays: list = None,
                 flow_durations: list = None,
                 fairness_metric: str = 'r_fair',
                 w_r_fair: float = 25.0,
                 r_fair_cap: float = 1.0,
                 w_jain: float = 25.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._flow_start_delays = [
            max(0.0, float(v)) for v in (flow_start_delays or [0.0])
        ]
        raw_durations = list(flow_durations or [])
        self._flow_durations = []
        for idx in range(len(self._flow_start_delays)):
            value = raw_durations[idx] if idx < len(raw_durations) else None
            if value in ('', None):
                self._flow_durations.append(None)
            else:
                self._flow_durations.append(max(0.0, float(value)))
        metric = str(fairness_metric or 'r_fair').strip().lower()
        if metric not in _FAIRNESS_METRICS:
            raise ValueError(
                f'reward.fairness_metric={fairness_metric!r} is invalid; '
                f'expected one of {_FAIRNESS_METRICS}')
        self._fairness_metric = metric
        self._w_r_fair = float(w_r_fair)
        self._r_fair_cap = float(r_fair_cap)
        self._w_jain = float(w_jain)

    def _elapsed(self, info: dict) -> float:
        if isinstance(info, dict) and 'time_s' in info:
            try:
                return max(0.0, float(info.get('time_s') or 0.0))
            except (TypeError, ValueError):
                pass
        return max(0.0, time.monotonic() - self._episode_start)

    def _active_flow_count(self, info: dict = None) -> int:
        elapsed = self._elapsed(info)
        active = 0
        for start, duration in zip(self._flow_start_delays, self._flow_durations):
            if elapsed + 1e-6 < start:
                continue
            if duration is not None and elapsed > start + duration:
                continue
            active += 1
        return max(1, active)

    def _current_fair_bw(self, info: dict = None) -> tuple:
        link_bw = max(self._current_link_bw(info), 1.0)
        active = self._active_flow_count(info)
        return max(link_bw / float(active), 1.0), link_bw, active

    def step(self, info: dict) -> float:
        avg_thr = float(info.get('avg_thr', 0))
        avg_urtt = float(info.get('avg_urtt', 0))
        srtt_us = self._srtt_us(info, fallback_us=avg_urtt)

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        bw_ref, link_bw, active_flows = self._current_fair_bw(info)
        rtt_ref = max(self._current_link_rtt_us(info), 1.0)

        thr_ratio = min(avg_thr / bw_ref, 1.0)
        rtt_rate = min(rtt_ref / avg_urtt, 1.0) if avg_urtt > 0 else 1.0
        rtt_penalty = 1.0 - rtt_rate ** 2
        base_reward = 25.0 * thr_ratio * (1.0 - rtt_penalty)

        self._last_srtt_us = srtt_us

        # Single-agent fairness metrics. Both Astraea's R_fair (EuroSys'24,
        # Eq. 6) and the Jain index need every active flow's throughput, but a
        # single-agent worker only sees its own avg_thr. Under the assumptions
        # that (a) the bottleneck is fully utilized and (b) the other n-1 flows
        # split the remaining capacity (link_bw - avg_thr) equally, both global
        # metrics collapse to closed-form local expressions in (avg_thr, n, C):
        #   R_fair = |avg_thr - fair_share| / (sqrt(n-1) * C)        in [0, ~0.5]
        #   Jain   = C^2 / ( n * (avg_thr^2 + (C-avg_thr)^2/(n-1)) ) in [1/n, 1]
        # With one active flow there is no fairness notion: R_fair=0, Jain=1.
        # Each is turned into an "unfairness cost" (Jain via 1 - Jain) and only
        # the selected metric is subtracted from the reward.
        share_ratio = avg_thr / bw_ref if bw_ref > 0 else 0.0
        if active_flows >= 2 and link_bw > 0:
            r_fair = abs(avg_thr - bw_ref) / (math.sqrt(active_flows - 1) * link_bw)
            sum_sq = avg_thr ** 2 + (link_bw - avg_thr) ** 2 / (active_flows - 1)
            jain = (link_bw ** 2) / (active_flows * sum_sq) if sum_sq > 0 else 1.0
        else:
            r_fair = 0.0
            jain = 1.0
        r_fair_cost = self._w_r_fair * min(r_fair, self._r_fair_cap)
        jain_cost = self._w_jain * max(0.0, 1.0 - jain)
        fairness_cost = jain_cost if self._fairness_metric == 'jain' else r_fair_cost

        self._step_count += 1
        sparse_bonus = 0.0
        if self._step_count % self._steps_per_bonus == 0:
            if srtt_us > 0 and abs(srtt_us - rtt_ref) <= self._bonus_tol_us:
                rtt_scale = max(rtt_ref / max(self._initial_rtt, 1.0), 1.0)
                sparse_bonus = self._bonus_val * rtt_scale

        unclipped = base_reward + sparse_bonus - fairness_cost
        reward = max(0.0, unclipped)
        self.last_components = {
            'base': base_reward,
            'sparse': sparse_bonus,
            'r_fair': r_fair,
            'jain': jain,
            'r_fair_cost': r_fair_cost,
            'jain_cost': jain_cost,
            'fairness_cost': fairness_cost,
            'cost': fairness_cost,
            'unclipped': unclipped,
            'fair_bw_mbps': bw_ref * 8.0 / 1e6,
            'link_bw_mbps': link_bw * 8.0 / 1e6,
            'active_flows': float(active_flows),
            'throughput_share_ratio': share_ratio,
        }
        return reward


def make_reward_calc() -> RewardCalc:
    """Build from environment variables set by orchestrator."""
    bw_mbps, base_rtt_us, link_schedule = link_context.context_values()
    episode_start = float(os.environ.get('OC_EPISODE_START', '0')) or time.monotonic()
    interval_ms = float(os.environ.get('OC_INTERVAL_MS', '20'))
    flow_start_delays = _parse_json_list(
        os.environ.get('OC_FAIR_FLOW_START_DELAYS'), default=[0.0])
    flow_durations = _parse_json_list(
        os.environ.get('OC_FAIR_FLOW_DURATIONS'), default=[])
    fairness_metric = os.environ.get('OC_FAIRNESS_METRIC', 'r_fair')
    w_r_fair = float(os.environ.get('OC_W_R_FAIR', '25.0'))
    r_fair_cap = float(os.environ.get('OC_R_FAIR_CAP', '1.0'))
    w_jain = float(os.environ.get('OC_W_JAIN', '25.0'))

    return RewardCalc(
        initial_bw_bytes_s=bw_mbps * 1e6 / 8.0,
        initial_rtt_us=base_rtt_us,
        link_schedule=link_schedule,
        episode_start=episode_start,
        interval_ms=interval_ms,
        flow_start_delays=flow_start_delays,
        flow_durations=flow_durations,
        fairness_metric=fairness_metric,
        w_r_fair=w_r_fair,
        r_fair_cap=r_fair_cap,
        w_jain=w_jain,
    )
