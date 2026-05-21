"""Tempest reward with a fair-share bandwidth reference for joining flows."""

import json
import math
import os
import time

from single_agent_olympus.rewards.tempest import RewardCalc as TempestRewardCalc
from single_agent_olympus.rewards.kalman_fit import RewardStateContractError


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
                 w_fairness: float = 2.0,
                 fairness_cap: float = 10.0,
                 w_overshoot: float = 3.0,
                 overshoot_cap: float = 10.0,
                 overshoot_threshold: float = 1.5,
                 w_kalman_drift: float = 2.0,
                 kalman_drift_cap: float = 4.0,
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
        self._w_fairness = float(w_fairness)
        self._fairness_cap = float(fairness_cap)
        self._w_overshoot = float(w_overshoot)
        self._overshoot_cap = float(overshoot_cap)
        self._overshoot_threshold = float(overshoot_threshold)
        self._w_kalman_drift = float(w_kalman_drift)
        self._kalman_drift_cap = float(kalman_drift_cap)
        self._last_kalman_min_rtt_us = 0.0

    def _active_flow_count(self) -> int:
        elapsed = max(0.0, time.monotonic() - self._episode_start)
        active = 0
        for start, duration in zip(self._flow_start_delays, self._flow_durations):
            if elapsed + 1e-6 < start:
                continue
            if duration is not None and elapsed > start + duration:
                continue
            active += 1
        return max(1, active)

    def _current_fair_bw(self) -> tuple:
        link_bw = max(self._current_link_bw(), 1.0)
        active = self._active_flow_count()
        return max(link_bw / float(active), 1.0), link_bw, active

    def _require_kalman_min_rtt(self, info: dict) -> float:
        raw = info.get('kalman_min_rtt_us', None)
        try:
            kalman_us = float(raw)
        except (TypeError, ValueError):
            kalman_us = 0.0
        if math.isfinite(kalman_us) and kalman_us > 0.0:
            return kalman_us

        state_name = (os.environ.get('SAO_STATE') or os.environ.get('OC_STATE')
                      or '<unset>')
        raise RewardStateContractError(
            'reward/state contract mismatch: tempest_fairness has '
            f'w_kalman_drift={self._w_kalman_drift:g}, so RewardCalc.step() '
            "requires info['kalman_min_rtt_us'] to be populated by the "
            f'selected state normalizer. Current state is {state_name!r}, '
            'which did not provide a positive kalman_min_rtt_us value. '
            'Use a Tempest/Kalman state that provides kalman_min_rtt_us, or '
            'disable this reward term with reward.w_kalman_drift: 0.0 / '
            'OC_W_KALMAN_DRIFT=0.'
        )

    def step(self, info: dict) -> float:
        avg_thr = float(info.get('avg_thr', 0))
        avg_urtt = float(info.get('avg_urtt', 0))
        cwnd_pkts = max(float(info.get('cwnd', 1)), 1.0)
        srtt_us = self._srtt_us(info, fallback_us=avg_urtt)

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        bw_ref, link_bw, active_flows = self._current_fair_bw()
        rtt_ref = max(self._current_link_rtt_us(), 1.0)

        thr_ratio = min(avg_thr / bw_ref, 1.0)
        rtt_rate = min(rtt_ref / avg_urtt, 1.0) if avg_urtt > 0 else 1.0
        rtt_penalty = 1.0 - rtt_rate ** 2
        base_reward = 25.0 * thr_ratio * (1.0 - rtt_penalty)

        sched_bdp_pkts = max(bw_ref * rtt_ref / 1e6 / 1460.0, 1.0)
        overshoot_ratio = max(
            cwnd_pkts / sched_bdp_pkts - self._overshoot_threshold, 0.0)
        overshoot_cost = self._w_overshoot * min(
            overshoot_ratio ** 2, self._overshoot_cap)

        self._last_srtt_us = srtt_us
        kalman_us = (self._require_kalman_min_rtt(info)
                     if self._w_kalman_drift > 0.0 else
                     float(info.get('kalman_min_rtt_us', 0.0) or 0.0))
        self._last_kalman_min_rtt_us = kalman_us
        if rtt_ref > 0 and kalman_us > 0:
            drift_ratio = max(kalman_us / rtt_ref - 1.0, 0.0)
            drift_cost = self._w_kalman_drift * min(
                drift_ratio ** 2, self._kalman_drift_cap)
        else:
            drift_cost = 0.0

        share_ratio = avg_thr / bw_ref if bw_ref > 0 else 0.0
        overshare_ratio = max(share_ratio - 1.0, 0.0)
        fairness_cost = self._w_fairness * min(
            overshare_ratio ** 2, self._fairness_cap)

        self._step_count += 1
        sparse_bonus = 0.0
        if self._step_count % self._steps_per_bonus == 0:
            if srtt_us > 0 and abs(srtt_us - rtt_ref) <= self._bonus_tol_us:
                rtt_scale = max(rtt_ref / max(self._initial_rtt, 1.0), 1.0)
                sparse_bonus = self._bonus_val * rtt_scale

        cost = overshoot_cost + drift_cost + fairness_cost
        unclipped = base_reward + sparse_bonus - cost
        reward = max(0.0, unclipped)
        self.last_components = {
            'base': base_reward,
            'sparse': sparse_bonus,
            'overshoot_cost': overshoot_cost,
            'kalman_drift_cost': drift_cost,
            'srtt_drift_cost': 0.0,
            'drift_cost': drift_cost,
            'fairness_cost': fairness_cost,
            'cost': cost,
            'unclipped': unclipped,
            'fair_bw_mbps': bw_ref * 8.0 / 1e6,
            'link_bw_mbps': link_bw * 8.0 / 1e6,
            'active_flows': float(active_flows),
            'throughput_share_ratio': share_ratio,
        }
        return reward


def make_reward_calc() -> RewardCalc:
    """Build from environment variables set by orchestrator."""
    bw_mbps = float(os.environ.get('OC_LINK_BW', '100'))
    base_rtt_us = float(os.environ.get('OC_BASE_RTT_US', '20000'))
    episode_start = float(os.environ.get('OC_EPISODE_START', '0')) or time.monotonic()
    link_schedule = json.loads(os.environ.get('OC_LINK_SCHEDULE', '[]'))
    interval_ms = float(os.environ.get('OC_INTERVAL_MS', '20'))
    w_overshoot = float(os.environ.get('OC_W_OVERSHOOT', '3.0'))
    overshoot_cap = float(os.environ.get('OC_OVERSHOOT_CAP', '10.0'))
    overshoot_threshold = float(os.environ.get('OC_OVERSHOOT_THRESHOLD', '1.5'))
    w_kalman_drift = float(os.environ.get(
        'OC_W_KALMAN_DRIFT', os.environ.get('OC_W_SRTT_DRIFT', '2.0')))
    kalman_drift_cap = float(os.environ.get(
        'OC_KALMAN_DRIFT_CAP', os.environ.get('OC_SRTT_DRIFT_CAP', '4.0')))
    flow_start_delays = _parse_json_list(
        os.environ.get('OC_FAIR_FLOW_START_DELAYS'), default=[0.0])
    flow_durations = _parse_json_list(
        os.environ.get('OC_FAIR_FLOW_DURATIONS'), default=[])
    w_fairness = float(os.environ.get('OC_W_FAIRNESS', '2.0'))
    fairness_cap = float(os.environ.get('OC_FAIRNESS_CAP', '10.0'))

    return RewardCalc(
        initial_bw_bytes_s=bw_mbps * 1e6 / 8.0,
        initial_rtt_us=base_rtt_us,
        link_schedule=link_schedule,
        episode_start=episode_start,
        w_overshoot=w_overshoot,
        overshoot_cap=overshoot_cap,
        overshoot_threshold=overshoot_threshold,
        w_kalman_drift=w_kalman_drift,
        kalman_drift_cap=kalman_drift_cap,
        interval_ms=interval_ms,
        flow_start_delays=flow_start_delays,
        flow_durations=flow_durations,
        w_fairness=w_fairness,
        fairness_cap=fairness_cap,
    )
