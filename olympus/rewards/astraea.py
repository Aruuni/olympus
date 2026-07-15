"""Astraea reward plugin (EuroSys '24, Liao et al., section 3.3 "Reward block").

The paper's reward is GLOBAL — a linear combination over all active flows:

    R = c0*R_thr - c1*R_lat - c2*R_loss - c3*R_fair - c4*R_stab      (Eq. 8)

with (Eq. 4-6):
    R_thr  = sum_i(thr_i) / c                (link utilization)
    R_lat  = max(mean_i(lat_i) - (1+beta)*d0, 0) * p_rate
    R_loss = mean_i(loss_i / thr_i)
    R_fair = std_i(avg_thr_i) / sum_i(avg_thr_i)     (across flows)
    R_stab = mean_i(std_w(thr_i) / avg_thr_i)        (across w MTPs)

and is bounded to (-c0, c0). Only the learner sees all flows, so the true
team reward is assembled there (``algorithms/astraea/model.astraea_team_reward``)
from the per-flow statistics each worker pushes. This worker-side plugin
provides:

  * the flow's OWN view of the non-relational terms (R_thr/R_lat/R_loss with
    n=1) — logged per step and useful as a local diagnostic;
  * ``flow_present`` scheduling (identical to tempest_fairness_ma) so starved
    but scheduled flows keep pushing experiences (MARL mask semantics);
  * the per-step link context (``link_bw_bytes``, ``rtt_ref_us``) that the
    worker copies into each Experience for the learner's global reward and
    global critic state.

Weights default to the paper's tech-report Table 4 values: c0=0.1, c1=0.02,
c2=1.0, c3=0.02, c4=0.01, with the reward bounded to (-0.1, 0.1) as stated
in section 3.3. beta is not published; default 0.5. d0 is taken as the
flow's observed min_rtt (its propagation-delay baseline). Override any of
them in the config under ``algorithms.astraea.reward``.
"""

from olympus.common import runtime_config
from olympus.rewards.tempest_fairness_ma import (
    RewardCalc as FairShareRewardCalc,
    calc_kwargs_from_env,
)


MSS_BYTES = 1460.0

DEFAULT_WEIGHTS = {
    'throughput_weight': 0.1,   # c0
    'latency_weight': 0.02,     # c1
    'loss_weight': 1.0,         # c2
    'fairness_weight': 0.02,    # c3 (learner-side only)
    'stability_weight': 0.01,   # c4 (learner-side only)
    'delay_coefficient': 0.5,   # beta
    'reward_clip': 0.1,         # bound of Eq. 8
}


def reward_weights(cfg: dict = None) -> dict:
    """Resolve the Astraea reward coefficients from the run config."""
    cfg = cfg if cfg is not None else runtime_config.load_config()
    out = dict(DEFAULT_WEIGHTS)
    for key in out:
        value = runtime_config.reward_value(cfg, key)
        if value is not None:
            out[key] = float(value)
    return out


def latency_threshold_us(min_rtt_us, delay_coefficient):
    """(1 + beta) * d0 in microseconds (scalar or numpy array).

    d0 is the flow's observed ``min_rtt`` — its propagation-delay baseline.
    R_lat is meant to stay zero until queueing inflates latency beyond the
    no-load baseline (times 1+beta); since our latency signal ``avg_urtt``
    is a round-trip time, the matching baseline is the round-trip min_rtt.
    Using anything below the no-load RTT as d0 puts the threshold under the
    zero-queue floor, turning R_lat into an always-on penalty proportional
    to pacing rate — under such a reward starvation is optimal (the failure
    mode of the astraea_20260714-023749 run).
    """
    return (1.0 + float(delay_coefficient)) * min_rtt_us


class RewardCalc(FairShareRewardCalc):
    """Per-flow local Astraea reward + scheduling/link-context components."""

    def __init__(self, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights = dict(weights or reward_weights())

    def step(self, info):
        avg_thr = float(info.get('avg_thr', 0.0) or 0.0)
        avg_urtt = float(info.get('avg_urtt', 0.0) or 0.0)
        min_rtt = float(info.get('min_rtt', 0.0) or 0.0)
        loss_ratio = float(info.get('loss_ratio', 0.0) or 0.0)
        pacing_rate = float(info.get('pacing_rate', 0.0) or 0.0)

        if avg_thr > self._max_tput:
            self._max_tput = avg_thr

        elapsed = self._elapsed(info)
        link_bw = max(self._current_link_bw(info), 1.0)
        rtt_ref = max(self._current_link_rtt_us(info), 1.0)
        active_flows = self._active_flow_count(elapsed)
        flow_present = self._flow_is_active(self._agent_id, elapsed)
        self._last_srtt_us = self._srtt_us(info, fallback_us=avg_urtt)

        w = self._weights
        r_thr = avg_thr / link_bw
        # d0 = the flow's own min_rtt; no latency signal yet -> no penalty.
        if avg_urtt > 0.0 and min_rtt > 0.0:
            threshold_us = latency_threshold_us(
                min_rtt, w['delay_coefficient'])
            lat_excess_s = max(avg_urtt - threshold_us, 0.0) / 1e6
        else:
            lat_excess_s = 0.0
        r_lat = lat_excess_s * (pacing_rate / MSS_BYTES)
        r_loss = loss_ratio / avg_thr if avg_thr > 0.0 else 0.0

        unclipped = (
            w['throughput_weight'] * r_thr
            - w['latency_weight'] * r_lat
            - w['loss_weight'] * r_loss
        )
        clip = w['reward_clip']
        reward = float(min(max(unclipped, -clip), clip))

        self._step_count += 1
        self.last_components = {
            'base': reward,
            'unclipped': unclipped,
            'r_thr': r_thr,
            'r_lat': r_lat,
            'r_loss': r_loss,
            'link_bw_bytes': link_bw,
            'link_bw_mbps': link_bw * 8.0 / 1e6,
            'rtt_ref_us': rtt_ref,
            'active_flows': float(active_flows),
            'flow_present': float(flow_present),
        }
        return reward


def make_reward_calc():
    return RewardCalc(**calc_kwargs_from_env())


__all__ = [
    'DEFAULT_WEIGHTS',
    'MSS_BYTES',
    'RewardCalc',
    'latency_threshold_us',
    'make_reward_calc',
    'reward_weights',
]
