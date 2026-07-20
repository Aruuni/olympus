"""Orca reward (Soheil-ab/Orca rl-module/envwrapper.py), as a runtime plugin.

This is Orca's native reward, previously computed inside the learner's
OrcaRepoTransform. Ported here so it is selected by ``runtime.reward: orca``
like every other reward — the orca worker now reads its reward from this
plugin, not from the state transform.

Formula (Orca's ``use_normalizer=False`` default path):

    reward = ((throughput - 5*loss_rate) / max_bw) * delay_metric

where

    max_bw       running maximum of ``throughput`` this flow has seen
    delay_metric min(1, delay_margin_coef * min_rtt / srtt), the same
                 queueing-delay discount used in Orca's state
    throughput   bytes/s over the interval
    loss_rate    lost bytes/s over the interval

``max_bw`` is per-flow state: one RewardCalc is built per worker/flow, so its
running maximum matches the transform's (which also normalises by the flow's
own peak throughput). Before any positive throughput is seen the reward is 0.

The normalizer (Welford) path of the original transform is a *state* concern
and is not reproduced here; Orca is deployed with the normalizer off.
"""

import os


_MSS_MIN = 1e-12


def _finite(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if v != v or v in (float('inf'), float('-inf')):
        return float(default)
    return v


class RewardCalc:
    def __init__(self, delay_margin_coef: float = 1.25,
                 interval_s: float = 0.02):
        self.delay_margin_coef = float(delay_margin_coef)
        self._interval_s = max(float(interval_s), 1e-6)
        self._max_bw = 0.0
        self._max_tput = 1.0
        self._last_min_rtt_us = 0.0
        self._warmed_up = False
        self.last_components = {}

    def _delta_t(self, info: dict) -> float:
        for key in ('delta_t', 'interval_s'):
            if key in info:
                dt = _finite(info.get(key), 0.0)
                if dt > 0.0:
                    return dt
        if 'interval_ms' in info:
            dt = _finite(info.get('interval_ms'), 0.0) / 1000.0
            if dt > 0.0:
                return dt
        return self._interval_s

    def step(self, info: dict) -> float:
        throughput = max(_finite(info.get('throughput',
                                          info.get('avg_thr', 0.0))), 0.0)
        delta_t = self._delta_t(info)

        loss_rate = max(_finite(info.get('lost_rate',
                                         info.get('loss_rate', 0.0))), 0.0)
        if loss_rate <= 0.0:
            loss_bytes = max(_finite(info.get('loss_bytes',
                                              info.get('lost_bytes', 0.0))), 0.0)
            loss_rate = loss_bytes / delta_t

        avg_urtt_us = max(_finite(info.get('avg_urtt',
                                           info.get('delay_us', 0.0))), 0.0)
        srtt_raw = _finite(info.get('srtt_us', 0.0))
        srtt_ms = ((srtt_raw / 8.0) if srtt_raw > 0.0 else avg_urtt_us) / 1000.0

        min_rtt_ms = info.get('min_rtt_ms', None)
        if min_rtt_ms is None:
            min_rtt_us = max(_finite(info.get('min_rtt',
                                              info.get('min_rtt_us', 0.0))), 0.0)
            min_rtt_ms = min_rtt_us / 1000.0
        else:
            min_rtt_ms = max(_finite(min_rtt_ms), 0.0)
            min_rtt_us = min_rtt_ms * 1000.0
        self._last_min_rtt_us = min_rtt_us

        if srtt_ms > _MSS_MIN:
            delay_metric = min(1.0, self.delay_margin_coef * min_rtt_ms / srtt_ms)
        else:
            delay_metric = 1.0

        if throughput > self._max_bw:
            self._max_bw = throughput
        if throughput > self._max_tput:
            self._max_tput = throughput

        # Guard the loss term against a garbage startup measurement. On a
        # connection's first tick loss_bytes/delta_t reads a large cumulative
        # counter over a tiny interval, so loss_rate comes out enormous and the
        # -5*loss term produces a spurious -1000s reward that dominates the
        # whole episode return (orca_20260718-* runs). The first tick has no
        # valid loss baseline, so skip the loss penalty for the flow's first
        # reward; on every later tick cap loss_rate at throughput (you cannot
        # lose more than you send in an interval), which bounds the reward to
        # [-4, 1] and leaves real (sub-throughput) loss signals untouched.
        if self._warmed_up:
            loss_rate = min(loss_rate, throughput)
        else:
            loss_rate = 0.0
            self._warmed_up = True

        if self._max_bw > _MSS_MIN:
            reward = (throughput - 5.0 * loss_rate) / self._max_bw * delay_metric
        else:
            reward = 0.0

        self.last_components = {
            'reward': float(reward),
            'throughput': float(throughput),
            'loss_rate': float(loss_rate),
            'max_bw': float(self._max_bw),
            'delay_metric': float(delay_metric),
            'min_rtt_ms': float(min_rtt_ms),
            'srtt_ms': float(srtt_ms),
        }
        return float(reward)

    @property
    def max_tput(self) -> float:
        return self._max_tput

    @property
    def min_rtt_us(self) -> float:
        return self._last_min_rtt_us

    # No Kalman estimator here; expose the kernel min_rtt under the same name
    # so worker logging (reward_calc.kalman_min_rtt_us) keeps working.
    @property
    def kalman_min_rtt_us(self) -> float:
        return self._last_min_rtt_us


def make_reward_calc() -> RewardCalc:
    """Build from environment variables set by the orchestrator."""
    delay_margin_coef = _finite(
        os.environ.get('OC_ORCA_DELAY_MARGIN_COEF', 1.25), 1.25)
    interval_s = _finite(os.environ.get('SAO_INTERVAL_MS', 20.0), 20.0) / 1000.0
    return RewardCalc(delay_margin_coef=delay_margin_coef,
                      interval_s=interval_s)
