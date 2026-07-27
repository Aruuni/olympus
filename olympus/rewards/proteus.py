"""Proteus: a ground-truth-free network-utility reward for continual RL.

This plugin keeps Dreamer's per-flow reward interface and its use of the raw
connection minimum RTT, but deliberately does not read the configured link
bandwidth, scheduled base RTT, link schedule, or number of competing flows.
Every input is available from sender-side TCP telemetry, so the same reward can
label transitions in RayNet, Mininet, and a continual-learning deployment.

The reward is a scalarized proportional-fair utility::

    log((goodput_bps + eps) / goodput_reference_bps)
      - delay_weight * max(
            log(rtt / ((1 + rtt_headroom) * min_rtt)), 0)
      - loss_weight * log(1 + loss_scale * loss_fraction)
      - stall_penalty * I[goodput_bps < stall_goodput_bps]

The logarithms make throughput and RTT trade-offs relative rather than tied to
one link scale.  A small RTT headroom avoids punishing normal measurement
jitter around the observed propagation-delay floor.  Rewards are clipped only
as a final numerical guard.
"""

import math
import os

from olympus.common import runtime_config


DEFAULTS = {
    'throughput_reference_bps': 1.0e6,
    'throughput_epsilon_bps': 1.0e4,
    'rtt_headroom': 0.05,
    'delay_weight': 2.0,
    'loss_weight': 1.0,
    'loss_scale': 100.0,
    'stall_goodput_bps': 1.0e4,
    'stall_penalty': 1.0,
    'reward_clip': 10.0,
}


def _finite(value, default=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def reward_options(cfg: dict = None) -> dict:
    """Resolve this reward's numeric options from the active config."""
    cfg = cfg if cfg is not None else runtime_config.load_config()
    options = dict(DEFAULTS)
    for key, default in DEFAULTS.items():
        raw = runtime_config.reward_value(cfg, key)
        if raw is not None:
            options[key] = _finite(raw, default)
    return options


class RewardCalc:
    """Causal, per-flow reward calculator compatible with Olympus workers."""

    def __init__(self, interval_ms: float = 20.0, **options):
        resolved = dict(DEFAULTS)
        resolved.update({key: value for key, value in options.items()
                         if key in resolved and value is not None})

        self.interval_s = max(_finite(interval_ms, 20.0) / 1000.0, 1e-6)
        self.throughput_reference_bps = max(
            _finite(resolved['throughput_reference_bps'], 1.0e6), 1.0)
        self.throughput_epsilon_bps = max(
            _finite(resolved['throughput_epsilon_bps'], 1.0e4), 1.0)
        self.rtt_headroom = max(_finite(resolved['rtt_headroom'], 0.05), 0.0)
        self.delay_weight = max(_finite(resolved['delay_weight'], 2.0), 0.0)
        self.loss_weight = max(_finite(resolved['loss_weight'], 1.0), 0.0)
        self.loss_scale = max(_finite(resolved['loss_scale'], 100.0), 0.0)
        self.stall_goodput_bps = max(
            _finite(resolved['stall_goodput_bps'], 1.0e4), 0.0)
        self.stall_penalty = max(
            _finite(resolved['stall_penalty'], 1.0), 0.0)
        self.reward_clip = max(_finite(resolved['reward_clip'], 10.0), 0.0)

        self._max_tput = 1.0  # bytes/s, matching the existing reward contract
        self._last_srtt_us = 0.0
        self._last_min_rtt_us = 0.0
        self._warmed_up = False
        self.last_components = {}

    @staticmethod
    def _read_min_rtt_us(info: dict) -> float:
        value = _finite(info.get('min_rtt', info.get('min_rtt_us', 0.0)))
        return max(value, 0.0)

    @staticmethod
    def _srtt_us(info: dict, fallback_us: float = 0.0) -> float:
        # Linux exports srtt_us with three fractional bits.
        raw = _finite(info.get('srtt_us', 0.0))
        value = raw / 8.0 if raw > 0.0 else _finite(fallback_us)
        return max(value, 0.0)

    def _delta_t(self, info: dict) -> float:
        for key in ('delta_t', 'interval_s'):
            value = _finite(info.get(key, 0.0))
            if value > 0.0:
                return value
        interval_ms = _finite(info.get('interval_ms', 0.0))
        return interval_ms / 1000.0 if interval_ms > 0.0 else self.interval_s

    def _loss_fraction(self, info: dict, throughput_bytes_s: float) -> float:
        # A backend may provide an already dimensionless fraction explicitly.
        if 'loss_fraction' in info:
            return min(max(_finite(info.get('loss_fraction')), 0.0), 1.0)

        loss_rate = 0.0
        for key in ('loss_rate', 'lost_rate', 'loss_ratio'):
            if key in info:
                loss_rate = max(_finite(info.get(key)), 0.0)
                break
        else:
            # Kernel loss_bytes is an interval counter, but its first read can
            # contain a cumulative startup value.  Ignore that first fallback
            # sample just as the Orca reward does.
            if not self._warmed_up:
                return 0.0
            loss_bytes = max(_finite(
                info.get('loss_bytes', info.get('lost_bytes', 0.0))), 0.0)
            loss_rate = loss_bytes / self._delta_t(info)

        total_rate = throughput_bytes_s + loss_rate
        if total_rate <= 0.0:
            return 0.0
        return min(max(loss_rate / total_rate, 0.0), 1.0)

    def step(self, info: dict) -> float:
        throughput_bytes_s = max(_finite(
            info.get('throughput', info.get('avg_thr', 0.0))), 0.0)
        goodput_bps = 8.0 * throughput_bytes_s
        avg_urtt_us = max(_finite(
            info.get('avg_urtt', info.get('delay_us', 0.0))), 0.0)
        srtt_us = self._srtt_us(info, fallback_us=avg_urtt_us)
        current_rtt_us = avg_urtt_us if avg_urtt_us > 0.0 else srtt_us
        min_rtt_us = self._read_min_rtt_us(info)
        loss_fraction = self._loss_fraction(info, throughput_bytes_s)
        self._warmed_up = True

        self._max_tput = max(self._max_tput, throughput_bytes_s)
        self._last_srtt_us = srtt_us
        self._last_min_rtt_us = min_rtt_us

        throughput_utility = math.log(
            (goodput_bps + self.throughput_epsilon_bps)
            / self.throughput_reference_bps)

        delay_excess_log = 0.0
        min_rtt_closeness = 0.0
        if current_rtt_us > 0.0 and min_rtt_us > 0.0:
            min_rtt_closeness = min(
                current_rtt_us / min_rtt_us,
                min_rtt_us / current_rtt_us,
            )
            delay_threshold_us = (1.0 + self.rtt_headroom) * min_rtt_us
            delay_excess_log = max(
                math.log(current_rtt_us / delay_threshold_us), 0.0)

        delay_cost = self.delay_weight * delay_excess_log
        loss_cost = self.loss_weight * math.log1p(
            self.loss_scale * loss_fraction)
        stalled = goodput_bps < self.stall_goodput_bps
        stall_cost = self.stall_penalty if stalled else 0.0
        unclipped = throughput_utility - delay_cost - loss_cost - stall_cost

        if self.reward_clip > 0.0:
            reward = min(max(unclipped, -self.reward_clip), self.reward_clip)
        else:
            reward = unclipped

        self.last_components = {
            # Compatibility names used by Dreamer-oriented diagnostics.
            'base': throughput_utility,
            'min_rtt_proximity': -delay_cost,
            'min_rtt_closeness': min_rtt_closeness,
            'min_rtt_error_us': (
                abs(current_rtt_us - min_rtt_us)
                if current_rtt_us > 0.0 and min_rtt_us > 0.0
                else math.inf),
            'min_rtt_us': min_rtt_us,
            'rtt_ref_us': min_rtt_us,
            # CRL-specific audit fields.
            'goodput_bps': goodput_bps,
            'current_rtt_us': current_rtt_us,
            'throughput_utility': throughput_utility,
            'delay_excess_log': delay_excess_log,
            'delay_cost': delay_cost,
            'loss_fraction': loss_fraction,
            'loss_cost': loss_cost,
            'stalled': float(stalled),
            'stall_cost': stall_cost,
            'unclipped': unclipped,
        }
        return float(reward)

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
        """Raw observed minimum RTT under the legacy logging property."""
        return self._last_min_rtt_us


def make_reward_calc() -> RewardCalc:
    cfg = runtime_config.load_config()
    interval_ms = _finite(
        os.environ.get('SAO_INTERVAL_MS', os.environ.get('OC_INTERVAL_MS', 20.0)),
        20.0,
    )
    return RewardCalc(interval_ms=interval_ms, **reward_options(cfg))


__all__ = [
    'DEFAULTS',
    'RewardCalc',
    'make_reward_calc',
    'reward_options',
]
