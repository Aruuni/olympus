"""
training/reward.py — episode-aware reward for Olympus option-critic agent.

Reward formula (all terms in [0, 1]):

    r = w_bw  * clip(tput / link_bw,          0, 1)   # fill the pipe
      + w_rtt * clip(rtt_floor / rtt,          0, 1)   # stay near unloaded RTT
      - w_loss * tanh(lost_pkts / 5.0)                 # penalise loss

RTT reference — two-tier:
  rtt_floor = max(
      episode_min_rtt,          # persistent minimum since flow start
      path_min_rtt_500ms,       # 500ms sliding window (tracks delay schedule changes)
  )

  The episode minimum never forgets, so a CC that builds a persistent queue
  (e.g. cubic filling the buffer) is always penalised relative to the first
  clean RTT seen.  The 500ms sliding window allows the floor to rise when the
  orchestrator genuinely increases propagation delay mid-episode.

Link capacity (bw) and baseline delay are set by the orchestrator per episode via
environment variables:
  OC_LINK_BW     — link bandwidth in Mbps  (default 100)

Reward weights (optional):
  OC_REWARD_W_BW   — default 0.5
  OC_REWARD_W_RTT  — default 0.5
  OC_REWARD_W_LOSS — default 0.5
"""

from __future__ import annotations

import json
import math
import os
import time


class RewardCalc:
    """
    Stateful reward calculator for one flow.

    Call `step(rtt_us, delivery_rate, lost)` once per state frame to get the
    instantaneous reward.  The object maintains the RTT sliding-window state.

    If a link_schedule is provided (list of {t, bw?, delay?} dicts sorted by t,
    plus episode_start as a monotonic timestamp), the bandwidth normaliser updates
    automatically to match whatever the Mininet link is doing at that moment.
    """

    def __init__(
        self,
        link_bw_mbps:   float = 100.0,
        w_bw:   float = 0.5,
        w_rtt:  float = 0.5,
        w_loss: float = 0.5,
        link_schedule:  list  = None,   # [{t, bw?, delay?}, ...] sorted by t
        episode_start:  float = None,   # time.monotonic() at iperf start
    ) -> None:
        self._base_bw_bytes_s = link_bw_mbps * 1e6 / 8.0
        self.link_bw_bytes_s  = self._base_bw_bytes_s

        self.w_bw   = w_bw
        self.w_rtt  = w_rtt
        self.w_loss = w_loss

        # BW schedule: list of (t_offset_s, bw_bytes_s) sorted ascending
        self._bw_schedule:    list = []
        # Delay schedule: list of (t_offset_s, prop_rtt_us) sorted ascending
        # prop_rtt_us = 2 × one-way delay in µs — used as RTT floor when the
        # path delay is genuinely increased by the orchestrator.
        self._delay_schedule: list = []
        self._base_prop_rtt_us: int = 0  # set from OC_LINK_DELAY if available

        if link_schedule and episode_start is not None:
            self._episode_start = episode_start
            for entry in link_schedule:
                if 'bw' in entry:
                    self._bw_schedule.append(
                        (float(entry['t']), float(entry['bw']) * 1e6 / 8.0)
                    )
                if 'delay' in entry:
                    self._delay_schedule.append(
                        (float(entry['t']), int(float(entry['delay']) * 1000))
                    )
        else:
            self._episode_start = None

        # Episode minimum RTT — never forgets, so persistent queue build-up
        # (e.g. cubic filling the buffer) is always penalised.
        self._rtt_floor_ep: int = 10 ** 9

    # ── Schedule tracking ─────────────────────────────────────────────────────

    def _active_value(self, schedule: list, base: float) -> float:
        """Return the currently active value from a sorted (t, val) schedule."""
        if not schedule or self._episode_start is None:
            return base
        elapsed = time.monotonic() - self._episode_start
        active = base
        for t_offset, val in schedule:
            if elapsed >= t_offset:
                active = val
            else:
                break
        return active

    def _update_link_bw(self) -> None:
        self.link_bw_bytes_s = self._active_value(
            self._bw_schedule, self._base_bw_bytes_s)

    # ── RTT reference ─────────────────────────────────────────────────────────

    def _update_rtt_floor(self, rtt_us: int) -> int:
        """
        RTT floor = max(episode_min, scheduled_prop_rtt).

        The episode minimum ensures persistent queue build-up is penalised.
        The scheduled prop RTT ensures the floor rises correctly when the
        orchestrator genuinely increases propagation delay mid-episode —
        so the reward is not incorrectly penalising a higher-delay phase.
        """
        if rtt_us < self._rtt_floor_ep:
            self._rtt_floor_ep = rtt_us
        sched_prop = int(self._active_value(
            self._delay_schedule, self._base_prop_rtt_us))
        return max(self._rtt_floor_ep, sched_prop)

    # ── Main interface ────────────────────────────────────────────────────────

    def step(self, rtt_us: int, delivery_rate: int, lost: int) -> float:
        """
        Compute instantaneous reward for one state frame.

        Parameters
        ----------
        rtt_us        : smoothed RTT in microseconds (from TCP_INFO srtt_us)
        delivery_rate : kernel delivery-rate estimate in bytes/s
        lost          : number of unrecovered lost packets (TCP_INFO lost)

        Returns
        -------
        float reward in roughly [-0.5, 1.0] with default weights.
        """
        rtt_us = max(rtt_us, 1)

        self._update_link_bw()
        rtt_floor = self._update_rtt_floor(rtt_us)

        # bandwidth utilisation: how much of the link we are filling
        bw_term = min(delivery_rate / self.link_bw_bytes_s, 1.0)

        # queuing-delay term: 1 = no queue, <1 = queue is building
        # rtt_floor is the episode-minimum RTT (never forgets persistent queues)
        rtt_term = min(rtt_floor / rtt_us, 1.0)

        # loss penalty: 0 at no loss, saturates to 1 around lost≈15 pkts
        loss_term = math.tanh(max(lost, 0) / 5.0)

        return (self.w_bw * bw_term
                + self.w_rtt * rtt_term
                - self.w_loss * loss_term)


def make_reward_calc() -> RewardCalc:
    """
    Build a RewardCalc from environment variables set by the orchestrator.

    OC_LINK_BW        — initial link bandwidth in Mbps (default 100)
    OC_EPISODE_START  — time.monotonic() at iperf start (float, set by orchestrator)
    OC_LINK_SCHEDULE  — JSON list of {t, bw?, delay?} dicts (set by orchestrator)
    OC_REWARD_W_BW    — default 0.5
    OC_REWARD_W_RTT   — default 0.5
    OC_REWARD_W_LOSS  — default 0.5
    """
    schedule_raw = os.environ.get('OC_LINK_SCHEDULE', '')
    link_schedule = json.loads(schedule_raw) if schedule_raw else []

    episode_start_raw = os.environ.get('OC_EPISODE_START', '')
    episode_start = float(episode_start_raw) if episode_start_raw else None

    calc = RewardCalc(
        link_bw_mbps   = float(os.environ.get('OC_LINK_BW',      '100.0')),
        w_bw           = float(os.environ.get('OC_REWARD_W_BW',  '0.5')),
        w_rtt          = float(os.environ.get('OC_REWARD_W_RTT', '0.5')),
        w_loss         = float(os.environ.get('OC_REWARD_W_LOSS', '0.5')),
        link_schedule  = link_schedule,
        episode_start  = episode_start,
    )
    # Base propagation RTT in µs: 2 × one-way delay
    link_delay_ms = float(os.environ.get('OC_LINK_DELAY', '0.0'))
    calc._base_prop_rtt_us = int(link_delay_ms * 1000)
    return calc
