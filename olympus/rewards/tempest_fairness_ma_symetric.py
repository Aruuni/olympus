"""Fair-share Tempest reward with a symmetric throughput term.

One change versus ``tempest_fairness_ma``: the worker-side throughput factor
is symmetric around the fair share — ``min(thr / fair_share, fair_share /
thr)`` — peaking at exactly the fair share and falling off on both sides, the
same shape the RTT term has around the propagation RTT. Overshooting
therefore costs the flow locally instead of merely earning nothing extra.

The learner combines the local rewards exactly like ``tempest_fairness_ma``
(and Astraea): one shared team scalar, ``mean(r) - fairness_weight *
min(R_fair, cap)``, copied to every agent.

Optional: set ``reward.per_agent_credit: true`` to instead train each agent
on its own credit-assigned reward

    R_i = team_alpha * r_i + (1 - team_alpha) * mean(r)
          - fairness_weight * min(R_fair, fairness_cap)
          - over_weight * max(0, x_i - mean(x)) / sum(x)

where only the flow above its fair share pays the targeted ``over_weight``
penalty (knobs: ``reward.team_alpha``, ``reward.over_weight``; the global
world model's reward head becomes per-agent, so checkpoints are not
weight-compatible with the shared-scalar mode).

Select with ``runtime.reward: tempest_fairness_ma_symetric``.
"""

from olympus.rewards.tempest_fairness_ma import (
    RewardCalc as FairShareRewardCalc,
    calc_kwargs_from_env,
)
from olympus.common.marl_team_reward import (
    over_share_fraction,
    per_agent_team_reward,
    r_fair_from_avg_throughput,
)


class RewardCalc(FairShareRewardCalc):
    def _thr_term(self, avg_thr: float, fair_bw: float) -> float:
        """Symmetric around the fair share: 1 at thr == fair_share, falling
        off ratio-wise on both sides (2x over scores like 2x under)."""
        if avg_thr <= 0.0:
            return 0.0
        return min(avg_thr / fair_bw, fair_bw / avg_thr)


def make_reward_calc():
    return RewardCalc(**calc_kwargs_from_env())


__all__ = [
    'RewardCalc',
    'make_reward_calc',
    'over_share_fraction',
    'per_agent_team_reward',
    'r_fair_from_avg_throughput',
]
