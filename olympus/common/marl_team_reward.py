"""Shared multi-agent team-reward helpers.

These functions express the attached average-throughput fairness metric and the
shared (team) Tempest reward. They live in ``common`` so that both the
MA-Dreamer algorithm (``algorithms/ma_dreamer/model.py``) and the multi-agent
fairness reward (``rewards/tempest_fairness_ma.py``) can use them without a
reward importing from an algorithm (sibling packages must not import each
other; shared helpers belong here).

Both helpers accept NumPy arrays or torch tensors and reduce the final
dimension, so the same code serves the per-step worker reward and the batched
learner-side computation.
"""

import numpy as np
import torch


def r_fair_from_avg_throughput(avg_throughput, mask=None, eps: float = 1e-9):
    """Compute the attached unfairness metric from average throughput.

        R_fair = sqrt(
            sum_i (x_i - mean(x))^2
            / (n * (sum_i x_i)^2)
        )

    Lower is fairer and equal allocation is zero. Inactive agents are excluded
    through ``mask``. The function supports NumPy arrays and torch tensors and
    reduces the final dimension.
    """
    if isinstance(avg_throughput, torch.Tensor):
        x = avg_throughput.float().clamp_min(0.0)
        valid = torch.ones_like(x) if mask is None else mask.to(x).clamp(0.0, 1.0)
        n = valid.sum(dim=-1)
        total = (x * valid).sum(dim=-1)
        mean = total / n.clamp_min(1.0)
        numerator = (((x - mean.unsqueeze(-1)) * valid) ** 2).sum(dim=-1)
        denominator = n * total.square()
        value = torch.sqrt(numerator / denominator.clamp_min(float(eps)))
        good = (n > 1.0) & (total > float(eps))
        return torch.where(good, value, torch.zeros_like(value))

    x = np.asarray(avg_throughput, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    valid = np.ones_like(x) if mask is None else np.asarray(mask, dtype=np.float64)
    valid = np.clip(valid, 0.0, 1.0)
    n = valid.sum(axis=-1)
    total = (x * valid).sum(axis=-1)
    mean = total / np.maximum(n, 1.0)
    numerator = np.square((x - np.expand_dims(mean, -1)) * valid).sum(axis=-1)
    denominator = n * np.square(total)
    value = np.sqrt(numerator / np.maximum(denominator, float(eps)))
    return np.where((n > 1.0) & (total > float(eps)), value, 0.0)


def shared_team_reward(local_rewards, avg_throughput, mask,
                       fairness_weight: float = 25.0,
                       fairness_cap: float = 1.0):
    """Mean Tempest reward minus the attached average-throughput unfairness."""
    if isinstance(local_rewards, torch.Tensor):
        valid = mask.to(local_rewards)
        base = (local_rewards * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        fairness = r_fair_from_avg_throughput(avg_throughput, valid)
        cost = float(fairness_weight) * fairness.clamp(max=float(fairness_cap))
        return base - cost, fairness, cost

    valid = np.asarray(mask, dtype=np.float64)
    rewards = np.asarray(local_rewards, dtype=np.float64)
    base = (rewards * valid).sum(axis=-1) / np.maximum(valid.sum(axis=-1), 1.0)
    fairness = r_fair_from_avg_throughput(avg_throughput, valid)
    cost = float(fairness_weight) * np.minimum(fairness, float(fairness_cap))
    return base - cost, fairness, cost


def over_share_fraction(avg_throughput, mask=None, eps: float = 1e-9):
    """Each agent's throughput above the active-agent mean, as a fraction of
    the total: ``max(0, x_i - mean(x)) / sum(x)``.

    Zero for agents at or below the mean, for inactive agents, and whenever
    fewer than two agents are active. Supports NumPy arrays and torch tensors;
    the trailing dimension is the agent axis and is preserved.
    """
    if isinstance(avg_throughput, torch.Tensor):
        x = avg_throughput.float().clamp_min(0.0)
        valid = torch.ones_like(x) if mask is None else mask.to(x).clamp(0.0, 1.0)
        x = x * valid
        n = valid.sum(dim=-1)
        total = x.sum(dim=-1)
        mean = total / n.clamp_min(1.0)
        over = (x - mean.unsqueeze(-1)).clamp_min(0.0) / total.clamp_min(
            float(eps)).unsqueeze(-1)
        good = ((n > 1.0) & (total > float(eps))).unsqueeze(-1)
        return torch.where(good, over * valid, torch.zeros_like(over))

    x = np.clip(np.asarray(avg_throughput, dtype=np.float64), 0.0, None)
    valid = np.ones_like(x) if mask is None else np.clip(
        np.asarray(mask, dtype=np.float64), 0.0, 1.0)
    x = x * valid
    n = valid.sum(axis=-1)
    total = x.sum(axis=-1)
    mean = total / np.maximum(n, 1.0)
    over = np.clip(x - np.expand_dims(mean, -1), 0.0, None) / np.expand_dims(
        np.maximum(total, float(eps)), -1)
    good = np.expand_dims((n > 1.0) & (total > float(eps)), -1)
    return np.where(good, over * valid, 0.0)


def per_agent_team_reward(local_rewards, avg_throughput, mask,
                          team_alpha: float = 0.5,
                          fairness_weight: float = 25.0,
                          fairness_cap: float = 1.0,
                          over_weight: float = 50.0):
    """Credit-assigned team reward: each agent keeps a stake in its own local
    reward and only the flow above its fair share pays a targeted penalty.

        R_i = team_alpha * r_i + (1 - team_alpha) * mean(r)
              - fairness_weight * min(R_fair, fairness_cap)
              - over_weight * max(0, x_i - mean(x)) / sum(x)

    Returns ``(per_agent_rewards, r_fair, fairness_cost)`` where the rewards
    keep the trailing agent dimension (zeroed for inactive agents) and the
    fairness terms reduce it, matching ``shared_team_reward``.
    """
    alpha = float(team_alpha)
    if isinstance(local_rewards, torch.Tensor):
        valid = mask.to(local_rewards)
        team_mean = (local_rewards * valid).sum(dim=-1) / valid.sum(
            dim=-1).clamp_min(1.0)
        fairness = r_fair_from_avg_throughput(avg_throughput, valid)
        cost = float(fairness_weight) * fairness.clamp(max=float(fairness_cap))
        over = over_share_fraction(avg_throughput, valid)
        rewards = (
            alpha * local_rewards
            + (1.0 - alpha) * team_mean.unsqueeze(-1)
            - cost.unsqueeze(-1)
            - float(over_weight) * over
        ) * valid
        return rewards, fairness, cost

    valid = np.asarray(mask, dtype=np.float64)
    locals_ = np.asarray(local_rewards, dtype=np.float64)
    team_mean = (locals_ * valid).sum(axis=-1) / np.maximum(
        valid.sum(axis=-1), 1.0)
    fairness = r_fair_from_avg_throughput(avg_throughput, valid)
    cost = float(fairness_weight) * np.minimum(fairness, float(fairness_cap))
    over = over_share_fraction(avg_throughput, valid)
    rewards = (
        alpha * locals_
        + (1.0 - alpha) * np.expand_dims(team_mean, -1)
        - np.expand_dims(cost, -1)
        - float(over_weight) * over
    ) * valid
    return rewards, fairness, cost
