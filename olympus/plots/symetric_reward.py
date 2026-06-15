#!/usr/bin/env python3
"""Explainer figure for the tempest_fairness_ma_symetric reward.

Four panels:
  1. Worker throughput term: flat-cap (shared reward) vs symmetric (credit).
  2. Worker RTT term, the shape the symmetric throughput term mirrors.
  3. Two-flow sweep of the learner-side training reward: the shared team
     scalar vs the per-agent credit rewards R_i.
  4. Decomposition of R_i for the 75/25 example.

The learner-side curves are computed with the actual training helpers
(shared_team_reward / per_agent_team_reward), not re-derived math.

Usage:
    python olympus/plot_credit_reward.py [--out tmp/credit_reward_explainer]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from olympus.common.marl_team_reward import (  # noqa: E402
    per_agent_team_reward,
    shared_team_reward,
)

# Constants of the running example: 100 Mbps bottleneck, 2 flows, drained
# queues (RTT term = 1), base scale 25, learner defaults.
LINK = 100.0
FAIR = LINK / 2.0
SCALE = 25.0
TEAM_ALPHA = 0.5
FAIRNESS_WEIGHT = 25.0
OVER_WEIGHT = 50.0


def thr_term_shared(ratio):
    return np.minimum(ratio, 1.0)


def thr_term_credit(ratio):
    ratio = np.asarray(ratio, dtype=np.float64)
    return np.where(ratio > 0, np.minimum(ratio, 1.0 / np.maximum(ratio, 1e-9)), 0.0)


def two_flow_sweep():
    """Sweep flow A's share of a fully utilized link; flow B gets the rest."""
    share_a = np.linspace(0.0, LINK, 401)
    thr = np.stack([share_a, LINK - share_a], axis=-1)
    mask = np.ones_like(thr)

    local_shared = SCALE * thr_term_shared(thr / FAIR)
    local_credit = SCALE * thr_term_credit(thr / FAIR)

    team_old, _, _ = shared_team_reward(
        local_shared, thr, mask, fairness_weight=FAIRNESS_WEIGHT)
    team_credit, _, _ = shared_team_reward(
        local_credit, thr, mask, fairness_weight=FAIRNESS_WEIGHT)
    per_agent, _, _ = per_agent_team_reward(
        local_credit, thr, mask,
        team_alpha=TEAM_ALPHA, fairness_weight=FAIRNESS_WEIGHT,
        fairness_cap=1.0, over_weight=OVER_WEIGHT)
    return share_a, team_old, team_credit, per_agent


def credit_breakdown(share_a=75.0):
    """Stacked decomposition of R_i at a given split of the running example."""
    thr = np.asarray([share_a, LINK - share_a])
    local = SCALE * thr_term_credit(thr / FAIR)
    mean_r = local.mean()
    r_fair = np.sqrt(((thr - thr.mean()) ** 2).sum() / (2.0 * thr.sum() ** 2))
    cost = FAIRNESS_WEIGHT * min(r_fair, 1.0)
    over = np.maximum(0.0, thr - thr.mean()) / thr.sum()
    return {
        'own stake  α·rᵢ': TEAM_ALPHA * local,
        'team stake  (1−α)·mean(r)': np.full(2, (1 - TEAM_ALPHA) * mean_r),
        'shared cost  −w_fair·R_fair': np.full(2, -cost),
        'over-share  −w_over·overᵢ': -OVER_WEIGHT * over,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out', default=os.path.join('tmp', 'symetric_reward_explainer'),
        help='output path without extension (.png and .pdf are written)')
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    fig.suptitle(
        'tempest_fairness_ma_symetric — reward explainer '
        '(100 Mbps link, 2 flows, drained queues)',
        fontsize=13)

    # ── 1. Worker throughput term ────────────────────────────────────────────
    ax = axes[0, 0]
    ratio = np.linspace(0.0, 3.0, 301)
    ax.plot(ratio, SCALE * thr_term_shared(ratio), lw=2, color='steelblue',
            label='shared (old): 25·min(thr/fair, 1)')
    ax.plot(ratio, SCALE * thr_term_credit(ratio), lw=2, color='firebrick',
            label='credit (new): 25·min(thr/fair, fair/thr)')
    ax.axvline(1.0, color='grey', ls=':', lw=1)
    ax.set_ylim(0, 28)
    ax.annotate('fair share', xy=(1.0, 26.0), ha='center', fontsize=9,
                color='grey')
    ax.fill_between(ratio, SCALE * thr_term_credit(ratio),
                    SCALE * thr_term_shared(ratio),
                    where=ratio > 1.0, alpha=0.15, color='firebrick')
    ax.annotate('overshoot now\ncosts locally', xy=(2.0, 18.0), fontsize=9,
                color='firebrick', ha='center')
    ax.set_xlabel('throughput / fair share')
    ax.set_ylabel('base reward (RTT term = 1)')
    ax.set_title('1 · worker throughput term')
    ax.legend(fontsize=9, loc='lower center')
    ax.grid(alpha=0.3)

    # ── 2. Worker RTT term (the shape being mirrored) ────────────────────────
    ax = axes[0, 1]
    rtt_ratio = np.linspace(0.5, 3.0, 301)
    rtt_rate = np.minimum(1.0 / rtt_ratio, 1.0) ** 2
    ax.plot(rtt_ratio, SCALE * rtt_rate, lw=2, color='seagreen',
            label='25·min(rtt_ref/rtt, 1)²')
    ax.axvline(1.0, color='grey', ls=':', lw=1)
    ax.set_ylim(0, 28)
    ax.annotate('propagation RTT', xy=(1.0, 26.0), ha='center', fontsize=9,
                color='grey')
    ax.annotate('queueing delay\npenalized', xy=(2.1, 12.0), fontsize=9,
                color='seagreen', ha='center')
    ax.set_xlabel('RTT / propagation RTT')
    ax.set_ylabel('base reward (thr term = 1)')
    ax.set_title('2 · worker RTT term — same peak-at-target shape')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 3. Learner-side training reward, two-flow sweep ──────────────────────
    ax = axes[1, 0]
    share_a, team_old, team_credit, per_agent = two_flow_sweep()
    ax.plot(share_a, team_old, lw=2, color='steelblue', ls='--',
            label='old reward: shared scalar, flat-cap locals')
    ax.plot(share_a, team_credit, lw=2, color='firebrick',
            label='credit (default): shared scalar, symmetric locals')
    ax.plot(share_a, per_agent[:, 0], lw=1, color='firebrick', alpha=0.45,
            label='optional per_agent_credit: flow A / flow B')
    ax.plot(share_a, per_agent[:, 1], lw=1, color='darkorange', alpha=0.45)
    ax.axvline(FAIR, color='grey', ls=':', lw=1)
    # Mark the 75/25 worked example on the two shared-scalar curves.
    idx = int(np.argmin(np.abs(share_a - 75.0)))
    ax.scatter([75, 75], [team_old[idx], team_credit[idx]],
               color=['steelblue', 'firebrick'], zorder=5, s=30)
    ax.annotate('75/25 example', xy=(75, team_old[idx]),
                xytext=(80, team_old[idx] + 4), fontsize=9, color='grey',
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.8))
    ax.axhline(0.0, color='black', lw=0.6)
    ax.set_xlabel("flow A's share of the link (Mbps);  flow B gets the rest")
    ax.set_ylabel('training reward per step')
    ax.set_title('3 · learner reward: both flows get one scalar (Astraea-style)')
    ax.legend(fontsize=8.5, loc='lower center')
    ax.grid(alpha=0.3)

    # ── 4. Decomposition of R_i at 75/25 ─────────────────────────────────────
    ax = axes[1, 1]
    parts = credit_breakdown(75.0)
    labels = ['greedy flow (75)', 'starved flow (25)']
    x = np.arange(2)
    colors = ['firebrick', 'darkorange', 'steelblue', 'dimgrey']
    pos = np.zeros(2)
    neg = np.zeros(2)
    for (name, values), color in zip(parts.items(), colors):
        bottom = np.where(values >= 0, pos, neg)
        ax.bar(x, values, 0.55, bottom=bottom, label=name, color=color,
               alpha=0.85)
        pos += np.maximum(values, 0)
        neg += np.minimum(values, 0)
    totals = sum(parts.values())
    ax.scatter(x, totals, color='black', zorder=5, s=45, marker='D',
               label='total R_i')
    for xi, total in zip(x, totals):
        ax.annotate(f'{total:.2f}', xy=(xi, total), xytext=(xi + 0.12, total),
                    fontsize=10, fontweight='bold', va='center')
    ax.axhline(0.0, color='black', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('reward contribution per step')
    ax.set_title('4 · R_i decomposition at 75/25 (optional per_agent_credit)')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=160)
        print(f'saved {args.out}.{ext}')


if __name__ == '__main__':
    main()
