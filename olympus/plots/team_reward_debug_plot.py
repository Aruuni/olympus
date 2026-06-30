"""Debug plot for MA team-reward decomposition from per-agent traces."""

import os

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

from olympus.plots.multi_flow_episode_plot import (
    _COLORS,
    _aligned_matrix,
    load_agent_traces,
)


def _team_reward_plot_path(episode_plot_path):
    root, ext = os.path.splitext(str(episode_plot_path))
    return f'{root}_team_reward_debug{ext or ".pdf"}'


def _r_fair(values):
    x = np.clip(values.astype(np.float64), 0.0, None)
    n = float(x.shape[0])
    total = x.sum(axis=0)
    mean = total / max(n, 1.0)
    numerator = np.square(x - mean).sum(axis=0)
    denominator = n * np.square(total)
    out = np.zeros_like(total, dtype=np.float64)
    valid = (n > 1.0) & (total > 1e-9)
    out[valid] = np.sqrt(
        numerator[valid] / np.maximum(denominator[valid], 1e-9))
    return out.astype(np.float32)


def _over_share(values):
    x = np.clip(values.astype(np.float64), 0.0, None)
    n = float(x.shape[0])
    total = x.sum(axis=0)
    mean = total / max(n, 1.0)
    over = np.zeros_like(x, dtype=np.float64)
    valid = (n > 1.0) & (total > 1e-9)
    over[:, valid] = np.maximum(
        x[:, valid] - mean[valid], 0.0) / total[valid]
    return over.astype(np.float32)


def _team_rewards(local_reward, avg_thr_mbps, *,
                  per_agent_credit=True,
                  team_alpha=0.5,
                  fairness_weight=25.0,
                  fairness_cap=1.0,
                  over_weight=50.0):
    r_fair = _r_fair(avg_thr_mbps)
    fair_cost = float(fairness_weight) * np.minimum(
        r_fair, float(fairness_cap))
    team_mean = local_reward.mean(axis=0)

    if per_agent_credit:
        over = _over_share(avg_thr_mbps)
        rewards = (
            float(team_alpha) * local_reward
            + (1.0 - float(team_alpha)) * team_mean
            - fair_cost
            - float(over_weight) * over
        )
    else:
        over = np.zeros_like(local_reward, dtype=np.float32)
        shared = team_mean - fair_cost
        rewards = np.repeat(shared[np.newaxis, :], local_reward.shape[0], axis=0)
    return rewards.astype(np.float32), r_fair, fair_cost, over


def _plot_agents(ax, t, values, traces, ylabel, title=None,
                 zero_line=False, linewidth=0.9):
    for index, item in enumerate(traces):
        agent_id = item['agent_id']
        ax.plot(
            t,
            values[index],
            color=_COLORS[index % len(_COLORS)],
            linewidth=linewidth,
            alpha=0.9,
            label=f'agent {agent_id}',
        )
    if zero_line:
        ax.axhline(0.0, color='black', linewidth=0.6, alpha=0.45)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9, loc='left')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85)


def plot(state_log_path, output, title=None, n_agents=None, trim_tail_s=5.0,
         reward_config=None):
    traces = load_agent_traces(
        state_log_path, n_agents=n_agents, trim_tail_s=trim_tail_s)
    if len(traces) < 2:
        print(f'[team_reward_plot] need >=2 traces for: {state_log_path}',
              flush=True)
        return None

    t, local_reward = _aligned_matrix(traces, 'reward')
    _, avg_thr_mbps = _aligned_matrix(traces, 'avg_thr_mbps')
    _, cwnd = _aligned_matrix(traces, 'cwnd')
    _, act_mu = _aligned_matrix(traces, 'act_mu')
    if len(t) == 0:
        print(f'[team_reward_plot] no aligned rows for: {state_log_path}',
              flush=True)
        return None

    reward_config = dict(reward_config or {})
    per_agent_credit = bool(reward_config.get('per_agent_credit', False))
    team_rewards, r_fair, fair_cost, over = _team_rewards(
        local_reward,
        avg_thr_mbps,
        per_agent_credit=per_agent_credit,
        team_alpha=float(reward_config.get('team_alpha', 0.5)),
        fairness_weight=float(reward_config.get('fairness_weight', 25.0)),
        fairness_cap=float(reward_config.get('fairness_cap', 1.0)),
        over_weight=float(reward_config.get('over_weight', 50.0)),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig, axes = plt.subplots(7, 1, figsize=(14, 20), sharex=True)
    if title:
        fig.suptitle(f'{title}  team reward debug',
                     fontsize=12, fontweight='bold')
    else:
        fig.suptitle('Team reward debug', fontsize=12, fontweight='bold')

    _plot_agents(
        axes[0], t, local_reward, traces,
        'local reward', 'Worker local rewards', zero_line=True)
    _plot_agents(
        axes[1], t, team_rewards, traces,
        'team reward', 'Learner-side team reward target', zero_line=True)

    ax = axes[2]
    ax.plot(t, r_fair, color='#4878cf', linewidth=1.0, label='R_fair')
    ax2 = ax.twinx()
    ax2.plot(t, fair_cost, color='#e15759', linewidth=1.0,
             label='fairness cost')
    ax.set_ylabel('R_fair', fontsize=9, color='#4878cf')
    ax2.set_ylabel('cost', fontsize=9, color='#e15759')
    ax.tick_params(axis='y', labelcolor='#4878cf')
    ax2.tick_params(axis='y', labelcolor='#e15759')
    ax.set_title('Fairness penalty', fontsize=9, loc='left')
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines],
              fontsize=7, loc='upper right', framealpha=0.85)

    _plot_agents(
        axes[3], t, over, traces,
        'over-share', 'Per-agent over-share fraction', zero_line=True)
    _plot_agents(
        axes[4], t, avg_thr_mbps, traces,
        'Mbps', 'Throughput used by learner reward')
    _plot_agents(
        axes[5], t, cwnd, traces,
        'pkts', 'CWND')
    _plot_agents(
        axes[6], t, act_mu, traces,
        'actor mu', 'Actor mean action', zero_line=True)

    axes[-1].set_xlabel('Episode time (s)')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'[team_reward_plot] saved -> {output}', flush=True)
    return output


__all__ = ['plot', '_team_reward_plot_path']
