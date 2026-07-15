"""Networks and math for MA-Dreamer shared imagination.

The implementation follows the central mechanism from MA-Dreamer:

* A parameter-shared local world model maps each flow's Tempest observation to
  a local RSSM latent. The local model and actor are sufficient for execution.
* A training-only global world model consumes the masked joint observation and
  joint action. It predicts each local latent so imagined joint trajectories
  can be translated back into the input space of the decentralized policies.
* Policy and value learning use rollouts from the global model, keeping every
  agent on one consistent imagined trajectory.

The congestion-control agents are homogeneous, so the paper's per-agent world
models and policies share parameters here. Each worker still maintains its own
recurrent local latent state.
"""

from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.common.action_plugins import (
    current_action_meta,
)
from olympus.common.state_plugins import (
    assert_state_compatible,
    current_state_name,
    load_state_module,
)
from olympus.algorithms.dreamer_v3.model import (
    ACTION_DIM,
    Actor,
    ContinueHead,
    Critic,
    Decoder,
    Encoder,
    MLP,
    RSSM,
    ReturnEMA,
    RewardHead,
    TwoHot,
    WorldModelInfo,
    actor_log_prob,
    kl_categorical,
    lambda_return,
    soft_update,
    symlog,
    symexp,
)
STATE_NAME = current_state_name(default='tempest')
_STATE_PLUGIN = load_state_module('ma_dreamer', STATE_NAME)
STATE_DIM = int(_STATE_PLUGIN.STATE_DIM)
STATE_FEATURES = list(_STATE_PLUGIN.STATE_FEATURES)
STATE_FEATURE_VERSION = str(_STATE_PLUGIN.STATE_FEATURE_VERSION)
normalize_state = _STATE_PLUGIN.normalize_state


def reset_state(initial_rtt_us: float = None) -> None:
    """Reset whichever per-flow state tracker the configured plugin owns."""
    for name in ('reset_tempest_kalman', 'reset_kalman', 'reset_oracle'):
        reset = getattr(_STATE_PLUGIN, name, None)
        if reset is not None:
            reset(initial_rtt_us)
            return


# Backward-compatible name used by older workers/importers.
reset_tempest_kalman = reset_state

Experience = namedtuple(
    'Experience',
    [
        'state',
        'action',
        'reward',
        'next_state',
        'avg_throughput',
        'agent_id',
        'group_id',
        'group_step',
        'done',
        'traj_id',
        'step_in_traj',
    ],
)

GlobalWorldModelInfo = namedtuple(
    'GlobalWorldModelInfo',
    [
        'loss',
        'recon',
        'reward',
        'cont',
        'kl_dyn',
        'kl_rep',
        'agent_h',
        'agent_z',
    ],
)


def state_meta() -> dict:
    return {
        'state_name': STATE_NAME,
        'state_dim': STATE_DIM,
        'state_feature_version': STATE_FEATURE_VERSION,
        'state_features': STATE_FEATURES,
        'action_meta': current_action_meta(),
    }


def assert_checkpoint_state_compatible(ckpt: dict, source='checkpoint') -> None:
    assert_state_compatible(state_meta(), ckpt, source=source)


# The team-reward helpers live in common/ so the multi-agent fairness reward
# can use them without a reward importing from an algorithm. Re-exported here
# (see __all__) so the learner and tests keep importing them from this module.
from olympus.common.marl_team_reward import (  # noqa: E402
    per_agent_team_reward,
    r_fair_from_avg_throughput,
    shared_team_reward,
)


class LocalWorldModel(nn.Module):
    """Tempest observation model used independently by every flow."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=256, embed_dim=128, h_dim=256,
                 latent_groups=8, latent_classes=8, reward_bins=255,
                 reward_low=-20.0, reward_high=20.0):
        super().__init__()
        self.encoder = Encoder(state_dim, embed_dim, hidden)
        self.rssm = RSSM(
            embed_dim, action_dim, latent_groups, latent_classes, h_dim, hidden)
        self.latent_dim = latent_groups * latent_classes
        self.decoder = Decoder(h_dim, self.latent_dim, state_dim, hidden)
        self.reward_head = RewardHead(
            h_dim, self.latent_dim, reward_bins, hidden)
        self.continue_head = ContinueHead(h_dim, self.latent_dim, hidden)
        self.twohot_reward = TwoHot(reward_low, reward_high, reward_bins)
        self.h_dim = h_dim
        self.action_dim = action_dim

    def loss(self, states, actions, rewards, dones, mask,
             free_bits=1.0, kl_balance=0.8, kl_weight=1.0,
             reward_weight=1.0, continue_weight=1.0, recon_weight=1.0):
        embeds = self.encoder(states)
        out = self.rssm.observe(embeds, actions)
        h_t = out['h'].transpose(0, 1)
        z_t = out['z'].transpose(0, 1)
        prior = out['prior'].transpose(0, 1)
        post = out['post'].transpose(0, 1)

        recon = self.decoder(h_t, z_t)
        recon_per = ((recon - symlog(states)) ** 2).mean(dim=-1)
        recon_loss = (recon_per * mask).sum() / mask.sum().clamp_min(1.0)

        reward_logits = self.reward_head(h_t, z_t)
        reward_target = self.twohot_reward.encode(symlog(rewards))
        reward_per = -(
            reward_target * F.log_softmax(reward_logits, dim=-1)).sum(dim=-1)
        reward_loss = (reward_per * mask).sum() / mask.sum().clamp_min(1.0)

        continue_logits = self.continue_head(h_t, z_t)
        continue_per = F.binary_cross_entropy_with_logits(
            continue_logits, 1.0 - dones, reduction='none')
        continue_loss = (
            continue_per * mask).sum() / mask.sum().clamp_min(1.0)

        kl_per, kl_dyn, kl_rep = kl_categorical(
            post, prior, free_bits, kl_balance)
        kl_loss = (kl_per * mask).sum() / mask.sum().clamp_min(1.0)

        loss = (
            recon_weight * recon_loss
            + reward_weight * reward_loss
            + continue_weight * continue_loss
            + kl_weight * kl_loss
        )
        # Keep these as detached on-device scalars; the learner only reads
        # them on logging steps, so we avoid a GPU->CPU sync every update.
        info = WorldModelInfo(
            loss=loss,
            recon=recon_loss.detach(),
            reward=reward_loss.detach(),
            cont=continue_loss.detach(),
            kl_dyn=kl_dyn.detach(),
            kl_rep=kl_rep.detach(),
        )
        return loss, info, h_t.detach(), z_t.detach()


class PerAgentRewardHead(nn.Module):
    """Twohot reward logits for every agent from the joint latent."""

    def __init__(self, h_dim: int, latent_dim: int, n_agents: int,
                 n_bins: int, hidden: int = 256):
        super().__init__()
        self.n_agents = int(n_agents)
        self.n_bins = int(n_bins)
        self.net = MLP(
            h_dim + latent_dim, hidden, self.n_agents * self.n_bins,
            n_layers=2)

    def forward(self, h, z):
        out = self.net(torch.cat([h, z], dim=-1))
        return out.reshape(*out.shape[:-1], self.n_agents, self.n_bins)


class GlobalWorldModel(nn.Module):
    """Training-only joint RSSM and global-to-local latent translator.

    ``reward_dim=1`` keeps the original shared team-reward head; setting it to
    ``max_agents`` predicts one reward per agent for credit-assigned rewards
    (``tempest_fairness_ma_symetric``).
    """

    def __init__(self, max_agents: int, state_dim=STATE_DIM, hidden=256,
                 embed_dim=128, h_dim=256, latent_groups=8,
                 latent_classes=8, local_h_dim=256, local_latent_groups=8,
                 local_latent_classes=8, reward_bins=255,
                 reward_low=-20.0, reward_high=20.0, reward_dim=1):
        super().__init__()
        self.max_agents = int(max_agents)
        self.state_dim = int(state_dim)
        self.joint_state_dim = self.max_agents * self.state_dim + self.max_agents
        self.encoder = Encoder(self.joint_state_dim, embed_dim, hidden)
        self.rssm = RSSM(
            embed_dim, self.max_agents, latent_groups, latent_classes, h_dim, hidden)
        self.latent_dim = latent_groups * latent_classes
        self.decoder = Decoder(
            h_dim, self.latent_dim, self.joint_state_dim, hidden)
        self.reward_dim = int(reward_dim)
        if self.reward_dim > 1:
            self.reward_head = PerAgentRewardHead(
                h_dim, self.latent_dim, self.reward_dim, reward_bins, hidden)
        else:
            self.reward_head = RewardHead(
                h_dim, self.latent_dim, reward_bins, hidden)
        self.continue_head = ContinueHead(h_dim, self.latent_dim, hidden)
        self.twohot_reward = TwoHot(reward_low, reward_high, reward_bins)

        self.h_dim = int(h_dim)
        self.local_h_dim = int(local_h_dim)
        self.local_latent_groups = int(local_latent_groups)
        self.local_latent_classes = int(local_latent_classes)
        self.local_latent_dim = (
            self.local_latent_groups * self.local_latent_classes)
        feature_dim = self.h_dim + self.latent_dim
        self.agent_h_head = MLP(
            feature_dim, hidden, self.max_agents * self.local_h_dim, n_layers=2)
        self.agent_z_head = MLP(
            feature_dim, hidden, self.max_agents * self.local_latent_dim,
            n_layers=2)

    def pack_joint_state(self, states, agent_mask):
        flat_states = states.reshape(*states.shape[:-2], -1)
        return torch.cat([flat_states, agent_mask], dim=-1)

    def agent_latents(self, h, z, sample=True):
        features = torch.cat([h, z], dim=-1)
        prefix = features.shape[:-1]
        agent_h = self.agent_h_head(features).reshape(
            *prefix, self.max_agents, self.local_h_dim)
        logits = self.agent_z_head(features).reshape(
            *prefix,
            self.max_agents,
            self.local_latent_groups,
            self.local_latent_classes,
        )
        probs = torch.softmax(logits, dim=-1)
        probs = 0.99 * probs + 0.01 / self.local_latent_classes
        if sample:
            onehot = torch.distributions.OneHotCategorical(probs=probs).sample()
            agent_z = onehot + probs - probs.detach()
        else:
            agent_z = probs
        agent_z = agent_z.reshape(*prefix, self.max_agents, self.local_latent_dim)
        return agent_h, agent_z, logits

    def loss(self, states, actions, rewards, dones, agent_mask,
             local_h_target, local_z_target,
             free_bits=1.0, kl_balance=0.8, kl_weight=1.0,
             reward_weight=1.0, continue_weight=1.0, recon_weight=1.0,
             latent_h_weight=1.0, latent_z_weight=1.0):
        row_mask = (agent_mask.sum(dim=-1) > 0.0).float()
        joint_state = self.pack_joint_state(states, agent_mask)
        joint_action = (actions.squeeze(-1) * agent_mask)

        embeds = self.encoder(joint_state)
        out = self.rssm.observe(embeds, joint_action)
        h_t = out['h'].transpose(0, 1)
        z_t = out['z'].transpose(0, 1)
        prior = out['prior'].transpose(0, 1)
        post = out['post'].transpose(0, 1)

        recon = self.decoder(h_t, z_t)
        recon_per = ((recon - symlog(joint_state)) ** 2).mean(dim=-1)
        recon_loss = (
            recon_per * row_mask).sum() / row_mask.sum().clamp_min(1.0)

        reward_logits = self.reward_head(h_t, z_t)
        reward_target = self.twohot_reward.encode(symlog(rewards))
        reward_per = -(
            reward_target * F.log_softmax(reward_logits, dim=-1)).sum(dim=-1)
        # Per-agent rewards are masked per agent; the shared scalar per row.
        reward_mask = agent_mask if self.reward_dim > 1 else row_mask
        reward_loss = (
            reward_per * reward_mask).sum() / reward_mask.sum().clamp_min(1.0)

        continue_logits = self.continue_head(h_t, z_t)
        continue_per = F.binary_cross_entropy_with_logits(
            continue_logits, 1.0 - dones, reduction='none')
        continue_loss = (
            continue_per * row_mask).sum() / row_mask.sum().clamp_min(1.0)

        kl_per, kl_dyn, kl_rep = kl_categorical(
            post, prior, free_bits, kl_balance)
        kl_loss = (
            kl_per * row_mask).sum() / row_mask.sum().clamp_min(1.0)

        agent_h, _, agent_z_logits = self.agent_latents(h_t, z_t, sample=False)
        latent_denom = agent_mask.sum().clamp_min(1.0)
        h_per = ((agent_h - local_h_target.detach()) ** 2).mean(dim=-1)
        agent_h_loss = (h_per * agent_mask).sum() / latent_denom

        target_z = local_z_target.detach().reshape(
            *local_z_target.shape[:-1],
            self.local_latent_groups,
            self.local_latent_classes,
        )
        z_per = -(
            target_z * F.log_softmax(agent_z_logits, dim=-1)
        ).sum(dim=-1).mean(dim=-1)
        agent_z_loss = (z_per * agent_mask).sum() / latent_denom

        loss = (
            recon_weight * recon_loss
            + reward_weight * reward_loss
            + continue_weight * continue_loss
            + kl_weight * kl_loss
            + latent_h_weight * agent_h_loss
            + latent_z_weight * agent_z_loss
        )
        # Detached on-device scalars; converted to floats by the learner only
        # on logging steps to avoid a GPU->CPU sync every update.
        info = GlobalWorldModelInfo(
            loss=loss,
            recon=recon_loss.detach(),
            reward=reward_loss.detach(),
            cont=continue_loss.detach(),
            kl_dyn=kl_dyn.detach(),
            kl_rep=kl_rep.detach(),
            agent_h=agent_h_loss.detach(),
            agent_z=agent_z_loss.detach(),
        )
        return loss, info, h_t.detach(), z_t.detach()


__all__ = [
    'ACTION_DIM',
    'Actor',
    'Critic',
    'Experience',
    'GlobalWorldModel',
    'GlobalWorldModelInfo',
    'LocalWorldModel',
    'ReturnEMA',
    'STATE_DIM',
    'STATE_FEATURES',
    'STATE_FEATURE_VERSION',
    'TwoHot',
    'actor_log_prob',
    'assert_checkpoint_state_compatible',
    'lambda_return',
    'normalize_state',
    'per_agent_team_reward',
    'PerAgentRewardHead',
    'r_fair_from_avg_throughput',
    'reset_tempest_kalman',
    'shared_team_reward',
    'soft_update',
    'state_meta',
    'symlog',
    'symexp',
]
