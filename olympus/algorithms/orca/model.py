"""ORCA-repo-style actor, critic, and TD3 losses.

This follows the public Soheil-ab/Orca implementation more closely than the
generic recurrent TD3 path:

* The observation (repo's 7 compact features + flat `rec_dim` history, instead
  of an LSTM) is a runtime state plugin — algorithms/orca/states/orca_repo.py —
  loaded here via `load_state_module`; the reward is likewise a runtime plugin
  (rewards/orca.py). Neither is baked into this model.
* Actor actions remain bounded in `[-1, 1]`; the selected Olympus action
  plugin maps them to live CWND updates.
* The actor is a TensorFlow-style MLP: Dense + BatchNorm + LeakyReLU.
"""

from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.common.action_plugins import (
    current_action_meta,
    load_action_module,
)
from olympus.common.state_plugins import (
    assert_state_compatible,
    current_state_name,
    load_state_module,
)

_ACTION_PLUGIN = load_action_module()

# The observation (Orca's 7 features + rec_dim flat history) is a runtime state
# plugin, resolved through the normal state-plugin system — see
# algorithms/orca/states/orca_repo.py. It defaults to orca_repo but is swappable
# via runtime.state; the learner only needs STATE_DIM / normalize_state / meta.
STATE_NAME = current_state_name(default='orca_repo')
_STATE_PLUGIN = load_state_module('orca', STATE_NAME)
STATE_FEATURE_VERSION = _STATE_PLUGIN.STATE_FEATURE_VERSION
STATE_FEATURES = list(getattr(_STATE_PLUGIN, 'STATE_FEATURES', []))
STATE_DIM = int(_STATE_PLUGIN.STATE_DIM)
REC_DIM = int(getattr(_STATE_PLUGIN, 'REC_DIM', 1))
normalize_state = _STATE_PLUGIN.normalize_state

ACTION_DIM = int(_ACTION_PLUGIN.ACTION_DIM)
ACTION_MIN = float(_ACTION_PLUGIN.ACTION_MIN)
ACTION_MAX = float(_ACTION_PLUGIN.ACTION_MAX)

Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'traj_id', 'step_in_traj'])
CriticInfo = namedtuple('CriticInfo', ['loss', 'q1_mean', 'q2_mean', 'td_abs'])
ActorInfo = namedtuple('ActorInfo', ['loss', 'q_mean', 'a_mean', 'a_abs'])



def model_state_meta() -> dict:
    action = current_action_meta()
    # Legacy ORCA checkpoints used the upstream 4**a range (0.25x..4x), not
    # the framework's default 0.5x..2x action plugin.
    action['legacy_default'] = False
    return {
        'state_name': STATE_NAME,
        'state_dim': int(STATE_DIM),
        'state_feature_version': STATE_FEATURE_VERSION,
        'state_features': list(STATE_FEATURES),
        'action_meta': action,
    }


def assert_checkpoint_state_compatible(ckpt: dict, source='checkpoint') -> None:
    assert_state_compatible(model_state_meta(), ckpt, source=source)


def _tf_scale_false_bn(num_features: int) -> nn.BatchNorm1d:
    """BatchNorm1d that matches tf.layers.batch_normalization(scale=False).

    TF's scale=False fixes gamma at 1.0 (no scaling parameter) but keeps a
    learnable beta. PyTorch's affine=False drops both. We emulate the TF
    behaviour by keeping affine=True and freezing the weight at 1.
    """
    bn = nn.BatchNorm1d(num_features, affine=True)
    nn.init.ones_(bn.weight)
    nn.init.zeros_(bn.bias)
    bn.weight.requires_grad_(False)
    return bn


class Actor(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256,
                 head_hidden: int = 256):
        super().__init__()
        self.hidden = int(hidden)
        self.head_hidden = int(head_hidden)
        self.fc1 = nn.Linear(state_dim, self.hidden)
        self.bn1 = _tf_scale_false_bn(self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.head_hidden)
        self.bn2 = _tf_scale_false_bn(self.head_hidden)
        self.action_head = nn.Linear(self.head_hidden, ACTION_DIM)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        x = self.fc1(s)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return torch.tanh(self.action_head(x))

    @torch.no_grad()
    def act(self, s_np: np.ndarray, noise_std: float = 0.0):
        device = next(self.parameters()).device
        s = torch.from_numpy(np.asarray(s_np, dtype=np.float32)).unsqueeze(0).to(device)
        a = self.forward(s)[0, 0]
        if noise_std > 0.0:
            a = (a + noise_std * torch.randn_like(a)).clamp(
                ACTION_MIN, ACTION_MAX)
        action = float(a.item())
        return action, float(_ACTION_PLUGIN.to_multiplier(action))


class SingleCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256,
                 head_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden + ACTION_DIM, head_hidden)
        self.q_head = nn.Linear(head_hidden, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        x = F.leaky_relu(self.fc1(s), negative_slope=0.01)
        x = torch.cat([x, a], dim=-1)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        return self.q_head(x).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256,
                 head_hidden: int = 256):
        super().__init__()
        self.q1 = SingleCritic(state_dim, hidden, head_hidden)
        self.q2 = SingleCritic(state_dim, hidden, head_hidden)

    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)

    def q1_only(self, s, a):
        return self.q1(s, a)


def action_to_percent(action: float) -> int:
    return int(round(float(_ACTION_PLUGIN.to_multiplier(action)) * 100.0))


def action_to_mult(a: torch.Tensor) -> torch.Tensor:
    return _ACTION_PLUGIN.to_multiplier(a)


def soft_update(online: nn.Module, target: nn.Module, tau: float = 0.001):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


def actor_arch_from_state_dict(state_dict: dict, hidden: int = 256,
                               head_hidden: int = 256) -> tuple:
    if not state_dict:
        return int(hidden), int(head_hidden)
    fc1 = state_dict.get('fc1.weight')
    fc2 = state_dict.get('fc2.weight')
    if fc1 is not None:
        hidden = int(fc1.shape[0])
    if fc2 is not None:
        head_hidden = int(fc2.shape[0])
    return int(hidden), int(head_hidden)


def actor_arch_from_checkpoint(ckpt: dict, hidden: int = 256,
                               head_hidden: int = 256) -> tuple:
    actor_state = (ckpt or {}).get('actor') or (ckpt or {}).get('actor_state_dict')
    return actor_arch_from_state_dict(actor_state, hidden, head_hidden)


def actor_model_meta(actor: nn.Module) -> dict:
    return {
        'algorithm': 'orca',
        'state_dim': int(STATE_DIM),
        'hidden': int(actor.hidden),
        'head_hidden': int(actor.head_hidden),
        'rec_dim': int(REC_DIM),
    }


def critic_loss(
    critic: TwinCritic,
    actor_target: Actor,
    critic_target: TwinCritic,
    s_batch: torch.Tensor,
    a_batch: torch.Tensor,
    r_batch: torch.Tensor,
    s2_batch: torch.Tensor,
    d_batch: torch.Tensor,
    gamma: float = 0.995,
    target_noise_std: float = 0.1,
    target_noise_clip: float = 0.2,
    reward_clip: float = 0.0,
) -> CriticInfo:
    with torch.no_grad():
        a_next = actor_target(s2_batch).squeeze(-1)
        noise = (torch.randn_like(a_next) * target_noise_std).clamp(
            -target_noise_clip, target_noise_clip)
        a_next = (a_next + noise).clamp(ACTION_MIN, ACTION_MAX)
        q1_next, q2_next = critic_target(s2_batch, a_next)
        q_next = torch.min(q1_next, q2_next)
        r = r_batch
        if reward_clip > 0.0:
            # Bound Orca's unbounded-below reward (-5*loss term) so hard-loss
            # episodes cannot drive the MSE target to large negatives.
            r = r.clamp(-reward_clip, reward_clip)
        y = r + gamma * q_next * (1.0 - d_batch)
        if reward_clip > 0.0:
            # With reward in [-reward_clip, reward_clip], the true return — and
            # thus Q — cannot leave +/-reward_clip/(1-gamma). Clamp the target
            # to that hard ceiling so a diverging bootstrap has nowhere to run.
            # (At gamma=0.995 the bound is loose; grad_clip and the reward clip
            # do the primary work — this is the backstop.)
            y_bound = reward_clip / max(1.0 - gamma, 1e-6)
            y = y.clamp(-y_bound, y_bound)

    q1, q2 = critic(s_batch, a_batch)
    loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    with torch.no_grad():
        td_abs = (q1 - y).abs().mean()
    return CriticInfo(
        loss=loss,
        q1_mean=float(q1.mean().item()),
        q2_mean=float(q2.mean().item()),
        td_abs=float(td_abs.item()),
    )


def actor_loss(actor: Actor, critic: TwinCritic, s_batch: torch.Tensor) -> ActorInfo:
    a = actor(s_batch).squeeze(-1)
    q1 = critic.q1_only(s_batch, a)
    loss = -q1.mean()
    return ActorInfo(
        loss=loss,
        q_mean=float(q1.mean().item()),
        a_mean=float(a.mean().item()),
        a_abs=float(a.abs().mean().item()),
    )
