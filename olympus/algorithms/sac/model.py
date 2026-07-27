"""
model.py — recurrent Soft Actor-Critic (SAC) actor-critic for direct CWND control.

SAC (Haarnoja et al. 2018, "Soft Actor-Critic" + "Applications" v2 with
automatic temperature tuning) is the maximum-entropy off-policy successor to
TD3.  It keeps TD3's twin clipped-double-Q critic but replaces the
deterministic actor + injected exploration noise + target-policy smoothing with
a *stochastic* squashed-Gaussian policy and an entropy bonus.  Empirically SAC
matches or beats TD3 on continuous control while being far less sensitive to
hyper-parameters (no policy-delay, no target-noise clip to tune, exploration is
learned rather than a hand-set schedule), so it is the "better-understood"
choice here.

This variant is *recurrent* (LSTM trunk) to match the rest of this codebase —
congestion control is partially observed (throughput/RTT depend on unobserved
link state), so an LSTM over the last few control steps is the established
design for every algorithm in olympus/algorithms.  The state / action / reward
pipeline is identical to the td3 package so rollout data is directly comparable.

  Actor (stochastic squashed Gaussian):
    input_proj : Linear(STATE_DIM → hidden) → LayerNorm → ReLU
    LSTM       : hidden → hidden, 1 layer
    post       : Linear(hidden → hidden//2) → ReLU
    mean_head  : Linear(hidden//2 → ACTION_DIM)
    log_std_head : Linear(hidden//2 → ACTION_DIM)   (clamped to [LOG_STD_MIN, MAX])
    → sample u ~ N(mean, std) via reparameterisation, a = tanh(u) ∈ [-1, 1]
    → log π(a|s) = log N(u) − Σ log(1 − tanh(u)² + ε)   (tanh Jacobian correction)

    The tanh squashing lives INSIDE the actor so the critic always sees a
    bounded action ∈ [-1, 1] and replay stores that same squashed value — the
    standard SAC/TD3 contract, identical to td3/model.py.

  Critic (twin Q — same as TD3):
    input_proj : Linear(STATE_DIM + ACTION_DIM → hidden) → LayerNorm → ReLU
    LSTM       : hidden → hidden
    post       : Linear(hidden → hidden//2) → ReLU
    q_head     : Linear(hidden//2 → 1)
    — two independent copies Q1, Q2; clipped double-Q min for the target.

  Temperature α: automatically tuned toward a target entropy of −ACTION_DIM
  (SAC v2).  There is NO target actor in SAC — the entropy term provides the
  smoothing that TD3 got from target-policy noise, so only the critics have
  Polyak-averaged targets.

Off-policy: the learner samples random seq_len chunks from the replay buffer;
each chunk runs through the LSTM from a zero-initialised hidden state — the
canonical burn-in-skipping simplification, matching the td3 learner.
"""

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.common.state_plugins import (
    assert_state_compatible,
    current_state_name,
    load_state_module,
    state_meta,
)
from olympus.common.action_plugins import load_action_module
from olympus.common import policy as policy_contract


# ── State / action constants ──────────────────────────────────────────────────

_ACTION_PLUGIN = load_action_module()
ACTION_DIM = int(_ACTION_PLUGIN.ACTION_DIM)
ACTION_MIN = float(_ACTION_PLUGIN.ACTION_MIN)
ACTION_MAX = float(_ACTION_PLUGIN.ACTION_MAX)

# Squashed-Gaussian log-std clamp. [-5, 2] is a stable range for this control
# task (std ∈ [~6.7e-3, ~7.4]); the vanilla SAC paper uses [-20, 2] but the
# tighter floor avoids collapsed-variance NaNs early in training here.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
# Numerical epsilon for the tanh Jacobian log(1 - tanh(u)^2) correction.
_TANH_EPS = 1e-6

Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'traj_id', 'step_in_traj'])

CriticInfo = namedtuple('CriticInfo', ['loss', 'q1_mean', 'q2_mean', 'td_abs'])
ActorInfo  = namedtuple('ActorInfo',
    ['loss', 'q_mean', 'logp_mean', 'a_mean', 'a_abs', 'entropy'])


# ── State normalisation (loaded from the shared state plugin) ─────────────────

_STATE_PLUGIN = load_state_module('sac')
STATE_NAME = current_state_name()
STATE_FEATURE_VERSION = _STATE_PLUGIN.STATE_FEATURE_VERSION
STATE_FEATURES = list(getattr(_STATE_PLUGIN, 'STATE_FEATURES', []))
STATE_DIM = int(_STATE_PLUGIN.STATE_DIM)
_STATE_LOW = _STATE_PLUGIN.STATE_LOW
_STATE_HIGH = _STATE_PLUGIN.STATE_HIGH
normalize_state = _STATE_PLUGIN.normalize_state


def model_state_meta() -> dict:
    return state_meta(_STATE_PLUGIN, STATE_NAME)


def assert_checkpoint_state_compatible(ckpt: dict, source='checkpoint') -> None:
    assert_state_compatible(model_state_meta(), ckpt, source=source)


def actor_arch_from_state_dict(state_dict: dict, hidden: int = 128,
                               head_hidden: int = None) -> tuple:
    """Infer SAC actor widths from a saved actor state dict."""
    if head_hidden is None:
        head_hidden = int(hidden) // 2
    if not state_dict:
        return int(hidden), int(head_hidden)

    input_weight = state_dict.get('input_proj.0.weight')
    mean_weight = state_dict.get('mean_head.weight')
    if input_weight is not None:
        hidden = int(input_weight.shape[0])
    if mean_weight is not None:
        head_hidden = int(mean_weight.shape[1])
    return int(hidden), int(head_hidden)


def actor_arch_from_checkpoint(ckpt: dict, hidden: int = 128,
                               head_hidden: int = None) -> tuple:
    actor_state = (ckpt or {}).get('actor') or (ckpt or {}).get('actor_state_dict')
    return actor_arch_from_state_dict(actor_state, hidden, head_hidden)


def actor_model_meta(actor: nn.Module) -> dict:
    return {
        'algorithm': 'sac',
        'state_dim': int(STATE_DIM),
        'hidden': int(actor.hidden),
        'head_hidden': int(actor.head_hidden),
    }


# ── Actor (stochastic squashed Gaussian) ──────────────────────────────────────

class Actor(nn.Module):
    """
    Stochastic policy: u ~ N(μ(s,h), σ(s,h)), a = tanh(u) ∈ [-1, 1]. The
    selected action plugin maps that bounded action to its TCP control meaning.

    Hidden state h is (h, c), each (1, B, hidden). At inference the worker
    carries it across steps; at training we zero-init per seq_len chunk.

    `hidden` is the LSTM width; `head_hidden` is the post-LSTM dense width
    (defaults to `hidden // 2`).
    """
    MULT_MIN = float(_ACTION_PLUGIN.MULTIPLIER_MIN)
    MULT_MAX = float(_ACTION_PLUGIN.MULTIPLIER_MAX)

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128,
                 head_hidden: int = None):
        super().__init__()
        if head_hidden is None:
            head_hidden = hidden // 2
        self.hidden = hidden
        self.head_hidden = head_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.post = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(head_hidden, ACTION_DIM)
        self.log_std_head = nn.Linear(head_hidden, ACTION_DIM)
        # Small init on the mean head keeps the initial mean near tanh(0)=0 →
        # mult≈1.0, no change, so early rollouts are calm. Bias the log-std
        # toward a moderate exploration level (std≈exp(-1)≈0.37) at start.
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def _trunk(self, s: torch.Tensor, h=None):
        """
        s : (B, T, D).
        Returns (feat: (B, T, head_hidden), h_new: (h, c) pair).
        """
        if s.dim() == 2:
            s = s.unsqueeze(1)
        B, T, D = s.shape
        x = self.input_proj(s.reshape(B * T, D)).reshape(B, T, self.hidden)
        if h is None:
            h0 = torch.zeros(1, B, self.hidden, device=s.device)
            c0 = torch.zeros(1, B, self.hidden, device=s.device)
            h = (h0, c0)
        lstm_out, h_new = self.lstm(x, h)
        feat = self.post(lstm_out.reshape(B * T, self.hidden)).reshape(B, T, -1)
        return feat, h_new

    def _mean_log_std(self, s: torch.Tensor, h=None):
        feat, h_new = self._trunk(s, h)
        mean = self.mean_head(feat)                       # (B, T, A)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, h_new

    def sample_sequence(self, s: torch.Tensor, h=None):
        """
        Reparameterised sample over a whole sequence (training).

        Returns (a, logp, h_new):
          a    : (B, T, A) squashed action ∈ (-1, 1)
          logp : (B, T)    log π(a|s) with the tanh Jacobian correction, summed
                           over the action dimension.
        """
        mean, log_std, h_new = self._mean_log_std(s, h)
        std = log_std.exp()
        eps = torch.randn_like(std)
        u = mean + std * eps                              # reparameterised pre-tanh
        a = torch.tanh(u)                                 # (B, T, A) ∈ (-1, 1)

        # log N(u | mean, std) − Σ log(1 − tanh(u)² + ε)   (SAC eq. 21)
        log_normal = (-0.5 * ((u - mean) / std) ** 2
                      - log_std - 0.5 * math.log(2.0 * math.pi))
        log_correction = torch.log(1.0 - a.pow(2) + _TANH_EPS)
        logp = (log_normal - log_correction).sum(dim=-1)  # (B, T)
        return a, logp, h_new

    @torch.no_grad()
    def act(self, s_np: np.ndarray, h=None, deterministic: bool = False):
        """
        Inference: returns (action_float, mult_float, log_std_float, h_new).
          action ∈ [-1, 1] (post-tanh, clipped). When `deterministic`, use the
          distribution mean (greedy) — standard SAC evaluation. Otherwise draw
          a stochastic sample; SAC needs no external exploration noise because
          the policy itself is stochastic.
          Replay stores this same bounded value so the critic and actor
          gradient pipelines see identical quantities.
        """
        device = next(self.parameters()).device
        s = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0).to(device)
        mean, log_std, h_new = self._mean_log_std(s, h)
        if deterministic:
            u = mean
        else:
            u = mean + log_std.exp() * torch.randn_like(mean)
        a = torch.tanh(u)[0, -1]                          # (A,)
        a = a.clamp(ACTION_MIN, ACTION_MAX)
        a0 = a[0]
        mult = float(_ACTION_PLUGIN.to_multiplier(a0).item())
        return float(a0.item()), mult, float(log_std[0, -1, 0].item()), h_new


def action_to_mult(a: torch.Tensor) -> torch.Tensor:
    """Differentiable bounded-action to multiplier mapping."""
    return _ACTION_PLUGIN.to_multiplier(a)


# ── Critic (twin Q — identical contract to td3/model.py) ──────────────────────

class SingleCritic(nn.Module):
    """One Q(s, a) network with an LSTM trunk over (s, a) concatenated."""
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128,
                 head_hidden: int = None):
        super().__init__()
        if head_hidden is None:
            head_hidden = hidden // 2
        self.hidden = hidden
        self.head_hidden = head_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + ACTION_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.post = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(head_hidden, 1)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.zeros_(self.q_head.bias)

    def forward_sequence(self, s: torch.Tensor, a: torch.Tensor, h=None):
        """
        s : (B, T, state_dim)
        a : (B, T, action_dim)
        Returns (q: (B, T), h_new). a is the squashed action (matches Actor).
        """
        if s.dim() == 2: s = s.unsqueeze(1)
        if a.dim() == 2: a = a.unsqueeze(-1)
        if a.dim() == 1: a = a.unsqueeze(-1).unsqueeze(-1)
        x_in = torch.cat([s, a], dim=-1)
        B, T, D = x_in.shape
        x = self.input_proj(x_in.reshape(B * T, D)).reshape(B, T, self.hidden)
        if h is None:
            h0 = torch.zeros(1, B, self.hidden, device=s.device)
            c0 = torch.zeros(1, B, self.hidden, device=s.device)
            h = (h0, c0)
        lstm_out, h_new = self.lstm(x, h)
        feat = self.post(lstm_out.reshape(B * T, self.hidden)).reshape(B, T, -1)
        q = self.q_head(feat).squeeze(-1)   # (B, T)
        return q, h_new


class TwinCritic(nn.Module):
    """Two independent SingleCritic towers — standard clipped double-Q."""
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128,
                 head_hidden: int = None):
        super().__init__()
        self.q1 = SingleCritic(state_dim, hidden, head_hidden)
        self.q2 = SingleCritic(state_dim, hidden, head_hidden)

    def forward_sequence(self, s, a, h=None):
        q1, _ = self.q1.forward_sequence(s, a, h)
        q2, _ = self.q2.forward_sequence(s, a, h)
        return q1, q2

    def q1_only(self, s, a, h=None):
        q1, _ = self.q1.forward_sequence(s, a, h)
        return q1


# ── Target-net soft update ────────────────────────────────────────────────────

def soft_update(online: nn.Module, target: nn.Module, tau: float = 0.005):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


# ── SAC losses ────────────────────────────────────────────────────────────────

def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (t * mask).sum() / mask.sum().clamp_min(1.0)


def actor_distillation_loss(
    actor: Actor,
    teacher: Actor,
    s_batch: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """KL from a frozen teacher's pre-tanh Gaussian to the current actor.

    Distilling the Gaussian parameters, rather than sampled actions, preserves
    both the teacher's mean control decision and its learned uncertainty.  The
    recurrent trunks consume the complete anchor sequence before the masked
    mean is taken, so valid later timesteps retain their LSTM context.
    """
    with torch.no_grad():
        old_mean, old_log_std, _ = teacher._mean_log_std(s_batch)
    new_mean, new_log_std, _ = actor._mean_log_std(s_batch)

    old_var = (2.0 * old_log_std).exp()
    new_var = (2.0 * new_log_std).exp().clamp_min(1e-8)
    kl = (
        new_log_std - old_log_std
        + (old_var + (old_mean - new_mean).pow(2)) / (2.0 * new_var)
        - 0.5
    ).sum(dim=-1)
    return _masked_mean(kl, mask)


def critic_distillation_loss(
    critic: TwinCritic,
    teacher: TwinCritic,
    s_batch: torch.Tensor,
    a_batch: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Match both current Q towers to a frozen teacher on anchor actions."""
    with torch.no_grad():
        old_q1, old_q2 = teacher.forward_sequence(s_batch, a_batch)
    new_q1, new_q2 = critic.forward_sequence(s_batch, a_batch)
    return (
        _masked_mean((new_q1 - old_q1).pow(2), mask)
        + _masked_mean((new_q2 - old_q2).pow(2), mask)
    )


def critic_loss(
    critic:        TwinCritic,
    actor:         Actor,
    critic_target: TwinCritic,
    s_batch:       torch.Tensor,   # (B, T, D)
    a_batch:       torch.Tensor,   # (B, T)   squashed actions the worker took
    r_batch:       torch.Tensor,   # (B, T)
    s2_batch:      torch.Tensor,   # (B, T, D)
    d_batch:       torch.Tensor,   # (B, T)
    mask:          torch.Tensor,   # (B, T)
    alpha:         float = 0.2,
    gamma:         float = 0.99,
) -> CriticInfo:
    """
    Soft clipped-double-Q target (SAC).  Unlike TD3 there is no target actor
    and no target-policy noise: the next action is sampled from the *current*
    stochastic policy, and its entropy −α·log π enters the bootstrap target.

      y = r + γ (1−d) [ min_i Q_i^target(s', ã') − α·log π(ã'|s') ],  ã' ~ π(·|s')

    Mask ignores padding timesteps from the replay buffer's ragged chunks.
    """
    with torch.no_grad():
        a_next, logp_next, _ = actor.sample_sequence(s2_batch)   # (B,T,A),(B,T)
        q1_next, q2_next = critic_target.forward_sequence(s2_batch, a_next)
        q_next = torch.min(q1_next, q2_next) - alpha * logp_next
        y = r_batch + gamma * q_next * (1.0 - d_batch)

    q1, q2 = critic.forward_sequence(s_batch, a_batch)
    loss = _masked_mean((q1 - y) ** 2, mask) + _masked_mean((q2 - y) ** 2, mask)

    with torch.no_grad():
        q1_mean = _masked_mean(q1, mask)
        q2_mean = _masked_mean(q2, mask)
        td_abs  = _masked_mean((q1 - y).abs(), mask)

    return CriticInfo(
        loss    = loss,
        q1_mean = float(q1_mean.item()),
        q2_mean = float(q2_mean.item()),
        td_abs  = float(td_abs.item()),
    )


def actor_loss(
    actor:   Actor,
    critic:  TwinCritic,
    s_batch: torch.Tensor,     # (B, T, D)
    mask:    torch.Tensor,     # (B, T)
    alpha:   float = 0.2,
):
    """
    Maximum-entropy policy improvement (SAC).  Minimise
        E_{ã~π}[ α·log π(ã|s) − min_i Q_i(s, ã) ]
    using a fresh reparameterised sample so the gradient flows through both the
    Gaussian and the tanh squashing.  Uses the min of BOTH critics (standard
    SAC), not Q1-only as in TD3.

    Returns (ActorInfo, logp) — the per-timestep log-probs are reused by the
    temperature update so we only sample the policy once.
    """
    a, logp, _ = actor.sample_sequence(s_batch)     # (B,T,A), (B,T)
    q1, q2 = critic.forward_sequence(s_batch, a)
    q = torch.min(q1, q2)                            # (B, T)

    loss = _masked_mean(alpha * logp - q, mask)

    with torch.no_grad():
        info = ActorInfo(
            loss      = loss,
            q_mean    = float(_masked_mean(q, mask).item()),
            logp_mean = float(_masked_mean(logp, mask).item()),
            a_mean    = float(_masked_mean(a.mean(dim=-1), mask).item()),
            a_abs     = float(_masked_mean(a.abs().mean(dim=-1), mask).item()),
            entropy   = float(_masked_mean(-logp, mask).item()),
        )
    return info, logp


def temperature_loss(
    log_alpha:      torch.Tensor,
    logp:           torch.Tensor,   # (B, T)  from actor_loss
    target_entropy: float,
    mask:           torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """
    Automatic temperature tuning (SAC v2).  Minimise
        E[ −log α · (log π + H_target) ]
    which drives the policy entropy toward the target H_target = −ACTION_DIM.
    `logp` is detached so this only updates log_alpha, not the policy.
    """
    return _masked_mean(-log_alpha * (logp.detach() + target_entropy), mask)


# ── Deployment factory ────────────────────────────────────────────────────────

def build_policy(ckpt, agent_cfg=None, training_cfg=None, device='cpu',
                 deterministic=True):
    """Load this checkpoint's actor as an inference Policy.

    SAC's policy is stochastic, so `deterministic` selects the distribution
    mean — the standard SAC evaluation mode, and what worker.py does when its
    own deterministic flag is set. See olympus/common/policy.py.
    """
    hidden = policy_contract.resolved_hidden(agent_cfg, 'hidden', 128)
    head_hidden = (agent_cfg or {}).get('head_hidden')
    head_hidden = int(head_hidden) if head_hidden not in (None, '') else None
    hidden, head_hidden = actor_arch_from_checkpoint(ckpt, hidden, head_hidden)
    actor = Actor(STATE_DIM, hidden, head_hidden)
    actor.load_state_dict(policy_contract.actor_state_dict(ckpt))
    actor.to(device).eval()
    return policy_contract.StochasticRecurrentActorPolicy(
        'sac', STATE_DIM, actor, deterministic)
