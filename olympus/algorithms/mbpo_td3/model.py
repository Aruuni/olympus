"""
model.py — MBPO on top of recurrent TD3.

The Actor / TwinCritic / soft_update / actor_loss / critic_loss are copied
verbatim from algorithms/orca_td3/model.py — only the forward-model ensemble
that produces imagined transitions is new.

Forward model:
  ProbabilisticModel : Linear(s+a → hidden) → SiLU → Linear → SiLU → Linear → SiLU
                       → Linear(hidden → 2·(state_dim + 1))
                       Outputs (mean, logvar) for (Δs, r). Soft-bounded logvar
                       (PETS trick) so a single huge variance can't dominate the
                       NLL gradient.

Ensemble:
  N independent ProbabilisticModels with different random init. At rollout time
  each batch element samples from a uniformly chosen member, capturing epistemic
  uncertainty without needing per-member bootstrap samplers.
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
STATE_DIM  = 11
ACTION_DIM = int(_ACTION_PLUGIN.ACTION_DIM)
ACTION_MIN = float(_ACTION_PLUGIN.ACTION_MIN)
ACTION_MAX = float(_ACTION_PLUGIN.ACTION_MAX)
# Kept verbatim so existing checkpoints' `model_meta.state_features` still match.
STATE_FEATURE_VERSION = 'kalman_observation_v1_srtt_unshifted'

Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'traj_id', 'step_in_traj'])

CriticInfo = namedtuple('CriticInfo', ['loss', 'q1_mean', 'q2_mean', 'td_abs'])
ActorInfo  = namedtuple('ActorInfo',  ['loss', 'q_mean', 'a_mean', 'a_abs'])
ModelInfo  = namedtuple('ModelInfo',  ['loss', 'mse', 'r_mse'])

_STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    dtype=np.float32,
)
STATE_LOW_T  = torch.from_numpy(_STATE_LOW)
STATE_HIGH_T = torch.from_numpy(_STATE_HIGH)


# ── State normalisation ───────────────────────────────────────────────────────

def normalize_state(info: dict) -> np.ndarray:
    """11-dim observation: TCP/kernel fields + worker-injected history.

    Identical contract to orca_td3/model.py.normalize_state.
    """
    cwnd        = max(int(info.get('cwnd', 1)), 1)
    avg_thr     = float(info.get('avg_thr', 0))
    avg_urtt    = float(info.get('avg_urtt', 0))
    srtt_raw    = float(info.get('srtt_us', 0))
    srtt_us     = (srtt_raw / 8.0) if srtt_raw > 0 else avg_urtt
    srtt_us     = max(srtt_us, 1.0)
    pacing_rate = float(info.get('pacing_rate', 0))
    packets_out = float(info.get('packets_out', 0))
    retrans_out = float(info.get('retrans_out', 0))
    prev_urtt   = float(info.get('prev_urtt', avg_urtt))
    prev_cwnd   = float(info.get('prev_cwnd', cwnd))
    peak_thr    = float(info.get('peak_thr', 0))
    kalman_min_rtt = float(info.get('kalman_min_rtt_us', avg_urtt))

    bw_ref = max(peak_thr, pacing_rate, 1.0)
    delta_rtt = float(np.clip((avg_urtt - prev_urtt) / max(prev_urtt, 1.0),
                              -1.0, 1.0))
    delta_cwnd = float(np.clip((cwnd - prev_cwnd) / max(prev_cwnd, 1.0),
                               -1.0, 1.0))
    cwnd_log = math.log1p(float(cwnd)) / math.log1p(10_000.0)

    s = np.array([
        delta_cwnd,
        avg_urtt / 1e5,
        cwnd_log,
        max(avg_thr / bw_ref, 0.0),
        max(pacing_rate / bw_ref, 0.0),
        max(packets_out / max(cwnd, 1), 0.0),
        delta_rtt,
        min(retrans_out / max(packets_out, 1.0), 1.0),
        avg_thr / 1e7,
        srtt_us / 1e5,
        kalman_min_rtt / 1e5,
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=_STATE_HIGH, neginf=_STATE_LOW)
    return np.clip(s, _STATE_LOW, _STATE_HIGH)


_STATE_PLUGIN = load_state_module('mbpo_td3')
STATE_NAME = current_state_name()
STATE_FEATURE_VERSION = _STATE_PLUGIN.STATE_FEATURE_VERSION
STATE_FEATURES = list(getattr(_STATE_PLUGIN, 'STATE_FEATURES', []))
STATE_DIM = int(_STATE_PLUGIN.STATE_DIM)
_STATE_LOW = _STATE_PLUGIN.STATE_LOW
_STATE_HIGH = _STATE_PLUGIN.STATE_HIGH
STATE_LOW_T = _STATE_PLUGIN.STATE_LOW_T
STATE_HIGH_T = _STATE_PLUGIN.STATE_HIGH_T
normalize_state = _STATE_PLUGIN.normalize_state


def model_state_meta() -> dict:
    return state_meta(_STATE_PLUGIN, STATE_NAME)


def assert_checkpoint_state_compatible(ckpt: dict, source='checkpoint') -> None:
    assert_state_compatible(model_state_meta(), ckpt, source=source)


# ── Actor (copied from orca_td3/model.py) ────────────────────────────────────

class Actor(nn.Module):
    MULT_MIN = float(_ACTION_PLUGIN.MULTIPLIER_MIN)
    MULT_MAX = float(_ACTION_PLUGIN.MULTIPLIER_MAX)

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        head_in = hidden // 2
        self.post = nn.Sequential(
            nn.Linear(hidden, head_in),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(head_in, ACTION_DIM)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def _trunk(self, s: torch.Tensor, h=None):
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

    def forward_sequence(self, s: torch.Tensor, h=None):
        feat, h_new = self._trunk(s, h)
        a = torch.tanh(self.action_head(feat))
        return a, h_new

    @torch.no_grad()
    def act(self, s_np: np.ndarray, h=None, noise_std: float = 0.0):
        device = next(self.parameters()).device
        s = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0).to(device)
        a, h_new = self.forward_sequence(s, h)
        a = a[0, -1, 0]
        if noise_std > 0.0:
            a = (a + noise_std * torch.randn_like(a)).clamp(
                ACTION_MIN, ACTION_MAX)
        mult = float(_ACTION_PLUGIN.to_multiplier(a).item())
        return float(a.item()), mult, h_new


def action_to_mult(a: torch.Tensor) -> torch.Tensor:
    return _ACTION_PLUGIN.to_multiplier(a)


# ── Critic (copied from orca_td3/model.py) ───────────────────────────────────

class SingleCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + ACTION_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        head_in = hidden // 2
        self.post = nn.Sequential(
            nn.Linear(hidden, head_in),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(head_in, 1)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.zeros_(self.q_head.bias)

    def forward_sequence(self, s, a, h=None):
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
        q = self.q_head(feat).squeeze(-1)
        return q, h_new


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()
        self.q1 = SingleCritic(state_dim, hidden)
        self.q2 = SingleCritic(state_dim, hidden)

    def forward_sequence(self, s, a, h=None):
        q1, _ = self.q1.forward_sequence(s, a, h)
        q2, _ = self.q2.forward_sequence(s, a, h)
        return q1, q2

    def q1_only(self, s, a, h=None):
        q1, _ = self.q1.forward_sequence(s, a, h)
        return q1


def soft_update(online: nn.Module, target: nn.Module, tau: float = 0.005):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


# ── TD3 losses (copied from orca_td3/model.py) ───────────────────────────────

def critic_loss(critic, actor_target, critic_target,
                s_batch, a_batch, r_batch, s2_batch, d_batch, mask,
                gamma=0.99, target_noise_std=0.2, target_noise_clip=0.5):
    with torch.no_grad():
        a_next, _ = actor_target.forward_sequence(s2_batch)
        a_next = a_next.squeeze(-1)
        noise = (torch.randn_like(a_next) * target_noise_std
                ).clamp(-target_noise_clip, target_noise_clip)
        a_next = (a_next + noise).clamp(ACTION_MIN, ACTION_MAX)
        q1_next, q2_next = critic_target.forward_sequence(s2_batch, a_next)
        q_next = torch.min(q1_next, q2_next)
        y = r_batch + gamma * q_next * (1.0 - d_batch)

    q1, q2 = critic.forward_sequence(s_batch, a_batch)
    td1 = (q1 - y) ** 2
    td2 = (q2 - y) ** 2

    def _masked_mean(t):
        return (t * mask).sum() / mask.sum().clamp_min(1.0)

    loss = _masked_mean(td1) + _masked_mean(td2)

    with torch.no_grad():
        q1_mean = _masked_mean(q1)
        q2_mean = _masked_mean(q2)
        td_abs  = _masked_mean((q1 - y).abs())

    return CriticInfo(
        loss    = loss,
        q1_mean = float(q1_mean.item()),
        q2_mean = float(q2_mean.item()),
        td_abs  = float(td_abs.item()),
    )


def actor_loss(actor, critic, s_batch, mask):
    a, _ = actor.forward_sequence(s_batch)
    a = a.squeeze(-1)
    q1 = critic.q1_only(s_batch, a)

    def _masked_mean(t):
        return (t * mask).sum() / mask.sum().clamp_min(1.0)

    loss = -_masked_mean(q1)

    with torch.no_grad():
        q_mean = -loss
        a_mean = _masked_mean(a)
        a_abs  = _masked_mean(a.abs())

    return ActorInfo(
        loss   = loss,
        q_mean = float(q_mean.item()),
        a_mean = float(a_mean.item()),
        a_abs  = float(a_abs.item()),
    )


# ── Forward model ensemble (NEW, MBPO-specific) ──────────────────────────────

class ProbabilisticModel(nn.Module):
    """
    Single ensemble member: predicts (Δs, reward) ~ N(mean, exp(logvar)) given (s, a).

    Predicting Δs (residual) instead of s' lets the network learn small corrections
    around the identity map; predicting reward jointly lets a single forward pass
    drive the imagined critic target.
    """
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 200):
        super().__init__()
        self.state_dim = state_dim
        in_dim  = state_dim + action_dim
        out_dim = (state_dim + 1) * 2   # mean + logvar for (Δs, r)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        # PETS-style trainable logvar bounds — keep variances numerically sane.
        self.max_logvar = nn.Parameter(torch.full((state_dim + 1,),  0.5))
        self.min_logvar = nn.Parameter(torch.full((state_dim + 1,), -10.0))

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        if a.dim() == s.dim() - 1:
            a = a.unsqueeze(-1)
        out = self.net(torch.cat([s, a], dim=-1))
        mean   = out[..., :self.state_dim + 1]
        logvar = out[..., self.state_dim + 1:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


class ForwardModelEnsemble(nn.Module):
    """
    N independent ProbabilisticModels. Each rollout step samples from a uniformly
    chosen member per batch element so epistemic uncertainty propagates.
    """
    def __init__(self, n_models: int = 5,
                 state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 200):
        super().__init__()
        self.n_models  = n_models
        self.state_dim = state_dim
        self.models = nn.ModuleList([
            ProbabilisticModel(state_dim, action_dim, hidden)
            for _ in range(n_models)
        ])

    def loss(self, s: torch.Tensor, a: torch.Tensor,
             delta_s_target: torch.Tensor, r_target: torch.Tensor) -> ModelInfo:
        """Sum of Gaussian-NLL across all members on the same batch."""
        targets = torch.cat([delta_s_target, r_target.unsqueeze(-1)], dim=-1)
        nll_total = torch.zeros((), device=s.device)
        bound_reg = torch.zeros((), device=s.device)
        with torch.no_grad():
            mse_acc = torch.zeros((), device=s.device)
            r_mse_acc = torch.zeros((), device=s.device)
        for m in self.models:
            mean, logvar = m(s, a)
            inv_var = (-logvar).exp()
            nll = 0.5 * ((targets - mean) ** 2 * inv_var + logvar)
            nll_total = nll_total + nll.mean()
            bound_reg = bound_reg + (m.max_logvar.sum() - m.min_logvar.sum())
            with torch.no_grad():
                mse_acc   = mse_acc + ((mean[..., :self.state_dim] - delta_s_target) ** 2).mean()
                r_mse_acc = r_mse_acc + ((mean[..., self.state_dim] - r_target) ** 2).mean()
        loss = nll_total + 0.01 * bound_reg
        return ModelInfo(
            loss  = loss,
            mse   = float(mse_acc.item() / self.n_models),
            r_mse = float(r_mse_acc.item() / self.n_models),
        )

    @torch.no_grad()
    def sample(self, s: torch.Tensor, a: torch.Tensor):
        """Sample (Δs, r) per batch element from a uniformly-chosen member."""
        if a.dim() == s.dim() - 1:
            a = a.unsqueeze(-1)
        means, logvars = [], []
        for m in self.models:
            mean, logvar = m(s, a)
            means.append(mean)
            logvars.append(logvar)
        means   = torch.stack(means,   dim=0)   # (M, B, state_dim+1)
        logvars = torch.stack(logvars, dim=0)
        idx = torch.randint(0, self.n_models, (s.shape[0],), device=s.device)
        idx_exp = idx.view(1, -1, 1).expand(1, -1, means.shape[-1])
        mean   = means.gather(0, idx_exp).squeeze(0)
        logvar = logvars.gather(0, idx_exp).squeeze(0)
        std = (0.5 * logvar).exp()
        sample = mean + std * torch.randn_like(mean)
        delta_s = sample[..., :self.state_dim]
        reward  = sample[..., self.state_dim]
        return delta_s, reward


# ── Deployment factory ────────────────────────────────────────────────────────

def build_policy(ckpt, agent_cfg=None, training_cfg=None, device='cpu',
                 deterministic=True):
    """Load this checkpoint's actor as an inference Policy.

    Mirrors worker.py: a single `hidden` width, no head override. See
    olympus/common/policy.py for the contract.
    """
    hidden = policy_contract.resolved_hidden(agent_cfg, 'hidden', 128)
    actor = Actor(STATE_DIM, hidden)
    actor.load_state_dict(policy_contract.actor_state_dict(ckpt))
    actor.to(device).eval()
    return policy_contract.RecurrentActorPolicy(
        'mbpo_td3', STATE_DIM, actor, deterministic)
