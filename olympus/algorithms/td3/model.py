"""
model.py — recurrent TD3 actor-critic for direct CWND control.

Matches the setup in Soheil-ab/Orca (Abbasloo et al. SIGCOMM 2020) adapted to
this codebase's state / action / reward pipeline:

  Actor:
    input_proj : Linear(STATE_DIM → hidden) → LayerNorm → ReLU
    LSTM       : hidden → hidden, 1 layer
    post       : Linear(hidden → hidden//2) → ReLU
    action_mu  : Linear(hidden//2 → 1) → tanh           ∈ [-1, 1]
    → selected action plugin maps `a` to the TCP CWND update

  The tanh lives INSIDE the actor so the critic always sees a bounded
  action ∈ [-1, 1].  Putting tanh outside (worker only) caused the actor
  to push pre-tanh to ±10⁵ — the worker's actual action saturated at the
  bounds while the critic kept chasing larger Q.  Standard TD3 contract.

  Critic (twin Q for TD3):
    input_proj : Linear(STATE_DIM + 1 → hidden) → LayerNorm → ReLU
    LSTM       : hidden → hidden
    post       : Linear(hidden → hidden//2) → ReLU
    q_head     : Linear(hidden//2 → 1)
    — two independent copies Q1, Q2 for target-smoothing (TD3).

State (11 dims) and `normalize_state(...)` are identical to the OC/PPO pipeline
so rollout data is directly comparable across algorithms.

Exploration uses Gaussian noise on the pre-tanh action, decayed from
`noise_start → noise_end` over `noise_decay_steps`.  TD3 target policy
smoothing adds clipped noise inside the critic update itself.

Off-policy: the learner samples random seq_len chunks from the replay buffer;
each chunk is run through the LSTM with a zero-initialised hidden state — the
canonical "burn-in skipping" simplification that works well for short rollouts.
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

_STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    dtype=np.float32,
)


# ── State normalisation ───────────────────────────────────────────────────────

def normalize_state(info: dict) -> np.ndarray:
    """11-dim observation: TCP/kernel fields + worker-injected history.

    Identical contract to option_critic/model.py.normalize_state — duplicated
    here so each algorithm is the authoritative source for its own state.
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


_STATE_PLUGIN = load_state_module('td3')
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
    """Infer ORCA actor widths from a saved actor state dict."""
    if head_hidden is None:
        head_hidden = int(hidden) // 2
    if not state_dict:
        return int(hidden), int(head_hidden)

    input_weight = state_dict.get('input_proj.0.weight')
    action_weight = state_dict.get('action_head.weight')
    if input_weight is not None:
        hidden = int(input_weight.shape[0])
    if action_weight is not None:
        head_hidden = int(action_weight.shape[1])
    return int(hidden), int(head_hidden)


def actor_arch_from_checkpoint(ckpt: dict, hidden: int = 128,
                               head_hidden: int = None) -> tuple:
    actor_state = (ckpt or {}).get('actor') or (ckpt or {}).get('actor_state_dict')
    return actor_arch_from_state_dict(actor_state, hidden, head_hidden)


def actor_model_meta(actor: nn.Module) -> dict:
    return {
        'algorithm': 'td3',
        'state_dim': int(STATE_DIM),
        'hidden': int(actor.hidden),
        'head_hidden': int(actor.head_hidden),
    }


# ── Actor ─────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Deterministic policy: μ(s, h) = tanh(head(trunk(s, h))). The selected
    action plugin maps that bounded action to its TCP control meaning.

    Hidden state h is (1, B, hidden).  At inference the worker carries it
    across steps.  At training we zero-initialise per seq_len chunk.

    `hidden` is the LSTM width (h1 in the Orca reference). `head_hidden` is the
    post-LSTM dense width (h2). When `head_hidden` is None it defaults to
    `hidden // 2`, matching the original architecture in this repo.
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
        self.action_head = nn.Linear(head_hidden, ACTION_DIM)
        # Small init on the head keeps the initial action near tanh(0)=0 →
        # mult=1.0, no change, so early rollouts are calm.
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def _trunk(self, s: torch.Tensor, h=None):
        """
        s : (B, T, D).
        Returns (feat: (B, T, head_hidden), h_new: (h, c) pair, each (1, B, hidden)).
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

    def forward_sequence(self, s: torch.Tensor, h=None):
        """
        Training forward: returns (a: (B, T, 1), h_new) where a is the
        SQUASHED action ∈ [-1, 1].  Putting tanh inside the actor so the
        critic always sees a bounded action — this is essential.  Without
        it, the actor's gradient dQ/d(pre-tanh) has no saturation and
        the actor pushes pre-tanh to ±∞ (worker action saturates while
        the critic chases ever-larger Q).
        """
        feat, h_new = self._trunk(s, h)
        a = torch.tanh(self.action_head(feat))   # (B, T, 1)  ∈ [-1, 1]
        return a, h_new

    @torch.no_grad()
    def act(self, s_np: np.ndarray, h=None,
            noise_std: float = 0.0):
        """
        Inference: returns (action_float, mult_float, h_new).
          action ∈ [-1, 1] (post-tanh, post-noise, clipped).
          Replay stores this same bounded value so the critic and actor
          gradient pipelines see identical quantities — the standard TD3
          contract.
        """
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
    """Differentiable bounded-action to multiplier mapping."""
    return _ACTION_PLUGIN.to_multiplier(a)


# ── Critic (twin Q) ───────────────────────────────────────────────────────────

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
        Returns (q: (B, T), h_new).  a is the pre-tanh action (matches Actor).
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
    """Two independent SingleCritic towers — standard TD3."""
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


# ── TD3 losses ────────────────────────────────────────────────────────────────

def critic_loss(
    critic:       TwinCritic,
    actor_target: Actor,
    critic_target: TwinCritic,
    s_batch:      torch.Tensor,   # (B, T, D)
    a_batch:      torch.Tensor,   # (B, T)  pre-tanh actions the worker took
    r_batch:      torch.Tensor,   # (B, T)
    s2_batch:     torch.Tensor,   # (B, T, D)
    d_batch:      torch.Tensor,   # (B, T)
    mask:         torch.Tensor,   # (B, T)
    gamma:        float = 0.99,
    target_noise_std:  float = 0.2,
    target_noise_clip: float = 0.5,
) -> CriticInfo:
    """
    Clipped double-Q target with target-policy smoothing (TD3 §4.1).
    Mask ignores padding timesteps produced by the replay buffer's ragged
    trajectory chunks.
    """
    with torch.no_grad():
        a_next, _ = actor_target.forward_sequence(s2_batch)        # (B, T, 1) ∈ [-1, 1]
        a_next = a_next.squeeze(-1)                                # (B, T)
        # TD3 target-policy smoothing: clipped Gaussian noise on the
        # squashed target action, re-clamped into the action support.
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


def actor_loss(
    actor:  Actor,
    critic: TwinCritic,
    s_batch: torch.Tensor,     # (B, T, D)
    mask:    torch.Tensor,     # (B, T)
) -> ActorInfo:
    """
    Deterministic policy gradient on Q1 only (TD3 §4.2).  Maximise
    E[Q1(s, μ(s))].
    """
    a, _ = actor.forward_sequence(s_batch)        # (B, T, 1) ∈ [-1, 1]
    a = a.squeeze(-1)                              # (B, T)
    q1 = critic.q1_only(s_batch, a)                # (B, T)

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
