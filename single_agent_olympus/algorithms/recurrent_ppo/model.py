"""
model.py — Recurrent PPO network for direct CWND control.

Ported from recurrent_ppo_clean_slate/model.py with two adaptations for the
single_agent_olympus framework:
  - Uses the hot-swappable `normalize_state` contract
    so rollout data is directly comparable across algorithms.
  - State contract is owned by this module (`STATE_DIM`, `STATE_FEATURE_VERSION`)
    rather than imported from a shared module.

Architecture:
  Shared:  Linear(STATE_DIM → hidden) → LayerNorm → ReLU
           → GRU(hidden → hidden, 1 layer)
           → Linear(hidden → hidden//2) → ReLU
  Actor:  Linear(hidden//2 → 1)  → μ      (state-dependent mean, pre-tanh)
          Linear(hidden//2 → 1)  → logσ   (state-dependent log-stddev)
  Critic: Linear(hidden//2 → 1)  → V(s)

Action:
  raw  ~ Normal(μ, σ)
  log_mult = _LOG_MULT_MID + _LOG_MULT_HALF · tanh(raw)   ∈ [log 0.5, log 2.0]
  mult     = exp(log_mult)                                 ∈ [0.5, 2.0]
"""

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from single_agent_olympus.common.state_plugins import (
    assert_state_compatible,
    current_state_name,
    load_state_module,
    state_meta,
)


# ── State / action constants ──────────────────────────────────────────────────

STATE_DIM = 11
ACTION_DIM = 1
# Kept verbatim so existing checkpoints' `model_meta.state_features` still match.
STATE_FEATURE_VERSION = 'kalman_observation_v1_srtt_unshifted'

Experience = namedtuple('Experience',
    ['state', 'action_raw', 'log_prob', 'value',
     'reward', 'done', 'traj_id', 'step_in_traj'])

# Log-space bounds for the CWND multiplier
_LOG_MULT_MIN  = math.log(0.5)
_LOG_MULT_MAX  = math.log(2.0)
_LOG_MULT_MID  = (_LOG_MULT_MAX + _LOG_MULT_MIN) / 2.0
_LOG_MULT_HALF = (_LOG_MULT_MAX - _LOG_MULT_MIN) / 2.0

LossInfo = namedtuple('LossInfo',
    ['loss',
     'policy', 'value', 'entropy', 'kl',
     'clip_frac', 'explained_var',
     'approx_ent_bits'])

_STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    dtype=np.float32,
)


# ── Legacy state normalisation, replaced by the selected state plugin below ──

def normalize_state(info: dict) -> np.ndarray:
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


_STATE_PLUGIN = load_state_module('recurrent_ppo')
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


# ── Network ───────────────────────────────────────────────────────────────────

class RecurrentPPONet(nn.Module):
    """
    Shared-trunk recurrent actor-critic.

    forward_sequence(s, h)  — training-time full sequence forward
        s : (B, T, STATE_DIM)
        h : (1, B, hidden) or None
        Returns (mu, logsig, value, h_new) shaped (B, T) for mu/logsig/value.

    act(s, h) — single-step inference; returns (action_raw, mult, log_prob,
                value, h_new, mu, sigma).
    """

    MULT_MIN = 0.5
    MULT_MAX = 2.0

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256,
                 logsig_init: float = -1.0):
        super().__init__()
        self.hidden = hidden

        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)

        head_in = hidden // 2
        self.post = nn.Sequential(
            nn.Linear(hidden, head_in),
            nn.ReLU(),
        )

        self.actor_mu     = nn.Linear(head_in, 1)
        self.actor_logsig = nn.Linear(head_in, 1)
        nn.init.zeros_(self.actor_mu.bias)
        nn.init.constant_(self.actor_logsig.bias, logsig_init)

        self.critic = nn.Linear(head_in, 1)
        nn.init.zeros_(self.critic.bias)

        nn.init.orthogonal_(self.actor_mu.weight,     gain=0.01)
        nn.init.orthogonal_(self.actor_logsig.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight,       gain=1.0)

    def _trunk(self, s: torch.Tensor, h: torch.Tensor = None):
        if s.dim() == 2:
            s = s.unsqueeze(1)
        B, T, D = s.shape
        x = self.input_proj(s.reshape(B * T, D)).reshape(B, T, self.hidden)
        if h is None:
            h = torch.zeros(1, B, self.hidden, device=s.device)
        gru_out, h_new = self.gru(x, h)
        feat = self.post(gru_out.reshape(B * T, self.hidden)).reshape(B, T, -1)
        return feat, h_new

    def forward_sequence(self, s: torch.Tensor, h: torch.Tensor = None):
        feat, h_new = self._trunk(s, h)
        mu     = self.actor_mu(feat)
        logsig = self.actor_logsig(feat).clamp(-2.0, 1.0)
        value  = self.critic(feat).squeeze(-1)
        return mu.squeeze(-1), logsig.squeeze(-1), value, h_new

    @torch.no_grad()
    def act(self, s_np: np.ndarray, h: torch.Tensor = None,
            deterministic: bool = False):
        device = next(self.parameters()).device
        s = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0).to(device)
        mu, logsig, value, h_new = self.forward_sequence(s, h)
        mu_s     = mu[0, -1]
        logsig_s = logsig[0, -1]
        sigma    = logsig_s.exp()

        if deterministic:
            action_raw = mu_s
        else:
            action_raw = mu_s + sigma * torch.randn_like(mu_s)

        log_prob = (-0.5 * ((action_raw - mu_s) / (sigma + 1e-8)) ** 2
                    - logsig_s - 0.5 * math.log(2.0 * math.pi))

        log_mult = _LOG_MULT_MID + _LOG_MULT_HALF * torch.tanh(action_raw)
        mult     = float(torch.exp(log_mult).clamp(self.MULT_MIN, self.MULT_MAX).item())

        return (float(action_raw.item()),
                mult,
                float(log_prob.item()),
                float(value.reshape(-1)[-1].item()),
                h_new,
                float(mu_s.item()),
                float(sigma.item()))


# ── PPO loss + GAE ────────────────────────────────────────────────────────────

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: float,
                gamma: float = 0.99, lam: float = 0.95):
    """Generalised Advantage Estimation over one trajectory."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        non_term = 1.0 - float(dones[t])
        next_val = last_value if t == T - 1 else values[t + 1]
        delta    = rewards[t] + gamma * next_val * non_term - values[t]
        last_gae = delta + gamma * lam * non_term * last_gae
        adv[t]   = last_gae
    returns = adv + values
    return adv, returns


def ppo_loss(net: RecurrentPPONet,
             s_batch: torch.Tensor,
             a_raw_batch: torch.Tensor,
             old_logp_batch: torch.Tensor,
             adv_batch: torch.Tensor,
             ret_batch: torch.Tensor,
             old_value_batch: torch.Tensor,
             mask: torch.Tensor,
             clip_eps: float = 0.2,
             c_value: float = 0.5,
             c_entropy: float = 0.01,
             value_clip: float = 0.2) -> LossInfo:
    """Clipped-objective PPO loss over masked sequences."""
    mu, logsig, value, _ = net.forward_sequence(s_batch)

    sigma    = logsig.exp()
    log_prob = (-0.5 * ((a_raw_batch - mu) / (sigma + 1e-8)) ** 2
                - logsig - 0.5 * math.log(2.0 * math.pi))
    entropy  = logsig + 0.5 * math.log(2.0 * math.pi * math.e)

    mvalid = mask.bool()
    if mvalid.any():
        adv_mean = adv_batch[mvalid].mean()
        adv_std  = adv_batch[mvalid].std().clamp_min(1e-8)
    else:
        adv_mean = torch.zeros((), device=adv_batch.device)
        adv_std  = torch.ones((), device=adv_batch.device)
    adv_norm = (adv_batch - adv_mean) / adv_std

    ratio     = (log_prob - old_logp_batch).exp()
    unclipped = ratio * adv_norm
    clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    pg_loss   = -torch.min(unclipped, clipped)

    v_clipped = old_value_batch + torch.clamp(value - old_value_batch,
                                              -value_clip, value_clip)
    v_loss_1  = (value     - ret_batch) ** 2
    v_loss_2  = (v_clipped - ret_batch) ** 2
    v_loss    = 0.5 * torch.max(v_loss_1, v_loss_2)

    ent_loss  = -entropy

    def _masked_mean(t):
        return (t * mask).sum() / mask.sum().clamp_min(1.0)

    l_policy  = _masked_mean(pg_loss)
    l_value   = _masked_mean(v_loss)
    l_entropy = _masked_mean(ent_loss)
    loss      = l_policy + c_value * l_value + c_entropy * l_entropy

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float()
        clip_frac = _masked_mean(clip_frac)
        kl_approx = _masked_mean(old_logp_batch - log_prob)
        ret_valid = ret_batch[mvalid] if mvalid.any() else ret_batch
        val_valid = value[mvalid]    if mvalid.any() else value
        if ret_valid.numel() > 1:
            var_ret   = ret_valid.var().clamp_min(1e-8)
            explained = 1.0 - (ret_valid - val_valid).var() / var_ret
        else:
            explained = torch.zeros((), device=ret_batch.device)
        ent_bits = _masked_mean(entropy) / math.log(2.0)

    return LossInfo(
        loss      = loss,
        policy    = l_policy.item(),
        value     = l_value.item(),
        entropy   = l_entropy.item(),
        kl        = float(kl_approx.item()),
        clip_frac = float(clip_frac.item()),
        explained_var = float(explained.item()),
        approx_ent_bits = float(ent_bits.item()),
    )
