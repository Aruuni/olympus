"""
model.py — Multi-Agent Transformer (MAT-CTDE).

Architecture (parameter-shared across agents):

  Per-agent obs encoder:
    Linear(STATE_DIM → hidden) → LayerNorm → ReLU
    → TransformerEncoder(2 layers, n_heads=4, dim=hidden)
        Self-attention is over the AGENT dimension for the centralized critic.

  Actor head (decentralized at exec, parameter-shared):
    local single-agent embedding → Linear(hidden → hidden//2) → ReLU
    → mu_head:     Linear(hidden//2 → 1)       (state-dependent pre-tanh mean)
    → logsig_head: Linear(hidden//2 → 1)       (state-dependent log std)

  Centralized critic (training only):
    encoder over agents → mean-pool across agents → Linear → V(s_joint)

Action:
  raw  ~ Normal(μ, σ)
  bounded = tanh(raw)
  mult = selected action plugin mapping

State (11 dims) and `normalize_state(...)` are identical to the
olympus algorithms so rollout data is directly comparable across
single- and multi-agent runs.

Centralized training, decentralized execution (CTDE):
  - The actor is always queried from each agent's local observation embedding,
    both during rollout and PPO updates.
  - The critic always sees the joint state (concatenated agent embeddings) and
    outputs a scalar joint value.
"""

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from olympus.common.action_plugins import (
    assert_action_compatible,
    current_action_meta,
    load_action_module,
)
from olympus.states.kalman import update_kalman_min_rtt


# ── State / action constants ──────────────────────────────────────────────────

_ACTION_PLUGIN = load_action_module()
STATE_DIM = 11
ACTION_DIM = int(_ACTION_PLUGIN.ACTION_DIM)
STATE_FEATURE_VERSION = 'shared_no_bdp_observation_v1_srtt_unshifted'

Experience = namedtuple('Experience',
    ['state', 'action_raw', 'log_prob', 'value', 'reward', 'throughput',
     'agent_id', 'group_id', 'group_step', 'done', 'traj_id', 'step_in_traj'])

LossInfo = namedtuple('LossInfo',
    ['loss', 'policy', 'value', 'entropy', 'kl',
     'clip_frac', 'explained_var', 'fairness_mean'])

_STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)


def model_action_meta() -> dict:
    return current_action_meta()


def assert_checkpoint_action_compatible(ckpt: dict,
                                        source='checkpoint') -> None:
    assert_action_compatible(model_action_meta(), ckpt, source=source)
_STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    dtype=np.float32,
)


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
    kalman_min_rtt = update_kalman_min_rtt(info)
    info['kalman_min_rtt_us'] = kalman_min_rtt

    bw_ref = max(peak_thr, pacing_rate, 1.0)
    delta_rtt = float(np.clip((avg_urtt - prev_urtt) / max(prev_urtt, 1.0),
                              -1.0, 1.0))
    delta_cwnd = float(np.clip((cwnd - prev_cwnd) / max(prev_cwnd, 1.0),
                               -1.0, 1.0))
    cwnd_log = math.log1p(float(cwnd)) / math.log1p(10_000.0)

    s = np.array([
        delta_cwnd, avg_urtt / 1e5, cwnd_log,
        max(avg_thr / bw_ref, 0.0), max(pacing_rate / bw_ref, 0.0),
        max(packets_out / max(cwnd, 1), 0.0),
        delta_rtt, min(retrans_out / max(packets_out, 1.0), 1.0),
        avg_thr / 1e7, srtt_us / 1e5, kalman_min_rtt / 1e5,
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=_STATE_HIGH, neginf=_STATE_LOW)
    return np.clip(s, _STATE_LOW, _STATE_HIGH)


# ── Transformer encoder ──────────────────────────────────────────────────────

class AgentTransformerEncoder(nn.Module):
    """Multi-layer transformer encoder over the AGENT dimension.

    Input  : (B, N, STATE_DIM)   — N agents per step in a slot
    Output : (B, N, hidden)      — per-agent contextualised embeddings
    """
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=4 * hidden, dropout=dropout,
            batch_first=True, activation='gelu',
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, agent_states: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        agent_states     : (B, N, STATE_DIM)
        key_padding_mask : (B, N) bool with True for padding positions.
        Returns          : (B, N, hidden)
        """
        x = self.input_proj(agent_states)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


# ── Actor + Critic heads ─────────────────────────────────────────────────────

class ActorHead(nn.Module):
    """Per-agent Gaussian policy head over the encoder output."""
    def __init__(self, hidden: int = 128, logsig_init: float = -1.0):
        super().__init__()
        head_in = hidden // 2
        self.post = nn.Sequential(
            nn.Linear(hidden, head_in),
            nn.ReLU(),
        )
        self.mu     = nn.Linear(head_in, ACTION_DIM)
        self.logsig = nn.Linear(head_in, ACTION_DIM)
        nn.init.zeros_(self.mu.bias)
        nn.init.constant_(self.logsig.bias, logsig_init)
        nn.init.orthogonal_(self.mu.weight,     gain=0.01)
        nn.init.orthogonal_(self.logsig.weight, gain=0.01)

    def forward(self, agent_embed: torch.Tensor):
        feat = self.post(agent_embed)
        mu = self.mu(feat).squeeze(-1)
        logsig = self.logsig(feat).clamp(-2.0, 1.0).squeeze(-1)
        return mu, logsig


class CentralCriticHead(nn.Module):
    """Joint-state value head: pools encoder output across agents → V(s_joint)."""
    def __init__(self, hidden: int = 128):
        super().__init__()
        head_in = hidden // 2
        self.post = nn.Sequential(
            nn.Linear(hidden, head_in),
            nn.ReLU(),
        )
        self.v = nn.Linear(head_in, 1)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, agent_embeds: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        agent_embeds     : (B, N, hidden)
        key_padding_mask : (B, N) — True for padding positions to exclude.
        Returns          : (B,) joint value for each timestep.
        """
        if key_padding_mask is None:
            pooled = agent_embeds.mean(dim=1)
        else:
            valid = (~key_padding_mask).float().unsqueeze(-1)         # (B, N, 1)
            pooled = (agent_embeds * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.v(self.post(pooled)).squeeze(-1)


# ── Top-level network ────────────────────────────────────────────────────────

class MATPolicy(nn.Module):
    """
    Single transformer encoder shared by actor and critic. Actor reads each
    agent's contextualised embedding; critic reads the pooled joint embedding.

    Decentralized execution: at inference the worker calls `act(state)` with
    its own (1, 1, STATE_DIM) observation, the transformer attends only to
    that single token, and the actor produces μ, logσ for that one agent.

    Centralized training: the learner stacks all N agents at each timestep
    along the agent axis, runs the encoder once, and computes per-agent log
    probs (actor) and a single joint value (critic) per timestep.
    """
    MULT_MIN = float(_ACTION_PLUGIN.MULTIPLIER_MIN)
    MULT_MAX = float(_ACTION_PLUGIN.MULTIPLIER_MAX)

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128,
                 n_layers: int = 2, n_heads: int = 4, logsig_init: float = -1.0):
        super().__init__()
        self.encoder = AgentTransformerEncoder(state_dim, hidden,
                                               n_layers=n_layers, n_heads=n_heads)
        self.actor   = ActorHead(hidden, logsig_init=logsig_init)
        self.critic  = CentralCriticHead(hidden)

    def forward_joint(self, joint_states: torch.Tensor,
                      key_padding_mask: torch.Tensor = None):
        """
        joint_states     : (B, N, STATE_DIM)
        key_padding_mask : (B, N) bool padding mask
        Returns          : (mu, logsig, value)
                             mu, logsig : (B, N)
                             value       : (B,)
        """
        # Actor must remain decentralized: PPO evaluates the same single-agent
        # transformer path that produced rollout log-probs in each worker, but
        # independently for each agent. The joint transformer output is reserved
        # for the centralized critic.
        B, N, D = joint_states.shape
        local_embed = self.encoder(joint_states.reshape(B * N, 1, D))
        local_embed = local_embed.reshape(B, N, -1)
        joint_embed = self.encoder(joint_states, key_padding_mask=key_padding_mask)
        mu, logsig = self.actor(local_embed)
        value = self.critic(joint_embed, key_padding_mask=key_padding_mask)
        return mu, logsig, value, joint_embed

    @torch.no_grad()
    def act(self, state_np: np.ndarray, deterministic: bool = False):
        """
        Decentralized single-agent inference. state_np : (STATE_DIM,) numpy.
        Returns (action_raw, mult, log_prob, value, mu, sigma).
        """
        device = next(self.parameters()).device
        s = torch.from_numpy(state_np).reshape(1, 1, -1).to(device)   # (B=1, N=1, D)
        mu, logsig, value, _ = self.forward_joint(s)
        mu_s     = mu[0, 0]
        logsig_s = logsig[0, 0]
        sigma    = logsig_s.exp()
        action_raw = mu_s if deterministic else (mu_s + sigma * torch.randn_like(mu_s))
        log_prob = (-0.5 * ((action_raw - mu_s) / (sigma + 1e-8)) ** 2
                    - logsig_s - 0.5 * math.log(2.0 * math.pi))
        mult = float(_ACTION_PLUGIN.to_multiplier(
            torch.tanh(action_raw)).item())
        return (float(action_raw.item()),
                mult,
                float(log_prob.item()),
                float(value.reshape(-1)[-1].item()),
                float(mu_s.item()),
                float(sigma.item()))


# ── GAE ──────────────────────────────────────────────────────────────────────

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: float,
                gamma: float = 0.99, lam: float = 0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        non_term = 1.0 - float(dones[t])
        next_val = last_value if t == T - 1 else values[t + 1]
        delta    = rewards[t] + gamma * next_val * non_term - values[t]
        last_gae = delta + gamma * lam * non_term * last_gae
        adv[t]   = last_gae
    return adv, adv + values


# ── MAPPO loss (joint critic, per-agent actor) ───────────────────────────────

def mappo_loss(net: MATPolicy,
               s_batch:           torch.Tensor,    # (B, N, D)
               a_raw_batch:       torch.Tensor,    # (B, N)
               old_logp_batch:    torch.Tensor,    # (B, N)
               adv_batch:         torch.Tensor,    # (B, N) per-agent advantage
               ret_batch:         torch.Tensor,    # (B,)   joint return
               old_value_batch:   torch.Tensor,    # (B,)   joint value
               agent_mask:        torch.Tensor,    # (B, N) 1 where agent is valid
               clip_eps: float = 0.2,
               c_value: float = 0.5,
               c_entropy: float = 0.01,
               value_clip: float = 0.2) -> LossInfo:
    """
    MAPPO with:
      - Per-agent advantage A_i = (R_joint - V_old(joint)) shared across agents,
        but per-agent log-prob ratio so the policy gradient is local.
      - Joint value loss (clipped) on the centralized critic.
      - Entropy bonus per agent.

    Per-agent advantage normalisation across the masked entries.
    """
    pad_mask = (agent_mask < 0.5)
    mu, logsig, value, _ = net.forward_joint(s_batch, key_padding_mask=pad_mask)

    sigma    = logsig.exp()
    log_prob = (-0.5 * ((a_raw_batch - mu) / (sigma + 1e-8)) ** 2
                - logsig - 0.5 * math.log(2.0 * math.pi))
    entropy  = logsig + 0.5 * math.log(2.0 * math.pi * math.e)

    mvalid = agent_mask.bool()
    row_valid = agent_mask.sum(dim=1) > 0.5
    if mvalid.any():
        adv_mean = adv_batch[mvalid].mean()
        # unbiased=True returns NaN for a single valid element. Mask-heavy
        # minibatches happen with late-joining flows, so use population std.
        adv_std  = adv_batch[mvalid].std(unbiased=False).clamp_min(1e-6)
    else:
        adv_mean = torch.zeros((), device=adv_batch.device)
        adv_std  = torch.ones((),  device=adv_batch.device)
    adv_norm = (adv_batch - adv_mean) / adv_std
    adv_norm = torch.where(mvalid, adv_norm, torch.zeros_like(adv_norm))

    log_ratio = torch.where(mvalid, log_prob - old_logp_batch,
                            torch.zeros_like(log_prob))
    ratio     = log_ratio.clamp(-20.0, 20.0).exp()
    unclipped = ratio * adv_norm
    clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    pg_loss   = torch.where(mvalid, -torch.min(unclipped, clipped),
                            torch.zeros_like(unclipped))

    v_clipped = old_value_batch + torch.clamp(value - old_value_batch,
                                              -value_clip, value_clip)
    v_loss_1  = (value     - ret_batch) ** 2
    v_loss_2  = (v_clipped - ret_batch) ** 2
    v_loss    = 0.5 * torch.max(v_loss_1, v_loss_2)
    ent_loss  = torch.where(mvalid, -entropy, torch.zeros_like(entropy))

    def _agent_mean(t):
        masked = torch.where(mvalid, t, torch.zeros_like(t))
        return masked.sum() / agent_mask.sum().clamp_min(1.0)

    def _row_mean(t):
        row_f = row_valid.float()
        return (t * row_f).sum() / row_f.sum().clamp_min(1.0)

    l_policy  = _agent_mean(pg_loss)
    l_entropy = _agent_mean(ent_loss)
    l_value   = _row_mean(v_loss)
    loss      = l_policy + c_value * l_value + c_entropy * l_entropy

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float()
        clip_frac = _agent_mean(clip_frac)
        kl_approx = _agent_mean((ratio - 1.0) - log_ratio)
        if row_valid.any():
            ret_v = ret_batch[row_valid]
            val_v = value[row_valid]
            var_ret = ret_v.var(unbiased=False).clamp_min(1e-8)
            explained = 1.0 - (ret_v - val_v).var(unbiased=False) / var_ret
        else:
            explained = torch.zeros((), device=ret_batch.device)

    return LossInfo(
        loss = loss,
        policy = l_policy.item(),
        value  = l_value.item(),
        entropy = l_entropy.item(),
        kl = float(kl_approx.item()),
        clip_frac = float(clip_frac.item()),
        explained_var = float(explained.item()),
        fairness_mean = 0.0,  # filled in by the learner
    )
