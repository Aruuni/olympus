"""
model.py — DreamerV3 networks for direct CWND control.

Components:

  RSSM        : recurrent state-space model.
                  Sequence model:  h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})
                  Prior   p(z_t | h_t)        — categorical (groups × classes)
                  Posterior q(z_t | h_t, e_t) — categorical, used when an obs is available
                Sampling uses straight-through one-hot so categorical latents are
                trainable end-to-end.

  Encoder/Decoder : MLPs over the 11-dim observation. The decoder is the world
                    model's reconstruction head; trained with symlog-MSE so the
                    obs scale is bounded.

  RewardHead, ContinueHead : (h, z) → reward / continue probability.
                             Reward uses twohot over a fixed grid of bins (after
                             symlog) so the head can represent multimodal returns
                             without exploding the loss scale.

  Actor       : (h, z) → squashed Normal over the bounded action ∈ [-1, 1].
                Continuous action distribution; entropy term added in the loss.

  Critic      : (h, z) → twohot over discounted return (after symlog).
                EMA target net + slow-moving return-percentile normaliser keeps
                advantages O(1) without hand-tuned reward scaling.

  Helpers     : symlog/symexp, twohot encoding, KL with free bits and balancing,
                lambda-return computation on imagined trajectories.

DreamerV3 quirks adopted:
  - symlog/symexp on rewards and value targets (bounded heads, multi-scale).
  - twohot categorical for value/reward heads (255 bins each).
  - KL balancing (0.8 dynamics / 0.2 representation) with 1.0-nat free bits.
  - Return EMA percentile normaliser for actor advantages.
  - Discrete latent: K groups × C classes one-hot per group, straight-through.

Quirks intentionally simplified for an 11-dim, fully-observed problem:
  - Obs encoder is a small MLP (no CNN needed).
  - K=8, C=8 latent (64-dim) instead of the paper's 32×32. Plenty for 11-dim obs.
  - Single hidden width for everything (no separate attention/conv variants).
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
STATE_FEATURE_VERSION = 'kalman_observation_v1_srtt_unshifted'

Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'traj_id', 'step_in_traj'])

WorldModelInfo = namedtuple('WorldModelInfo',
    ['loss', 'recon', 'reward', 'cont', 'kl_dyn', 'kl_rep'])
ActorInfo  = namedtuple('ActorInfo',  ['loss', 'entropy', 'a_mean', 'a_abs', 'q_mean'])
CriticInfo = namedtuple('CriticInfo', ['loss', 'value_mean', 'target_mean'])

_STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    dtype=np.float32,
)


# ── State normalisation (identical to orca_td3) ──────────────────────────────

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


_STATE_PLUGIN = load_state_module('dreamer_v3')
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


# ── DreamerV3 helpers ────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHot:
    """
    Twohot categorical over a fixed grid of bins. Encodes a continuous value as
    a distribution where mass is split between the two adjacent bins so the
    expectation matches the original value. Used for the reward and value heads
    so a single distribution can represent rewards/returns spanning many
    orders of magnitude after symlog.
    """
    def __init__(self, low: float = -20.0, high: float = 20.0, n_bins: int = 255):
        self.low    = float(low)
        self.high   = float(high)
        self.n_bins = int(n_bins)
        self.bins   = torch.linspace(self.low, self.high, self.n_bins)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., ) → (..., n_bins) twohot distribution. Caller may pass
        symlog-transformed values; the bin grid is fixed in symlog space."""
        bins = self.bins.to(x.device)
        x_c = x.clamp(self.low, self.high)
        idx = (x_c - self.low) / (self.high - self.low) * (self.n_bins - 1)
        idx_low  = idx.floor().long().clamp(0, self.n_bins - 1)
        idx_high = (idx_low + 1).clamp(0, self.n_bins - 1)
        weight_high = (idx - idx_low.float()).unsqueeze(-1)
        weight_low  = 1.0 - weight_high
        out = torch.zeros(*x.shape, self.n_bins, device=x.device, dtype=x.dtype)
        out.scatter_(-1, idx_low.unsqueeze(-1),  weight_low)
        out.scatter_(-1, idx_high.unsqueeze(-1), weight_high)
        return out

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """logits : (..., n_bins) → (...,) expected value over the bin grid."""
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bins.to(logits.device)).sum(dim=-1)


def kl_categorical(post_logits: torch.Tensor, prior_logits: torch.Tensor,
                   free_bits: float = 1.0,
                   balance: float = 0.8) -> tuple:
    """
    KL divergence between two factorised categorical distributions with KL
    balancing (DreamerV3) and free bits per group:

      kl_dyn = KL(stop_grad(post) || prior)   — drags the prior toward posterior
      kl_rep = KL(post || stop_grad(prior))   — drags the posterior toward prior
      total  = balance * max(kl_dyn, free) + (1 - balance) * max(kl_rep, free)

    Each set of logits has shape (..., groups, classes); KL is summed over groups.
    Returns (per-sample loss, mean kl_dyn, mean kl_rep) for diagnostics.
    """
    post     = torch.distributions.Categorical(logits=post_logits)
    prior    = torch.distributions.Categorical(logits=prior_logits)
    post_sg  = torch.distributions.Categorical(logits=post_logits.detach())
    prior_sg = torch.distributions.Categorical(logits=prior_logits.detach())

    kl_dyn = torch.distributions.kl_divergence(post_sg, prior).sum(dim=-1)
    kl_rep = torch.distributions.kl_divergence(post,    prior_sg).sum(dim=-1)
    fb     = torch.full_like(kl_dyn, float(free_bits))
    loss   = (balance * torch.maximum(kl_dyn, fb)
              + (1.0 - balance) * torch.maximum(kl_rep, fb))
    return loss, kl_dyn.detach().mean(), kl_rep.detach().mean()


def lambda_return(rewards: torch.Tensor,
                  values:  torch.Tensor,
                  conts:   torch.Tensor,
                  bootstrap: torch.Tensor,
                  lam: float = 0.95,
                  gamma: float = 0.99) -> torch.Tensor:
    """
    Generalised advantage / lambda-return (Sutton-style backward recursion).

      rewards : (T, B)     — predicted reward at imagined step t
      values  : (T, B)     — predicted value at imagined step t
      conts   : (T, B)     — predicted continue probability ∈ [0, 1]
      bootstrap : (B,)     — value at step T (terminal bootstrap)
      lam     : λ blend between 1-step and Monte-Carlo
    Returns (T, B) of lambda returns for each imagined step.
    """
    T = rewards.shape[0]
    G = torch.zeros_like(rewards)
    g = bootstrap
    for t in reversed(range(T)):
        v_next = values[t]
        g = rewards[t] + gamma * conts[t] * ((1.0 - lam) * v_next + lam * g)
        G[t] = g
    return G


# ── World model components ───────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, act=nn.SiLU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RSSM(nn.Module):
    """
    Recurrent state-space model. Maintains a deterministic GRU state h and a
    discrete stochastic latent z (groups × classes one-hot, flattened).
    """
    def __init__(self, embed_dim: int, action_dim: int = ACTION_DIM,
                 latent_groups: int = 8, latent_classes: int = 8,
                 h_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.latent_groups   = latent_groups
        self.latent_classes  = latent_classes
        self.latent_dim      = latent_groups * latent_classes
        self.h_dim           = h_dim

        self.gru_input = nn.Sequential(
            nn.Linear(self.latent_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.gru        = nn.GRUCell(hidden, h_dim)
        self.prior_net  = MLP(h_dim,             hidden, self.latent_dim, n_layers=1)
        self.post_net   = MLP(h_dim + embed_dim, hidden, self.latent_dim, n_layers=1)

    def initial(self, batch_size: int, device):
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def _sample_z(self, logits: torch.Tensor, sample: bool = True):
        """Return a categorical latent, optionally using its deterministic mode."""
        shape   = logits.shape[:-1] + (self.latent_groups, self.latent_classes)
        logits  = logits.reshape(*shape)
        probs   = torch.softmax(logits, dim=-1)
        # Avoid an exact 0 in any class (paper "unimix").
        probs   = 0.99 * probs + 0.01 / self.latent_classes
        if sample:
            one_hot = torch.distributions.OneHotCategorical(probs=probs).sample()
            z_st = one_hot + probs - probs.detach()
        else:
            z_st = F.one_hot(
                probs.argmax(dim=-1), num_classes=self.latent_classes
            ).to(dtype=probs.dtype)
        return z_st.reshape(*z_st.shape[:-2], self.latent_dim), logits

    def step(self, h_prev, z_prev, a_prev, embed, sample: bool = True):
        """One observed step: returns (h, z, prior_logits, post_logits)."""
        x  = self.gru_input(torch.cat([z_prev, a_prev], dim=-1))
        h  = self.gru(x, h_prev)
        prior_raw = self.prior_net(h)
        post_raw  = self.post_net(torch.cat([h, embed], dim=-1))
        z, post_logits  = self._sample_z(post_raw, sample=sample)
        # Only logits are consumed for the prior here; never waste an RNG draw.
        _, prior_logits = self._sample_z(prior_raw, sample=False)
        return h, z, prior_logits, post_logits

    def imagine_step(self, h_prev, z_prev, a_prev):
        """One imagined step using the prior (no observation available)."""
        x = self.gru_input(torch.cat([z_prev, a_prev], dim=-1))
        h = self.gru(x, h_prev)
        prior_raw = self.prior_net(h)
        z, _ = self._sample_z(prior_raw)
        return h, z

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor,
                init: tuple = None):
        """
        Run the RSSM through a sequence of observations.

          embeds  : (B, T, embed_dim)
          actions : (B, T, action_dim) — action a_{t-1} that produced embed_t.
                    (i.e. actions[:, 0] is the action taken before the first
                    observation; pass zeros for the very first step in a
                    fresh trajectory.)
        Returns dict with stacked (T, B, …) tensors.
        """
        B, T, _ = embeds.shape
        if init is None:
            h, z = self.initial(B, embeds.device)
        else:
            h, z = init
        Hs, Zs, Pri, Post = [], [], [], []
        for t in range(T):
            h, z, prior_logits, post_logits = self.step(h, z, actions[:, t], embeds[:, t])
            Hs.append(h); Zs.append(z); Pri.append(prior_logits); Post.append(post_logits)
        return dict(
            h     = torch.stack(Hs,   dim=0),  # (T, B, h_dim)
            z     = torch.stack(Zs,   dim=0),  # (T, B, latent_dim)
            prior = torch.stack(Pri,  dim=0),  # (T, B, K, C)
            post  = torch.stack(Post, dim=0),
        )


class Encoder(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, embed_dim: int = 128,
                 hidden: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = MLP(state_dim, hidden, embed_dim, n_layers=2)

    def forward(self, s):
        return self.net(symlog(s))


class Decoder(nn.Module):
    """Predicts symlog(s) from (h, z); reconstruction loss is MSE in symlog space."""
    def __init__(self, h_dim: int, latent_dim: int,
                 state_dim: int = STATE_DIM, hidden: int = 256):
        super().__init__()
        self.net = MLP(h_dim + latent_dim, hidden, state_dim, n_layers=2)

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))


class RewardHead(nn.Module):
    def __init__(self, h_dim: int, latent_dim: int, n_bins: int,
                 hidden: int = 256):
        super().__init__()
        self.net = MLP(h_dim + latent_dim, hidden, n_bins, n_layers=2)

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))


class ContinueHead(nn.Module):
    def __init__(self, h_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = MLP(h_dim + latent_dim, hidden, 1, n_layers=2)

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


# ── Actor and critic ─────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Squashed Gaussian over the bounded pre-tanh action ∈ [-1, 1].

    Inputs come from the world model: concat(h, z). At rollout time the worker
    runs the encoder + RSSM step + actor; at training time the actor is queried
    on imagined (h, z) states sampled from the world model rollout.
    """
    LOG_STD_MIN = -2.0
    LOG_STD_MAX = 0.0

    def __init__(self, h_dim: int, latent_dim: int, hidden: int = 256,
                 log_std_min: float = None, log_std_max: float = None):
        super().__init__()
        self.log_std_min = self.LOG_STD_MIN if log_std_min is None else float(log_std_min)
        self.log_std_max = self.LOG_STD_MAX if log_std_max is None else float(log_std_max)
        self.net = MLP(h_dim + latent_dim, hidden, 2 * ACTION_DIM, n_layers=2)
        # Small init on the head keeps the initial squashed action near 0.
        with torch.no_grad():
            last = [m for m in self.net.net if isinstance(m, nn.Linear)][-1]
            last.weight.mul_(0.01)
            last.bias.zero_()

    def forward(self, h, z):
        x = self.net(torch.cat([h, z], dim=-1))
        mu, log_std = x.chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def dist(self, h, z):
        mu, log_std = self(h, z)
        std = log_std.exp()
        base = torch.distributions.Normal(mu, std)
        return base, mu, log_std

    @torch.no_grad()
    def act(self, h, z, deterministic: bool = False):
        """Returns (action ∈ [-1, 1], mult > 0, mu, log_std)."""
        base, mu, log_std = self.dist(h, z)
        raw = mu if deterministic else base.rsample()
        a   = torch.tanh(raw)
        mult = _ACTION_PLUGIN.to_multiplier(a)
        return a, mult, mu, log_std


def action_to_mult(action):
    return _ACTION_PLUGIN.to_multiplier(action)


def actor_log_prob(base_dist, raw_action: torch.Tensor) -> torch.Tensor:
    """log π(a | s) under a tanh-squashed Normal: subtract the tanh Jacobian."""
    log_prob = base_dist.log_prob(raw_action).sum(dim=-1)
    # log|d tanh / dx| = log(1 - tanh(x)^2)
    log_prob = log_prob - (2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
                          ).sum(dim=-1)
    return log_prob


class Critic(nn.Module):
    """Twohot-categorical value head over symlog(return)."""
    def __init__(self, h_dim: int, latent_dim: int, n_bins: int,
                 hidden: int = 256):
        super().__init__()
        self.net = MLP(h_dim + latent_dim, hidden, n_bins, n_layers=2)

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))


# ── Return EMA percentile normaliser (DreamerV3 actor advantage scaling) ─────

class ReturnEMA:
    """
    Tracks low/high percentiles of imagined returns with EMA smoothing.
    Actor advantages are divided by max(1.0, high - low) so policy gradient
    scale stays consistent across reward magnitudes without hand tuning.
    """
    def __init__(self, decay: float = 0.99,
                 low_pct: float = 0.05, high_pct: float = 0.95):
        self.decay = float(decay)
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)
        self.low  = 0.0
        self.high = 1.0
        self._initialised = False

    def update(self, returns: torch.Tensor):
        with torch.no_grad():
            flat = returns.detach().flatten()
            if flat.numel() == 0:
                return
            low_b  = float(torch.quantile(flat, self.low_pct).item())
            high_b = float(torch.quantile(flat, self.high_pct).item())
            if not self._initialised:
                self.low, self.high = low_b, high_b
                self._initialised = True
            else:
                self.low  = self.decay * self.low  + (1.0 - self.decay) * low_b
                self.high = self.decay * self.high + (1.0 - self.decay) * high_b

    def scale(self) -> float:
        return max(1.0, float(self.high - self.low))


# ── World model wrapper ──────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """Encoder + RSSM + (decoder, reward, continue) — trained jointly on real
    sequences. The Actor/Critic are held separately so they can be trained on
    imagined rollouts without re-running gradients through the world model."""

    def __init__(self,
                 state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM,
                 hidden: int = 256,
                 embed_dim: int = 128,
                 h_dim: int = 256,
                 latent_groups: int = 8,
                 latent_classes: int = 8,
                 reward_bins: int = 255,
                 reward_low: float = -20.0,
                 reward_high: float = 20.0):
        super().__init__()
        self.encoder       = Encoder(state_dim, embed_dim, hidden)
        self.rssm          = RSSM(embed_dim, action_dim,
                                  latent_groups, latent_classes, h_dim, hidden)
        latent_dim         = latent_groups * latent_classes
        self.decoder       = Decoder(h_dim, latent_dim, state_dim, hidden)
        self.reward_head   = RewardHead(h_dim, latent_dim, reward_bins, hidden)
        self.continue_head = ContinueHead(h_dim, latent_dim, hidden)
        self.twohot_reward = TwoHot(reward_low, reward_high, reward_bins)
        self.h_dim         = h_dim
        self.latent_dim    = latent_dim
        self.action_dim    = action_dim

    def loss(self, states, actions, rewards, dones, mask,
             free_bits: float = 1.0, kl_balance: float = 0.8,
             kl_weight: float = 1.0, reward_weight: float = 1.0,
             continue_weight: float = 1.0, recon_weight: float = 1.0):
        """
        Compute the joint world-model loss on a batch of sequences.

          states  : (B, T, state_dim)  — observations
          actions : (B, T, action_dim) — action a_{t-1} preceding state_t
                                         (action[:, 0] should be zero for fresh episodes)
          rewards : (B, T)             — reward observed at step t
          dones   : (B, T)             — terminal flag at step t
          mask    : (B, T)             — 1 for valid timesteps, 0 for padding
        Returns (loss, info, posterior_h, posterior_z) so the caller can use
        the posterior states as imagination starting points.
        """
        embeds = self.encoder(states)                       # (B, T, embed_dim)
        out    = self.rssm.observe(embeds, actions)          # T-major dict
        h_t    = out['h'].transpose(0, 1)                    # (B, T, h_dim)
        z_t    = out['z'].transpose(0, 1)                    # (B, T, latent_dim)
        prior  = out['prior'].transpose(0, 1)                # (B, T, K, C)
        post   = out['post'].transpose(0, 1)

        # Reconstruction in symlog space
        recon  = self.decoder(h_t, z_t)
        recon_target = symlog(states)
        recon_per = ((recon - recon_target) ** 2).mean(dim=-1)
        recon_loss = (recon_per * mask).sum() / mask.sum().clamp_min(1.0)

        # Reward head: twohot CE on symlog reward
        r_logits  = self.reward_head(h_t, z_t)               # (B, T, n_bins)
        r_target  = self.twohot_reward.encode(symlog(rewards))
        log_probs = F.log_softmax(r_logits, dim=-1)
        reward_per = -(r_target * log_probs).sum(dim=-1)
        reward_loss = (reward_per * mask).sum() / mask.sum().clamp_min(1.0)

        # Continue head: BCE on (1 - done)
        c_logits = self.continue_head(h_t, z_t)
        cont_per = F.binary_cross_entropy_with_logits(
            c_logits, 1.0 - dones, reduction='none')
        cont_loss = (cont_per * mask).sum() / mask.sum().clamp_min(1.0)

        # KL between prior and posterior, with free bits and balancing
        kl_per, kl_dyn, kl_rep = kl_categorical(post, prior, free_bits, kl_balance)
        kl_loss = (kl_per * mask).sum() / mask.sum().clamp_min(1.0)

        loss = (recon_weight * recon_loss
                + reward_weight * reward_loss
                + continue_weight * cont_loss
                + kl_weight * kl_loss)

        info = WorldModelInfo(
            loss   = loss,
            recon  = float(recon_loss.detach().item()),
            reward = float(reward_loss.detach().item()),
            cont   = float(cont_loss.detach().item()),
            kl_dyn = float(kl_dyn.item()),
            kl_rep = float(kl_rep.item()),
        )
        return loss, info, h_t.detach(), z_t.detach()


# ── Soft target update ───────────────────────────────────────────────────────

def soft_update(online: nn.Module, target: nn.Module, tau: float = 0.02):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


# ── Deployment factory ────────────────────────────────────────────────────────

def build_policy(ckpt, agent_cfg=None, training_cfg=None, device='cpu',
                 deterministic=True):
    """Load this checkpoint's world model + actor as an inference Policy.

    Decentralized execution: world_model plus the shared actor, exactly what
    worker.py runs. The RSSM posterior is always sampled (worker.py never
    passes `sample=`); `deterministic` only selects the actor's mean action.
    See olympus/common/policy.py for the contract.
    """
    agent_cfg = agent_cfg or {}
    training_cfg = training_cfg or {}
    hidden = policy_contract.resolved_hidden(agent_cfg, 'hidden', 256)
    embed = policy_contract.resolved_hidden(agent_cfg, 'embed_dim', 128)
    h_dim = policy_contract.resolved_hidden(agent_cfg, 'h_dim', 256)
    groups = policy_contract.resolved_hidden(agent_cfg, 'latent_groups', 8)
    classes = policy_contract.resolved_hidden(agent_cfg, 'latent_classes', 8)
    bins = policy_contract.resolved_hidden(agent_cfg, 'reward_bins', 255)
    world = WorldModel(
        STATE_DIM, ACTION_DIM, hidden, embed, h_dim, groups, classes, bins)
    actor = Actor(
        h_dim, world.latent_dim, hidden,
        log_std_min=float(training_cfg.get('actor_log_std_min', -2.0)),
        log_std_max=float(training_cfg.get('actor_log_std_max', 0.0)))
    world.load_state_dict(ckpt['world_model'], strict=False)
    actor.load_state_dict(ckpt['actor'], strict=False)
    world.to(device).eval()
    actor.to(device).eval()
    return policy_contract.LatentWorldModelPolicy(
        'dreamer_v3', STATE_DIM, world, actor, ACTION_DIM, device,
        deterministic)
