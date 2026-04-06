"""
training/model.py — Option-Critic network for Olympus

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OPTION-CRITIC PRIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Standard RL picks one primitive action per timestep.  Option-Critic
(Bacon et al. 2017) extends this: an *option* ω is a temporally-extended
commitment — a policy that runs for multiple steps before you reconsider.
Each option has three parts:

  π_ω(a|s)  — intra-option policy: what to do while inside ω
  β_ω(s)    — termination: probability of leaving ω in state s
  I_ω       — initiation set: where ω can be started (we use global)

In our domain, each option IS a CC algorithm.  The "intra-option policy"
is trivial — the kernel runs the CC.  What the network actually learns:

  π(ω|s)      — which CC to pick next               (option policy)
  β_ω(s)      — when to switch away from ω          (termination)
  Q_Ω(s, ω)  — how good option ω is in state s      (critic)
  T_ω(s)      — how long to commit to ω in ms        (dwell head)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DWELL TIME  (our simplification of β)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instead of re-evaluating termination every timestep, the model outputs
a dwell time T: "commit to this arm for T milliseconds, then re-query."
T is sampled from a per-option log-normal distribution whose parameters
(μ, log σ) are learned.  This gives the model a soft analogue of β:
short T ≈ high β (switches often), long T ≈ low β (sticky).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LOSS TERMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given experience tuple (s, ω, R, s', done):

  Advantage  A  = R + γ·max_{ω'} Q(s',ω')·(1-done) − Q(s,ω)

  L_value    = A²                                    [TD(0) critic]
  L_policy   = −log π(ω|s) · A.detach()             [policy gradient]
  L_entropy  = −H(π(·|s))                            [exploration bonus]
  L_term     = β_ω(s') · (Q(s',ω) − max Q(s',·) + margin).detach()
                                                     [termination gradient]
  L_dwell    = −log p(T | μ_ω, σ_ω) · A.detach()   [REINFORCE on dwell]

  Total: L_value + L_policy − c_ent·L_entropy + c_term·L_term + c_dwell·L_dwell

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NOTE ON GIL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module is imported by BOTH oc_worker.py (one per flow, spawned by
oc_listener.c via fork+execl — each has its own GIL) AND actor_learner.py
(a separate process with its own GIL).  No threading is used; all
parallelism is via OS processes + multiprocessing Queues.
"""

import math
import os
from collections import namedtuple

# Per-component loss breakdown returned by compute_loss
LossInfo = namedtuple('LossInfo', ['total', 'value', 'policy', 'entropy', 'term', 'dwell'])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal

# ── Arm registry (must mirror mutant.h / oc_listener.c) ──────────────────────

ARM_IDS   = [0,       2,     5,      12,    13]
ARM_NAMES = ['cubic','bbr1','vegas','bbr3','astraea']
N_OPTIONS = len(ARM_IDS)

# Map raw mutant arm_id ↔ model index 0..N_OPTIONS-1
_id_to_idx = {arm_id: i for i, arm_id in enumerate(ARM_IDS)}

def arm_id_to_idx(arm_id: int) -> int:
    return _id_to_idx[arm_id]

def idx_to_arm_id(idx: int) -> int:
    return ARM_IDS[idx]

# ── Experience tuple (worker → learner) ──────────────────────────────────────

Experience = namedtuple('Experience', [
    'cport',       # int  — experiment identifier
    'flow_id',     # int  — flow within experiment
    'state',       # np.ndarray shape (STATE_DIM,)
    'option_idx',  # int  — model index (0..N_OPTIONS-1)
    'dwell_ms',    # float — actual dwell used
    'reward',      # float — accumulated reward over the dwell
    'next_state',  # np.ndarray shape (STATE_DIM,)
    'done',        # bool
])

# ── State layout from oc_state_t (oc_listener.c) ─────────────────────────────
#
#   struct.unpack('<8IQ', data)  →  8 uint32 + 1 uint64
#   index  field           unit
#     0    cur_arm         raw mutant arm_id
#     1    rtt_us          µs
#     2    rttvar_us       µs
#     3    min_rtt_us      µs
#     4    snd_cwnd        MSS units
#     5    lost            packets
#     6    retrans         packets
#     7    delivered       packets (cumulative)
#     8    delivery_rate   bytes/s

STATE_DIM = 9   # 8 metrics + normalised cur_arm

def normalize_state(raw: tuple) -> np.ndarray:
    """
    Map raw oc_state_t fields to a compact float32 vector.

    All features are log1p-scaled to compress the wide dynamic range of
    TCP metrics.  Fixed divisors are chosen so typical values land in [0, 1].

    raw = (cur_arm, rtt_us, rttvar_us, min_rtt_us, snd_cwnd,
            lost, retrans, delivered, delivery_rate)
    """
    (cur_arm, rtt_us, rttvar_us, min_rtt_us,
     snd_cwnd, lost, retrans, delivered, delivery_rate) = raw

    return np.array([
        math.log1p(rtt_us)        / 13.0,   # log1p(200 ms) ≈ log1p(200000) ≈ 12.2
        math.log1p(rttvar_us)     / 10.0,
        math.log1p(min_rtt_us)    / 13.0,
        math.log1p(snd_cwnd)      /  7.0,   # log1p(1000 pkts) ≈ 6.9
        math.log1p(lost)          /  5.0,
        math.log1p(retrans)       /  5.0,
        math.log1p(delivered)     / 20.0,   # cumulative, grows large
        math.log1p(delivery_rate) / 20.0,   # log1p(10 MB/s) ≈ 16.1
        _id_to_idx.get(cur_arm, 0) / float(N_OPTIONS - 1),  # arm → [0, 1]
    ], dtype=np.float32)

# ── Network ───────────────────────────────────────────────────────────────────

class OptionCriticNet(nn.Module):
    """
    Option-Critic network.

    Forward pass returns a ForwardOutput namedtuple:

      option_probs  (B, N_OPTIONS)  — π(ω|s), policy over arms
      dwell_mu      (B, N_OPTIONS)  — log-normal μ for dwell time (ms)
      dwell_logsig  (B, N_OPTIONS)  — log-normal log σ for dwell time
      q_values      (B, N_OPTIONS)  — Q_Ω(s, ω) for each arm
      termination   (B, N_OPTIONS)  — β_ω(s) for each arm, in [0,1]

    Dwell time for the chosen option ω:
        T ~ LogNormal(dwell_mu[:, ω], exp(dwell_logsig[:, ω]))
        T_clamped = clamp(T, DWELL_MIN_MS, DWELL_MAX_MS)
    """

    DWELL_MIN_MS = 50.0
    DWELL_MAX_MS = 500.0

    ForwardOutput = namedtuple(
        'ForwardOutput',
        ['option_probs', 'dwell_mu', 'dwell_logsig', 'q_values', 'termination']
    )

    def __init__(self, state_dim: int = STATE_DIM,
                 n_options: int = N_OPTIONS,
                 hidden: int = 128):
        super().__init__()
        self.n_options = n_options

        # Shared encoder — all heads branch off this representation
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # π(ω|s): which arm to pick — zero bias so initial policy is uniform
        self.option_head = nn.Linear(hidden, n_options)
        nn.init.zeros_(self.option_head.bias)

        # T_ω(s): dwell time distribution (log-normal, per option)
        # Bias dwell_mu to log(500) so the initial median dwell ≈ 500 ms,
        # keeping it at the new DWELL_MIN_MS boundary and well off the clamp.
        self.dwell_mu     = nn.Linear(hidden, n_options)
        self.dwell_logsig = nn.Linear(hidden, n_options)
        # median = exp(μ) → target ~100ms, midpoint of [20, 500] on log scale
        nn.init.constant_(self.dwell_mu.bias, math.log(100.0))
        # σ = 0.3 → 2σ draw: exp(log(100) ± 0.6) = [55ms, 182ms] — within range
        nn.init.constant_(self.dwell_logsig.bias, math.log(0.3))

        # Q_Ω(s, ω): value of each option
        self.value_head = nn.Linear(hidden, n_options)

        # β_ω(s): termination probability for each option
        self.term_head = nn.Linear(hidden, n_options)

    def forward(self, s: torch.Tensor) -> 'OptionCriticNet.ForwardOutput':
        """s: (B, STATE_DIM) float32"""
        h = self.encoder(s)
        return self.ForwardOutput(
            option_probs = F.softmax(self.option_head(h), dim=-1),
            dwell_mu     = self.dwell_mu(h),
            dwell_logsig = self.dwell_logsig(h).clamp(-3.0, 0.7),
            q_values     = self.value_head(h),
            termination  = torch.sigmoid(self.term_head(h)),
        )

    @torch.no_grad()
    def act(self, s: np.ndarray, epsilon: float = 0.15):
        """
        ε-greedy stochastic action.  Returns (arm_id, dwell_ms).

        With probability epsilon a uniformly random arm is chosen,
        preventing permanent policy collapse to a single arm.
        s: normalised state vector, shape (STATE_DIM,)
        """
        # Build mask from OC_ARMS env var (comma-separated arm names).
        # Unavailable arms are zeroed out so they are never sampled.
        oc_arms_env = os.environ.get('OC_ARMS', '')
        if oc_arms_env:
            allowed_names = set(n.strip() for n in oc_arms_env.split(','))
            mask = torch.tensor(
                [1.0 if ARM_NAMES[i] in allowed_names else 0.0
                 for i in range(N_OPTIONS)],
                dtype=torch.float32)
            allowed_indices = [i for i in range(N_OPTIONS)
                               if ARM_NAMES[i] in allowed_names]
        else:
            mask = torch.ones(N_OPTIONS, dtype=torch.float32)
            allowed_indices = list(range(N_OPTIONS))

        t = torch.from_numpy(s).unsqueeze(0)    # (1, STATE_DIM)
        out = self.forward(t)

        # Apply mask to policy probs before sampling
        masked_probs = out.option_probs[0] * mask
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)

        if np.random.random() < epsilon:
            opt_idx = int(np.random.choice(allowed_indices))
        else:
            opt_idx = torch.multinomial(masked_probs.unsqueeze(0), 1).item()

        # Sample dwell from log-normal
        mu  = out.dwell_mu[0, opt_idx].item()
        sig = math.exp(out.dwell_logsig[0, opt_idx].item())
        T   = math.exp(mu + sig * torch.randn(1).item())
        dwell_min = float(os.environ.get('OC_DWELL_MIN_MS', str(self.DWELL_MIN_MS)))
        dwell_max = float(os.environ.get('OC_DWELL_MAX_MS', str(self.DWELL_MAX_MS)))
        T   = float(np.clip(T, dwell_min, dwell_max))

        return idx_to_arm_id(opt_idx), T

def soft_update_target(net: OptionCriticNet, target: OptionCriticNet, tau: float = 0.01) -> None:
    """
    Polyak-average target network weights toward the online network:
        θ_target ← τ·θ_online + (1-τ)·θ_target

    τ=0.01 means the target moves ~1% toward the online net each call.
    Call once per training step for a smooth, stable bootstrap target.
    """
    for p_online, p_target in zip(net.parameters(), target.parameters()):
        p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)


# ── Loss computation ──────────────────────────────────────────────────────────

def compute_loss(
    net: OptionCriticNet,
    batch: list,           # list of Experience
    gamma:        float = 0.99,
    c_entropy:    float = 0.01,
    c_term:       float = 0.1,
    c_dwell:      float = 0.1,
    term_margin:  float = 0.01,
    target_net:   'OptionCriticNet | None' = None,
) -> torch.Tensor:
    """
    Compute Option-Critic loss over a batch of Experience tuples.

    target_net: frozen copy of net used for TD bootstrapping.  When provided,
    Q(s',ω') is computed with target_net (no grad) instead of net — this
    stabilises training by decoupling the TD target from the weights being
    updated (the classic DQN fix for value collapse).  Update target_net
    periodically with soft_update_target() in the training loop.

    Returns a scalar loss tensor (call .backward() then optimizer.step()).
    """
    states      = torch.tensor(np.stack([e.state      for e in batch]), dtype=torch.float32)
    next_states = torch.tensor(np.stack([e.next_state for e in batch]), dtype=torch.float32)
    options     = torch.tensor([e.option_idx for e in batch], dtype=torch.long)
    rewards     = torch.tensor([e.reward     for e in batch], dtype=torch.float32)
    dwell_used  = torch.tensor([e.dwell_ms   for e in batch], dtype=torch.float32)
    dones       = torch.tensor([float(e.done) for e in batch], dtype=torch.float32)

    out  = net(states)
    # Use target network for next-state Q if available, otherwise fall back to net
    bootstrap_net = target_net if target_net is not None else net
    with torch.no_grad():
        outn = bootstrap_net(next_states)

    # Q values for the chosen option
    q_chosen = out.q_values.gather(1, options.unsqueeze(1)).squeeze(1)

    # TD target using max Q over next options (from frozen target network)
    with torch.no_grad():
        q_next_max = outn.q_values.max(dim=1).values
        target = rewards + gamma * q_next_max * (1.0 - dones)

    advantage = (target - q_chosen).detach()
    # Normalise advantage so large value spikes don't overwhelm policy gradient
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # ── Value loss ────────────────────────────────────────────────────────────
    l_value = F.mse_loss(q_chosen, target)

    # ── Policy gradient loss ──────────────────────────────────────────────────
    log_prob = torch.log(out.option_probs.gather(1, options.unsqueeze(1)).squeeze(1) + 1e-8)
    l_policy = -(log_prob * advantage).mean()

    # ── Entropy bonus (encourages exploration across arms) ───────────────────
    entropy = -(out.option_probs * (out.option_probs + 1e-8).log()).sum(dim=-1)
    l_entropy = -entropy.mean()

    # ── Termination loss ─────────────────────────────────────────────────────
    # β_ω(s') from the online net (needs grad); Q advantage from target net.
    beta_chosen = net(next_states).termination.gather(1, options.unsqueeze(1)).squeeze(1)
    adv_term = (outn.q_values.gather(1, options.unsqueeze(1)).squeeze(1)
                - outn.q_values.max(dim=1).values + term_margin).detach()
    l_term = (beta_chosen * adv_term).mean()

    # ── Dwell REINFORCE loss ──────────────────────────────────────────────────
    # Treat dwell as a continuous action sampled from LogNormal(μ_ω, σ_ω).
    mu_chosen  = out.dwell_mu.gather(1, options.unsqueeze(1)).squeeze(1)
    sig_chosen = out.dwell_logsig.gather(1, options.unsqueeze(1)).squeeze(1).exp()
    dist = LogNormal(mu_chosen, sig_chosen + 1e-6)
    log_prob_dwell = dist.log_prob(dwell_used.clamp(min=1.0))
    # Clamp log_prob to prevent REINFORCE variance explosion with random init.
    l_dwell = -(log_prob_dwell.clamp(-10.0, 10.0) * advantage).mean()

    total = (l_value
             + l_policy
             + c_entropy * l_entropy
             + c_term    * l_term
             + c_dwell   * l_dwell)
    return LossInfo(
        total=total,
        value=l_value.item(),
        policy=l_policy.item(),
        entropy=(-l_entropy).item(),   # positive = entropy (higher = more exploratory)
        term=l_term.item(),
        dwell=l_dwell.item(),
    )
