"""
model.py — Option-Critic network for direct CWND control.

Two backbones are supported:

  `gru`:
    Shared: Linear(state → hidden) → LayerNorm → ReLU
    Policy trunk : GRU(hidden → hidden) → Linear(hidden → hidden//2) → ReLU
    Value trunk  : GRU(hidden → hidden) → Linear(hidden → hidden//2) → ReLU

  `mlp`:
    Shared: Linear(state → hidden) → ReLU
    Policy trunk : Linear(hidden → hidden) → ReLU → Linear(hidden → hidden//2) → ReLU
    Value trunk  : Linear(hidden → hidden) → ReLU → Linear(hidden → hidden//2) → ReLU

The MLP path is closer to the typical feedforward reference implementations of
Option-Critic. The GRU path keeps temporal context in hidden state.
"""

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from single_agent_olympus.common.state_plugins import (
    assert_state_compatible,
    current_state_name,
    load_state_module,
    state_meta,
)

# ── State / action constants ──────────────────────────────────────────────────

STATE_DIM  = 11
N_OPTIONS  = 3
DEFAULT_OC_ARCH = 'gru'
DEFAULT_OC_ACTION_DIST = 'squashed_normal'
# Kept verbatim so existing checkpoints' `model_meta.state_features` still match.
STATE_FEATURE_VERSION = 'kalman_observation_v1_srtt_unshifted'

CriticInfo = namedtuple('CriticInfo', [
    'loss', 'value',
    'td_target_mean', 'td_target_abs',
    'q_chosen_mean', 'q_chosen_abs',
])
ActorInfo = namedtuple('ActorInfo', [
    'loss', 'action', 'term', 'mean_beta', 'entropy',
    'usage', 'action_div', 'beta_target', 'center', 'smooth', 'action_aux',
    'q_gap', 'option_mass', 'beta_by_option', 'mu_by_option', 'q_by_option',
])

Experience = namedtuple('Experience',
    ['state', 'option_idx', 'can_terminate', 'force_terminate',
     'can_terminate_next', 'force_terminate_next',
     'prev_cwnd_mult', 'cwnd_mult', 'reward', 'next_state', 'done', 'traj_id'])

# Log-space bounds for the CWND multiplier action
_LOG_MULT_MIN  = math.log(0.5)
_LOG_MULT_MAX  = math.log(2.0)
_LOG_MULT_MID  = (_LOG_MULT_MAX + _LOG_MULT_MIN) / 2.0
_LOG_MULT_HALF = (_LOG_MULT_MAX - _LOG_MULT_MIN) / 2.0

# Bufferbloat can push RTT/Kalman/in-flight features far outside the range the
# network saw early in training. Keep the observation vector bounded so a bad
# rollout cannot turn the value head into a numerical amplifier.
_STATE_LOW = np.array(
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_STATE_HIGH = np.array(
    [1.0, 8.0, 8.0, 2.0, 2.0, 4.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    dtype=np.float32,
)


def normalize_oc_arch(arch: str) -> str:
    arch = str(arch or DEFAULT_OC_ARCH).strip().lower()
    if arch not in ('gru', 'mlp'):
        raise ValueError(f'unsupported Option-Critic architecture: {arch}')
    return arch


def option_critic_kwargs_from_agent_cfg(agent_cfg: dict) -> dict:
    agent_cfg = agent_cfg or {}
    return {
        'n_options': int(agent_cfg.get('n_options', N_OPTIONS)),
        'hidden': int(agent_cfg.get('hidden', 256)),
        'beta_temperature': float(agent_cfg.get('beta_temperature', 1.0)),
        'beta_floor': float(agent_cfg.get('beta_floor', 0.0)),
        'arch': normalize_oc_arch(agent_cfg.get('arch', DEFAULT_OC_ARCH)),
        'mult_min': float(agent_cfg.get('mult_min', 0.5)),
        'mult_max': float(agent_cfg.get('mult_max', 2.0)),
        'action_init': str(agent_cfg.get('action_init', 'small_spread')),
        'action_init_scale': float(agent_cfg.get('action_init_scale', 0.05)),
        'q_value_clip': float(agent_cfg.get('q_value_clip', 1500.0)),
        'action_logsig_min': float(agent_cfg.get('action_logsig_min', -1.0)),
    }


def option_critic_meta_from_checkpoint(ckpt: dict) -> dict:
    meta = dict((ckpt or {}).get('model_meta', {}) or {})
    if 'arch' in meta:
        meta['arch'] = normalize_oc_arch(meta['arch'])
    meta.setdefault('beta_floor', 0.0)
    meta.setdefault('action_dist', 'legacy_clipped_lognormal')
    meta.setdefault('mult_min', 0.5)
    meta.setdefault('mult_max', 2.0)
    meta.setdefault('action_init', 'spread')
    meta.setdefault('action_init_scale', 1.0)
    meta.setdefault('q_value_clip', 0.0)
    meta.setdefault('action_logsig_min', -1.0)
    meta.setdefault('state_features', 'legacy_with_bdp_observation')
    return meta


# ── State normalisation ───────────────────────────────────────────────────────

def normalize_state(info: dict) -> np.ndarray:
    """
    Normalize state using only fields present in the Astraea IPC state message.
    No environment parameters (scheduled link BW or RTT) touch this function.
    State is identical at training and inference time.

    BW reference : peak_thr in bytes/s (max observed throughput this episode)
                   — falls back
                   to pacing_rate before peak is established.  Unlike pacing_rate
                   alone, peak_thr stays at the clean-pipe measurement even when
                   the buffer fills, so BW-ratio features correctly show
                   underutilisation at congestion instead of looking normal.

    RTT reference: srtt_us — kernel smoothed RTT, always observable.

    Dimensions:
      0  delta_cwnd               — fractional CWND change vs previous step [-1, 1]
      1  avg_urtt / 1e5           — actual RTT absolute scale
      2  log1p(cwnd) / log1p(10000)
                                  — absolute CWND scale, not BDP-derived
      3  avg_thr / peak_thr       — throughput utilisation of observed peak
      4  pacing_rate / peak_thr   — pacing overhead vs observed peak
      5  packets_out / cwnd       — in-flight ratio
      6  delta_rtt                — fractional RTT change vs previous step [-1, 1]
      7  retrans / packets        — retransmit fraction (loss proxy)
      8  avg_thr / 1e7            — absolute throughput hint
      9  srtt_us / 1e5            — srtt hint
     10  kalman_min_rtt_us / 1e5  — Kalman-tracked propagation floor (same scale as
                                    dim 1); agent can derive queueing as (dim1 - dim10).

    delta_cwnd + delta_rtt together give the GRU a causal signal: if CWND grew
    and RTT rose shortly after, that rise is controllable (queueing). If RTT changed
    without CWND changing, it is an external link delay change.
    """
    cwnd             = max(int(info.get('cwnd',             1)), 1)
    avg_thr          = float(info.get('avg_thr',            0))
    avg_urtt         = float(info.get('avg_urtt',           0))
    # Linux exposes tp->srtt_us shifted left by 3.  RewardCalc already unshifts
    # it before checking sparse RTT proximity; keep the observation on the same
    # microsecond scale as avg_urtt and the Kalman floor.
    srtt_raw         = float(info.get('srtt_us',            0))
    srtt_us          = (srtt_raw / 8.0) if srtt_raw > 0 else avg_urtt
    srtt_us          = max(srtt_us, 1.0)
    pacing_rate      = float(info.get('pacing_rate',        0))
    packets_out      = float(info.get('packets_out',        0))
    retrans_out      = float(info.get('retrans_out',        0))
    prev_urtt        = float(info.get('prev_urtt',          avg_urtt))
    prev_cwnd        = float(info.get('prev_cwnd',          cwnd))
    peak_thr         = float(info.get('peak_thr',           0))
    kalman_min_rtt   = float(info.get('kalman_min_rtt_us',  avg_urtt))

    # BW reference: observed throughput peak, falling back to pacing_rate
    bw_ref = max(peak_thr, pacing_rate, 1.0)

    delta_rtt  = float(np.clip((avg_urtt - prev_urtt) / max(prev_urtt, 1.0), -1.0, 1.0))
    delta_cwnd = float(np.clip((cwnd     - prev_cwnd) / max(prev_cwnd, 1.0), -1.0, 1.0))

    cwnd_log = math.log1p(float(cwnd)) / math.log1p(10_000.0)
    s = np.array([
        delta_cwnd,                                                 # 0  CWND rate of change
        avg_urtt / 1e5,                                             # 1  actual RTT absolute
        cwnd_log,                                                   # 2  absolute CWND scale, no BDP
        max(avg_thr     / bw_ref, 0.0),                                # 3  throughput utilisation
        max(pacing_rate / bw_ref, 0.0),                                # 4  pacing utilisation
        max(packets_out / max(cwnd, 1), 0.0),                          # 5  in-flight ratio
        delta_rtt,                                                  # 6  RTT rate of change
        min(retrans_out / max(packets_out, 1.0), 1.0),              # 7  retransmit fraction
        avg_thr / 1e7,                                              # 8  absolute throughput hint
        srtt_us / 1e5,                                              # 9  srtt hint
        kalman_min_rtt / 1e5,                                       # 10 Kalman propagation floor
    ], dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=_STATE_HIGH, neginf=_STATE_LOW)
    return np.clip(s, _STATE_LOW, _STATE_HIGH)


_STATE_PLUGIN = load_state_module('option_critic')
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

class OptionCriticNet(nn.Module):
    """
    Option-Critic network with selectable backbone.

    `gru` keeps temporal state in both policy and value trunks.
    `mlp` is the reference-style feedforward variant.

    forward(s, h) — training, runs both trunks
      s : (B, STATE_DIM) or (B, T, STATE_DIM)
      h : ignored — trunks always start from zero in BPTT when recurrent
      Returns 5-tuple: (mu, logsig, q, beta, None)

    act(s, h_pol) — inference, runs both trunks for option selection
      Returns (option_idx, cwnd_multiplier, h_pol_new)
    """

    MULT_MIN = 0.5
    MULT_MAX = 2.0

    def __init__(self, state_dim: int = STATE_DIM,
                 n_options: int = N_OPTIONS,
                 hidden: int = 256,
                 beta_temperature: float = 1.0,
                 beta_floor: float = 0.0,
                 arch: str = DEFAULT_OC_ARCH,
                 mult_min: float = 0.5,
                 mult_max: float = 2.0,
                 action_init: str = 'small_spread',
                 action_init_scale: float = 0.05,
                 q_value_clip: float = 1500.0,
                 action_logsig_min: float = -1.0):
        super().__init__()
        if mult_min <= 0.0 or mult_max <= mult_min:
            raise ValueError(f'invalid multiplier bounds: {mult_min}..{mult_max}')
        self.n_options = n_options
        self.hidden    = hidden
        self.beta_temperature = max(float(beta_temperature), 1e-3)
        self.beta_floor = min(max(float(beta_floor), 0.0), 0.49)
        self.arch = normalize_oc_arch(arch)
        self.is_recurrent = (self.arch == 'gru')
        self.action_dist = DEFAULT_OC_ACTION_DIST
        self.q_value_clip = max(float(q_value_clip), 0.0)
        self.action_logsig_min = min(max(float(action_logsig_min), -5.0), 0.0)
        self.MULT_MIN = float(mult_min)
        self.MULT_MAX = float(mult_max)
        self.log_mult_min = math.log(self.MULT_MIN)
        self.log_mult_max = math.log(self.MULT_MAX)
        self.log_mult_mid = 0.5 * (self.log_mult_min + self.log_mult_max)
        self.log_mult_half = 0.5 * (self.log_mult_max - self.log_mult_min)
        self.action_init = str(action_init or 'small_spread').strip().lower()
        self.action_init_scale = max(float(action_init_scale), 0.0)

        if self.is_recurrent:
            # ── Shared input projection ───────────────────────────────────────
            self.input_proj = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )

            # ── Policy / value GRU trunks ────────────────────────────────────
            self.policy_core = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
            self.value_core  = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
            self.policy_post = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )
            self.value_post = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )
        else:
            # Reference-like feedforward Option-Critic backbone.
            self.input_proj = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
            )
            self.policy_core = nn.Identity()
            self.value_core  = nn.Identity()
            self.policy_post = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )
            self.value_post = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )

        head_in = hidden // 2

        # No option_head: canonical OC selects options by ε-greedy over Q.
        # Keeping a parameterised softmax-over-options here trained with REINFORCE
        # is dead weight at inference AND pollutes the shared policy trunk during
        # training — removed to match Bacon 2017 / lweitkamp reference.

        self.action_mu     = nn.Linear(head_in, n_options)
        self.action_logsig = nn.Linear(head_in, n_options)
        # Start action means near mult=1.0 by default. Full-range spread made a
        # 2-option model begin life as "max shrink" vs "max grow", which the
        # policy gradient then tended to preserve as bang-bang control.
        with torch.no_grad():
            if self.action_init in ('center', 'centered', 'zero', 'zeros') or n_options == 1:
                init_log_mults = torch.zeros(n_options)
            elif self.action_init in ('small_spread', 'near_zero', 'near_hold'):
                span = min(self.action_init_scale, self.log_mult_half * 0.95)
                init_log_mults = torch.linspace(-span, span, n_options)
            elif self.action_init in ('spread', 'full_spread'):
                init_log_mults = torch.linspace(self.log_mult_min, self.log_mult_max, n_options)
            else:
                raise ValueError(f'unsupported action_init: {self.action_init}')
            init_log_mults = init_log_mults.clamp(self.log_mult_min + 1e-6,
                                                  self.log_mult_max - 1e-6)
            vals = ((init_log_mults - self.log_mult_mid) / self.log_mult_half).clamp(-0.999, 0.999)
            self.action_mu.bias.copy_(torch.atanh(vals))
        nn.init.constant_(self.action_logsig.bias, self.action_logsig_min)

        self.term_head = nn.Linear(head_in, n_options)
        # Initialise β close to 0 — options start sticky, termination loss must
        # actively push β up before switching happens.  Keep the effective
        # initial logit near -3 even when beta_temperature > 1.
        nn.init.constant_(self.term_head.bias, -3.0 * self.beta_temperature)

        self.value_head = nn.Linear(head_in, n_options)

        # ── Option discriminator (DIAYN-style) ───────────────────────────────
        # Predicts which option was active from (s, s').  Trained with CE;
        # log q(ω|s,s') − log(1/K) is added to the extrinsic reward as an
        # intrinsic signal, pushing options toward distinguishable transitions.
        # Operates on raw normalized states, NOT on the policy trunk features,
        # so its gradient doesn't re-shape the shared encoder.
        self.discriminator = nn.Sequential(
            nn.Linear(2 * state_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_options),
        )

    def model_meta(self) -> dict:
        return {
            'arch': self.arch,
            'hidden': int(self.hidden),
            'n_options': int(self.n_options),
            'beta_temperature': float(self.beta_temperature),
            'beta_floor': float(self.beta_floor),
            'action_dist': self.action_dist,
            'mult_min': float(self.MULT_MIN),
            'mult_max': float(self.MULT_MAX),
            'action_init': self.action_init,
            'action_init_scale': float(self.action_init_scale),
            'q_value_clip': float(self.q_value_clip),
            'action_logsig_min': float(self.action_logsig_min),
            'state_features': STATE_FEATURE_VERSION,
            'state_name': STATE_NAME,
            'state_dim': STATE_DIM,
            'state_feature_version': STATE_FEATURE_VERSION,
        }

    def initial_hidden(self, batch_size: int = 1, device=None):
        if not self.is_recurrent:
            return None
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden, device=device)

    def _run_core(self, core, x: torch.Tensor, h):
        if self.is_recurrent:
            if h is None:
                h = self.initial_hidden(batch_size=x.shape[0], device=x.device)
            out, h_new = core(x, h)
            return out, h_new
        return core(x), h

    def _log_mult_from_raw_loc(self, raw_loc: torch.Tensor) -> torch.Tensor:
        return self.log_mult_mid + self.log_mult_half * torch.tanh(raw_loc)

    def _raw_loc_from_log_mult(self, log_mult: torch.Tensor) -> torch.Tensor:
        scaled = ((log_mult - self.log_mult_mid) / self.log_mult_half).clamp(-0.999999, 0.999999)
        return torch.atanh(scaled)

    def _q_from_head(self, val_feat: torch.Tensor) -> torch.Tensor:
        q = self.value_head(val_feat)
        if self.q_value_clip > 0.0:
            scale = float(self.q_value_clip)
            q = scale * torch.tanh(q / scale)
        return q

    def forward(self, s: torch.Tensor, h=None):
        """
        Training forward pass — runs both trunks.
        h is ignored (always zero-initialised for BPTT when recurrent).
        Returns 5-tuple: (action_mu, action_logsig, q_values, term, None).
        """
        seq = (s.dim() == 3)
        if not seq:
            s = s.unsqueeze(1)

        B, T, D = s.shape
        x = self.input_proj(s.reshape(B * T, D)).reshape(B, T, self.hidden)

        pol_h0 = self.initial_hidden(batch_size=B, device=s.device)
        val_h0 = self.initial_hidden(batch_size=B, device=s.device)
        pol_out, _ = self._run_core(self.policy_core, x, pol_h0)
        val_out, _ = self._run_core(self.value_core, x, val_h0)

        pol_feat = self.policy_post(pol_out)
        val_feat = self.value_post(val_out)

        action_mu = self._log_mult_from_raw_loc(self.action_mu(pol_feat))
        action_ls = self.action_logsig(pol_feat).clamp(self.action_logsig_min, 0.0)
        term      = self._beta_from_logits(self.term_head(pol_feat))
        q_values  = self._q_from_head(val_feat)

        if not seq:
            action_mu = action_mu[:, -1, :]
            action_ls = action_ls[:, -1, :]
            q_values  = q_values[:, -1, :]
            term      = term[:, -1, :]

        return (action_mu, action_ls, q_values, term, None)

    def _beta_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # A temperature > 1 slows β saturation without changing the loss shape.
        beta = torch.sigmoid(logits / self.beta_temperature)
        if self.beta_floor > 0.0:
            beta = self.beta_floor + (1.0 - 2.0 * self.beta_floor) * beta
        return beta

    @torch.no_grad()
    def act(self, s: np.ndarray, h_pol: torch.Tensor, h_val: torch.Tensor = None,
            epsilon: float = 0.0,
            cur_option: int = -1,
            deterministic: bool = False,
            term_threshold: float = 0.5,
            can_terminate: bool = True,
            force_terminate: bool = False):
        """
        Single-step policy rollout.  Training-identical semantics: Bernoulli
        termination on β, bounded squashed-Normal action sample, ε-greedy option
        override on switch.  For offline evaluation, set deterministic=True
        so β uses a fixed threshold and the action uses exp(μ) instead of a
        fresh sample.  That keeps rollouts stable and makes the configured
        inference termination threshold actually matter.

        Returns (option_idx, cwnd_multiplier, h_pol_new, h_val_new,
                 mu, sigma, terminated).
        """
        device  = next(self.parameters()).device
        t       = torch.from_numpy(s).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D)
        B, T, D = t.shape

        x = self.input_proj(t.reshape(B * T, D)).reshape(B, T, self.hidden)
        pol_out, h_pol_new = self._run_core(self.policy_core, x, h_pol)
        val_out, h_val_new = self._run_core(self.value_core, x,  h_val)

        pol_feat = self.policy_post(pol_out[:, -1, :])
        val_feat = self.value_post(val_out[:, -1, :])

        raw_loc = self.action_mu(pol_feat)
        mu     = self._log_mult_from_raw_loc(raw_loc)
        logsig = self.action_logsig(pol_feat).clamp(self.action_logsig_min, 0.0)
        beta   = self._beta_from_logits(self.term_head(pol_feat))
        q_vals = self._q_from_head(val_feat)   # (1, n_options)

        # Training uses Bernoulli(β).  Deterministic inference uses a fixed
        # threshold so options do not chatter purely from sampling noise.
        if cur_option >= 0:
            beta_val  = beta[0, cur_option].item()
            if force_terminate:
                terminate = True
            elif not can_terminate:
                terminate = False
            elif deterministic:
                terminate = (beta_val >= term_threshold)
            else:
                terminate = (torch.rand(1).item() < beta_val)
        else:
            terminate = True  # first step

        # Option: Q-argmax, with ε-greedy override only during stochastic rollout.
        if not terminate:
            opt = cur_option
        elif (not deterministic) and epsilon > 0.0 and np.random.random() < epsilon:
            opt = np.random.randint(self.n_options)
        else:
            opt = int(q_vals[0].argmax().item())

        # Action: use a bounded squashed Normal in log-multiplier space. This
        # keeps rollout and training consistent at the action bounds instead of
        # sampling an unbounded LogNormal and clipping it afterwards.
        m_val = mu[0, opt].item()
        s_val = math.exp(max(-4.0, min(0.0, logsig[0, opt].item())))
        if deterministic:
            raw_arg = max(-8.0, min(8.0, raw_loc[0, opt].item()))
        else:
            raw_arg = max(-8.0, min(8.0, raw_loc[0, opt].item() + s_val * np.random.randn()))
        log_mult = self.log_mult_mid + self.log_mult_half * math.tanh(raw_arg)
        mult  = float(math.exp(log_mult))

        return int(opt), mult, h_pol_new, h_val_new, m_val, s_val, bool(terminate)


def soft_update(online: OptionCriticNet, target: OptionCriticNet, tau: float = 0.005):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


# ── Losses (closer to canonical OC) ──────────────────────────────────────────

def _batch_to_tensors(batch: list, device):
    states      = torch.tensor(np.stack([e.state      for e in batch]), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.stack([e.next_state for e in batch]), dtype=torch.float32).to(device)
    options     = torch.tensor([e.option_idx for e in batch], dtype=torch.long).to(device)
    can_term    = torch.tensor([float(e.can_terminate) for e in batch], dtype=torch.float32).to(device)
    force_term  = torch.tensor([float(e.force_terminate) for e in batch], dtype=torch.float32).to(device)
    can_term_n  = torch.tensor([float(e.can_terminate_next) for e in batch], dtype=torch.float32).to(device)
    force_term_n = torch.tensor([float(e.force_terminate_next) for e in batch], dtype=torch.float32).to(device)
    prev_mults  = torch.tensor([e.prev_cwnd_mult for e in batch], dtype=torch.float32).to(device)
    mults       = torch.tensor([e.cwnd_mult  for e in batch], dtype=torch.float32).to(device)
    rewards     = torch.tensor([e.reward     for e in batch], dtype=torch.float32).to(device)
    dones       = torch.tensor([float(e.done) for e in batch], dtype=torch.float32).to(device)
    return (states, next_states, options, can_term, force_term,
            can_term_n, force_term_n, prev_mults, mults, rewards, dones)


def _td_target(net: OptionCriticNet, target_net: OptionCriticNet,
               next_states: torch.Tensor,
               options: torch.Tensor,
               can_terminate_next: torch.Tensor,
               force_terminate_next: torch.Tensor,
               rewards: torch.Tensor,
               dones: torch.Tensor,
               gamma: float):
    with torch.no_grad():
        _, _, qn, _, _ = target_net(next_states)
        _, _, _, beta_td, _ = net(next_states)
        q_next_cur = qn.gather(1, options.unsqueeze(1)).squeeze(1)
        q_next_max = qn.max(dim=1).values
        beta_allowed = beta_td.gather(1, options.unsqueeze(1)).squeeze(1) * can_terminate_next
        beta_o     = torch.where(force_terminate_next > 0.5,
                                 torch.ones_like(beta_allowed),
                                 beta_allowed)
        q_next_mix = (1.0 - beta_o) * q_next_cur + beta_o * q_next_max
        td = rewards + gamma * q_next_mix * (1.0 - dones)
        if net.q_value_clip > 0.0:
            td = td.clamp(-net.q_value_clip, net.q_value_clip)
        return td


def compute_critic_loss(net: OptionCriticNet, target_net: OptionCriticNet,
                        batch: list,
                        gamma: float = 0.99,
                        c_value: float = 1.0,
                        seq_len: int = 1,
                        critic_loss: str = 'huber',
                        huber_delta: float = 10.0) -> CriticInfo:
    """
    Reference-style critic update:
      0.5 * (Q(s, ω) - gt)^2

    Still supports sequence batches so the recurrent value trunk can train on
    coherent windows, but the target itself stays canonical Option-Critic.
    Dwell masks are included in the target: β is forced to 0 before min dwell
    and forced to 1 after max dwell so training matches rollout semantics.
    """
    device = next(net.parameters()).device
    (states, next_states, options, _, _, can_term_n, force_term_n,
     _, _, rewards, dones) = _batch_to_tensors(batch, device)

    if seq_len > 1:
        n_seqs = len(batch) // seq_len

        def _seq(t, extra_dims=()):
            return t[:n_seqs * seq_len].reshape(n_seqs, seq_len, *extra_dims)

        s_seq  = _seq(states,      (STATE_DIM,))
        ns_seq = _seq(next_states, (STATE_DIM,))
        o_seq  = _seq(options)
        ct_seq = _seq(can_term_n)
        ft_seq = _seq(force_term_n)
        r_seq  = _seq(rewards)
        d_seq  = _seq(dones)

        _, _, q, _, _ = net(s_seq)
        with torch.no_grad():
            _, _, qn, _, _ = target_net(ns_seq)
            _, _, _, beta_td, _ = net(ns_seq)
            o_seq_idx  = o_seq.unsqueeze(-1)
            q_next_cur = qn.gather(-1, o_seq_idx).squeeze(-1)
            q_next_max = qn.max(dim=-1).values
            beta_allowed = beta_td.gather(-1, o_seq_idx).squeeze(-1) * ct_seq
            beta_o     = torch.where(ft_seq > 0.5,
                                     torch.ones_like(beta_allowed),
                                     beta_allowed)
            q_next_mix = (1.0 - beta_o) * q_next_cur + beta_o * q_next_max
            td_target  = r_seq + gamma * q_next_mix * (1.0 - d_seq)
            if net.q_value_clip > 0.0:
                td_target = td_target.clamp(-net.q_value_clip, net.q_value_clip)

        q         = q.reshape(n_seqs * seq_len, -1)
        options   = o_seq.reshape(n_seqs * seq_len)
        td_target = td_target.reshape(n_seqs * seq_len)
    else:
        _, _, q, _, _ = net(states)
        td_target = _td_target(net, target_net, next_states, options,
                               can_term_n, force_term_n, rewards, dones, gamma)

    q_chosen   = q.gather(1, options.unsqueeze(1)).squeeze(1)
    td_target_det = td_target.detach()
    td_error = q_chosen - td_target_det
    if str(critic_loss).strip().lower() == 'mse':
        value_loss = 0.5 * td_error.pow(2).mean()
    else:
        delta = max(float(huber_delta), 1e-6)
        abs_err = td_error.abs()
        value_loss = torch.where(
            abs_err <= delta,
            0.5 * td_error.pow(2),
            delta * (abs_err - 0.5 * delta),
        ).mean()
    return CriticInfo(
        loss=c_value * value_loss,
        value=value_loss.item(),
        td_target_mean=float(td_target_det.mean().item()),
        td_target_abs=float(td_target_det.abs().mean().item()),
        q_chosen_mean=float(q_chosen.detach().mean().item()),
        q_chosen_abs=float(q_chosen.detach().abs().mean().item()),
    )


def compute_actor_loss(net: OptionCriticNet, target_net: OptionCriticNet,
                       batch: list,
                       gamma: float = 0.99,
                       c_term: float = 1.0,
                       c_action: float = 1.0,
                       c_entropy: float = 0.01,
                       c_bdp_action: float = 0.0,
                       bdp_action_gain: float = 0.15,
                       term_margin: float = 0.01,
                       c_action_center: float = 0.0,
                       c_action_smooth: float = 0.0,
                       c_option_usage: float = 0.0,
                       c_action_div: float = 0.0,
                       option_kl_target: float = 0.25,
                       option_usage_temp: float = 1.0,
                       c_beta_target: float = 0.0,
                       beta_target: float = 0.20,
                       advantage_norm: bool = True,
                       action_raw_clip: float = 2.0,
                       policy_logp_clip: float = 6.0) -> ActorInfo:
    """
    Reference-style OC actor update on fresh transitions.

    This follows the Bacon/lweitkamp structure much more closely than the
    replay-based actor update:
      termination_loss = β(s,ω) * (Q(s,ω) - max Q(s,·) + ξ)
      policy_loss      = -log π(a|s,ω) * (gt - Q(s,ω))
      actor_loss       = policy_loss + termination_loss - entropy_reg * H

    In this codebase actions are continuous bounded multipliers instead of a
    categorical primitive action distribution, so only the action-distribution
    family differs from the reference.
    A hard dwell mask zeros the termination loss on steps where the option is
    not yet allowed to end.
    """
    device = next(net.parameters()).device
    (states, next_states, options, can_term, force_term, can_term_n,
     force_term_n, prev_mults, mults, rewards, dones) = _batch_to_tensors(batch, device)

    mu, logsig, q, beta, _ = net(states)
    gt = _td_target(net, target_net, next_states, options, can_term_n,
                    force_term_n, rewards, dones, gamma)

    mu_sel      = mu.gather(1, options.unsqueeze(1)).squeeze(1)
    logsig_sel  = logsig.gather(1, options.unsqueeze(1)).squeeze(1)
    sig_sel     = logsig_sel.exp()
    # The action head represents a squashed Normal in log-multiplier space:
    #   u ~ Normal(raw_loc, sigma)
    #   log_mult = mid + half * tanh(u)
    #   mult = exp(log_mult)
    #
    # Compute log π(mult|s,ω) with the inverse transform so the policy loss is
    # consistent with the action the worker actually executed.
    raw_loc_sel = net._raw_loc_from_log_mult(mu_sel)
    log_mult_act = mults.clamp(min=net.MULT_MIN + 1e-6,
                               max=net.MULT_MAX - 1e-6).log()
    raw_act = net._raw_loc_from_log_mult(log_mult_act)
    raw_clip = float(action_raw_clip)
    if raw_clip > 0.0:
        raw_act = raw_act.clamp(-raw_clip, raw_clip)
    base_dist = Normal(raw_loc_sel, sig_sel + 1e-6)
    # The tanh/exp transform Jacobian is parameter-independent for replayed
    # actions, so it has zero policy-gradient effect. Using the base Normal
    # log-prob keeps boundary actions from adding a large constant to the loss.
    logp = base_dist.log_prob(raw_act)
    logp_clip = float(policy_logp_clip)
    if logp_clip > 0.0:
        logp = logp.clamp(min=-logp_clip, max=logp_clip)
    entropy = (0.5 * (1.0 + math.log(2.0 * math.pi)) + logsig_sel).mean()

    # Small stabilizers for bounded continuous control:
    # 1) center the option policy around mult=1.0 unless reward strongly says otherwise
    # 2) discourage abrupt changes relative to the previous applied multiplier
    prev_log_mult = prev_mults.clamp(min=net.MULT_MIN, max=net.MULT_MAX).log()
    action_center_loss = mu_sel.pow(2).mean()
    action_smooth_loss = (mu_sel - prev_log_mult).pow(2).mean()
    # The state no longer contains an explicit cwnd/BDP feature.  Keep this
    # legacy auxiliary strictly disabled even if old configs still pass a gain.
    action_aux_loss = torch.zeros((), device=device)

    q_sel_det   = q.gather(1, options.unsqueeze(1)).squeeze(1).detach()
    v_det       = q.max(dim=1).values.detach()
    beta_cur    = beta.gather(1, options.unsqueeze(1)).squeeze(1)

    termination_loss = (beta_cur * can_term * (1.0 - force_term)
                        * (q_sel_det - v_det + term_margin)
                        * (1.0 - dones)).mean()
    advantage = gt.detach() - q_sel_det
    if advantage_norm and advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    policy_loss = -(logp * advantage).mean()

    opt_pref = torch.softmax(q / max(float(option_usage_temp), 1e-6), dim=1)
    opt_mass_soft = opt_pref.mean(dim=0)
    uniform = torch.full_like(opt_mass_soft, 1.0 / net.n_options)
    usage_loss = (opt_mass_soft * (opt_mass_soft.clamp_min(1e-8).log()
                  - uniform.log())).sum()

    raw_loc_all = net._raw_loc_from_log_mult(mu)
    scale_all = logsig.exp().clamp_min(1e-4)
    pair_losses = []
    kl_target = max(float(option_kl_target), 0.0)
    for i in range(net.n_options):
        for j in range(i + 1, net.n_options):
            loc_i, loc_j = raw_loc_all[:, i], raw_loc_all[:, j]
            sig_i, sig_j = scale_all[:, i], scale_all[:, j]
            kl_ij = torch.log(sig_j / sig_i) + (
                sig_i.pow(2) + (loc_i - loc_j).pow(2)
            ) / (2.0 * sig_j.pow(2)) - 0.5
            kl_ji = torch.log(sig_i / sig_j) + (
                sig_j.pow(2) + (loc_j - loc_i).pow(2)
            ) / (2.0 * sig_i.pow(2)) - 0.5
            sym_kl = 0.5 * (kl_ij + kl_ji)
            pair_losses.append(torch.relu(kl_target - sym_kl).mean())
    action_div_loss = (torch.stack(pair_losses).mean()
                       if pair_losses else torch.zeros((), device=device))

    beta_by_option_t = beta.mean(dim=0)
    beta_target_loss = (beta_by_option_t - float(beta_target)).pow(2).mean()

    q_sorted = torch.sort(q.detach(), dim=1, descending=True).values
    q_gap = (q_sorted[:, 0] - q_sorted[:, 1]).mean() if net.n_options > 1 else torch.zeros((), device=device)
    option_mass_t = torch.bincount(options, minlength=net.n_options).float()
    option_mass_t = option_mass_t / option_mass_t.sum().clamp_min(1.0)
    mu_by_option_t = mu.detach().mean(dim=0)
    q_by_option_t = q.detach().mean(dim=0)

    actor_loss       = (
        c_action * policy_loss
        + c_term * termination_loss
        - c_entropy * entropy
        + c_action_center * action_center_loss
        + c_action_smooth * action_smooth_loss
        + c_bdp_action * action_aux_loss
        + c_option_usage * usage_loss
        + c_action_div * action_div_loss
        + c_beta_target * beta_target_loss
    )

    return ActorInfo(
        loss      = actor_loss,
        action    = policy_loss.item(),
        term      = termination_loss.item(),
        mean_beta = float(beta.detach().mean().item()),
        entropy   = float(entropy.item()),
        usage     = float(usage_loss.detach().item()),
        action_div = float(action_div_loss.detach().item()),
        beta_target = float(beta_target_loss.detach().item()),
        center    = float(action_center_loss.detach().item()),
        smooth    = float(action_smooth_loss.detach().item()),
        action_aux = float(action_aux_loss.detach().item()),
        q_gap     = float(q_gap.detach().item()),
        option_mass = [float(x) for x in option_mass_t.detach().cpu().tolist()],
        beta_by_option = [float(x) for x in beta_by_option_t.detach().cpu().tolist()],
        mu_by_option = [float(x) for x in mu_by_option_t.detach().cpu().tolist()],
        q_by_option = [float(x) for x in q_by_option_t.detach().cpu().tolist()],
    )
