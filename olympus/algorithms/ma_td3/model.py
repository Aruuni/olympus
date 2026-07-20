"""MA-TD3 networks and reward/global-state math for the Astraea recreation.

Follows "Astraea: Towards Fair and Efficient Learning-based Congestion
Control" (EuroSys '24) and the released inference agent
(``astraea/agent/agent.py`` in this repo):

* Actor — deterministic policy on the flow's stacked local state (the last
  ``w`` MTP observations, ``rec_dim`` in the original code), 3-layer MLP
  (h1 -> h2 -> ceil(h2/2)) with BatchNorm and a tanh head in [-1, 1].
  Decentralized execution needs only this network.
* TwinCritic — two independent Q towers (TD3 clipped double-Q). Each sees
  [stacked local state ++ 12-dim global state] and folds the action in after
  the first layer, exactly like the original Critic.build; ``is_global`` is
  always on for multi-agent training (paper section 3.4).
* astraea_team_reward — the global reward, Eq. 8:
      R = c0*R_thr - c1*R_lat - c2*R_loss - c3*R_fair - c4*R_stab
  computed by the learner over the joint window of simultaneous flows.
* global_state_from_stats — the paper's Table 2 global state, with the exact
  normalization of ``astraea.agent.definitions.transform_state`` (the
  released agent bundle).

Local per-step observations come from the ``astraea`` state plugin (10
features); history stacking happens in the worker (a shift register, zero
initialized) and in the learner (from replayed sequence chunks).
"""

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from olympus.common.action_plugins import load_action_module
from olympus.common.state_plugins import (
    assert_state_compatible,
    current_state_name,
    load_state_module,
    state_meta,
)
from olympus.rewards.astraea import (
    DEFAULT_WEIGHTS,
    MSS_BYTES,
    latency_threshold_us,
)


_ACTION_PLUGIN = load_action_module()
ACTION_DIM = int(_ACTION_PLUGIN.ACTION_DIM)
ACTION_MIN = float(_ACTION_PLUGIN.ACTION_MIN)
ACTION_MAX = float(_ACTION_PLUGIN.ACTION_MAX)

STATE_NAME = current_state_name(default='astraea')
_STATE_PLUGIN = load_state_module('ma_td3', STATE_NAME)
STATE_DIM = int(_STATE_PLUGIN.STATE_DIM)
STATE_FEATURES = list(_STATE_PLUGIN.STATE_FEATURES)
STATE_FEATURE_VERSION = str(_STATE_PLUGIN.STATE_FEATURE_VERSION)
normalize_state = _STATE_PLUGIN.normalize_state

# Paper Table 2 (the released agent's GLOBAL_DIM): ovr/min/max throughput,
# avg latency, min/max/avg cwnd, avg loss, flow count, one-way delay, BDP,
# link bandwidth.
GLOBAL_DIM = 12
# w — MTPs of local-state history fed to the policy (rec_dim in astraea.json)
# and the window of the fairness/stability reward metrics (Eq. 6-7).
HISTORY_LEN_DEFAULT = 5


Experience = namedtuple(
    'Experience',
    [
        'state',          # normalized 10-dim observation before the action
        'action',         # tanh-bounded scalar the worker applied
        'reward',         # worker-local reward (diagnostic; learner rebuilds)
        'next_state',     # normalized observation after the action
        'throughput',     # avg_thr (bytes/s) observed at next_state
        'latency_us',     # avg_urtt (us) observed at next_state
        'loss_ratio',     # lost bytes/s observed at next_state
        'pacing_rate',    # pacing rate (bytes/s) observed at next_state
        'cwnd',           # cwnd (packets) observed at next_state
        'min_rtt_us',     # flow's observed min RTT (d0 baseline) at this step
        'rtt_ref_us',     # scheduled base RTT (training-only link oracle)
        'link_bw',        # bottleneck bandwidth (bytes/s) at this step
        'buffer_bytes',   # bottleneck queue size (training-only link oracle)
        'agent_id',
        'group_id',
        'group_step',
        'done',
        'traj_id',
        'step_in_traj',
    ],
)


def model_state_meta() -> dict:
    return state_meta(_STATE_PLUGIN, STATE_NAME)


def assert_checkpoint_state_compatible(ckpt: dict, source='checkpoint') -> None:
    assert_state_compatible(model_state_meta(), ckpt, source=source)
    history = (ckpt or {}).get('history_len')
    if history is not None and int(history) != int(_expected_history[0]):
        raise ValueError(
            f'{source} history mismatch: history_len={history} vs '
            f'{_expected_history[0]}')
    actor_norm = (ckpt or {}).get('actor_norm')
    if (actor_norm is not None
            and str(actor_norm).lower() != _expected_actor_norm[0]):
        raise ValueError(
            f'{source} actor normalization mismatch: actor_norm={actor_norm} '
            f'vs {_expected_actor_norm[0]}')


# The worker/learner set this once from config so checkpoint payloads can be
# validated against the constructed network width.
_expected_history = [HISTORY_LEN_DEFAULT]
_expected_actor_norm = ['batchnorm']


def set_expected_history(history_len: int) -> None:
    _expected_history[0] = int(history_len)


def set_expected_actor_norm(actor_norm: str) -> None:
    norm = str(actor_norm or 'batchnorm').strip().lower()
    if norm not in ('batchnorm', 'layernorm'):
        raise ValueError(
            f'unsupported Astraea actor normalization {actor_norm!r}; '
            'expected batchnorm or layernorm')
    _expected_actor_norm[0] = norm


# ── Networks ──────────────────────────────────────────────────────────────────

class PaperBatchNorm1d(nn.Module):
    """TensorFlow ``layers.batch_normalization(scale=False)`` equivalent."""

    def __init__(self, width: int):
        super().__init__()
        # TensorFlow defaults: decay/momentum=0.99 and epsilon=1e-3. PyTorch's
        # momentum is the NEW-sample weight, hence 1 - 0.99 = 0.01.
        self.norm = nn.BatchNorm1d(
            int(width), eps=1e-3, momentum=0.01, affine=False)
        # scale=False removes gamma but TensorFlow still learns beta/offset.
        self.bias = nn.Parameter(torch.zeros(int(width)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value) + self.bias


class Actor(nn.Module):
    """mu(stacked local state) -> tanh action in [-1, 1].

    Mirrors the released 3-layer actor (h1=256, h2=128, h3=ceil(h2/2)).
    BatchNorm is the paper-faithful default; LayerNorm remains available only
    for loading experiments made with the earlier Olympus recreation.
    """

    def __init__(self, state_dim: int = STATE_DIM,
                 history_len: int = HISTORY_LEN_DEFAULT,
                 h1: int = 256, h2: int = 128,
                 actor_norm: str = 'batchnorm'):
        super().__init__()
        self.state_dim = int(state_dim)
        self.history_len = int(history_len)
        self.input_dim = self.state_dim * self.history_len
        self.actor_norm = str(actor_norm or 'batchnorm').strip().lower()
        if self.actor_norm not in ('batchnorm', 'layernorm'):
            raise ValueError(
                f'unsupported Astraea actor normalization {actor_norm!r}')
        h3 = math.ceil(h2 / 2)

        def norm(width):
            # The released TensorFlow actor uses batch normalization at all
            # three hidden layers.  BatchNorm works with batch-size-one
            # inference because workers always put the actor in eval mode.
            if self.actor_norm == 'batchnorm':
                return PaperBatchNorm1d(width)
            return nn.LayerNorm(width)

        # negative_slope=0.2 matches the released agent's tf.nn.leaky_relu
        # (TF default alpha=0.2); PyTorch's nn.LeakyReLU() default is 0.01.
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            norm(h1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(h1, h2),
            norm(h2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(h2, h3),
            norm(h3),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.action_head = nn.Linear(h3, ACTION_DIM)
        # TensorFlow dense layers use Glorot initialization by default.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.action_head(self.net(stacked)))

    @torch.no_grad()
    def act(self, stacked_np: np.ndarray, noise_std: float = 0.0):
        device = next(self.parameters()).device
        stacked = torch.from_numpy(
            np.asarray(stacked_np, dtype=np.float32)).unsqueeze(0).to(device)
        action = self.forward(stacked)[0, 0]
        if noise_std > 0.0:
            action = (action + noise_std * torch.randn_like(action)).clamp(
                ACTION_MIN, ACTION_MAX)
        mult = float(_ACTION_PLUGIN.to_multiplier(action).item())
        return float(action.item()), mult


class SingleCritic(nn.Module):
    """Q([stacked local ++ global], a) — original layout: the action joins
    after the first dense layer (Critic.build in astraea/agent/agent.py)."""

    def __init__(self, input_dim: int, h1: int = 256, h2: int = 128):
        super().__init__()
        h3 = math.ceil(h2 / 2)
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1 + ACTION_DIM, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.q_head = nn.Linear(h3, 1)
        # negative_slope=0.2 matches the released agent's tf.nn.leaky_relu.
        self.act_fn = nn.LeakyReLU(negative_slope=0.2)
        for module in (self.fc1, self.fc2, self.fc3, self.q_head):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        x = self.act_fn(self.fc1(s))
        x = self.act_fn(self.fc2(torch.cat([x, a], dim=-1)))
        x = self.act_fn(self.fc3(x))
        return self.q_head(x).squeeze(-1)


class TwinCritic(nn.Module):
    """Two independent critics — TD3 clipped double-Q (critic/critic2)."""

    def __init__(self, state_dim: int = STATE_DIM,
                 history_len: int = HISTORY_LEN_DEFAULT,
                 global_dim: int = GLOBAL_DIM,
                 h1: int = 256, h2: int = 128):
        super().__init__()
        input_dim = int(state_dim) * int(history_len) + int(global_dim)
        self.q1 = SingleCritic(input_dim, h1, h2)
        self.q2 = SingleCritic(input_dim, h1, h2)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        return self.q1(s, a), self.q2(s, a)

    def q1_only(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q1(s, a)


def soft_update(online: nn.Module, target: nn.Module, tau: float = 0.005):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)
    # BatchNorm running statistics are buffers, not parameters.  Keep target
    # statistics synchronized just like target weights; integer counters are
    # copied because interpolation is undefined for them.
    online_buffers = dict(online.named_buffers())
    for name, target_buffer in target.named_buffers():
        online_buffer = online_buffers.get(name)
        if online_buffer is None:
            continue
        if target_buffer.is_floating_point():
            target_buffer.data.mul_(1.0 - tau).add_(
                online_buffer.data, alpha=tau)
        else:
            target_buffer.data.copy_(online_buffer.data)


# ── Global state (paper Table 2) ──────────────────────────────────────────────

def _masked_stat(values, mask, fn, fill):
    """Reduce ``values`` over the last (agent) axis honoring ``mask``."""
    masked = np.where(mask > 0.0, values, fill)
    out = fn(masked, axis=-1)
    any_present = mask.sum(axis=-1) > 0.0
    return np.where(any_present, out, 0.0)


def global_state_from_stats(throughput, latency_us, cwnd, loss_ratio, mask,
                            link_bw, rtt_ref_us, buffer_bytes) -> np.ndarray:
    """12-dim global state (paper Table 2) from per-agent stats at ONE step.

    Table 2: ovr/min/max throughput, avg latency, min/max/avg cwnd, avg loss
    ratio, flow count, d0 (base one-way delay of the link), buf (bottleneck
    buffer size), c (link bandwidth). Arrays are (..., N); ``link_bw``
    (bytes/s), ``rtt_ref_us`` and ``buffer_bytes`` are per agent (flows may
    propagate at their own base RTT — the aggregate d0 entry uses the
    presence-masked mean). Feature scaling follows the released agent bundle
    (``astraea.agent.definitions.transform_state`` global branch); the
    training-only link oracle supplies d0/buf/c.
    """
    throughput = np.asarray(throughput, dtype=np.float64)
    latency_us = np.asarray(latency_us, dtype=np.float64)
    cwnd = np.asarray(cwnd, dtype=np.float64)
    loss_ratio = np.asarray(loss_ratio, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    link_bw = np.asarray(link_bw, dtype=np.float64)
    rtt_ref_us = np.asarray(rtt_ref_us, dtype=np.float64)
    buffer_bytes = np.asarray(buffer_bytes, dtype=np.float64)

    n = mask.sum(axis=-1)
    denom = np.maximum(n, 1.0)

    thr = throughput / 5e7
    lat = latency_us / 5e5
    win = cwnd / 1000.0
    loss = loss_ratio / 1e6

    bw_bytes = _masked_stat(link_bw, mask, np.max, 0.0)
    bw_mbps = bw_bytes * 8.0 / 1e6
    rtt_ms = ((rtt_ref_us * mask).sum(axis=-1) / denom) / 1e3
    one_way_delay_ms = rtt_ms / 2.0
    buffer_packets = _masked_stat(buffer_bytes, mask, np.max, 0.0) / MSS_BYTES

    g = np.stack([
        (thr * mask).sum(axis=-1),
        _masked_stat(thr, mask, np.min, np.inf),
        _masked_stat(thr, mask, np.max, -np.inf),
        (lat * mask).sum(axis=-1) / denom,
        _masked_stat(win, mask, np.min, np.inf),
        _masked_stat(win, mask, np.max, -np.inf),
        (win * mask).sum(axis=-1) / denom,
        (loss * mask).sum(axis=-1) / denom,
        n / 10.0,
        one_way_delay_ms / 500.0,
        buffer_packets / 10.0,
        bw_mbps / 500.0,
    ], axis=-1)
    return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ── Global reward (paper Eq. 4-8) ─────────────────────────────────────────────

def astraea_team_reward(throughput, latency_us, loss_ratio, pacing_rate,
                        mask, link_bw, min_rtt_us, weights=None):
    """Global convergence reward over a window of joint steps.

    Arrays are (B, W, N): the last ``w`` MTPs of every agent in the group,
    newest last. Instantaneous terms (R_thr/R_lat/R_loss) use the final step;
    fairness uses the window-averaged per-flow throughputs (Eq. 6-7) and
    stability the per-flow throughput deviation across the window.
    ``min_rtt_us`` is each flow's observed min RTT — the d0 baseline of the
    latency term. Returns ``(reward (B,), components dict)``; the reward is
    shared by every agent of the group (the paper's single global scalar).
    """
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})

    throughput = np.asarray(throughput, dtype=np.float64)
    latency_us = np.asarray(latency_us, dtype=np.float64)
    loss_ratio = np.asarray(loss_ratio, dtype=np.float64)
    pacing_rate = np.asarray(pacing_rate, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    link_bw = np.asarray(link_bw, dtype=np.float64)
    min_rtt_us = np.asarray(min_rtt_us, dtype=np.float64)

    now = mask[:, -1, :]                       # (B, N) flows present now
    n_now = np.maximum(now.sum(axis=-1), 1.0)  # avoid /0; empty rows -> 0 terms

    # R_thr: overall utilization of the bottleneck.
    bw_now = _masked_stat(link_bw[:, -1, :], now, np.max, 0.0)
    r_thr = (throughput[:, -1, :] * now).sum(axis=-1) / np.maximum(bw_now, 1.0)

    # R_lat (paper Eq. 5): the flows' average latency excess over the
    # (1+beta)*d0 threshold, thresholded AFTER averaging exactly as in the
    # paper, then multiplied by the pacing rate — "the total increased
    # latency of all sending packets" in an MTP, i.e. pacing in packets/s.
    # d0 is each flow's own observed min_rtt, so the zero-queue baseline
    # stays zero under heterogeneous per-flow RTTs; flows without latency or
    # min_rtt samples yet are left out of the average.
    lat_now = latency_us[:, -1, :]
    min_rtt_now = min_rtt_us[:, -1, :]
    lat_mask = now * (lat_now > 0.0) * (min_rtt_now > 0.0)
    n_lat = np.maximum(lat_mask.sum(axis=-1), 1.0)
    thresholds = latency_threshold_us(
        min_rtt_now, w['delay_coefficient'])
    mean_diff_s = (
        ((lat_now - thresholds) / 1e6) * lat_mask).sum(axis=-1) / n_lat
    mean_excess = np.maximum(mean_diff_s, 0.0)
    mean_pacing_pkts = (
        (pacing_rate[:, -1, :] / MSS_BYTES) * now).sum(axis=-1) / n_now
    r_lat = mean_excess * mean_pacing_pkts
    # Bound R_lat before weighting: on deep-buffer episodes a few ms of queue
    # with no loss sends this into the tens, and c1*R_lat then dwarfs the 0.1
    # throughput ceiling, storing an ordinary CWND overshoot as the maximum
    # penalty and driving the pessimistic TD3 actor to the CWND floor. Any
    # value above reward_clip/latency_weight already saturates the clip, so
    # this loses no gradient. Cap is inf by default (paper Eq. 5); see config.
    r_lat = np.minimum(r_lat, w['latency_cap'])

    # R_loss: mean per-flow loss-to-throughput ratio.
    thr_now = throughput[:, -1, :]
    loss_term = np.where(thr_now > 0.0, loss_ratio[:, -1, :] /
                         np.maximum(thr_now, 1e-9), 0.0)
    r_loss = (loss_term * now).sum(axis=-1) / n_now

    # avg_thr_i over the window (Eq. 7) — masked per step.
    steps_i = np.maximum(mask.sum(axis=1), 1.0)          # (B, N)
    avg_thr = (throughput * mask).sum(axis=1) / steps_i  # (B, N)

    # R_fair (Eq. 6): dispersion of window-averaged throughputs across the
    # flows present now, normalized by their sum.
    mean_avg = (avg_thr * now).sum(axis=-1, keepdims=True) / n_now[:, None]
    sq_dev = ((avg_thr - mean_avg) ** 2) * now
    total_avg = (avg_thr * now).sum(axis=-1)
    fair_denom = n_now * np.maximum(total_avg, 1e-9) ** 2
    r_fair = np.sqrt(sq_dev.sum(axis=-1) / fair_denom)
    r_fair = np.where(total_avg > 0.0, r_fair, 0.0)

    # R_stab (Eq. 6): per-flow throughput deviation across the window,
    # normalized by that flow's window average; mean across flows.
    dev = ((throughput - avg_thr[:, None, :]) ** 2) * mask
    stab_denom = steps_i * np.maximum(avg_thr, 1e-9) ** 2
    stab_i = np.sqrt(dev.sum(axis=1) / stab_denom)
    stab_i = np.where(avg_thr > 0.0, stab_i, 0.0)
    r_stab = (stab_i * now).sum(axis=-1) / n_now

    reward = (
        w['throughput_weight'] * r_thr
        - w['latency_weight'] * r_lat
        - w['loss_weight'] * r_loss
        - w['fairness_weight'] * r_fair
        - w['stability_weight'] * r_stab
    )
    clip = w['reward_clip']
    reward = np.clip(reward, -clip, clip).astype(np.float32)
    components = {
        'r_thr': r_thr,
        'r_lat': r_lat,
        'r_loss': r_loss,
        'r_fair': r_fair,
        'r_stab': r_stab,
    }
    return reward, components


__all__ = [
    'ACTION_DIM',
    'ACTION_MAX',
    'ACTION_MIN',
    'Actor',
    'Experience',
    'GLOBAL_DIM',
    'HISTORY_LEN_DEFAULT',
    'STATE_DIM',
    'STATE_FEATURES',
    'STATE_FEATURE_VERSION',
    'SingleCritic',
    'TwinCritic',
    'assert_checkpoint_state_compatible',
    'astraea_team_reward',
    'global_state_from_stats',
    'model_state_meta',
    'normalize_state',
    'set_expected_history',
    'set_expected_actor_norm',
    'soft_update',
]
