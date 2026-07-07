"""ORCA-repo-style actor, critic, state transform, and TD3 losses.

This follows the public Soheil-ab/Orca implementation more closely than the
generic recurrent TD3 path:

* 15 raw TCP/ORCA inputs are transformed into the repo's 7 compact features.
* A flat `rec_dim=10` history is the policy input instead of an LSTM.
* Actor actions remain bounded in `[-1, 1]`; the selected Olympus action
  plugin maps them to live CWND updates.
* The actor is a TensorFlow-style MLP: Dense + BatchNorm + LeakyReLU.
"""

import json
import math
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.common.action_plugins import (
    current_action_meta,
    load_action_module,
)
from olympus.common import runtime_config
from olympus.common.state_plugins import assert_state_compatible

_ACTION_PLUGIN = load_action_module()
_RUNTIME_CFG = runtime_config.load_config()

STATS_FILENAME = 'stats.json'


RAW_FEATURES = [
    'delay_ms',
    'throughput',
    'samples',
    'delta_t',
    'target',
    'cwnd',
    'pacing_rate',
    'loss_rate',
    'srtt_ms',
    'snd_ssthresh',
    'packets_out',
    'retrans_out',
    'max_packets_out',
    'mss',
    'min_rtt_ms',
]
BASE_STATE_FEATURES = [
    'thr_over_max_bw',
    'pacing_over_max_bw',
    'loss_penalty_over_max_bw',
    'samples_over_cwnd',
    'delta_t',
    'min_rtt_over_srtt',
    'delay_metric',
]

RAW_DIM = 15
BASE_STATE_DIM = 7
REC_DIM = int(runtime_config.agent_value(
    _RUNTIME_CFG, 'rec_dim', env='SAO_ORCA_REC_DIM',
    default=os.environ.get('SAO_REC_DIM', '10')))
STATE_DIM = BASE_STATE_DIM * REC_DIM
ACTION_DIM = int(_ACTION_PLUGIN.ACTION_DIM)
ACTION_MIN = float(_ACTION_PLUGIN.ACTION_MIN)
ACTION_MAX = float(_ACTION_PLUGIN.ACTION_MAX)
STATE_NAME = 'orca_repo'
STATE_FEATURE_VERSION = f'orca_repo_state_v1_rec{REC_DIM}'
STATE_FEATURES = [
    f'h{hist}:{name}'
    for hist in range(REC_DIM)
    for name in BASE_STATE_FEATURES
]

Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'traj_id', 'step_in_traj'])
CriticInfo = namedtuple('CriticInfo', ['loss', 'q1_mean', 'q2_mean', 'td_abs'])
ActorInfo = namedtuple('ActorInfo', ['loss', 'q_mean', 'a_mean', 'a_abs'])


def _finite(value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


class OnlineNormalizer:
    """The upstream ORCA Welford normalizer with per-dimension min tracking.

    Mirrors Normalizer in Soheil-ab/Orca rl-module/envwrapper.py, including the
    stats.json save/load protocol so workers can resume the running statistics
    across runs.
    """

    def __init__(self, dim: int = RAW_DIM):
        self.dim = int(dim)
        self.n = 1e-5
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.mean_diff = np.zeros(self.dim, dtype=np.float64)
        self.var = np.zeros(self.dim, dtype=np.float64)
        self.min = np.zeros(self.dim, dtype=np.float64)

    def observe(self, x: np.ndarray) -> None:
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = self.mean_diff / self.n

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.n <= 2.0:
            return np.zeros_like(x, dtype=np.float64)
        std = np.sqrt(np.maximum(self.var, 1e-12))
        z = (x - self.mean) / std
        self.min = np.minimum(self.min, z)
        return z

    def stats(self) -> np.ndarray:
        return self.min

    def save_stats(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            'n': float(self.n),
            'mean': self.mean.tolist(),
            'mean_diff': self.mean_diff.tolist(),
            'var': self.var.tolist(),
            'min': self.min.tolist(),
        }
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as fp:
            json.dump(payload, fp)
        os.replace(tmp, path)

    def load_stats(self, path: str) -> bool:
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path) as fp:
                history = json.load(fp)
        except (OSError, ValueError):
            return False
        try:
            mean = np.asarray(history['mean'], dtype=np.float64)
            if mean.shape != (self.dim,):
                return False
            self.n = float(history['n'])
            self.mean = mean
            self.mean_diff = np.asarray(history['mean_diff'], dtype=np.float64)
            self.var = np.asarray(history['var'], dtype=np.float64)
            self.min = np.asarray(history['min'], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        return True


class OrcaRepoTransform:
    """State/reward transform from Soheil-ab/Orca rl-module/envwrapper.py."""

    def __init__(self, delay_margin_coef: float = 1.25,
                 use_normalizer: bool = False):
        self.delay_margin_coef = float(delay_margin_coef)
        self.use_normalizer = bool(use_normalizer)
        self.normalizer = OnlineNormalizer(RAW_DIM) if self.use_normalizer else None
        self.max_bw = 0.0
        self.max_cwnd = 0.0
        self.max_smp = 0.0
        self.min_del = 9_999_999.0
        self.last_components = {}

    def raw_vector(self, info: dict, interval_s: float, target: float = 0.0) -> np.ndarray:
        avg_urtt_us = _finite(info.get('avg_urtt', info.get('delay_us', 0.0)))
        delay_ms = avg_urtt_us / 1000.0
        throughput = max(_finite(info.get('throughput', info.get('avg_thr', 0.0))), 0.0)
        samples = max(_finite(info.get('count', info.get('cnt', 0.0))), 0.0)
        delta_t = max(float(interval_s or 0.0), 1e-6)
        cwnd = max(_finite(info.get('cwnd', 1.0), 1.0), 1.0)
        pacing_rate = max(_finite(info.get('pacing_rate', 0.0)), 0.0)

        loss_rate = max(_finite(info.get('lost_rate',
                                          info.get('loss_rate', 0.0))), 0.0)
        if loss_rate <= 0.0:
            loss_bytes = max(_finite(info.get('loss_bytes',
                                              info.get('lost_bytes', 0.0))), 0.0)
            loss_rate = loss_bytes / delta_t

        srtt_raw = _finite(info.get('srtt_us', 0.0))
        srtt_ms = ((srtt_raw / 8.0) if srtt_raw > 0.0 else avg_urtt_us) / 1000.0
        min_rtt_ms = _finite(info.get('min_rtt_ms', None), None)
        if min_rtt_ms is None:
            min_rtt_ms = _finite(info.get('min_rtt',
                                          info.get('min_rtt_us', 0.0))) / 1000.0

        return np.array([
            delay_ms,
            throughput,
            samples,
            delta_t,
            max(_finite(info.get('target', target)), 0.0),
            cwnd,
            pacing_rate,
            loss_rate,
            max(srtt_ms, 0.0),
            max(_finite(info.get('snd_ssthresh', 0.0)), 0.0),
            max(_finite(info.get('packets_out', 0.0)), 0.0),
            max(_finite(info.get('retrans_out', 0.0)), 0.0),
            max(_finite(info.get('max_packets_out', 0.0)), 0.0),
            max(_finite(info.get('mss', info.get('mss_cache', 0.0))), 0.0),
            max(min_rtt_ms, 0.0),
        ], dtype=np.float64)

    def transform_raw(self, raw: np.ndarray, evaluation: bool = False):
        if self.use_normalizer:
            if not evaluation:
                self.normalizer.observe(raw)
            norm = self.normalizer.normalize(raw)
            mins = self.normalizer.stats()
        else:
            norm = raw.astype(np.float64, copy=True)
            mins = np.zeros_like(norm)

        d_n = norm[0] - mins[0]
        thr_n_min = norm[1] - mins[1]
        samples_n_min = norm[2] - mins[2]
        # Matches envwrapper.py:253 — delta_t_n is the normalized value, not
        # the min-shifted one. With use_normalizer=False (Orca's default),
        # norm[3] == raw[3] because transform_raw copies raw into norm.
        delta_t_n = norm[3]
        cwnd_n_min = norm[5] - mins[5]
        pacing_rate_n_min = norm[6] - mins[6]
        loss_rate_n_min = norm[7] - mins[7]
        srtt_ms_min = norm[8] - mins[8]
        min_rtt_min = raw[14] - mins[14]

        self.max_bw = max(self.max_bw, thr_n_min)
        self.max_cwnd = max(self.max_cwnd, cwnd_n_min)
        self.max_smp = max(self.max_smp, samples_n_min)
        self.min_del = min(self.min_del, d_n)

        if srtt_ms_min > 1e-12:
            delay_metric = min(1.0, self.delay_margin_coef * min_rtt_min / srtt_ms_min)
        else:
            delay_metric = 1.0

        if self.max_bw > 1e-12:
            reward = ((thr_n_min - 5.0 * loss_rate_n_min) / self.max_bw
                      * delay_metric)
            state0 = thr_n_min / self.max_bw
            pacing = min(pacing_rate_n_min / self.max_bw, 10.0)
            loss = 5.0 * loss_rate_n_min / self.max_bw
        else:
            reward = 0.0
            state0 = pacing = loss = 0.0

        samples = raw[2]
        cwnd = max(raw[5], 1.0)
        min_over_srtt = min_rtt_min / srtt_ms_min if srtt_ms_min > 1e-12 else 0.0
        state = np.array([
            state0,
            pacing,
            loss,
            samples / cwnd,
            delta_t_n,
            min_over_srtt,
            delay_metric,
        ], dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        self.last_components = {
            'reward': float(reward),
            'thr_n_min': float(thr_n_min),
            'loss_rate_n_min': float(loss_rate_n_min),
            'max_bw': float(self.max_bw),
            'delay_metric': float(delay_metric),
            'min_rtt_min': float(min_rtt_min),
            'srtt_ms_min': float(srtt_ms_min),
        }
        return state, float(reward)

    def step(self, info: dict, interval_s: float, target: float = 0.0,
             evaluation: bool = False):
        return self.transform_raw(
            self.raw_vector(info, interval_s=interval_s, target=target),
            evaluation=evaluation,
        )

    def save_stats(self, path: str) -> bool:
        if not self.use_normalizer or self.normalizer is None or not path:
            return False
        self.normalizer.save_stats(path)
        return True

    def load_stats(self, path: str) -> bool:
        if not self.use_normalizer or self.normalizer is None or not path:
            return False
        return self.normalizer.load_stats(path)


class HistoryStack:
    """Flat rec_dim history buffer used by upstream ORCA recurrent mode."""

    def __init__(self, rec_dim: int = REC_DIM):
        self.rec_dim = int(rec_dim)
        self.state = np.zeros(self.rec_dim * BASE_STATE_DIM, dtype=np.float32)

    def push(self, base_state: np.ndarray) -> np.ndarray:
        base_state = np.asarray(base_state, dtype=np.float32).reshape(BASE_STATE_DIM)
        self.state = np.concatenate([self.state[BASE_STATE_DIM:], base_state]).astype(np.float32)
        return self.state.copy()


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
) -> CriticInfo:
    with torch.no_grad():
        a_next = actor_target(s2_batch).squeeze(-1)
        noise = (torch.randn_like(a_next) * target_noise_std).clamp(
            -target_noise_clip, target_noise_clip)
        a_next = (a_next + noise).clamp(ACTION_MIN, ACTION_MAX)
        q1_next, q2_next = critic_target(s2_batch, a_next)
        q_next = torch.min(q1_next, q2_next)
        y = r_batch + gamma * q_next * (1.0 - d_batch)

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
