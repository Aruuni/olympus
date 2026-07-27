"""
learner.py — recurrent Soft Actor-Critic (SAC) learner for olympus.

Off-policy: workers push one `Experience(s, a_squash, r, s', done, …)` per
control step into a replay buffer.  The learner samples random length-L chunks
from per-trajectory lists, zero-inits the LSTM hidden state, and runs the
canonical SAC v2 update every step:

  1. Critic: soft clipped-double-Q target with the entropy bonus baked in.
     y = r + γ(1−d)[ min_i Q_i^target(s', ã') − α·log π(ã'|s') ],  ã' ~ π(·|s')
     L = Σ_i (Q_i(s, a) − y)²
  2. Actor: maximum-entropy policy gradient on min(Q1, Q2) with reparameterised
     samples, minimising E[α·log π − min_i Q_i].
  3. Temperature α: automatically tuned toward target entropy −ACTION_DIM
     (SAC v2), so exploration is learned rather than hand-scheduled.
  4. Critic targets: Polyak-averaged with τ every step.  There is NO target
     actor in SAC.

Differences from the td3 learner: stochastic actor + entropy term, learnable
α, no policy_delay / target-noise knobs, and the actor update fires every step.
The IPC / replay / checkpoint scaffolding is intentionally shared idiom with
the td3 learner so runs are directly comparable.

Usage (started by orchestrator):
  python olympus/algorithms/sac/learner.py --config olympus/config.yaml
"""

import argparse
import copy
import csv
import io
import json
import multiprocessing
import os
import pickle
import random
import secrets
import signal
import sys
import time
from collections import defaultdict, deque
from multiprocessing.managers import BaseManager

import numpy as np
import torch
import torch.optim as optim
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from olympus.algorithms.sac.model import (
    Actor, TwinCritic, Experience,
    STATE_DIM, ACTION_DIM,
    actor_arch_from_checkpoint, actor_distillation_loss, actor_loss,
    actor_model_meta, assert_checkpoint_state_compatible,
    critic_distillation_loss, critic_loss, model_state_meta, soft_update,
    temperature_loss,
)
from olympus.common.mixed_replay import build_mixed_replay


def _continual_config(cfg: dict) -> dict:
    """Return merged top-level and deployment continual-learning options."""
    options = dict((cfg.get('continual') or {}))
    options.update(dict(((cfg.get('deployment') or {}).get('continual') or {})))
    return options


def _reward_signature(cfg: dict) -> str:
    """Stable identity for scalar rewards stored in persistent replay."""
    runtime = cfg.get('runtime') or {}
    value = {
        'name': _canonical_reward_name(runtime.get('reward', '')),
        'options': cfg.get('reward') or {},
    }
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _canonical_reward_name(name) -> str:
    name = str(name)
    return 'proteus' if name == 'crl_network_utility' else name


def _canonical_reward_signature(signature: str) -> str:
    """Normalize reward renames when restoring an existing replay file."""
    try:
        value = json.loads(signature)
    except (TypeError, ValueError):
        return signature
    if not isinstance(value, dict):
        return signature
    value['name'] = _canonical_reward_name(value.get('name', ''))
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


class UpdateBudget:
    """Bound gradient work by the number of newly collected transitions."""

    def __init__(self, updates_per_transition: float, max_burst: int):
        self.ratio = max(float(updates_per_transition), 0.0)
        self.max_burst = max(int(max_burst), 1)
        self.credit = 0.0

    def add(self, transitions: int) -> None:
        self.credit = min(
            float(self.max_burst),
            self.credit + max(int(transitions), 0) * self.ratio,
        )

    def take(self) -> bool:
        if self.credit < 1.0:
            return False
        self.credit -= 1.0
        return True


# ── Manager IPC ───────────────────────────────────────────────────────────────

_exp_queue   = multiprocessing.Queue(maxsize=200_000)
_PARAM_CACHE_BYTES = int(os.environ.get('SAO_PARAM_CACHE_BYTES', str(64 * 1024 * 1024)))
_param_lock  = multiprocessing.Lock()
_param_size  = multiprocessing.Value('i', 0)
_param_blob  = multiprocessing.RawArray('B', _PARAM_CACHE_BYTES)


def _push_exp(exp, source=None):
    try:    _exp_queue.put_nowait((source, exp))
    except: pass

def _push_exp_batch(exps, source=None):
    for exp in exps:
        try:    _exp_queue.put_nowait((source, exp))
        except: pass

def _pull_params():
    try:
        with _param_lock:
            n = int(_param_size.value)
            if n <= 0:
                return None
            return bytes(_param_blob[:n])
    except Exception:
        return None

def _set_latest_params(payload: bytes):
    if len(payload) > _PARAM_CACHE_BYTES:
        print(f'[learner] param payload too large for cache: {len(payload)} > '
              f'{_PARAM_CACHE_BYTES}', flush=True)
        return
    with _param_lock:
        _param_blob[:len(payload)] = payload
        _param_size.value = len(payload)


class _QueueManager(BaseManager): pass
_QueueManager.register('push_exp',       callable=_push_exp)
_QueueManager.register('push_exp_batch', callable=_push_exp_batch)
_QueueManager.register('pull_params',    callable=_pull_params)


def _manager_handle_error(self, c, msg):   # noqa: ARG001
    import traceback as _tb, io as _io
    buf = _io.StringIO()
    _tb.print_exc(file=buf)
    text = buf.getvalue()
    if not any(x in text for x in ('BrokenPipeError', 'ConnectionResetError',
                                   'EOFError', 'ConnectionRefusedError')):
        sys.stderr.write('[learner] manager server error:\n' + text)
        sys.stderr.flush()

from multiprocessing.managers import Server as _MgrServer
_MgrServer.handle_error = _manager_handle_error


# ── Running reward stats ──────────────────────────────────────────────────────

class _RunningStats:
    """Welford running mean/variance — shared idiom with the td3/PPO learners."""
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        if x.size == 0: return
        bmean, bvar, bcount = float(x.mean()), float(x.var()), float(x.size)
        delta = bmean - self.mean
        tot   = self.count + bcount
        new_mean = self.mean + delta * bcount / tot
        M2 = (self.var * self.count + bvar * bcount
              + delta ** 2 * self.count * bcount / tot)
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    @property
    def std(self) -> float:
        return max(self.var ** 0.5, 1e-6)


def _episode_from_traj_id(traj_id) -> int:
    """Workers encode traj_id as '<algorithm>_<cport>_<episode>_<flow>'."""
    try:
        return int(str(traj_id).rsplit('_', 2)[-2])
    except (IndexError, TypeError, ValueError):
        return -1


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Per-trajectory FIFO store.  Each trajectory is a list of Experience tuples
    appended in order.  Sampling grabs random length-L chunks from random
    trajectories; if a trajectory is shorter than L we pad with a mask.

    Staleness is controlled by `max_transitions` — the newest trajectories are
    kept and older ones evicted in FIFO order.  Identical to the td3 buffer so
    rollout consumption is comparable across algorithms.
    """
    def __init__(self, max_transitions: int = 400_000):
        self._trajs = {}                       # traj_id -> list[Experience]
        self._order = deque()                  # traj_id in insertion order
        self._total = 0
        self._cap   = int(max_transitions)

    def push(self, exp: Experience) -> None:
        tid = exp.traj_id
        if tid not in self._trajs:
            self._trajs[tid] = []
            self._order.append(tid)
        self._trajs[tid].append(exp)
        self._total += 1
        while self._total > self._cap and self._order:
            old = self._order.popleft()
            evicted = self._trajs.pop(old, None)
            if evicted is not None:
                self._total -= len(evicted)

    def size(self) -> int:
        return self._total

    def n_trajs(self) -> int:
        return len(self._trajs)

    def sample_chunks(self, n_chunks: int, seq_len: int):
        """
        Sample `n_chunks` length-`seq_len` windows.  Returns a dict of numpy
        arrays, (n, T, …), plus a (n, T) mask.  Drops chunks that would be
        entirely padding.
        """
        if not self._trajs:
            return None
        tids = list(self._trajs.keys())
        out_s, out_a, out_r, out_s2, out_d, out_m = [], [], [], [], [], []
        attempts = 0
        while len(out_s) < n_chunks and attempts < n_chunks * 8:
            attempts += 1
            tid = random.choice(tids)
            traj = self._trajs[tid]
            T = len(traj)
            if T < 2:
                continue
            start = random.randint(0, max(T - 2, 0))
            end   = min(start + seq_len, T)
            L     = end - start

            s   = np.zeros((seq_len, STATE_DIM),  dtype=np.float32)
            a   = np.zeros((seq_len, ACTION_DIM), dtype=np.float32)
            r   = np.zeros(seq_len, dtype=np.float32)
            s2  = np.zeros((seq_len, STATE_DIM),  dtype=np.float32)
            d   = np.zeros(seq_len, dtype=np.float32)
            m   = np.zeros(seq_len, dtype=np.float32)

            for t in range(L):
                e = traj[start + t]
                s [t] = e.state
                a [t, 0] = e.action
                r [t] = e.reward
                s2[t] = e.next_state
                d [t] = float(e.done)
                m [t] = 1.0

            out_s.append(s); out_a.append(a); out_r.append(r)
            out_s2.append(s2); out_d.append(d); out_m.append(m)

        if not out_s:
            return None
        return dict(
            state      = np.stack(out_s),
            action     = np.stack(out_a),
            reward     = np.stack(out_r),
            next_state = np.stack(out_s2),
            done       = np.stack(out_d),
            mask       = np.stack(out_m),
        )

    def snapshot_rewards(self, n: int = 10_000) -> np.ndarray:
        """Return up to n random recent rewards for RMS updates."""
        tids = list(self._trajs.keys())
        out  = []
        budget = n
        while budget > 0 and tids:
            tid = random.choice(tids)
            traj = self._trajs[tid]
            if not traj:
                tids.remove(tid); continue
            take = min(len(traj), budget)
            for e in random.sample(traj, take):
                out.append(e.reward)
            budget -= take
            if len(out) >= n:
                break
        return np.array(out, dtype=np.float32)


class DistillationMemory:
    """Lifetime reservoir of recurrent replay sequences used as anchors.

    FIFO replay adapts to recent traffic and eventually forgets old regimes.
    This much smaller reservoir samples uniformly across every sequence ever
    offered, so old recurrent contexts retain a bounded chance of surviving
    for the lifetime of the service.
    """

    def __init__(self, capacity: int = 2048):
        self.capacity = max(int(capacity), 0)
        self.seen = 0
        self._items = []

    def __len__(self) -> int:
        return len(self._items)

    def add_batch(self, batch: dict) -> None:
        if self.capacity <= 0 or not batch:
            return
        states = np.asarray(batch['state'], dtype=np.float32)
        actions = np.asarray(batch['action'], dtype=np.float32)
        masks = np.asarray(batch['mask'], dtype=np.float32)
        for index in range(states.shape[0]):
            if not np.any(masks[index] > 0.0):
                continue
            item = {
                'state': states[index].copy(),
                'action': actions[index].copy(),
                'mask': masks[index].copy(),
            }
            self.seen += 1
            if len(self._items) < self.capacity:
                self._items.append(item)
                continue
            slot = random.randrange(self.seen)
            if slot < self.capacity:
                self._items[slot] = item

    def sample(self, n_sequences: int):
        if not self._items or n_sequences <= 0:
            return None
        count = min(int(n_sequences), len(self._items))
        items = random.sample(self._items, count)
        return {
            key: np.stack([item[key] for item in items])
            for key in ('state', 'action', 'mask')
        }

    def to_payload(self) -> dict:
        return {
            'format': 1,
            'capacity': self.capacity,
            'seen': self.seen,
            'items': self._items,
        }

    def restore(self, payload: dict) -> None:
        if not isinstance(payload, dict) or payload.get('format') != 1:
            raise ValueError('unsupported distillation-memory format')
        restored = []
        for raw in payload.get('items') or []:
            if not isinstance(raw, dict):
                continue
            state = np.asarray(raw.get('state'), dtype=np.float32)
            action = np.asarray(raw.get('action'), dtype=np.float32)
            mask = np.asarray(raw.get('mask'), dtype=np.float32)
            if (state.ndim != 2 or state.shape[-1] != STATE_DIM
                    or action.ndim != 2 or action.shape[-1] != ACTION_DIM
                    or mask.ndim != 1 or mask.shape[0] != state.shape[0]
                    or action.shape[0] != state.shape[0]):
                continue
            restored.append({
                'state': state.copy(),
                'action': action.copy(),
                'mask': mask.copy(),
            })
            if len(restored) >= self.capacity:
                break
        self._items = restored
        self.seen = max(int(payload.get('seen', len(restored))), len(restored))


# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)
        self.batch_size            = int(t.get('batch_size',            256))
        self.seq_len               = int(t.get('seq_len',               32))
        self.min_replay            = int(t.get('min_replay',            5_000))
        self.replay_capacity       = int(t.get('replay_capacity',       400_000))
        self.gamma                 = float(t.get('gamma',               0.99))
        self.tau                   = float(t.get('target_tau',          0.005))
        self.lr_actor              = float(t.get('lr_actor',            3e-4))
        self.lr_critic             = float(t.get('lr_critic',           3e-4))
        self.lr_alpha              = float(t.get('lr_alpha',            3e-4))
        self.lr_critic_decay_after = int(t.get('lr_critic_decay_after_steps', 0))
        self.lr_critic_final       = float(t.get('lr_critic_final',     self.lr_critic))
        self.use_reward_normalizer = bool(t.get('use_reward_normalizer', True))
        # Automatic temperature tuning (SAC v2). Set autotune_alpha: false to
        # hold α fixed at `alpha_init`.
        self.autotune_alpha        = bool(t.get('autotune_alpha',       True))
        self.alpha_init            = float(t.get('alpha_init',          0.2))
        # Target entropy defaults to -ACTION_DIM (the SAC heuristic); override
        # via config to tune exploration pressure.
        target_entropy_cfg         = t.get('target_entropy', None)
        self.target_entropy        = (float(target_entropy_cfg)
                                      if target_entropy_cfg is not None
                                      else -float(ACTION_DIM))
        self.grad_clip             = float(t.get('grad_clip',           5.0))
        self.save_every            = int(t.get('save_every',            500))
        self.save_every_episodes   = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 50))
        self.updates_per_step      = int(t.get('updates_per_step',      1))
        continual                  = _continual_config(cfg)
        self.continual_enabled     = bool(continual.get('enabled', False))
        self.update_budget         = (UpdateBudget(
            continual.get('updates_per_transition', 0.25),
            continual.get('max_update_burst', 100),
        ) if self.continual_enabled else None)
        self.persist_replay        = bool(
            continual.get('persist_replay', self.continual_enabled))
        self.replay_save_every_s   = max(float(
            continual.get('replay_save_every_s', 300.0)), 0.0)
        distillation = dict(t.get('distillation') or {})
        distillation.update(dict(continual.get('distillation') or {}))
        self.distill_enabled       = bool(distillation.get('enabled', False))
        self.distill_actor_weight  = max(float(
            distillation.get('actor_weight', 0.05)), 0.0)
        self.distill_critic_weight = max(float(
            distillation.get('critic_weight', 0.05)), 0.0)
        self.distill_anchor_capacity = max(int(
            distillation.get('anchor_capacity', 2048)), 0)
        self.distill_anchor_min    = max(int(
            distillation.get('anchor_min_sequences', 64)), 1)
        self.distill_batch_sequences = max(int(
            distillation.get('batch_sequences', 8)), 1)
        self.distill_capture_every = max(int(
            distillation.get('capture_every_updates', 25)), 1)
        self.distill_teacher_update_every = max(int(
            distillation.get('teacher_update_every', 0)), 0)
        self.reward_signature      = _reward_signature(cfg)
        self.total_received        = 0
        ckpt_path                  = t.get('checkpoint',
                                           os.path.join(_PKG, 'data', 'checkpoints',
                                                        'sac_cwnd_model.pt'))
        resume_from                = t.get('resume_from', '')
        agent_cfg                  = cfg.get('agent', {}) or {}
        hidden                     = int(agent_cfg.get('hidden', 128))
        head_hidden_cfg            = agent_cfg.get('head_hidden')
        head_hidden                = (int(head_hidden_cfg)
                                      if head_hidden_cfg is not None else None)
        load_path                  = resume_from or ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(f'[learner] resume checkpoint not found: {resume_from}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        ckpt_to_load = None
        if load_path and os.path.exists(load_path):
            try:
                ckpt_to_load = torch.load(
                    load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_state_compatible(ckpt_to_load, source=load_path)
                ckpt_hidden, ckpt_head_hidden = actor_arch_from_checkpoint(
                    ckpt_to_load, hidden, head_hidden)
                current_head_hidden = head_hidden if head_hidden is not None else hidden // 2
                if (ckpt_hidden, ckpt_head_hidden) != (hidden, current_head_hidden):
                    print(f'[learner] using checkpoint actor config '
                          f'hidden={ckpt_hidden} head_hidden={ckpt_head_hidden}',
                          flush=True)
                hidden, head_hidden = ckpt_hidden, ckpt_head_hidden
            except Exception as e:
                print(f'[learner] ckpt load failed ({e}), fresh start', flush=True)
                ckpt_to_load = None

        self.actor           = Actor(STATE_DIM, hidden, head_hidden).to(self.device)
        self.critic          = TwinCritic(STATE_DIM, hidden, head_hidden).to(self.device)
        self.critic_target   = TwinCritic(STATE_DIM, hidden, head_hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters(): p.requires_grad_(False)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Optimise log α for positivity + stable gradients (SAC v2).
        self.log_alpha = torch.tensor(
            float(np.log(max(self.alpha_init, 1e-8))),
            device=self.device, requires_grad=self.autotune_alpha)
        self.opt_alpha = (optim.Adam([self.log_alpha], lr=self.lr_alpha)
                          if self.autotune_alpha else None)

        self.buf = build_mixed_replay(
            cfg, lambda cap: ReplayBuffer(max_transitions=cap),
            self.replay_capacity)
        self._mixed_enabled = self.buf is not None
        if not self._mixed_enabled:
            self.buf = ReplayBuffer(max_transitions=self.replay_capacity)
        self.step      = 0
        self.ckpt_path = ckpt_path
        default_replay_path = (
            os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                         'sac_replay.pkl') if ckpt_path else '')
        self.replay_path = str(
            continual.get('replay_path') or default_replay_path)
        self._last_replay_save = time.monotonic()
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)
        self._r_rms    = _RunningStats()
        self.distill_memory = DistillationMemory(
            capacity=self.distill_anchor_capacity)
        self.teacher_actor = None
        self.teacher_critic = None
        self.distill_teacher_step = 0

        restored_ckpt = None
        if ckpt_to_load is not None:
            try:
                ckpt = ckpt_to_load
                assert_checkpoint_state_compatible(ckpt, source=load_path)
                self.actor.load_state_dict(ckpt['actor'],  strict=False)
                self.critic.load_state_dict(ckpt['critic'], strict=False)
                self.critic_target.load_state_dict(
                    ckpt.get('critic_target', ckpt['critic']), strict=False)
                if 'opt_actor' in ckpt:
                    self.opt_actor.load_state_dict(ckpt['opt_actor'])
                if 'opt_critic' in ckpt:
                    self.opt_critic.load_state_dict(ckpt['opt_critic'])
                if 'log_alpha' in ckpt:
                    with torch.no_grad():
                        self.log_alpha.copy_(torch.as_tensor(
                            ckpt['log_alpha'], device=self.device))
                if self.opt_alpha is not None and 'opt_alpha' in ckpt:
                    self.opt_alpha.load_state_dict(ckpt['opt_alpha'])
                self.opt_actor.param_groups[0]['lr'] = self.lr_actor
                self.opt_critic.param_groups[0]['lr'] = self.lr_critic
                self.step = ckpt.get('step', 0)
                reward_stats = ckpt.get('reward_stats') or {}
                self._r_rms.mean = float(reward_stats.get(
                    'mean', self._r_rms.mean))
                self._r_rms.var = max(float(reward_stats.get(
                    'var', self._r_rms.var)), 0.0)
                self._r_rms.count = max(float(reward_stats.get(
                    'count', self._r_rms.count)), 1e-4)
                continual_state = ckpt.get('continual_state') or {}
                self.total_received = int(continual_state.get(
                    'total_received', 0))
                if self.update_budget is not None:
                    self.update_budget.credit = min(
                        max(float(continual_state.get('update_credit', 0.0)), 0.0),
                        float(self.update_budget.max_burst),
                    )
                src = 'resume checkpoint' if resume_from else 'checkpoint'
                print(f'[learner] loaded {src} step={self.step} path={load_path}', flush=True)
                restored_ckpt = ckpt
            except Exception as e:
                print(f'[learner] ckpt load failed ({e}), fresh start', flush=True)

        self._initialize_distillation(restored_ckpt)

        if self.persist_replay:
            self._restore_replay()

        log_path = t.get('log_path',
                         ckpt_path.replace('.pt', '_log.csv') if ckpt_path else '')
        self._csv_file, self._csv_writer = None, None
        self._csv_has_distillation = False
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            existed = os.path.exists(log_path)
            if existed:
                try:
                    with open(log_path, newline='') as existing_file:
                        existing_header = next(csv.reader(existing_file), [])
                    self._csv_has_distillation = (
                        'actor_distill_loss' in existing_header)
                except OSError:
                    pass
            else:
                self._csv_has_distillation = True
            self._csv_file   = open(log_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow([
                    'step','critic_loss','actor_loss','alpha','q1_mean','q2_mean',
                    'td_abs','entropy','logp_mean','a_mean','a_abs','r_std',
                    'buf','trajs','lr_a','lr_c',
                    'actor_distill_loss','critic_distill_loss',
                    'distill_anchors','distill_teacher_step',
                ])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[learner] manager on port {port}', flush=True)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @staticmethod
    def _freeze(module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _initialize_distillation(self, ckpt: dict = None) -> None:
        if not self.distill_enabled:
            return
        self.teacher_actor = copy.deepcopy(self.actor).to(self.device)
        self.teacher_critic = copy.deepcopy(self.critic).to(self.device)
        state = dict((ckpt or {}).get('distillation_state') or {})
        try:
            if state.get('teacher_actor'):
                self.teacher_actor.load_state_dict(
                    state['teacher_actor'], strict=True)
            if state.get('teacher_critic'):
                self.teacher_critic.load_state_dict(
                    state['teacher_critic'], strict=True)
            self.distill_teacher_step = int(
                state.get('teacher_step', self.step))
        except Exception as exc:
            print(f'[learner] distillation teacher restore failed ({exc}); '
                  'using current actor/critic', flush=True)
            self.teacher_actor.load_state_dict(self.actor.state_dict())
            self.teacher_critic.load_state_dict(self.critic.state_dict())
            self.distill_teacher_step = self.step
        self._freeze(self.teacher_actor)
        self._freeze(self.teacher_critic)
        print('[learner] distillation enabled '
              f'actor_weight={self.distill_actor_weight:g} '
              f'critic_weight={self.distill_critic_weight:g} '
              f'anchors={self.distill_anchor_capacity} '
              f'teacher_step={self.distill_teacher_step}', flush=True)

    def _refresh_distillation_teacher(self) -> None:
        if not self.distill_enabled:
            return
        self.teacher_actor.load_state_dict(self.actor.state_dict())
        self.teacher_critic.load_state_dict(self.critic.state_dict())
        self.distill_teacher_step = self.step
        self._freeze(self.teacher_actor)
        self._freeze(self.teacher_critic)
        print(f'[learner] refreshed distillation teacher step={self.step}',
              flush=True)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.actor.eval(); self.critic.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_hb = time.monotonic()
        total_drained = 0

        while True:
            drained = 0
            try:
                while True:
                    source, exp = _exp_queue.get_nowait()
                    if self._mixed_enabled:
                        self.buf.push(exp, source)
                    else:
                        self.buf.push(exp)
                    self._note_episode_done(exp)
                    drained += 1
            except Exception:
                pass
            self.total_received += drained
            if self.update_budget is not None:
                self.update_budget.add(drained)
            total_drained += drained

            now = time.monotonic()
            if now - last_hb >= 5.0:
                dt   = now - last_hb
                rate = total_drained / max(dt, 1e-6)
                mix  = (''.join(f' {k}={v}' for k, v in self.buf.sizes().items())
                        if self._mixed_enabled else '')
                print(f'[learner] hb buf={self.buf.size()}/{self.replay_capacity}{mix}  '
                      f'recv={total_drained} ({rate:.0f}/s)  trajs={self.buf.n_trajs()}',
                      flush=True)
                last_hb = now; total_drained = 0

            self._maybe_save_replay(now)

            _gate = ((not self.buf.ready(self.min_replay))
                     if self._mixed_enabled
                     else (self.buf.size() < self.min_replay))
            if _gate:
                if drained == 0:
                    time.sleep(0.05)
                continue

            if self.update_budget is None:
                update_count = max(self.updates_per_step, 0)
            else:
                update_count = 0
                while self.update_budget.take():
                    update_count += 1

            update_start_step = self.step
            for _ in range(update_count):
                if self.step % 50 == 0:
                    sample = self.buf.snapshot_rewards(n=8_000)
                    if sample.size:
                        self._r_rms.update(sample)
                prev_step = self.step
                self._sac_update()
                if self.step != prev_step:
                    self._maybe_save_latest_checkpoint()
            crossed_broadcast = (
                update_count
                and self.param_broadcast_every > 0
                and self.step // self.param_broadcast_every
                > update_start_step // self.param_broadcast_every)
            if crossed_broadcast:
                self._broadcast()
            if update_count == 0 and drained == 0:
                time.sleep(0.01)

    # ── SAC update ───────────────────────────────────────────────────────────

    def _sac_update(self):
        if (self.lr_critic_decay_after > 0
                and self.step >= self.lr_critic_decay_after
                and self.opt_critic.param_groups[0]['lr'] > self.lr_critic_final):
            for pg in self.opt_critic.param_groups:
                pg['lr'] = self.lr_critic_final
            print(f'[learner] critic LR → {self.lr_critic_final:g} '
                  f'at step={self.step}', flush=True)

        n_seqs = max(1, self.batch_size // self.seq_len)
        batch  = self.buf.sample_chunks(n_seqs, self.seq_len)
        if batch is None:
            return
        if (self.distill_enabled
                and self.step % self.distill_capture_every == 0):
            self.distill_memory.add_batch(batch)
        anchor = None
        if (self.distill_enabled
                and len(self.distill_memory) >= self.distill_anchor_min):
            anchor = self.distill_memory.sample(self.distill_batch_sequences)
        dev = self.device
        s   = torch.from_numpy(batch['state']).to(dev)
        a   = torch.from_numpy(batch['action']).to(dev).squeeze(-1)   # (B, T)
        r   = torch.from_numpy(batch['reward']).to(dev)
        s2  = torch.from_numpy(batch['next_state']).to(dev)
        d   = torch.from_numpy(batch['done']).to(dev)
        m   = torch.from_numpy(batch['mask']).to(dev)

        r_std = float(self._r_rms.std)
        r_scaled = r / r_std if self.use_reward_normalizer else r

        alpha = self.alpha
        self.actor.train(); self.critic.train()
        actor_distill = torch.zeros((), device=dev)
        critic_distill = torch.zeros((), device=dev)
        anchor_s = anchor_a = anchor_m = None
        if anchor is not None:
            anchor_s = torch.from_numpy(anchor['state']).to(dev)
            anchor_a = torch.from_numpy(anchor['action']).to(dev)
            anchor_m = torch.from_numpy(anchor['mask']).to(dev)

        # 1. Critic ------------------------------------------------------------
        c_info = critic_loss(
            self.critic, self.actor, self.critic_target,
            s_batch  = s,
            a_batch  = a,
            r_batch  = r_scaled,
            s2_batch = s2,
            d_batch  = d,
            mask     = m,
            alpha    = alpha,
            gamma    = self.gamma,
        )
        if not torch.isfinite(c_info.loss):
            print('[learner] non-finite critic loss — skip update', flush=True)
            self.actor.eval(); self.critic.eval()
            return

        if anchor_s is not None and self.distill_critic_weight > 0.0:
            critic_distill = critic_distillation_loss(
                self.critic, self.teacher_critic,
                anchor_s, anchor_a, anchor_m)
        critic_total = (
            c_info.loss + self.distill_critic_weight * critic_distill)

        self.opt_critic.zero_grad()
        critic_total.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()

        # 2. Actor (every step — SAC has no policy delay) ----------------------
        a_info, logp = actor_loss(self.actor, self.critic, s_batch=s, mask=m,
                                  alpha=alpha)
        if torch.isfinite(a_info.loss):
            if anchor_s is not None and self.distill_actor_weight > 0.0:
                actor_distill = actor_distillation_loss(
                    self.actor, self.teacher_actor, anchor_s, anchor_m)
            actor_total = (
                a_info.loss + self.distill_actor_weight * actor_distill)
            self.opt_actor.zero_grad()
            actor_total.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.opt_actor.step()
        else:
            print('[learner] non-finite actor loss — skip actor step', flush=True)

        # 3. Temperature α (SAC v2 automatic tuning) ---------------------------
        if self.autotune_alpha and torch.isfinite(a_info.loss):
            alpha_l = temperature_loss(
                self.log_alpha, logp, self.target_entropy, m)
            if torch.isfinite(alpha_l):
                self.opt_alpha.zero_grad()
                alpha_l.backward()
                self.opt_alpha.step()

        # 4. Target critics ----------------------------------------------------
        soft_update(self.critic, self.critic_target, self.tau)

        self.step += 1
        self.actor.eval(); self.critic.eval()
        if (self.distill_enabled
                and self.distill_teacher_update_every > 0
                and self.step % self.distill_teacher_update_every == 0
                and self.step != self.distill_teacher_step):
            self._refresh_distillation_teacher()

        if self.step % 10 == 0:
            lr_a = self.opt_actor.param_groups[0]['lr']
            lr_c = self.opt_critic.param_groups[0]['lr']
            print(
                f'[learner] step={self.step:6d}  '
                f'critic={c_info.loss.item():.4f}  actor={a_info.loss.item():.4f}  '
                f'alpha={alpha:.4f}  '
                f'(q1={c_info.q1_mean:.3f} q2={c_info.q2_mean:.3f} '
                f'|td|={c_info.td_abs:.3f} H={a_info.entropy:.3f} '
                f'a_mean={a_info.a_mean:.3f} a_abs={a_info.a_abs:.3f}) '
                f'distill_pi={actor_distill.item():.4f} '
                f'distill_q={critic_distill.item():.4f} '
                f'anchors={len(self.distill_memory)} '
                f'r_std={r_std:.3f}  buf={self.buf.size()}  '
                f'trajs={self.buf.n_trajs()}',
                flush=True)
            if self._csv_writer:
                self._csv_writer.writerow([
                    self.step,
                    f'{c_info.loss.item():.6f}',
                    f'{a_info.loss.item():.6f}',
                    f'{alpha:.6f}',
                    f'{c_info.q1_mean:.6f}', f'{c_info.q2_mean:.6f}',
                    f'{c_info.td_abs:.6f}',
                    f'{a_info.entropy:.6f}', f'{a_info.logp_mean:.6f}',
                    f'{a_info.a_mean:.6f}', f'{a_info.a_abs:.6f}',
                    f'{r_std:.6f}',
                    self.buf.size(), self.buf.n_trajs(),
                    f'{lr_a:.2e}', f'{lr_c:.2e}',
                ] + ([
                    f'{actor_distill.item():.6f}',
                    f'{critic_distill.item():.6f}',
                    len(self.distill_memory),
                    self.distill_teacher_step,
                ] if self._csv_has_distillation else []))

    # ── Broadcast / save ─────────────────────────────────────────────────────

    def _broadcast(self):
        buf = io.BytesIO()
        torch.save({
            'actor_state_dict': {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'step': self.step,
            'model_meta': actor_model_meta(self.actor),
            'state_meta': model_state_meta(),
        }, buf)
        payload = buf.getvalue()
        _set_latest_params(payload)

    def _episode_ckpt_path(self, episode_count: int) -> str:
        base, ext = os.path.splitext(self.ckpt_path)
        return f'{base}_ep{int(episode_count):06d}{ext or ".pt"}'

    def _maybe_save_latest_checkpoint(self) -> None:
        if not self.ckpt_path or self.save_every <= 0 or self.step <= 0:
            return
        if self.step % self.save_every != 0:
            return
        if self.step == getattr(self, '_last_latest_ckpt_step', -1):
            return
        self._save(reason=f'step={self.step} latest')
        self._last_latest_ckpt_step = self.step

    def _note_episode_done(self, exp) -> None:
        if self.save_every_episodes <= 0 or not bool(getattr(exp, 'done', False)):
            return
        episode = _episode_from_traj_id(getattr(exp, 'traj_id', ''))
        if episode < 0 or episode in self._completed_episode_ids:
            return
        self._completed_episode_ids.add(episode)
        completed = len(self._completed_episode_ids)
        while self._next_episode_ckpt and completed >= self._next_episode_ckpt:
            self._save(reason=f'episode={self._next_episode_ckpt} latest')
            path = self._episode_ckpt_path(self._next_episode_ckpt)
            self._save(path=path, reason=f'episode={self._next_episode_ckpt} snapshot')
            self._next_episode_ckpt += self.save_every_episodes

    def _checkpoint_payload(self) -> dict:
        distillation_state = {'enabled': bool(self.distill_enabled)}
        if self.distill_enabled:
            distillation_state.update({
                'teacher_actor': self.teacher_actor.state_dict(),
                'teacher_critic': self.teacher_critic.state_dict(),
                'teacher_step': self.distill_teacher_step,
                'anchor_size': len(self.distill_memory),
                'anchor_seen': self.distill_memory.seen,
                'actor_weight': self.distill_actor_weight,
                'critic_weight': self.distill_critic_weight,
            })
        return {
            'actor':         self.actor.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'opt_actor':     self.opt_actor.state_dict(),
            'opt_critic':    self.opt_critic.state_dict(),
            'log_alpha':     float(self.log_alpha.detach().cpu().item()),
            'opt_alpha':     self.opt_alpha.state_dict() if self.opt_alpha else None,
            'step':          self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'model_meta':    actor_model_meta(self.actor),
            'state_meta':    model_state_meta(),
            'reward_signature': self.reward_signature,
            'distillation_state': distillation_state,
            'reward_stats': {
                'mean': self._r_rms.mean,
                'var': self._r_rms.var,
                'count': self._r_rms.count,
            },
            'continual_state': {
                'total_received': self.total_received,
                'update_credit': (
                    self.update_budget.credit
                    if self.update_budget is not None else 0.0),
            },
        }

    def _restore_replay(self) -> None:
        if not self.replay_path or not os.path.exists(self.replay_path):
            return
        try:
            with open(self.replay_path, 'rb') as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or payload.get('format') != 1:
                raise ValueError('unsupported replay format')
            if (_canonical_reward_signature(payload.get('reward_signature')) !=
                    _canonical_reward_signature(self.reward_signature)):
                raise ValueError('reward configuration changed')
            if payload.get('state_meta') != model_state_meta():
                raise ValueError('state configuration changed')
            restored = payload.get('buffer')
            if not all(hasattr(restored, name) for name in (
                    'push', 'size', 'n_trajs', 'sample_chunks')):
                raise ValueError('replay payload has no compatible buffer')
            self.buf = restored
            self._mixed_enabled = bool(payload.get('mixed_enabled', False))
            if (getattr(self, 'distill_enabled', False)
                    and payload.get('distillation_memory')):
                self.distill_memory.restore(payload['distillation_memory'])
            print(f'[learner] restored replay size={self.buf.size()} '
                  f'anchors={len(getattr(self, "distill_memory", []))} '
                  f'path={self.replay_path}', flush=True)
        except Exception as exc:
            print(f'[learner] replay restore skipped ({exc}); fresh buffer',
                  flush=True)

    def _save_replay(self) -> None:
        if not self.persist_replay or not self.replay_path:
            return
        path = os.path.abspath(self.replay_path)
        tmp = f'{path}.tmp.{os.getpid()}'
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                'format': 1,
                'reward_signature': self.reward_signature,
                'state_meta': model_state_meta(),
                'mixed_enabled': self._mixed_enabled,
                'saved_at': time.time(),
                'buffer': self.buf,
                'distillation_memory': (
                    self.distill_memory.to_payload()
                    if getattr(self, 'distill_enabled', False) else None),
            }
            with open(tmp, 'wb') as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            self._last_replay_save = time.monotonic()
            print(f'[learner] saved replay size={self.buf.size()} path={path}',
                  flush=True)
        except Exception as exc:
            print(f'[learner] replay save failed: {exc}', flush=True)
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def _maybe_save_replay(self, now: float = None) -> None:
        if not self.persist_replay or self.replay_save_every_s <= 0.0:
            return
        now = time.monotonic() if now is None else float(now)
        if now - self._last_replay_save >= self.replay_save_every_s:
            self._save_replay()

    def _save(self, path: str = None, reason: str = 'ckpt'):
        path = path or self.ckpt_path
        if not path:
            return
        path = os.path.abspath(path)
        tmp = f'{path}.tmp.{os.getpid()}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            torch.save(self._checkpoint_payload(), tmp)
            os.replace(tmp, path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
        print(f'[learner] saved {reason} step={self.step} '
              f'episodes={len(self._completed_episode_ids)} path={path}', flush=True)

    def stop(self):
        self._save()
        self._save_replay()
        if self._csv_file:
            self._csv_file.close()
        self._mgr.shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--port',   type=int, default=6301)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    authkey = secrets.token_bytes(16)
    print(f'SAO_MANAGER_KEY={authkey.hex()}', flush=True)

    learner = Learner(cfg, port=args.port, authkey=authkey)

    def _sigterm(_signum, _frame):
        learner.stop()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)

    try:
        learner.run()
    except KeyboardInterrupt:
        learner.stop()
