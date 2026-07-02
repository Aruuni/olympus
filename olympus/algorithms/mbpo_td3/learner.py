"""
learner.py — MBPO on top of recurrent TD3.

Same skeleton as algorithms/orca_td3/learner.py: BaseManager IPC, per-trajectory
ReplayBuffer with chunk sampling, twin-Q TD3 update, Welford reward std, periodic
broadcast/save. The MBPO additions are confined to:

  - ForwardModelEnsemble + an Adam optimiser, trained periodically on the real
    replay buffer.
  - A second ReplayBuffer (`model_buf`) for synthetic transitions.
  - `_train_model()` — N epochs of NLL on real chunks.
  - `_generate_rollouts()` — branches H-step imagined trajectories from random
    real states using the current actor + ensemble; pushes them to model_buf.
  - `_td3_update()` now draws a mixed batch (real_ratio of chunks from real,
    rest from model) and runs the standard critic+actor update on the mix.

Usage (started by orchestrator):
  python olympus/algorithms/mbpo_td3/learner.py \
    --config olympus/config.yaml
"""

import argparse
import csv
import io
import multiprocessing
import os
import random
import secrets
import signal
import sys
import time
from collections import deque
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

from olympus.algorithms.mbpo_td3.model import (
    Actor, TwinCritic, ForwardModelEnsemble, Experience,
    STATE_DIM, ACTION_DIM, STATE_LOW_T, STATE_HIGH_T,
    actor_loss, assert_checkpoint_state_compatible, critic_loss,
    model_state_meta, soft_update,
)
from olympus.common.mixed_replay import MixedReplay, mixed_settings


# ── IPC queues ────────────────────────────────────────────────────────────────

_exp_queue   = multiprocessing.Queue(maxsize=200_000)
_param_queue = multiprocessing.Queue(maxsize=200)


def _push_exp(exp, source=None):
    try:    _exp_queue.put_nowait((source, exp))
    except: pass

def _push_exp_batch(exps, source=None):
    for exp in exps:
        try:    _exp_queue.put_nowait((source, exp))
        except: pass

def _pull_params():
    try:    return _param_queue.get_nowait()
    except: return None


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


# ── Replay buffer (copied from orca_td3, adds sample_states helper) ──────────

class ReplayBuffer:
    def __init__(self, max_transitions: int = 400_000):
        self._trajs = {}
        self._order = deque()
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

    def sample_transitions(self, n: int):
        """Flat random sample of (s, a, r, s') tuples — used for model training."""
        if not self._trajs:
            return None
        tids = list(self._trajs.keys())
        s_, a_, r_, s2_ = [], [], [], []
        attempts = 0
        while len(s_) < n and attempts < n * 8:
            attempts += 1
            tid = random.choice(tids)
            traj = self._trajs[tid]
            if not traj:
                continue
            e = random.choice(traj)
            s_.append(e.state); a_.append([e.action])
            r_.append(e.reward); s2_.append(e.next_state)
        if not s_:
            return None
        return (np.stack(s_).astype(np.float32),
                np.stack(a_).astype(np.float32),
                np.array(r_, dtype=np.float32),
                np.stack(s2_).astype(np.float32))

    def sample_states(self, n: int):
        """Random sample of states only — used as imagined-rollout seeds."""
        if not self._trajs:
            return None
        tids = list(self._trajs.keys())
        out = []
        attempts = 0
        while len(out) < n and attempts < n * 8:
            attempts += 1
            tid = random.choice(tids)
            traj = self._trajs[tid]
            if not traj:
                continue
            out.append(random.choice(traj).state)
        if not out:
            return None
        return np.stack(out).astype(np.float32)

    def snapshot_rewards(self, n: int = 10_000) -> np.ndarray:
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
        self.lr_actor              = float(t.get('lr_actor',            1e-4))
        self.lr_critic             = float(t.get('lr_critic',           1e-3))
        self.lr_critic_decay_after = int(t.get('lr_critic_decay_after_steps', 0))
        self.lr_critic_final       = float(t.get('lr_critic_final',     self.lr_critic))
        self.policy_delay          = int(t.get('policy_delay',          2))
        self.target_noise_std      = float(t.get('target_noise_std',    0.2))
        self.target_noise_clip     = float(t.get('target_noise_clip',   0.5))
        self.grad_clip             = float(t.get('grad_clip',           5.0))
        self.save_every            = int(t.get('save_every',            500))
        self.save_every_episodes   = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 50))
        self.updates_per_step      = int(t.get('updates_per_step',      1))

        # ── MBPO knobs ────────────────────────────────────────────────────────
        self.model_hidden          = int(t.get('model_hidden',          200))
        self.model_ensemble_size   = int(t.get('model_ensemble_size',   5))
        self.model_lr              = float(t.get('model_lr',            1e-3))
        self.model_batch_size      = int(t.get('model_batch_size',      256))
        self.model_train_every     = int(t.get('model_train_every',     250))
        self.model_epochs          = int(t.get('model_epochs',          5))
        self.rollout_horizon       = max(2, int(t.get('rollout_horizon', 3)))
        self.rollouts_per_cycle    = int(t.get('rollouts_per_cycle',    1000))
        self.rollout_every         = int(t.get('rollout_every',         250))
        self.model_buffer_capacity = int(t.get('model_buffer_capacity', 400_000))
        self.real_ratio            = float(t.get('real_ratio',          0.5))
        self.model_min_real        = int(t.get('model_min_real',        5_000))
        self.rollout_noise_std     = float(t.get('rollout_noise_std',   0.1))

        ckpt_path                  = t.get('checkpoint',
                                           os.path.join(_PKG, 'data', 'checkpoints',
                                                        'mbpo_td3_cwnd_model.pt'))
        resume_from                = t.get('resume_from', '')
        hidden                     = int(cfg.get('agent', {}).get('hidden', 128))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        self.actor           = Actor(STATE_DIM, hidden).to(self.device)
        self.actor_target    = Actor(STATE_DIM, hidden).to(self.device)
        self.critic          = TwinCritic(STATE_DIM, hidden).to(self.device)
        self.critic_target   = TwinCritic(STATE_DIM, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.actor_target.parameters():  p.requires_grad_(False)
        for p in self.critic_target.parameters(): p.requires_grad_(False)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # ── Forward model ensemble ────────────────────────────────────────────
        self.model_ensemble = ForwardModelEnsemble(
            n_models   = self.model_ensemble_size,
            state_dim  = STATE_DIM,
            action_dim = ACTION_DIM,
            hidden     = self.model_hidden,
        ).to(self.device)
        self.opt_model = optim.Adam(self.model_ensemble.parameters(), lr=self.model_lr)
        self._state_low_t  = STATE_LOW_T.to(self.device)
        self._state_high_t = STATE_HIGH_T.to(self.device)
        self._model_trained = False

        self._mixed_enabled, self._mix_frac, _mix_cap = mixed_settings(cfg)
        if self._mixed_enabled:
            _cap = _mix_cap or self.replay_capacity
            self.buf = MixedReplay(
                lambda: ReplayBuffer(max_transitions=_cap), self._mix_frac)
            print(f'[learner] mixed collection enabled '
                  f'emulation_fraction={self._mix_frac} '
                  f'buffer_capacity={_cap} (x2)', flush=True)
        else:
            self.buf   = ReplayBuffer(max_transitions=self.replay_capacity)
        self.model_buf = ReplayBuffer(max_transitions=self.model_buffer_capacity)
        self.step      = 0
        self.ckpt_path = ckpt_path
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)
        self._r_rms    = _RunningStats()

        load_path = resume_from or ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(f'[learner] resume checkpoint not found: {resume_from}')
        if load_path and os.path.exists(load_path):
            try:
                ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_state_compatible(ckpt, source=load_path)
                self.actor.load_state_dict(ckpt['actor'],  strict=False)
                self.critic.load_state_dict(ckpt['critic'], strict=False)
                self.actor_target.load_state_dict(ckpt.get('actor_target',  ckpt['actor']),  strict=False)
                self.critic_target.load_state_dict(ckpt.get('critic_target', ckpt['critic']), strict=False)
                if 'opt_actor' in ckpt:
                    self.opt_actor.load_state_dict(ckpt['opt_actor'])
                if 'opt_critic' in ckpt:
                    self.opt_critic.load_state_dict(ckpt['opt_critic'])
                if 'model_ensemble' in ckpt:
                    self.model_ensemble.load_state_dict(ckpt['model_ensemble'], strict=False)
                    self._model_trained = True
                if 'opt_model' in ckpt:
                    self.opt_model.load_state_dict(ckpt['opt_model'])
                self.opt_actor.param_groups[0]['lr'] = self.lr_actor
                self.opt_critic.param_groups[0]['lr'] = self.lr_critic
                self.opt_model.param_groups[0]['lr'] = self.model_lr
                self.step = ckpt.get('step', 0)
                src = 'resume checkpoint' if resume_from else 'checkpoint'
                print(f'[learner] loaded {src} step={self.step} path={load_path}', flush=True)
            except Exception as e:
                print(f'[learner] ckpt load failed ({e}), fresh start', flush=True)

        log_path = t.get('log_path',
                         ckpt_path.replace('.pt', '_log.csv') if ckpt_path else '')
        self._csv_file, self._csv_writer = None, None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            existed = os.path.exists(log_path)
            self._csv_file   = open(log_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow([
                    'step','critic_loss','actor_loss','q1_mean','q2_mean',
                    'td_abs','a_mean','a_abs','r_std','buf','trajs',
                    'model_buf','model_loss','model_mse','model_r_mse',
                    'lr_a','lr_c',
                ])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[learner] manager on port {port}  '
              f'(MBPO: ensemble={self.model_ensemble_size} hidden={self.model_hidden} '
              f'horizon={self.rollout_horizon} real_ratio={self.real_ratio:.2f})',
              flush=True)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.actor.eval(); self.critic.eval(); self.model_ensemble.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_hb = time.monotonic()
        total_drained = 0
        last_model_loss = float('nan')
        last_model_mse  = float('nan')
        last_model_rmse = float('nan')

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
            total_drained += drained

            now = time.monotonic()
            if now - last_hb >= 5.0:
                dt   = now - last_hb
                rate = total_drained / max(dt, 1e-6)
                mix  = (f' emu={self.buf.size_emu()} sim={self.buf.size_sim()}'
                        if self._mixed_enabled else '')
                print(f'[learner] hb buf={self.buf.size()}/{self.replay_capacity}{mix}  '
                      f'model_buf={self.model_buf.size()}/{self.model_buffer_capacity}  '
                      f'recv={total_drained} ({rate:.0f}/s)  trajs={self.buf.n_trajs()}',
                      flush=True)
                last_hb = now; total_drained = 0

            _gate = ((not self.buf.ready(self.min_replay))
                     if self._mixed_enabled
                     else (self.buf.size() < self.min_replay))
            if _gate:
                if drained == 0:
                    time.sleep(0.05)
                continue

            if self.step % 50 == 0:
                sample = self.buf.snapshot_rewards(n=8_000)
                if sample.size:
                    self._r_rms.update(sample)

            # ── MBPO: periodic model training + rollout generation ───────────
            if (self.buf.size() >= self.model_min_real
                    and self.step % self.model_train_every == 0):
                info = self._train_model()
                if info is not None:
                    last_model_loss = info['loss']
                    last_model_mse  = info['mse']
                    last_model_rmse = info['r_mse']
                    print(f'[learner] model trained step={self.step}  '
                          f'loss={info["loss"]:.4f}  mse={info["mse"]:.4f}  '
                          f'r_mse={info["r_mse"]:.4f}', flush=True)

            if (self._model_trained
                    and self.buf.size() >= self.model_min_real
                    and self.step % self.rollout_every == 0):
                pushed = self._generate_rollouts()
                if pushed:
                    print(f'[learner] imagined {pushed} transitions  '
                          f'model_buf={self.model_buf.size()}', flush=True)

            for _ in range(self.updates_per_step):
                prev_step = self.step
                self._td3_update(last_model_loss, last_model_mse, last_model_rmse)
                if self.step != prev_step:
                    self._maybe_save_latest_checkpoint()
            if self.step % self.param_broadcast_every == 0:
                self._broadcast()

    # ── Model training ───────────────────────────────────────────────────────

    def _train_model(self):
        self.model_ensemble.train()
        info_acc = {'loss': 0.0, 'mse': 0.0, 'r_mse': 0.0}
        n_done = 0
        for _ in range(self.model_epochs):
            sample = self.buf.sample_transitions(self.model_batch_size)
            if sample is None:
                break
            s_np, a_np, r_np, s2_np = sample
            s   = torch.from_numpy(s_np).to(self.device)
            a   = torch.from_numpy(a_np).to(self.device)
            r   = torch.from_numpy(r_np).to(self.device)
            s2  = torch.from_numpy(s2_np).to(self.device)
            delta_s = s2 - s

            info = self.model_ensemble.loss(s, a, delta_s, r)
            if not torch.isfinite(info.loss):
                continue
            self.opt_model.zero_grad()
            info.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_ensemble.parameters(),
                                           self.grad_clip)
            self.opt_model.step()
            info_acc['loss']  += float(info.loss.item())
            info_acc['mse']   += info.mse
            info_acc['r_mse'] += info.r_mse
            n_done += 1
        self.model_ensemble.eval()
        if n_done == 0:
            return None
        self._model_trained = True
        return {k: v / n_done for k, v in info_acc.items()}

    # ── Imagined rollout generation ──────────────────────────────────────────

    @torch.no_grad()
    def _generate_rollouts(self) -> int:
        starts = self.buf.sample_states(self.rollouts_per_cycle)
        if starts is None:
            return 0
        s = torch.from_numpy(starts).to(self.device)            # (N, STATE_DIM)
        n = s.shape[0]
        traj_states = [[] for _ in range(n)]   # list of (s, a, r, s2)

        for h in range(self.rollout_horizon):
            a, _ = self.actor.forward_sequence(s.unsqueeze(1))  # (N, 1, 1)
            a = a.squeeze(-1).squeeze(-1)                       # (N,)
            if self.rollout_noise_std > 0:
                noise = (torch.randn_like(a) * self.rollout_noise_std
                        ).clamp(-0.5, 0.5)
                a = (a + noise).clamp(model.ACTION_MIN, model.ACTION_MAX)

            delta_s, reward = self.model_ensemble.sample(s, a)
            s_next = (s + delta_s).clamp(self._state_low_t, self._state_high_t)

            s_np      = s.cpu().numpy()
            s_next_np = s_next.cpu().numpy()
            a_np      = a.cpu().numpy()
            r_np      = reward.cpu().numpy()
            for i in range(n):
                traj_states[i].append((s_np[i], float(a_np[i]),
                                       float(r_np[i]), s_next_np[i]))
            s = s_next

        n_pushed = 0
        for i, steps in enumerate(traj_states):
            tid = f'imag_{self.step}_{i}'
            for h, (s_, a_, r_, s2_) in enumerate(steps):
                self.model_buf.push(Experience(
                    state=s_, action=a_, reward=r_, next_state=s2_,
                    done=False, traj_id=tid, step_in_traj=h,
                ))
                n_pushed += 1
        return n_pushed

    # ── TD3 update on mixed batch ────────────────────────────────────────────

    def _td3_update(self, last_model_loss, last_model_mse, last_model_rmse):
        if (self.lr_critic_decay_after > 0
                and self.step >= self.lr_critic_decay_after
                and self.opt_critic.param_groups[0]['lr'] > self.lr_critic_final):
            for pg in self.opt_critic.param_groups:
                pg['lr'] = self.lr_critic_final
            print(f'[learner] critic LR → {self.lr_critic_final:g} '
                  f'at step={self.step}', flush=True)

        n_seqs_total = max(1, self.batch_size // self.seq_len)
        use_model = self._model_trained and self.model_buf.size() > 0
        if use_model:
            n_real  = max(1, int(round(n_seqs_total * self.real_ratio)))
            n_model = max(0, n_seqs_total - n_real)
        else:
            n_real, n_model = n_seqs_total, 0

        real_batch  = self.buf.sample_chunks(n_real,  self.seq_len)
        model_batch = (self.model_buf.sample_chunks(n_model, self.seq_len)
                       if n_model > 0 else None)
        if real_batch is None and model_batch is None:
            return
        if real_batch is None:
            batch = model_batch
        elif model_batch is None:
            batch = real_batch
        else:
            batch = {k: np.concatenate([real_batch[k], model_batch[k]], axis=0)
                     for k in real_batch}

        dev = self.device
        s   = torch.from_numpy(batch['state']).to(dev)
        a   = torch.from_numpy(batch['action']).to(dev).squeeze(-1)
        r   = torch.from_numpy(batch['reward']).to(dev)
        s2  = torch.from_numpy(batch['next_state']).to(dev)
        d   = torch.from_numpy(batch['done']).to(dev)
        m   = torch.from_numpy(batch['mask']).to(dev)

        r_std = float(self._r_rms.std)
        r_scaled = r / r_std

        self.actor.train(); self.critic.train()

        c_info = critic_loss(
            self.critic, self.actor_target, self.critic_target,
            s_batch  = s,
            a_batch  = a,
            r_batch  = r_scaled,
            s2_batch = s2,
            d_batch  = d,
            mask     = m,
            gamma    = self.gamma,
            target_noise_std  = self.target_noise_std,
            target_noise_clip = self.target_noise_clip,
        )
        if not torch.isfinite(c_info.loss):
            print('[learner] non-finite critic loss — skip update', flush=True)
            self.actor.eval(); self.critic.eval()
            return

        self.opt_critic.zero_grad()
        c_info.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()

        a_info = None
        self.step += 1
        if self.step % self.policy_delay == 0:
            a_info = actor_loss(self.actor, self.critic, s_batch=s, mask=m)
            if torch.isfinite(a_info.loss):
                self.opt_actor.zero_grad()
                a_info.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.opt_actor.step()
                soft_update(self.actor,  self.actor_target,  self.tau)
                soft_update(self.critic, self.critic_target, self.tau)
            else:
                a_info = None

        self.actor.eval(); self.critic.eval()

        if self.step % 10 == 0:
            lr_a = self.opt_actor.param_groups[0]['lr']
            lr_c = self.opt_critic.param_groups[0]['lr']
            a_loss = a_info.loss.item() if a_info is not None else float('nan')
            a_mean = a_info.a_mean       if a_info is not None else float('nan')
            a_abs  = a_info.a_abs        if a_info is not None else float('nan')
            print(
                f'[learner] step={self.step:6d}  '
                f'critic={c_info.loss.item():.4f}  actor={a_loss:.4f}  '
                f'(q1={c_info.q1_mean:.3f} q2={c_info.q2_mean:.3f} '
                f'|td|={c_info.td_abs:.3f} '
                f'a_mean={a_mean:.3f} a_abs={a_abs:.3f}) '
                f'r_std={r_std:.3f}  buf={self.buf.size()}  '
                f'mbuf={self.model_buf.size()}  trajs={self.buf.n_trajs()}',
                flush=True)
            if self._csv_writer:
                self._csv_writer.writerow([
                    self.step,
                    f'{c_info.loss.item():.6f}',
                    f'{a_loss:.6f}',
                    f'{c_info.q1_mean:.6f}', f'{c_info.q2_mean:.6f}',
                    f'{c_info.td_abs:.6f}',
                    f'{a_mean:.6f}', f'{a_abs:.6f}',
                    f'{r_std:.6f}',
                    self.buf.size(), self.buf.n_trajs(),
                    self.model_buf.size(),
                    f'{last_model_loss:.6f}',
                    f'{last_model_mse:.6f}',
                    f'{last_model_rmse:.6f}',
                    f'{lr_a:.2e}', f'{lr_c:.2e}',
                ])

    # ── Broadcast / save ─────────────────────────────────────────────────────

    def _broadcast(self):
        buf = io.BytesIO()
        torch.save({
            'actor_state_dict': {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'step': self.step,
            'state_meta': model_state_meta(),
        }, buf)
        payload = buf.getvalue()
        try:
            while True: _param_queue.get_nowait()
        except Exception:
            pass
        try: _param_queue.put_nowait(payload)
        except Exception: pass

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
        return {
            'actor':          self.actor.state_dict(),
            'actor_target':   self.actor_target.state_dict(),
            'critic':         self.critic.state_dict(),
            'critic_target':  self.critic_target.state_dict(),
            'opt_actor':      self.opt_actor.state_dict(),
            'opt_critic':     self.opt_critic.state_dict(),
            'model_ensemble': self.model_ensemble.state_dict(),
            'opt_model':      self.opt_model.state_dict(),
            'step':           self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'state_meta':     model_state_meta(),
        }

    def _save(self, path: str = None, reason: str = 'ckpt'):
        path = path or self.ckpt_path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self._checkpoint_payload(), path)
        print(f'[learner] saved {reason} step={self.step} '
              f'episodes={len(self._completed_episode_ids)} path={path}', flush=True)

    def stop(self):
        self._save()
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
