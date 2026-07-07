"""
learner.py — recurrent TD3 learner for olympus.

Off-policy: workers push one `Experience(s, a_pre, r, s', done, …)` per
control step into a replay buffer.  The learner samples random length-L
chunks from per-trajectory lists, zero-inits the LSTM hidden state, and
runs the canonical TD3 update:

  1. Critic: clipped double-Q target with target-policy smoothing.
     y = r + γ · min_i Q_i^target(s', μ^target(s') + ε_clip)
     L = Σ_i (Q_i(s, a) - y)²
  2. Actor: deterministic PG on Q1 only, applied every `policy_delay`
     critic updates.
  3. Target nets: Polyak-averaged with τ every step.

A running-reward std is maintained (same trick as the PPO learner) so the
critic target scale stays O(1) instead of scaling with raw reward magnitude.

Usage (started by orchestrator):
  python olympus/algorithms/td3/learner.py \
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

from olympus.algorithms.td3.model import (
    Actor, TwinCritic, Experience,
    STATE_DIM, ACTION_DIM,
    actor_arch_from_checkpoint, actor_loss, actor_model_meta,
    assert_checkpoint_state_compatible, critic_loss, model_state_meta, soft_update,
)
from olympus.common.mixed_replay import build_mixed_replay


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
    """Welford running mean/variance — shared idiom with the PPO learner."""
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

    Staleness is controlled by `max_trajs` — the N newest trajectories are
    kept and older ones are evicted in FIFO order.  Very large replay
    capacities aren't useful for this setup because the environment
    distribution is non-stationary (link schedules change per-episode).
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
        # Evict oldest trajectories until under cap.
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
        self.use_reward_normalizer = bool(t.get('use_reward_normalizer', True))
        self.policy_delay          = int(t.get('policy_delay',          2))
        self.target_noise_std      = float(t.get('target_noise_std',    0.2))
        self.target_noise_clip     = float(t.get('target_noise_clip',   0.5))
        self.grad_clip             = float(t.get('grad_clip',           5.0))
        self.save_every            = int(t.get('save_every',            500))
        self.save_every_episodes   = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 50))
        self.updates_per_step      = int(t.get('updates_per_step',      1))
        ckpt_path                  = t.get('checkpoint',
                                           os.path.join(_PKG, 'data', 'checkpoints',
                                                        'td3_cwnd_model.pt'))
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
        self.actor_target    = Actor(STATE_DIM, hidden, head_hidden).to(self.device)
        self.critic          = TwinCritic(STATE_DIM, hidden, head_hidden).to(self.device)
        self.critic_target   = TwinCritic(STATE_DIM, hidden, head_hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.actor_target.parameters():  p.requires_grad_(False)
        for p in self.critic_target.parameters(): p.requires_grad_(False)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.buf = build_mixed_replay(
            cfg, lambda cap: ReplayBuffer(max_transitions=cap),
            self.replay_capacity)
        self._mixed_enabled = self.buf is not None
        if not self._mixed_enabled:
            self.buf = ReplayBuffer(max_transitions=self.replay_capacity)
        self.step      = 0
        self.ckpt_path = ckpt_path
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)
        self._r_rms    = _RunningStats()

        if ckpt_to_load is not None:
            try:
                ckpt = ckpt_to_load
                assert_checkpoint_state_compatible(ckpt, source=load_path)
                self.actor.load_state_dict(ckpt['actor'],  strict=False)
                self.critic.load_state_dict(ckpt['critic'], strict=False)
                self.actor_target.load_state_dict(ckpt.get('actor_target',  ckpt['actor']),  strict=False)
                self.critic_target.load_state_dict(ckpt.get('critic_target', ckpt['critic']), strict=False)
                if 'opt_actor' in ckpt:
                    self.opt_actor.load_state_dict(ckpt['opt_actor'])
                if 'opt_critic' in ckpt:
                    self.opt_critic.load_state_dict(ckpt['opt_critic'])
                self.opt_actor.param_groups[0]['lr'] = self.lr_actor
                self.opt_critic.param_groups[0]['lr'] = self.lr_critic
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
                    'td_abs','a_mean','a_abs','r_std','buf','trajs','lr_a','lr_c',
                ])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[learner] manager on port {port}', flush=True)

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

            _gate = ((not self.buf.ready(self.min_replay))
                     if self._mixed_enabled
                     else (self.buf.size() < self.min_replay))
            if _gate:
                if drained == 0:
                    time.sleep(0.05)
                continue

            # Refresh reward-std from a random sample of replay once per N steps.
            if self.step % 50 == 0:
                sample = self.buf.snapshot_rewards(n=8_000)
                if sample.size:
                    self._r_rms.update(sample)

            for _ in range(self.updates_per_step):
                prev_step = self.step
                self._td3_update()
                if self.step != prev_step:
                    self._maybe_save_latest_checkpoint()
            if self.step % self.param_broadcast_every == 0:
                self._broadcast()

    # ── TD3 update ───────────────────────────────────────────────────────────

    def _td3_update(self):
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
        dev = self.device
        s   = torch.from_numpy(batch['state']).to(dev)
        a   = torch.from_numpy(batch['action']).to(dev).squeeze(-1)   # (B, T)
        r   = torch.from_numpy(batch['reward']).to(dev)
        s2  = torch.from_numpy(batch['next_state']).to(dev)
        d   = torch.from_numpy(batch['done']).to(dev)
        m   = torch.from_numpy(batch['mask']).to(dev)

        # Standardise rewards so critic targets stay O(1) — disabled by config
        # if `use_reward_normalizer: false`. r_std is still computed/logged so
        # diagnostics stay comparable across runs.
        r_std = float(self._r_rms.std)
        r_scaled = r / r_std if self.use_reward_normalizer else r

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

        # Logging every 10 critic updates
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
                f'trajs={self.buf.n_trajs()}',
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
                    f'{lr_a:.2e}', f'{lr_c:.2e}',
                ])

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
        return {
            'actor':         self.actor.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'opt_actor':     self.opt_actor.state_dict(),
            'opt_critic':    self.opt_critic.state_dict(),
            'step':          self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'model_meta':    actor_model_meta(self.actor),
            'state_meta':    model_state_meta(),
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
