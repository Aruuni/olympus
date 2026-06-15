"""
learner.py — Central Recurrent-PPO learner for olympus.

Buffers per-flow rollouts pushed by workers, builds minibatches of fixed-length
subsequences, runs K epochs of clipped-objective PPO, then broadcasts the new
policy back. Strictly on-policy: once we have `rollout_steps` transitions the
update runs and the buffer is cleared.

Same IPC + checkpoint + per-run logging machinery as the off-policy learners
in this package, with PPO-specific additions:
  - `push_bootstrap` callable so workers can hand the learner a tail-state
    value estimate (used by GAE for truncated trajectories).
  - Welford running-std on returns so raw-reward magnitudes don't drown out
    the policy gradient through the value loss.

Usage (started by orchestrator):
  python olympus/algorithms/recurrent_ppo/learner.py \\
      --config olympus/config.yaml
"""

import argparse
import csv
import io
import multiprocessing
import os
import secrets
import signal
import sys
import time
from collections import defaultdict
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

from olympus.algorithms.recurrent_ppo.model import (
    RecurrentPPONet, Experience, compute_gae, ppo_loss,
    STATE_DIM, assert_checkpoint_state_compatible, model_state_meta,
)


# ── IPC queues ────────────────────────────────────────────────────────────────

_exp_queue       = multiprocessing.Queue(maxsize=200_000)
_bootstrap_queue = multiprocessing.Queue(maxsize=2_000)
_param_queue     = multiprocessing.Queue(maxsize=200)


def _push_exp(exp):
    try:    _exp_queue.put_nowait(exp)
    except: pass

def _push_exp_batch(exps):
    for exp in exps:
        try:    _exp_queue.put_nowait(exp)
        except: pass

def _push_bootstrap(traj_id, value):
    try:    _bootstrap_queue.put_nowait((traj_id, float(value)))
    except: pass

def _pull_params():
    try:    return _param_queue.get_nowait()
    except: return None


class _QueueManager(BaseManager): pass
_QueueManager.register('push_exp',       callable=_push_exp)
_QueueManager.register('push_exp_batch', callable=_push_exp_batch)
_QueueManager.register('push_bootstrap', callable=_push_bootstrap)
_QueueManager.register('pull_params',    callable=_pull_params)


def _manager_handle_error(self, c, msg):  # noqa: ARG001
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


def _episode_from_traj_id(traj_id) -> int:
    try:
        return int(str(traj_id).rsplit('_', 2)[-2])
    except (IndexError, TypeError, ValueError):
        return -1


# ── Running reward stats ──────────────────────────────────────────────────────

class _RunningStats:
    """Welford running mean/variance — same idiom as orca_td3 learner."""
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
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


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Per-trajectory store that's cleared after every PPO update."""
    def __init__(self):
        self._trajs = defaultdict(list)
        self._bootstrap = {}
        self._total = 0

    def push(self, exp):
        self._trajs[exp.traj_id].append(exp)
        self._total += 1

    def push_bootstrap(self, traj_id, value):
        self._bootstrap[traj_id] = float(value)

    def size(self):
        return self._total

    def n_trajs(self):
        return len(self._trajs)

    def extract(self):
        out = []
        for tid, traj in self._trajs.items():
            if len(traj) < 2:
                continue
            s   = np.stack([e.state for e in traj]).astype(np.float32)
            a   = np.array([e.action_raw for e in traj], dtype=np.float32)
            lp  = np.array([e.log_prob   for e in traj], dtype=np.float32)
            v   = np.array([e.value      for e in traj], dtype=np.float32)
            r   = np.array([e.reward     for e in traj], dtype=np.float32)
            d   = np.array([float(e.done) for e in traj], dtype=np.float32)
            last_v = self._bootstrap.get(tid, 0.0)
            if d[-1] > 0.5:
                last_v = 0.0
            out.append(dict(
                traj_id    = tid,
                state      = s,
                action_raw = a,
                log_prob   = lp,
                value      = v,
                reward     = r,
                done       = d,
                last_value = last_v,
            ))
        return out

    def clear(self):
        self._trajs.clear()
        self._bootstrap.clear()
        self._total = 0


# ── Minibatch builder ─────────────────────────────────────────────────────────

def _pad_sequences(trajs, seq_len):
    """Slice each trajectory into seq_len chunks; pad + mask the tail chunk."""
    chunks = []
    for tj in trajs:
        T = len(tj['state'])
        for i in range(0, T, seq_len):
            end = min(i + seq_len, T)
            L   = end - i
            chunk = dict(
                state      = np.zeros((seq_len, tj['state'].shape[1]), dtype=np.float32),
                action_raw = np.zeros(seq_len, dtype=np.float32),
                log_prob   = np.zeros(seq_len, dtype=np.float32),
                value      = np.zeros(seq_len, dtype=np.float32),
                adv        = np.zeros(seq_len, dtype=np.float32),
                ret        = np.zeros(seq_len, dtype=np.float32),
                mask       = np.zeros(seq_len, dtype=np.float32),
            )
            chunk['state'][:L]      = tj['state'][i:end]
            chunk['action_raw'][:L] = tj['action_raw'][i:end]
            chunk['log_prob'][:L]   = tj['log_prob'][i:end]
            chunk['value'][:L]      = tj['value'][i:end]
            chunk['adv'][:L]        = tj['adv'][i:end]
            chunk['ret'][:L]        = tj['ret'][i:end]
            chunk['mask'][:L]       = 1.0
            chunks.append(chunk)
    if not chunks:
        return None
    def _stack(k):
        return np.stack([c[k] for c in chunks])
    return dict(
        state      = torch.from_numpy(_stack('state')),
        action_raw = torch.from_numpy(_stack('action_raw')),
        log_prob   = torch.from_numpy(_stack('log_prob')),
        value      = torch.from_numpy(_stack('value')),
        adv        = torch.from_numpy(_stack('adv')),
        ret        = torch.from_numpy(_stack('ret')),
        mask       = torch.from_numpy(_stack('mask')),
    )


# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)
        a = cfg.get('agent',    {}) or {}

        self.rollout_steps         = int(t.get('rollout_steps',         20_000))
        self.seq_len               = int(t.get('seq_len',               64))
        self.n_epochs              = int(t.get('n_epochs',              4))
        self.minibatches_per_epoch = int(t.get('minibatches_per_epoch', 4))
        self.clip_eps              = float(t.get('clip_eps',            0.2))
        self.value_clip            = float(t.get('value_clip',          0.2))
        self.gamma                 = float(t.get('gamma',               0.99))
        self.lam                   = float(t.get('lam',                 0.95))
        self.c_value               = float(t.get('c_value',             0.5))
        self.c_entropy             = float(t.get('c_entropy',           0.01))
        self.lr                    = float(t.get('lr',                  3e-4))
        self.grad_clip             = float(t.get('grad_clip',           0.5))
        self.target_kl             = float(t.get('target_kl',           0.03))
        self.use_reward_normalizer = bool(t.get('use_reward_normalizer', True))
        self.save_every            = int(t.get('save_every',            5))
        self.save_every_episodes   = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 1))
        ckpt_path                  = t.get('checkpoint',
                                           os.path.join(_PKG, 'data', 'checkpoints',
                                                        'recurrent_ppo_cwnd_model.pt'))
        resume_from = t.get('resume_from', '')
        hidden      = int(a.get('hidden', 256))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        self.net = RecurrentPPONet(STATE_DIM, hidden).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)

        self.buf       = RolloutBuffer()
        self.step      = 0
        self.ckpt_path = ckpt_path
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)
        self._ret_rms  = _RunningStats()

        load_path = resume_from or ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(f'[learner] resume checkpoint not found: {resume_from}')
        if load_path and os.path.exists(load_path):
            try:
                ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_state_compatible(ckpt, source=load_path)
                self.net.load_state_dict(ckpt['model'], strict=False)
                if 'opt' in ckpt:
                    try: self.opt.load_state_dict(ckpt['opt'])
                    except Exception: pass
                self.opt.param_groups[0]['lr'] = self.lr
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
                    'step','loss','policy','value','entropy',
                    'kl','clip_frac','explained_var','ent_bits',
                    'r_std','n_trajs','n_trans','lr',
                ])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[learner] manager on port {port}  '
              f'(PPO: hidden={hidden} rollout_steps={self.rollout_steps} '
              f'seq_len={self.seq_len} epochs={self.n_epochs} '
              f'gamma={self.gamma} lam={self.lam})',
              flush=True)

    def run(self):
        self.net.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_hb = time.monotonic()
        total_drained = 0

        while True:
            drained = 0
            try:
                while True:
                    exp = _exp_queue.get_nowait()
                    self.buf.push(exp)
                    self._note_episode_done(exp)
                    drained += 1
            except Exception:
                pass
            total_drained += drained
            try:
                while True:
                    tid, v = _bootstrap_queue.get_nowait()
                    self.buf.push_bootstrap(tid, v)
            except Exception:
                pass

            now = time.monotonic()
            if now - last_hb >= 5.0:
                dt = now - last_hb
                rate = total_drained / max(dt, 1e-6)
                print(f'[learner] hb buf={self.buf.size()}/{self.rollout_steps}  '
                      f'recv={total_drained} ({rate:.0f}/s)  trajs={self.buf.n_trajs()}',
                      flush=True)
                last_hb = now
                total_drained = 0

            if self.buf.size() < self.rollout_steps:
                if drained == 0:
                    time.sleep(0.05)
                continue

            self._ppo_update()
            self.buf.clear()

            self.step += 1
            self._maybe_save_latest_checkpoint()
            if self.step % self.param_broadcast_every == 0:
                self._broadcast()

    def _ppo_update(self):
        raw_trajs = self.buf.extract()
        if not raw_trajs:
            return

        # Compute unscaled discounted returns for the running-reward std.
        all_returns = []
        for tj in raw_trajs:
            T = len(tj['reward'])
            ret = np.zeros(T, dtype=np.float32)
            g = tj['last_value']
            for t in reversed(range(T)):
                non_term = 1.0 - tj['done'][t]
                g = tj['reward'][t] + self.gamma * g * non_term
                ret[t] = g
            all_returns.append(ret)
        if self.use_reward_normalizer:
            self._ret_rms.update(np.concatenate(all_returns))
            r_std = float(self._ret_rms.std)
        else:
            r_std = 1.0

        trajs = []
        for tj in raw_trajs:
            r_scaled = tj['reward'] / r_std if self.use_reward_normalizer else tj['reward']
            adv, ret = compute_gae(
                r_scaled, tj['value'], tj['done'],
                tj['last_value'], gamma=self.gamma, lam=self.lam)
            trajs.append(dict(
                traj_id    = tj['traj_id'],
                state      = tj['state'],
                action_raw = tj['action_raw'],
                log_prob   = tj['log_prob'],
                value      = tj['value'],
                adv        = adv,
                ret        = ret,
            ))

        batch = _pad_sequences(trajs, self.seq_len)
        if batch is None:
            return
        batch = {k: v.to(self.device) for k, v in batch.items()}
        N = batch['state'].shape[0]

        self.net.train()
        n_mb = max(1, self.minibatches_per_epoch)
        mb_sz = max(1, N // n_mb)

        kl_running = 0.0
        kl_batches = 0
        early_stop = False
        last_info  = None
        n_total_mb = 0

        for epoch in range(self.n_epochs):
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, mb_sz):
                idx = perm[i:i + mb_sz]
                mb  = {k: v[idx] for k, v in batch.items()}

                info = ppo_loss(
                    self.net,
                    s_batch        = mb['state'],
                    a_raw_batch    = mb['action_raw'],
                    old_logp_batch = mb['log_prob'],
                    adv_batch      = mb['adv'],
                    ret_batch      = mb['ret'],
                    old_value_batch= mb['value'],
                    mask           = mb['mask'],
                    clip_eps       = self.clip_eps,
                    c_value        = self.c_value,
                    c_entropy      = self.c_entropy,
                    value_clip     = self.value_clip,
                )
                if not torch.isfinite(info.loss):
                    print('[learner] non-finite loss, skipping minibatch', flush=True)
                    continue
                self.opt.zero_grad()
                info.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

                kl_running += info.kl
                kl_batches += 1
                n_total_mb += 1
                last_info = info

            mean_kl = kl_running / max(kl_batches, 1)
            if mean_kl > self.target_kl * 1.5:
                print(f'[learner] early-stop epoch {epoch+1}/{self.n_epochs} '
                      f'(kl={mean_kl:.4f} > 1.5×target)', flush=True)
                early_stop = True
                break

        self.net.eval()

        if last_info is not None:
            n_trajs = len(trajs)
            n_trans = sum(len(t['state']) for t in trajs)
            lr = self.opt.param_groups[0]['lr']
            print(
                f'[learner] step={self.step:5d}  loss={last_info.loss.item():.4f}  '
                f'pol={last_info.policy:.4f} val={last_info.value:.4f} '
                f'ent={last_info.entropy:.4f} kl={last_info.kl:.4f} '
                f'clip={last_info.clip_frac:.3f} evar={last_info.explained_var:.3f}  '
                f'r_std={r_std:.3f}  '
                f'trajs={n_trajs} trans={n_trans} mb={n_total_mb}'
                f'{" [KL-stop]" if early_stop else ""}',
                flush=True)
            if self._csv_writer:
                self._csv_writer.writerow([
                    self.step,
                    f'{last_info.loss.item():.6f}',
                    f'{last_info.policy:.6f}', f'{last_info.value:.6f}',
                    f'{last_info.entropy:.6f}', f'{last_info.kl:.6f}',
                    f'{last_info.clip_frac:.6f}',
                    f'{last_info.explained_var:.6f}',
                    f'{last_info.approx_ent_bits:.6f}',
                    f'{r_std:.6f}',
                    n_trajs, n_trans, f'{lr:.2e}',
                ])

    def _broadcast(self):
        buf = io.BytesIO()
        torch.save({'state_dict': {k: v.cpu() for k, v in self.net.state_dict().items()},
                    'step': self.step,
                    'state_meta': model_state_meta()}, buf)
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
            self._save(path=self._episode_ckpt_path(self._next_episode_ckpt),
                       reason=f'episode={self._next_episode_ckpt} snapshot')
            self._next_episode_ckpt += self.save_every_episodes

    def _save(self, path: str = None, reason: str = 'ckpt'):
        path = path or self.ckpt_path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'model': self.net.state_dict(),
            'opt':   self.opt.state_dict(),
            'step':  self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'state_meta': model_state_meta(),
        }, path)
        print(f'[learner] saved {reason} step={self.step} path={path}', flush=True)

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
