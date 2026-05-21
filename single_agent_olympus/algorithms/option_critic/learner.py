"""
learner.py — Central Option-Critic learner for single_agent_olympus.

Receives experiences from per-flow workers via multiprocessing Manager,
trains the network, and broadcasts updated weights back to workers.

Usage (started automatically by orchestrator):
  python single_agent_olympus/algorithms/option_critic/learner.py \
    --config single_agent_olympus/config.yaml
"""

import argparse
import collections
import csv
import io
import json
import multiprocessing
import os
import queue
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

from single_agent_olympus.algorithms.option_critic.model import (
    OptionCriticNet,
    Experience,
    assert_checkpoint_state_compatible,
    compute_actor_loss,
    compute_critic_loss,
    model_state_meta,
    option_critic_kwargs_from_agent_cfg,
    option_critic_meta_from_checkpoint,
    soft_update,
    STATE_DIM,
    N_OPTIONS,
)


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_ROOT, path))


def _episode_from_traj_id(traj_id) -> int:
    try:
        return int(str(traj_id).rsplit('_', 2)[-2])
    except (IndexError, TypeError, ValueError):
        return -1


# ── Manager IPC (module-level so the forked server shares the same fds) ───────

_exp_queue   = multiprocessing.Queue(maxsize=50_000)
_PARAM_CACHE_BYTES = int(os.environ.get('SAO_PARAM_CACHE_BYTES', str(64 * 1024 * 1024)))
_param_lock  = multiprocessing.Lock()
_param_size  = multiprocessing.Value('i', 0)
_param_blob  = multiprocessing.RawArray('B', _PARAM_CACHE_BYTES)

def _push_exp(exp):
    try:    _exp_queue.put_nowait(exp)
    except: pass

def _push_exp_batch(exps):
    for exp in exps:
        try:    _exp_queue.put_nowait(exp)
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
_QueueManager.register('push_exp',    callable=_push_exp)
_QueueManager.register('push_exp_batch', callable=_push_exp_batch)
_QueueManager.register('pull_params', callable=_pull_params)

# Suppress BrokenPipeError / ConnectionResetError that the Manager server
# emits when a worker process dies mid-request.  These are expected — workers
# can be killed at any time — and should not pollute the learner log.
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

# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Trajectory-aware replay buffer with reservoir sampling eviction.

    Experiences are stored per trajectory (traj_id = unique per flow per episode).
    sample_sequences() only samples contiguous windows within a single trajectory
    so the GRU always sees coherent temporal context from one flow.

    Eviction uses reservoir sampling: when over capacity, a random existing
    trajectory is evicted rather than always the oldest.  This prevents bursts
    of hard-config episodes from flooding the buffer and evicting all good data
    — which caused periodic return crashes when the episode pool shuffled hard
    configs to the front.
    """
    def __init__(self, capacity: int = 100_000):
        self._capacity    = capacity
        self._trajs       = {}          # traj_id → list of Experience
        self._traj_keys   = []          # list of keys for O(1) random eviction
        self._opt_counts  = {}          # traj_id → {option_idx: count} — cached so
                                        # sample_sequences() doesn't rescan the buffer
        self._total       = 0

    def push(self, exp):
        tid = exp.traj_id
        if tid not in self._trajs:
            self._trajs[tid] = []
            self._traj_keys.append(tid)
            self._opt_counts[tid] = {}
        self._trajs[tid].append(exp)
        oc = self._opt_counts[tid]
        oc[exp.option_idx] = oc.get(exp.option_idx, 0) + 1
        self._total += 1
        # Reservoir eviction: evict a random trajectory (not always the oldest)
        # so no burst of new data can fully displace existing good trajectories.
        while self._total > self._capacity and self._traj_keys:
            idx = random.randrange(len(self._traj_keys))
            old = self._traj_keys[idx]
            # Skip the trajectory we just pushed to (don't immediately evict it)
            if old == tid and len(self._traj_keys) > 1:
                idx = (idx + 1) % len(self._traj_keys)
                old = self._traj_keys[idx]
            self._traj_keys[idx] = self._traj_keys[-1]
            self._traj_keys.pop()
            if old in self._trajs:
                self._total -= len(self._trajs.pop(old))
            self._opt_counts.pop(old, None)

    def sample_sequences(self, n_seqs: int, seq_len: int):
        """
        Sample n_seqs contiguous windows of seq_len steps, each from a single
        trajectory. Only trajectories with >= seq_len steps are eligible.

        Balanced per-option sampling: trajectories are grouped by their dominant
        option (most-used option across the trajectory) and the n_seqs slots are
        distributed round-robin across option buckets.  This prevents a dominant
        option from monopolising the batch and starving options with fewer
        trajectories of gradient signal.

        Returns a flat list of length n_seqs * seq_len.
        """
        eligible = [tid for tid in self._traj_keys if len(self._trajs[tid]) >= seq_len]
        if not eligible:
            all_exp = [e for t in self._trajs.values() for e in t]
            return (all_exp * ((n_seqs * seq_len) // max(len(all_exp), 1) + 1))[:n_seqs * seq_len]

        # Group trajectories by dominant option, using the cached counts so
        # this scales with n_trajectories, not total buffer size.
        buckets: dict = {}
        for tid in eligible:
            counts = self._opt_counts.get(tid)
            if not counts:
                continue
            dom = max(counts, key=counts.get)
            buckets.setdefault(dom, []).append(tid)

        bucket_list = [v for v in buckets.values() if v]
        if not bucket_list:
            bucket_list = [eligible]

        out = []
        for i in range(n_seqs):
            bucket = bucket_list[i % len(bucket_list)]
            tid    = random.choice(bucket)
            traj   = self._trajs[tid]
            start  = random.randint(0, len(traj) - seq_len)
            out.extend(traj[start:start + seq_len])
        return out

    def sample(self, n):
        """Random single-step sample across all trajectories."""
        all_exp = [e for t in self._trajs.values() for e in t]
        return random.sample(all_exp, min(n, len(all_exp)))

    def __len__(self):
        return self._total

# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)

        self.batch_size            = int(t.get('batch_size',            64))
        self.min_replay            = int(t.get('min_replay',            128))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 50))
        self.save_every            = int(t.get('save_every',            100))
        self.save_every_episodes   = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.actor_batch_size      = int(t.get('actor_batch_size',      64))
        self.actor_recent_window   = int(t.get('actor_recent_window',   4096))
        self.actor_warmup_steps    = int(t.get('actor_warmup_steps',    0))
        self.actor_update_every    = max(1, int(t.get('actor_update_every', 1)))
        self.reject_collapsed_checkpoint = bool(t.get('reject_collapsed_checkpoint', False))
        self.gamma                 = float(t.get('gamma',               0.99))
        self.target_tau            = float(t.get('target_tau',          0.005))
        self.c_value               = float(t.get('c_value',             1.0))
        self.c_term                = float(t.get('c_term',              1.0))
        self.c_action              = float(t.get('c_action',            1.0))
        self.c_entropy             = float(t.get('c_entropy',           0.01))
        self.c_action_center       = float(t.get('c_action_center',     0.0))
        self.c_action_smooth       = float(t.get('c_action_smooth',     0.0))
        configured_c_bdp_action   = float(t.get('c_bdp_action',        0.0))
        if abs(configured_c_bdp_action) > 1e-12:
            print('[learner] ignoring c_bdp_action because the state no longer '
                  'contains a precomputed BDP observation', flush=True)
        self.c_bdp_action          = 0.0
        self.bdp_action_gain       = float(t.get('bdp_action_gain',     0.15))
        self.c_option_usage        = float(t.get('c_option_usage',      0.0))
        self.c_action_div          = float(t.get('c_action_div',        0.0))
        self.option_kl_target      = float(t.get('option_kl_target',    0.25))
        self.option_usage_temp     = float(t.get('option_usage_temp',   1.0))
        self.c_beta_target         = float(t.get('c_beta_target',       0.0))
        self.beta_target           = float(t.get('beta_target',         0.20))
        self.advantage_norm        = bool(t.get('advantage_norm',       True))
        self.balanced_actor_options = bool(t.get('balanced_actor_options', True))
        self.critic_loss           = str(t.get('critic_loss',           'huber'))
        self.critic_huber_delta    = float(t.get('critic_huber_delta',  10.0))
        self.term_margin           = float(t.get('term_margin',         0.01))
        self.action_raw_clip       = float(t.get('action_raw_clip',      2.0))
        self.policy_logp_clip      = float(t.get('policy_logp_clip',     6.0))
        self.seq_len               = int(t.get('seq_len',               16))
        ckpt_path                  = _resolve_repo_path(
            t.get('checkpoint', os.path.join(_PKG, 'data', 'checkpoints',
                                             'option_critic_cwnd_model.pt')))
        resume_from_raw           = t.get('resume_from', '')
        resume_from               = (_resolve_repo_path(str(resume_from_raw))
                                     if resume_from_raw else '')

        agent_cfg = cfg.get('agent', {})
        self.model_kwargs = option_critic_kwargs_from_agent_cfg(agent_cfg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        self.net        = OptionCriticNet(STATE_DIM, **self.model_kwargs).to(self.device)
        self.target_net = OptionCriticNet(STATE_DIM, **self.model_kwargs).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        if not self.net.is_recurrent and self.seq_len != 1:
            print(f'[learner] arch={self.net.arch} ignores seq_len={self.seq_len}; '
                  'using flat replay batches for the critic', flush=True)

        lr = float(t.get('lr', 1e-4))
        self.lr = lr
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # These regularizers were useful as experiments, but they are not part
        # of canonical Option-Critic.  Keep the config keys for compatibility,
        # and ignore them when running in this more reference-like mode.
        ignored_terms = {
            'c_logsig':     float(t.get('c_logsig',     0.0)),
            'c_mu_repel':   float(t.get('c_mu_repel',   0.0)),
            'c_disc':       float(t.get('c_disc',       0.0)),
            'c_intrinsic':  float(t.get('c_intrinsic',  0.0)),
        }
        active_ignored = {k: v for k, v in ignored_terms.items() if abs(v) > 1e-12}
        if active_ignored:
            joined = ', '.join(f'{k}={v:g}' for k, v in active_ignored.items())
            print(f'[learner] reference-mode ignoring extra regularizers: {joined}', flush=True)

        self.buf       = ReplayBuffer(int(t.get('replay_capacity', 100_000)))
        self.actor_recent = deque(maxlen=self.actor_recent_window)
        self.step      = 0
        self.ckpt_path = ckpt_path
        self.loaded_checkpoint = False
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes

        # ── State envelope reservoir ──────────────────────────────────────────
        # Reservoir sample of normalized training states so {ckpt}.envelope.json
        # can be rewritten periodically with per-dim (p1, p99) bounds.  Used by
        # test_random_conditions to flag inference OOD steps.
        self._env_reservoir      = []
        self._env_reservoir_size = int(t.get('envelope_reservoir', 10_000))
        self._env_seen           = 0
        self._last_collapse_warn_step = -10_000

        # Load checkpoint if present. `resume_from` loads source weights while
        # saving future checkpoints into this run's ckpt_path.
        load_path = resume_from or ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(f'[learner] resume checkpoint not found: {resume_from}')
        if load_path and os.path.exists(load_path):
            try:
                ckpt = torch.load(load_path, map_location='cpu', weights_only=False)
                assert_checkpoint_state_compatible(ckpt, source=load_path)
                ckpt_meta = option_critic_meta_from_checkpoint(ckpt)
                cur_meta = self.net.model_meta()
                mismatches = []
                for key in (
                    'arch', 'hidden', 'n_options', 'action_dist', 'beta_floor',
                    'mult_min', 'mult_max', 'action_init', 'action_init_scale',
                    'q_value_clip', 'state_features', 'state_dim',
                ):
                    if key in ckpt_meta and ckpt_meta[key] != cur_meta[key]:
                        mismatches.append(f'{key}={ckpt_meta[key]} (ckpt) vs {cur_meta[key]} (config)')
                if mismatches:
                    joined = ', '.join(mismatches)
                    print(f'[learner] checkpoint model_meta mismatch ({joined}); '
                          'starting fresh for the configured architecture', flush=True)
                else:
                # strict=False lets checkpoints from previous architecture versions
                # load cleanly when heads are added/removed (e.g. option_head removed
                # to match canonical OC).  Missing heads stay at their fresh init.
                    self.net.load_state_dict(ckpt['model'], strict=False)
                    self.target_net.load_state_dict(ckpt.get('target', ckpt['model']), strict=False)
                    if 'optimizer' in ckpt:
                        self.opt.load_state_dict(ckpt['optimizer'])
                    # Reset LR to config value — ignore any decayed LR in old checkpoints.
                    self.opt.param_groups[0]['lr'] = self.lr
                    self.step = ckpt.get('step', 0)
                    self.loaded_checkpoint = True
                    src = 'resume checkpoint' if resume_from else 'checkpoint'
                    print(f'[learner] loaded {src} step={self.step} path={load_path}', flush=True)
                    if (self.reject_collapsed_checkpoint
                            and self._checkpoint_action_collapsed()):
                        print('[learner] checkpoint action head is collapsed; '
                              'starting fresh', flush=True)
                        self.net = OptionCriticNet(STATE_DIM, **self.model_kwargs).to(self.device)
                        self.target_net = OptionCriticNet(STATE_DIM, **self.model_kwargs).to(self.device)
                        self.target_net.load_state_dict(self.net.state_dict())
                        self.target_net.eval()
                        self.opt = optim.Adam(self.net.parameters(), lr=self.lr)
                        self.step = 0
                        self.loaded_checkpoint = False
            except Exception as e:
                print(f'[learner] checkpoint load failed ({e}), fresh start', flush=True)

        # CSV log
        log_path = t.get('log_path',
                         ckpt_path.replace('.pt', '_log.csv') if ckpt_path else '')
        self._csv_file, self._csv_writer = None, None
        log_header = [
            'step','loss','loss_value',
            'loss_action','loss_term','entropy',
            'mean_beta','td_target_mean','td_target_abs',
            'q_chosen_mean','q_chosen_abs',
            'loss_usage','loss_action_div','loss_beta_target',
            'loss_action_center','loss_action_smooth','loss_action_aux','q_gap',
            'option_mass','beta_by_option','mu_by_option','q_by_option',
            'lr','buf_size','actor_batch','actor_applied',
        ]
        if log_path:
            base, ext = os.path.splitext(log_path)
            suffix = 1
            while os.path.exists(log_path):
                try:
                    with open(log_path, newline='') as f:
                        existing_header = next(csv.reader(f), [])
                    if existing_header == log_header:
                        break
                except Exception:
                    break
                suffix += 1
                log_path = f'{base}_v{suffix}{ext}'
            existed = os.path.exists(log_path)
            if existed and not self.loaded_checkpoint and self.step == 0:
                ts = time.strftime('%Y%m%d-%H%M%S')
                rotated = f'{base}.stale-{ts}{ext}'
                os.replace(log_path, rotated)
                existed = False
                print(f'[learner] rotated stale diagnostics log to {rotated}',
                      flush=True)
            if suffix > 1 and not existed:
                print(f'[learner] log schema changed; writing diagnostics to {log_path}',
                      flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._csv_file   = open(log_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow(log_header)

        # Start manager
        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        self._exp_queue   = _exp_queue
        print(f'[learner] manager on port {port}', flush=True)

    # ── Loop ──────────────────────────────────────────────────────────────────

    def run(self):
        self.net.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        while True:
            # Drain experience queue
            drained = 0
            try:
                while True:
                    exp = self._exp_queue.get_nowait()
                    self.buf.push(exp)
                    self.actor_recent.append(exp)
                    self._envelope_add(exp.state)
                    self._note_episode_done(exp)
                    drained += 1
            except Exception:
                pass  # queue.Empty is expected — any other error is transient IPC noise

            if len(self.buf) < self.min_replay or len(self.actor_recent) < self.actor_batch_size:
                if drained == 0:
                    time.sleep(0.01)
                continue

            # Critic: sample contiguous sequences for real BPTT.
            critic_seq_len = self.seq_len if self.net.is_recurrent else 1
            if self.net.is_recurrent and critic_seq_len > 1:
                n_seqs = max(1, self.batch_size // critic_seq_len)
                critic_batch = self.buf.sample_sequences(n_seqs, critic_seq_len)
            else:
                critic_batch = self.buf.sample(self.batch_size)
            actor_batch = self._sample_actor_batch()
            try:
                self.net.train()
                critic_info = compute_critic_loss(
                    self.net, self.target_net, critic_batch,
                    gamma       = self.gamma,
                    c_value     = self.c_value,
                    seq_len     = critic_seq_len,
                    critic_loss = self.critic_loss,
                    huber_delta = self.critic_huber_delta,
                )
                actor_info = compute_actor_loss(
                    self.net, self.target_net, actor_batch,
                    gamma       = self.gamma,
                    c_term      = self.c_term,
                    c_action    = self.c_action,
                    c_entropy   = self.c_entropy,
                    c_bdp_action = self.c_bdp_action,
                    bdp_action_gain = self.bdp_action_gain,
                    term_margin = self.term_margin,
                    c_action_center = self.c_action_center,
                    c_action_smooth = self.c_action_smooth,
                    c_option_usage = self.c_option_usage,
                    c_action_div = self.c_action_div,
                    option_kl_target = self.option_kl_target,
                    option_usage_temp = self.option_usage_temp,
                    c_beta_target = self.c_beta_target,
                    beta_target = self.beta_target,
                    advantage_norm = self.advantage_norm,
                    action_raw_clip = self.action_raw_clip,
                    policy_logp_clip = self.policy_logp_clip,
                )
                actor_loss_for_step = actor_info.loss
                next_step = self.step + 1
                actor_applied = (
                    next_step >= self.actor_warmup_steps
                    and (next_step % self.actor_update_every) == 0
                )
                if not actor_applied:
                    actor_loss_for_step = actor_info.loss.detach() * 0.0
                total_loss = critic_info.loss + actor_loss_for_step
                if not torch.isfinite(total_loss):
                    print(f'[learner] non-finite loss critic={critic_info.loss.item():.4f} '
                          f'actor={actor_info.loss.item():.4f}, skipping step', flush=True)
                    self.net.eval()
                    continue
                self.opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.opt.step()
                soft_update(self.net, self.target_net, self.target_tau)
                self.net.eval()
                self.step += 1
            except Exception as e:
                import traceback
                print(f'[learner] train step error: {e}\n{traceback.format_exc()}',
                      flush=True)
                self.net.eval()
                continue

            if self.step % 10 == 0:
                lr = self.opt.param_groups[0]['lr']
                self._warn_if_collapsed(actor_info)
                print(
                    f'[learner] step={self.step:6d}  '
                    f'critic={critic_info.loss.item():.4f}  actor={actor_info.loss.item():.4f}  '
                    f'(val={critic_info.value:.3f} act={actor_info.action:.3f} '
                    f'term={actor_info.term:.3f} ent={actor_info.entropy:.3f} '
                    f'use={actor_info.usage:.3f} div={actor_info.action_div:.3f} '
                    f'aux={actor_info.action_aux:.3f} '
                    f'gap={actor_info.q_gap:.2f} '
                    f'|gt|={critic_info.td_target_abs:.2f} |q|={critic_info.q_chosen_abs:.2f} '
                    f'β̄={actor_info.mean_beta:.3f})  '
                    f'buf={len(self.buf)}  actor_batch={len(actor_batch)}  lr={lr:.2e}',
                    flush=True)
                if self._csv_writer:
                    self._csv_writer.writerow([
                        self.step,
                        f'{total_loss.item():.6f}',
                        f'{critic_info.value:.6f}', f'{actor_info.action:.6f}',
                        f'{actor_info.term:.6f}', f'{actor_info.entropy:.6f}',
                        f'{actor_info.mean_beta:.6f}',
                        f'{critic_info.td_target_mean:.6f}', f'{critic_info.td_target_abs:.6f}',
                        f'{critic_info.q_chosen_mean:.6f}', f'{critic_info.q_chosen_abs:.6f}',
                        f'{actor_info.usage:.6f}', f'{actor_info.action_div:.6f}',
                        f'{actor_info.beta_target:.6f}',
                        f'{actor_info.center:.6f}', f'{actor_info.smooth:.6f}',
                        f'{actor_info.action_aux:.6f}',
                        f'{actor_info.q_gap:.6f}',
                        self._fmt_vec(actor_info.option_mass),
                        self._fmt_vec(actor_info.beta_by_option),
                        self._fmt_vec(actor_info.mu_by_option),
                        self._fmt_vec(actor_info.q_by_option),
                        f'{lr:.2e}',
                        len(self.buf), len(actor_batch), int(actor_applied),
                    ])

            if self.step % self.param_broadcast_every == 0:
                self._broadcast()

            self._maybe_save_latest_checkpoint()

    def _sample_actor_batch(self):
        recent = list(self.actor_recent)
        if not self.balanced_actor_options:
            return random.sample(recent, self.actor_batch_size)

        buckets = {i: [] for i in range(self.net.n_options)}
        for exp in recent:
            if 0 <= int(exp.option_idx) < self.net.n_options:
                buckets[int(exp.option_idx)].append(exp)

        non_empty = [b for b in buckets.values() if b]
        if not non_empty:
            return random.sample(recent, self.actor_batch_size)

        out = []
        per_option = max(1, self.actor_batch_size // len(non_empty))
        for bucket in non_empty:
            if len(bucket) >= per_option:
                out.extend(random.sample(bucket, per_option))
            else:
                out.extend(random.choices(bucket, k=per_option))

        while len(out) < self.actor_batch_size:
            out.append(random.choice(recent))
        if len(out) > self.actor_batch_size:
            out = random.sample(out, self.actor_batch_size)
        random.shuffle(out)
        return out

    def _warn_if_collapsed(self, actor_info):
        if self.step - self._last_collapse_warn_step < 5000:
            return
        mus = actor_info.mu_by_option or []
        betas = actor_info.beta_by_option or []
        if not mus or not betas:
            return
        at_upper = all(m >= self.net.log_mult_max - 1e-3 for m in mus)
        at_lower = all(m <= self.net.log_mult_min + 1e-3 for m in mus)
        beta_floor = all(b <= self.net.beta_floor + 1e-3 for b in betas)
        if (at_upper or at_lower) and beta_floor:
            edge = 'upper' if at_upper else 'lower'
            print(
                f'[learner] collapse warning step={self.step}: all option means '
                f'at {edge} action bound and beta at floor; actor is no longer '
                f'exploring useful option switches',
                flush=True)
            self._last_collapse_warn_step = self.step

    def _checkpoint_action_collapsed(self):
        probes = [np.zeros(STATE_DIM, dtype=np.float32)]
        env_path = self.ckpt_path + '.envelope.json' if self.ckpt_path else ''
        try:
            with open(env_path) as f:
                env = json.load(f)
            p1 = np.asarray(env.get('p1', []), dtype=np.float32)
            p99 = np.asarray(env.get('p99', []), dtype=np.float32)
            if p1.shape == (STATE_DIM,) and p99.shape == (STATE_DIM,):
                probes.extend([p1, p99, 0.5 * (p1 + p99)])
        except Exception:
            probes.extend([
                np.array([0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
                         dtype=np.float32),
                np.array([0.0, 4.0, 8.0, 0.2, 0.5, 1.0, 0.0, 0.0, 0.5, 4.0, 4.0],
                         dtype=np.float32),
            ])

        with torch.no_grad():
            states = torch.tensor(np.stack(probes), dtype=torch.float32, device=self.device)
            mu, _, _, beta, _ = self.net(states)
            at_lower = mu <= self.net.log_mult_min + 1e-3
            at_upper = mu >= self.net.log_mult_max - 1e-3
            at_edge = at_lower | at_upper
            edge_frac = at_edge.float().mean().item()
            same_edge_frac = (at_lower.all(dim=1) | at_upper.all(dim=1)).float().mean().item()
            beta_floor_frac = (beta <= self.net.beta_floor + 1e-3).float().mean().item()
        action_collapsed = (
            edge_frac >= 0.75 and same_edge_frac >= 0.50 and beta_floor_frac >= 0.25
        )
        beta_collapsed = beta_floor_frac >= 0.75
        return action_collapsed or beta_collapsed

    @staticmethod
    def _fmt_vec(values):
        return ';'.join(f'{float(v):.6f}' for v in values)

    def _broadcast(self):
        # Serialise to bytes so workers pull a plain bytestring rather
        # than PyTorch tensors backed by shared-memory FDs.  With 10+ parallel
        # workers each pull opens new FDs; bytes avoid the Errno 24 exhaustion.
        buf = io.BytesIO()
        torch.save({'state_dict': {k: v.cpu() for k, v in self.net.state_dict().items()},
                    'step': self.step,
                    'model_meta': self.net.model_meta(),
                    'state_meta': model_state_meta()}, buf)
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
            self._save(path=self._episode_ckpt_path(self._next_episode_ckpt),
                       reason=f'episode={self._next_episode_ckpt} snapshot')
            self._next_episode_ckpt += self.save_every_episodes

    def _save(self, path: str = None, reason: str = 'checkpoint'):
        path = path or self.ckpt_path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'model':      self.net.state_dict(),
            'target':     self.target_net.state_dict(),
            'optimizer':  self.opt.state_dict(),
            'step':       self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'model_meta': self.net.model_meta(),
            'state_meta': model_state_meta(),
        }, path)
        self._save_envelope(path)
        print(f'[learner] saved {reason} step={self.step} path={path}', flush=True)

    # ── Envelope reservoir ────────────────────────────────────────────────────

    def _envelope_add(self, state):
        """Reservoir sampling — keeps a uniform sample of all training states."""
        self._env_seen += 1
        s = np.asarray(state, dtype=np.float32)
        if len(self._env_reservoir) < self._env_reservoir_size:
            self._env_reservoir.append(s)
        else:
            j = random.randint(0, self._env_seen - 1)
            if j < self._env_reservoir_size:
                self._env_reservoir[j] = s

    def _save_envelope(self, ckpt_path: str = None):
        """Save per-dim (p1, p99) envelope to {ckpt}.envelope.json."""
        ckpt_path = ckpt_path or self.ckpt_path
        if not ckpt_path or len(self._env_reservoir) < 100:
            return
        arr = np.stack(self._env_reservoir)
        env = {
            'p1':        np.percentile(arr, 1.0,  axis=0).tolist(),
            'p99':       np.percentile(arr, 99.0, axis=0).tolist(),
            'min':       arr.min(axis=0).tolist(),
            'max':       arr.max(axis=0).tolist(),
            'n_samples': int(arr.shape[0]),
            'n_seen':    int(self._env_seen),
            'step':      int(self.step),
            'state_features': self.net.model_meta().get('state_features', 'unknown'),
        }
        env_path = ckpt_path + '.envelope.json'
        tmp_path = env_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(env, f, indent=2)
        os.replace(tmp_path, env_path)

    def stop(self):
        self._save()
        if self._csv_file:
            self._csv_file.close()
        self._mgr.shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--port',   type=int, default=None)
    ap.add_argument('--key',    default='')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    port    = args.port or int(cfg.get('learner', {}).get('port', 6301))
    authkey = bytes.fromhex(args.key) if args.key else secrets.token_bytes(32)

    learner = Learner(cfg, port, authkey)  # CUDA init + checkpoint load + manager start
    print(f'SAO_MANAGER_KEY={authkey.hex()}', flush=True)  # signal readiness only after manager is up

    def _on_signal(sig, _):
        print(f'[learner] signal {sig}, saving', flush=True)
        learner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        learner.run()
    except Exception:
        import traceback
        print(f'[learner] FATAL crash:\n{traceback.format_exc()}', flush=True)
        try:
            learner._save()
        except Exception:
            pass
        sys.exit(1)
