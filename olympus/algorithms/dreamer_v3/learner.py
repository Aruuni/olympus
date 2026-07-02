"""
learner.py — DreamerV3 learner.

Three nested update loops, all driven off real transitions pushed by workers:

  1. World model update (joint encoder + RSSM + decoder + reward + continue heads)
       Sample (B, T) sequence chunks from the per-trajectory replay; minimise
       reconstruction + reward + continue + KL(prior || posterior) with KL
       balancing and free bits. Returns posterior (h, z) trajectories.

  2. Imagine H steps from every posterior step using the actor + RSSM prior.
       Each imagined transition has predicted (h, z), reward (twohot expectation,
       symexp'd), value (twohot expectation, symexp'd), and continue probability.
       Compute lambda returns backwards.

  3. Actor + critic update on imagined trajectories.
       Critic: twohot CE on symlog(returns) + EMA target net.
       Actor:  score-function policy gradient on detached samples plus entropy
               and action regularisation to avoid tanh-boundary collapse.

The IPC + checkpoint + per-run logging machinery is structured the same as
algorithms/orca_td3/learner.py so the orchestrator and plotting tooling work
unchanged.
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

from olympus.algorithms.dreamer_v3.model import (
    WorldModel, Actor, Critic, Experience,
    STATE_DIM, ACTION_DIM,
    TwoHot, ReturnEMA,
    assert_checkpoint_state_compatible, model_state_meta,
    symlog, symexp, lambda_return, actor_log_prob, soft_update,
    ActorInfo, CriticInfo,
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


def _episode_from_traj_id(traj_id) -> int:
    try:
        return int(str(traj_id).rsplit('_', 2)[-2])
    except (IndexError, TypeError, ValueError):
        return -1


# ── Replay buffer (per-trajectory chunks, like orca_td3) ─────────────────────

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
        """Sample n_chunks coherent (s, a, r, d, mask) windows of length seq_len.

        For DreamerV3 we also need the action that PRECEDED the first state
        in the chunk (the GRU consumes a_{t-1} when stepping into s_t). We pad
        the leading action with zero when the chunk starts at the very first
        step of a trajectory.
        """
        if not self._trajs:
            return None
        tids = list(self._trajs.keys())
        out_s, out_a, out_r, out_d, out_m = [], [], [], [], []
        attempts = 0
        while len(out_s) < n_chunks and attempts < n_chunks * 8:
            attempts += 1
            tid = random.choice(tids)
            traj = self._trajs[tid]
            T = len(traj)
            if T < 2:
                continue
            start = random.randint(0, max(T - seq_len, 0)) if T > seq_len else 0
            end   = min(start + seq_len, T)
            L     = end - start

            s   = np.zeros((seq_len, STATE_DIM),  dtype=np.float32)
            a   = np.zeros((seq_len, ACTION_DIM), dtype=np.float32)
            r   = np.zeros(seq_len, dtype=np.float32)
            d   = np.zeros(seq_len, dtype=np.float32)
            m   = np.zeros(seq_len, dtype=np.float32)

            for t in range(L):
                e = traj[start + t]
                s[t]  = e.state
                # Action that PRECEDED state e.state — i.e. the action stored
                # one step earlier in the trajectory. For start == 0 there is
                # no preceding action so we leave it at zero.
                if start + t > 0:
                    a[t, 0] = traj[start + t - 1].action
                r[t]  = e.reward
                d[t]  = float(e.done)
                m[t]  = 1.0

            out_s.append(s); out_a.append(a); out_r.append(r)
            out_d.append(d); out_m.append(m)

        if not out_s:
            return None
        return dict(
            state  = np.stack(out_s),
            action = np.stack(out_a),
            reward = np.stack(out_r),
            done   = np.stack(out_d),
            mask   = np.stack(out_m),
        )


# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)
        a = cfg.get('agent',    {}) or {}

        # Replay + IPC
        self.batch_size            = int(t.get('batch_size',           16))
        self.seq_len               = int(t.get('seq_len',              48))
        self.min_replay            = int(t.get('min_replay',           5_000))
        self.replay_capacity       = int(t.get('replay_capacity',      400_000))
        self.save_every            = int(t.get('save_every',           500))
        self.save_every_episodes   = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 50))
        self.updates_per_step      = int(t.get('updates_per_step',     1))
        self.grad_clip             = float(t.get('grad_clip',          1000.0))

        # World-model loss weights & KL knobs
        self.kl_weight       = float(t.get('kl_weight',       1.0))
        self.kl_balance      = float(t.get('kl_balance',      0.8))
        self.kl_free_bits    = float(t.get('kl_free_bits',    1.0))
        self.recon_weight    = float(t.get('recon_weight',    1.0))
        self.reward_weight   = float(t.get('reward_weight',   1.0))
        self.continue_weight = float(t.get('continue_weight', 1.0))

        # Optimisers
        self.lr_world  = float(t.get('lr_world',  1e-4))
        self.lr_actor  = float(t.get('lr_actor',  3e-5))
        self.lr_critic = float(t.get('lr_critic', 3e-5))

        # Imagination + actor-critic
        self.horizon           = int(t.get('horizon',            15))
        self.gamma             = float(t.get('gamma',            0.99))
        self.lam               = float(t.get('lam',              0.95))
        self.entropy_coef      = float(t.get('entropy_coef',     3e-4))
        self.action_l2_coef    = float(t.get('action_l2_coef',   0.02))
        self.action_smooth_coef = float(t.get('action_smooth_coef', 0.02))
        self.target_tau        = float(t.get('target_tau',       0.02))
        self.actor_warmup_step = int(t.get('actor_warmup_steps', 0))
        # Reward / value twohot bin range (in symlog space).
        self.value_bins  = int(t.get('value_bins',  255))
        self.value_low   = float(t.get('value_low', -20.0))
        self.value_high  = float(t.get('value_high', 20.0))
        self.reward_bins = int(t.get('reward_bins', 255))

        # World-model architecture (matches the worker's env-passed config)
        self.hidden        = int(a.get('hidden',        256))
        self.embed_dim     = int(a.get('embed_dim',     128))
        self.h_dim         = int(a.get('h_dim',         256))
        self.latent_groups = int(a.get('latent_groups', 8))
        self.latent_classes = int(a.get('latent_classes', 8))

        ckpt_path = t.get('checkpoint',
                          os.path.join(_PKG, 'data', 'checkpoints',
                                       'dreamer_v3_cwnd_model.pt'))
        resume_from = t.get('resume_from', '')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        # Build networks
        self.world = WorldModel(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            hidden=self.hidden, embed_dim=self.embed_dim, h_dim=self.h_dim,
            latent_groups=self.latent_groups, latent_classes=self.latent_classes,
            reward_bins=self.reward_bins,
            reward_low=self.value_low, reward_high=self.value_high,
        ).to(self.device)
        self.latent_dim = self.world.latent_dim

        self.actor          = Actor(
            self.h_dim, self.latent_dim, self.hidden,
            log_std_min=float(t.get('actor_log_std_min', Actor.LOG_STD_MIN)),
            log_std_max=float(t.get('actor_log_std_max', Actor.LOG_STD_MAX)),
        ).to(self.device)
        self.critic         = Critic(self.h_dim, self.latent_dim,
                                     self.value_bins, self.hidden).to(self.device)
        self.critic_target  = Critic(self.h_dim, self.latent_dim,
                                     self.value_bins, self.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.opt_world  = optim.Adam(self.world.parameters(),  lr=self.lr_world)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.value_twohot = TwoHot(self.value_low, self.value_high, self.value_bins)
        self.return_ema   = ReturnEMA()

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
        self.step      = 0
        self.ckpt_path = ckpt_path
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)

        load_path = resume_from or ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(f'[learner] resume checkpoint not found: {resume_from}')
        if load_path and os.path.exists(load_path):
            try:
                ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_state_compatible(ckpt, source=load_path)
                self.world.load_state_dict(ckpt['world_model'], strict=False)
                self.actor.load_state_dict(ckpt['actor'],  strict=False)
                self.critic.load_state_dict(ckpt['critic'], strict=False)
                self.critic_target.load_state_dict(ckpt.get('critic_target',
                                                            ckpt['critic']), strict=False)
                if 'opt_world' in ckpt:  self.opt_world.load_state_dict(ckpt['opt_world'])
                if 'opt_actor' in ckpt:  self.opt_actor.load_state_dict(ckpt['opt_actor'])
                if 'opt_critic' in ckpt: self.opt_critic.load_state_dict(ckpt['opt_critic'])
                self.opt_world.param_groups[0]['lr']  = self.lr_world
                self.opt_actor.param_groups[0]['lr']  = self.lr_actor
                self.opt_critic.param_groups[0]['lr'] = self.lr_critic
                self.step = ckpt.get('step', 0)
                src = 'resume checkpoint' if resume_from else 'checkpoint'
                print(f'[learner] loaded {src} step={self.step} path={load_path}',
                      flush=True)
            except Exception as e:
                print(f'[learner] ckpt load failed ({e}), fresh start', flush=True)

        # CSV log
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
                    'step',
                    'wm_loss','wm_recon','wm_reward','wm_cont','kl_dyn','kl_rep',
                    'critic_loss','value_mean','target_mean',
                    'actor_loss','entropy','a_mean','a_abs','q_mean',
                    'return_low','return_high','return_scale',
                    'buf','trajs',
                ])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[learner] manager on port {port}  '
              f'(DreamerV3: h={self.h_dim} embed={self.embed_dim} '
              f'latent={self.latent_groups}x{self.latent_classes}={self.latent_dim} '
              f'horizon={self.horizon} γ={self.gamma} λ={self.lam} '
              f'entropy={self.entropy_coef:g} '
              f'act_l2={self.action_l2_coef:g} '
              f'act_smooth={self.action_smooth_coef:g})',
              flush=True)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.world.eval(); self.actor.eval(); self.critic.eval()
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
                dt = now - last_hb
                rate = total_drained / max(dt, 1e-6)
                mix = (f' emu={self.buf.size_emu()} sim={self.buf.size_sim()}'
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

            for _ in range(self.updates_per_step):
                prev_step = self.step
                self._update()
                if self.step != prev_step:
                    self._maybe_save_latest_checkpoint()
            if self.step % self.param_broadcast_every == 0:
                self._broadcast()

    # ── One full DreamerV3 update ────────────────────────────────────────────

    def _update(self):
        batch = self.buf.sample_chunks(self.batch_size, self.seq_len)
        if batch is None:
            return

        s = torch.from_numpy(batch['state']).to(self.device)            # (B, T, D)
        a = torch.from_numpy(batch['action']).to(self.device)            # (B, T, action_dim)
        r = torch.from_numpy(batch['reward']).to(self.device)
        d = torch.from_numpy(batch['done']).to(self.device)
        m = torch.from_numpy(batch['mask']).to(self.device)

        # ── 1. World model update ────────────────────────────────────────────
        self.world.train()
        wm_loss, wm_info, h_post, z_post = self.world.loss(
            states=s, actions=a, rewards=r, dones=d, mask=m,
            free_bits=self.kl_free_bits, kl_balance=self.kl_balance,
            kl_weight=self.kl_weight, reward_weight=self.reward_weight,
            continue_weight=self.continue_weight, recon_weight=self.recon_weight,
        )
        if not torch.isfinite(wm_loss):
            print('[learner] non-finite world-model loss — skip update', flush=True)
            self.world.eval()
            return
        self.opt_world.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.grad_clip)
        self.opt_world.step()
        self.world.eval()
        self.step += 1

        # ── 2. Imagine H steps from every posterior step ─────────────────────
        # Flatten posterior states across (B, T) into one batch and roll forward
        # `horizon` steps using actor + RSSM prior. Mask out padded posterior
        # steps so imagined trajectories start only from valid timesteps.
        with torch.no_grad():
            valid_mask = m.reshape(-1).bool()                            # (B*T,)
        h0 = h_post.reshape(-1, self.h_dim)
        z0 = z_post.reshape(-1, self.latent_dim)
        h0 = h0[valid_mask]
        z0 = z0[valid_mask]
        if h0.shape[0] == 0:
            return

        H = self.horizon
        N = h0.shape[0]
        imag_h, imag_z = [h0], [z0]
        actions, log_probs, entropies = [], [], []
        h, z = h0, z0
        for _ in range(H):
            base, mu, log_std = self.actor.dist(h, z)
            raw   = base.rsample()
            a_i   = torch.tanh(raw)
            logp  = actor_log_prob(base, raw.detach())
            entropy = base.entropy().sum(dim=-1)
            with torch.no_grad():
                h_next, z_next = self.world.rssm.imagine_step(h.detach(), z.detach(),
                                                              a_i.detach())
            actions.append(a_i)
            log_probs.append(logp)
            entropies.append(entropy)
            imag_h.append(h_next)
            imag_z.append(z_next)
            h, z = h_next, z_next

        imag_h_stack = torch.stack(imag_h, dim=0)              # (H+1, N, h_dim)
        imag_z_stack = torch.stack(imag_z, dim=0)              # (H+1, N, latent_dim)
        log_probs_stack = torch.stack(log_probs, dim=0)        # (H, N)
        entropies_stack = torch.stack(entropies, dim=0)        # (H, N)

        # World-model predictions for reward and continue at each imagined step.
        with torch.no_grad():
            r_logits = self.world.reward_head(imag_h_stack[1:], imag_z_stack[1:])
            r_pred   = symexp(self.world.twohot_reward.expectation(r_logits))
            c_logits = self.world.continue_head(imag_h_stack[1:], imag_z_stack[1:])
            c_pred   = torch.sigmoid(c_logits)

            v_target_logits = self.critic_target(imag_h_stack[1:], imag_z_stack[1:])
            v_target        = symexp(self.value_twohot.expectation(v_target_logits))
            bootstrap_logits = self.critic_target(imag_h_stack[-1], imag_z_stack[-1])
            bootstrap = symexp(self.value_twohot.expectation(bootstrap_logits))

            returns = lambda_return(
                rewards   = r_pred,
                values    = v_target,
                conts     = c_pred,
                bootstrap = bootstrap,
                lam       = self.lam,
                gamma     = self.gamma,
            )                                                     # (H, N)
            self.return_ema.update(returns)

        # ── 3. Critic update: twohot CE on symlog(return) ────────────────────
        self.critic.train()
        v_logits   = self.critic(imag_h_stack[:-1].detach(), imag_z_stack[:-1].detach())
        v_target_t = self.value_twohot.encode(symlog(returns.detach()))
        log_probs_v = torch.nn.functional.log_softmax(v_logits, dim=-1)
        critic_per = -(v_target_t * log_probs_v).sum(dim=-1)
        critic_loss = critic_per.mean()
        if not torch.isfinite(critic_loss):
            self.critic.eval()
            return
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()
        soft_update(self.critic, self.critic_target, self.target_tau)
        self.critic.eval()

        # ── 4. Actor update: REINFORCE on normalised advantages ──────────────
        actor_info_loss = float('nan')
        actor_info_a_mean = float('nan')
        actor_info_a_abs  = float('nan')
        actor_info_ent    = float('nan')
        actor_info_q_mean = float('nan')
        if self.step >= self.actor_warmup_step:
            self.actor.train()
            actions_stack_for_loss = torch.stack(actions, dim=0)
            with torch.no_grad():
                v_now_logits = self.critic_target(imag_h_stack[:-1], imag_z_stack[:-1])
                v_now = symexp(self.value_twohot.expectation(v_now_logits))
                advantage = (returns - v_now) / self.return_ema.scale()
            actor_obj = (log_probs_stack * advantage.detach()).mean()
            entropy_obj = entropies_stack.mean()
            action_l2 = actions_stack_for_loss.pow(2).mean()
            if actions_stack_for_loss.shape[0] > 1:
                action_smooth = (
                    actions_stack_for_loss[1:] - actions_stack_for_loss[:-1]
                ).pow(2).mean()
            else:
                action_smooth = actions_stack_for_loss.new_tensor(0.0)
            actor_loss = (
                -(actor_obj + self.entropy_coef * entropy_obj)
                + self.action_l2_coef * action_l2
                + self.action_smooth_coef * action_smooth
            )
            if torch.isfinite(actor_loss):
                self.opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.opt_actor.step()
                actor_info_loss   = float(actor_loss.detach().item())
                actor_info_ent    = float(entropy_obj.detach().item())
                actor_info_q_mean = float(returns.detach().mean().item())
                actions_stack = torch.stack(actions, dim=0).detach()
                actor_info_a_mean = float(actions_stack.mean().item())
                actor_info_a_abs  = float(actions_stack.abs().mean().item())
            self.actor.eval()

        # ── Logging ──────────────────────────────────────────────────────────
        if self.step % 10 == 0:
            v_logits_mean = float(symexp(self.value_twohot.expectation(v_logits.detach())
                                        ).mean().item())
            print(
                f'[learner] step={self.step:6d}  '
                f'wm={wm_info.loss.item():.3f} '
                f'(rec={wm_info.recon:.3f} rew={wm_info.reward:.3f} '
                f'cnt={wm_info.cont:.3f} kl_d={wm_info.kl_dyn:.2f} '
                f'kl_r={wm_info.kl_rep:.2f})  '
                f'critic={critic_loss.item():.3f}  actor={actor_info_loss:.3f}  '
                f'ent={actor_info_ent:.3f}  '
                f'a={actor_info_a_mean:.2f}±{actor_info_a_abs:.2f}  '
                f'ret={actor_info_q_mean:.2f}  '
                f'ema=[{self.return_ema.low:.2f},{self.return_ema.high:.2f}]  '
                f'buf={self.buf.size()}  trajs={self.buf.n_trajs()}',
                flush=True)
            if self._csv_writer:
                self._csv_writer.writerow([
                    self.step,
                    f'{wm_info.loss.item():.6f}',
                    f'{wm_info.recon:.6f}', f'{wm_info.reward:.6f}',
                    f'{wm_info.cont:.6f}',
                    f'{wm_info.kl_dyn:.6f}', f'{wm_info.kl_rep:.6f}',
                    f'{critic_loss.item():.6f}',
                    f'{v_logits_mean:.6f}',
                    f'{float(returns.detach().mean().item()):.6f}',
                    f'{actor_info_loss:.6f}',
                    f'{actor_info_ent:.6f}',
                    f'{actor_info_a_mean:.6f}', f'{actor_info_a_abs:.6f}',
                    f'{actor_info_q_mean:.6f}',
                    f'{self.return_ema.low:.6f}', f'{self.return_ema.high:.6f}',
                    f'{self.return_ema.scale():.6f}',
                    self.buf.size(), self.buf.n_trajs(),
                ])

    # ── Broadcast / save ─────────────────────────────────────────────────────

    def _broadcast(self):
        buf = io.BytesIO()
        torch.save({
            'world_model': {k: v.cpu() for k, v in self.world.state_dict().items()},
            'actor':       {k: v.cpu() for k, v in self.actor.state_dict().items()},
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
            self._save(path=self._episode_ckpt_path(self._next_episode_ckpt),
                       reason=f'episode={self._next_episode_ckpt} snapshot')
            self._next_episode_ckpt += self.save_every_episodes

    def _save(self, path: str = None, reason: str = 'ckpt'):
        path = path or self.ckpt_path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'world_model':   self.world.state_dict(),
            'actor':         self.actor.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'opt_world':     self.opt_world.state_dict(),
            'opt_actor':     self.opt_actor.state_dict(),
            'opt_critic':    self.opt_critic.state_dict(),
            'step':          self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'state_meta':    model_state_meta(),
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
