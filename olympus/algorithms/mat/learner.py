"""
learner.py — MAT-CTDE learner.

Receives per-agent rollouts from workers and runs MAPPO over the parameter-
shared transformer policy with a centralized joint critic. Three things make
this multi-agent rather than single-agent:

  1. Experience tuples carry `(agent_id, group_id, throughput)` so the learner
     can group simultaneous flows on the same bottleneck.
  2. Before computing GAE the learner adds a Jain's-fairness reward bonus
     computed across the throughput vector at each timestep:

        r_aug[t] = r_local[t] + fairness_weight · J(throughput_vec[t])

     This means the policy gradient sees fairness in the TD targets without
     workers needing any cross-flow IPC.
  3. The critic is centralized: the joint state at each timestep is the
     concatenation of all N agents' observations. The transformer encoder
     attends across the agent dimension to produce a single joint value.

Otherwise this is a vanilla on-policy PPO learner — collect `rollout_steps`,
update for K epochs of M minibatches with clipping, broadcast, repeat.
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

from olympus.algorithms.mat.model import (
    MATPolicy, Experience, compute_gae, mappo_loss,
    STATE_DIM, ACTION_DIM, assert_checkpoint_action_compatible,
    model_action_meta,
)


# ── IPC queues ────────────────────────────────────────────────────────────────

_exp_queue       = multiprocessing.Queue(maxsize=400_000)
_bootstrap_queue = multiprocessing.Queue(maxsize=4_000)
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


def _max_flow_count(spec, default=2):
    if spec is None:
        spec = default
    if isinstance(spec, str):
        text = spec.strip()
        if '-' in text:
            lo, hi = text.split('-', 1)
            vals = range(int(lo), int(hi) + 1)
        else:
            vals = [int(text)]
    elif isinstance(spec, (list, tuple)):
        vals = spec
    else:
        vals = [spec]
    if not vals:
        vals = [default]
    return max(max(1, min(int(v), 4)) for v in vals)


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


# ── Rollout buffer (per-agent trajectories, group-aware) ─────────────────────

class RolloutBuffer:
    """Per-trajectory store, cleared after every PPO update.

    Trajectories are keyed by `traj_id` (per-agent, per-episode). The learner
    additionally indexes them by `(group_id, agent_id)` so simultaneous flows
    on the same bottleneck can be assembled into joint timesteps.
    """
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
        """Returns a dict keyed by group_id; each value is a dict of per-agent
        numpy arrays plus the bootstrap value of each agent's trajectory.
        `group_step` is included so the joint-batch builder can align
        late-joiners on the slot-wide timeline."""
        groups = defaultdict(dict)
        for tid, traj in self._trajs.items():
            if len(traj) < 2:
                continue
            traj = sorted(traj, key=lambda e: (
                int(getattr(e, 'group_step', getattr(e, 'step_in_traj', 0))),
                int(getattr(e, 'step_in_traj', 0)),
            ))
            agent_id = traj[0].agent_id
            group_id = traj[0].group_id
            s   = np.stack([e.state for e in traj]).astype(np.float32)
            a   = np.array([e.action_raw for e in traj], dtype=np.float32)
            lp  = np.array([e.log_prob   for e in traj], dtype=np.float32)
            v   = np.array([e.value      for e in traj], dtype=np.float32)
            r   = np.array([e.reward     for e in traj], dtype=np.float32)
            thr = np.array([e.throughput for e in traj], dtype=np.float32)
            d   = np.array([float(e.done) for e in traj], dtype=np.float32)
            gs  = np.array([int(getattr(e, 'group_step', i))
                            for i, e in enumerate(traj)], dtype=np.int64)
            finite = (
                np.isfinite(s).all(axis=1)
                & np.isfinite(a)
                & np.isfinite(lp)
                & np.isfinite(v)
                & np.isfinite(r)
                & np.isfinite(thr)
                & np.isfinite(d)
            )
            if finite.sum() < 2:
                continue
            if not finite.all():
                s, a, lp, v = s[finite], a[finite], lp[finite], v[finite]
                r, thr, d, gs = r[finite], thr[finite], d[finite], gs[finite]
            last_v = self._bootstrap.get(tid, 0.0)
            if not np.isfinite(last_v):
                last_v = 0.0
            if d[-1] > 0.5:
                last_v = 0.0
            groups[group_id][agent_id] = dict(
                traj_id=tid, state=s, action_raw=a, log_prob=lp,
                value=v, reward=r, throughput=thr, done=d, group_step=gs,
                last_value=last_v,
            )
        return groups

    def clear(self):
        self._trajs.clear()
        self._bootstrap.clear()
        self._total = 0


# ── Joint-batch builder ───────────────────────────────────────────────────────

def _build_joint_batch(groups, n_agents, fairness_weight, gamma, lam,
                       reward_normalizer=None, fairness_centered=True,
                       team_reward_aggregation='mean'):
    """
    For each group, align all agents on a common timeline and produce a joint
    batch. Adds the fairness reward to each agent's per-step reward, computes
    GAE per agent against the joint critic's value, and pads / masks agents
    that are missing at any timestep.

    Returns a dict of tensors:
      state      : (T_total, N, STATE_DIM)
      action_raw : (T_total, N)
      log_prob   : (T_total, N)
      adv        : (T_total, N)            per-agent advantage
      ret        : (T_total,)              joint return
      value      : (T_total,)              joint old-value estimate
      mask       : (T_total, N)            1 for valid agents
      fairness_mean : float                summary diagnostic
    where T_total is the sum of trajectory lengths across all groups.
    """
    out_state, out_a, out_lp = [], [], []
    out_adv, out_ret, out_val = [], [], []
    out_mask = []
    fairness_acc, fairness_count = 0.0, 0

    for group_id, per_agent in groups.items():
        if not per_agent:
            continue
        ids = sorted(per_agent.keys())
        ids = [i for i in ids if 0 <= i < n_agents]
        if not ids:
            continue
        # Slot-wide timeline length: max(group_step) + 1 across all agents in
        # the group. Late-joiners scatter into [join_step, ..., join_step + L_a].
        T = 0
        for a in ids:
            gs = per_agent[a]['group_step']
            if gs.size > 0:
                T = max(T, int(gs.max()) + 1)
        if T <= 1:
            continue

        joint_state  = np.zeros((T, n_agents, STATE_DIM), dtype=np.float32)
        joint_action = np.zeros((T, n_agents),           dtype=np.float32)
        joint_logp   = np.zeros((T, n_agents),           dtype=np.float32)
        joint_value  = np.zeros((T, n_agents),           dtype=np.float32)
        joint_thrput = np.zeros((T, n_agents),           dtype=np.float32)
        joint_reward = np.zeros((T, n_agents),           dtype=np.float32)
        joint_done   = np.zeros((T,),                    dtype=np.float32)
        agent_mask   = np.zeros((T, n_agents),           dtype=np.float32)
        last_v_arr   = np.zeros(n_agents,                dtype=np.float32)
        any_done_T   = False

        for a in ids:
            d = per_agent[a]
            gs = d['group_step']
            if gs.size == 0:
                continue
            # Clamp into [0, T-1] in case of out-of-range steps.
            gs_c = np.clip(gs, 0, T - 1)
            joint_state[gs_c, a]  = d['state']
            joint_action[gs_c, a] = d['action_raw']
            joint_logp[gs_c, a]   = d['log_prob']
            joint_value[gs_c, a]  = d['value']
            joint_thrput[gs_c, a] = d['throughput']
            joint_reward[gs_c, a] = d['reward']
            agent_mask[gs_c, a]   = 1.0
            last_v_arr[a]         = d['last_value']
            if d['done'].size > 0 and d['done'][-1] > 0.5:
                any_done_T = True
        row_active = agent_mask.sum(axis=1) > 0.0
        if int(row_active.sum()) <= 1:
            continue
        if not bool(row_active.all()):
            joint_state  = joint_state[row_active]
            joint_action = joint_action[row_active]
            joint_logp   = joint_logp[row_active]
            joint_value  = joint_value[row_active]
            joint_thrput = joint_thrput[row_active]
            joint_reward = joint_reward[row_active]
            agent_mask   = agent_mask[row_active]
            joint_done   = joint_done[row_active]
        joint_done[-1] = float(any_done_T)

        # Fairness on simultaneous throughputs (zero where masked). Always
        # compute the diagnostic so fairness_weight=0 ablations still log it.
        n_active = agent_mask.sum(axis=1).clip(min=1.0)              # (T,)
        num = (joint_thrput * agent_mask).sum(axis=1) ** 2           # (T,)
        sq_sum = ((joint_thrput * agent_mask) ** 2).sum(axis=1)      # (T,)
        den = np.maximum(n_active * sq_sum, 1e-9)                    # (T,)
        fairness = num / den                                         # (T,)
        fairness_acc   += float(fairness.sum())
        fairness_count += int(fairness.size)
        if fairness_centered:
            min_fair = 1.0 / n_active
            denom = np.maximum(1.0 - min_fair, 1e-6)
            fairness_signal = np.where(
                n_active > 1.0,
                np.clip((fairness - min_fair) / denom, 0.0, 1.0),
                0.0,
            )
        else:
            fairness_signal = fairness
        if fairness_weight != 0.0:
            # Add the bonus to each active agent's reward at that timestep.
            joint_reward = joint_reward + fairness_weight * fairness_signal[:, None] * agent_mask

        # Joint return target, used by the central critic. The aggregation
        # ('mean' | 'sum') controls whether we want a per-step return that
        # scales with N (sum) or stays bounded as N grows (mean). For an
        # untrained value head, 'mean' keeps targets O(1) and stops the value
        # loss from drowning out the policy gradient.
        joint_v_mean = (joint_value * agent_mask).sum(axis=1) / n_active
        active_last = [last_v_arr[a] for a in ids if np.isfinite(last_v_arr[a])]
        last_v_joint = float(np.mean(active_last)) if active_last else 0.0
        if any_done_T:
            last_v_joint = 0.0
        joint_r_sum = (joint_reward * agent_mask).sum(axis=1)           # (T,)
        if str(team_reward_aggregation).lower() == 'sum':
            joint_r_per_step = joint_r_sum
        else:
            joint_r_per_step = joint_r_sum / n_active
        # Standardise the JOINT per-step reward — not per-agent — so the
        # critic targets stay O(1) regardless of n_agents and aggregation.
        if reward_normalizer is not None:
            reward_normalizer.update(joint_r_per_step)
            r_std = float(reward_normalizer.std)
            joint_r_per_step = joint_r_per_step / r_std
        adv_joint, ret_joint = compute_gae(
            joint_r_per_step, joint_v_mean, joint_done,
            last_v_joint, gamma=gamma, lam=lam)

        out_state.append(joint_state)
        out_a.append(joint_action)
        out_lp.append(joint_logp)
        # Broadcast the joint advantage across agents (each agent feels the
        # same shared signal, but its policy gradient is local via its own
        # log-prob ratio).
        out_adv.append(np.broadcast_to(adv_joint[:, None], joint_action.shape).copy())
        out_ret.append(ret_joint)
        out_val.append(joint_v_mean)
        out_mask.append(agent_mask)

    if not out_state:
        return None

    fairness_mean = (fairness_acc / max(fairness_count, 1)) if fairness_count else 0.0
    return dict(
        state         = np.concatenate(out_state, axis=0),
        action_raw    = np.concatenate(out_a,     axis=0),
        log_prob      = np.concatenate(out_lp,    axis=0),
        adv           = np.concatenate(out_adv,   axis=0),
        ret           = np.concatenate(out_ret,   axis=0),
        old_value     = np.concatenate(out_val,   axis=0),
        agent_mask    = np.concatenate(out_mask,  axis=0),
        fairness_mean = fairness_mean,
    )


# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)
        a = cfg.get('agent',    {}) or {}

        self.rollout_steps         = int(t.get('rollout_steps',         20_000))
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
        self.fairness_weight       = float(t.get('fairness_weight',     5.0))
        self.fairness_centered     = bool(t.get('fairness_centered',    True))
        self.team_reward_aggregation = str(t.get('team_reward_aggregation', 'mean'))
        self.use_reward_normalizer = bool(t.get('use_reward_normalizer', True))
        self.save_every            = int(t.get('save_every',            5))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 1))
        self.n_agents              = _max_flow_count(cfg.get('sweep', {}).get('flows', 2))
        torch_threads              = int(t.get('torch_threads', 0) or 0)
        torch_interop_threads      = int(t.get('torch_interop_threads', 0) or 0)
        ckpt_path                  = t.get('checkpoint',
                                           os.path.join(_PKG, 'data', 'checkpoints',
                                                        'mat_cwnd_model.pt'))
        resume_from = t.get('resume_from', '')
        hidden      = int(a.get('hidden', 128))
        n_layers    = int(a.get('n_layers', 2))
        n_heads     = int(a.get('n_heads', 4))

        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
        if torch_interop_threads > 0:
            try:
                torch.set_num_interop_threads(torch_interop_threads)
            except RuntimeError:
                pass
        self.torch_threads = torch.get_num_threads()
        try:
            self.torch_interop_threads = torch.get_num_interop_threads()
        except Exception:
            self.torch_interop_threads = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device} '
              f'torch_threads={self.torch_threads} '
              f'inter_op={self.torch_interop_threads}',
              flush=True)

        self.net = MATPolicy(STATE_DIM, hidden,
                             n_layers=n_layers, n_heads=n_heads).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)

        self.buf       = RolloutBuffer()
        self.step      = 0
        self.ckpt_path = ckpt_path
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)
        self._ret_rms  = _RunningStats() if self.use_reward_normalizer else None

        load_path = resume_from or ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(f'[learner] resume checkpoint not found: {resume_from}')
        if load_path and os.path.exists(load_path):
            try:
                ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_action_compatible(ckpt, source=load_path)
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
                    'kl','clip_frac','explained_var','fairness_mean',
                    'r_std','n_groups','n_trans','n_agents','lr',
                ])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[learner] manager on port {port}  '
              f'(MAT-CTDE: hidden={hidden} layers={n_layers} heads={n_heads} '
              f'rollout={self.rollout_steps} epochs={self.n_epochs} '
              f'gamma={self.gamma} lam={self.lam} '
              f'fairness_w={self.fairness_weight} '
              f'fairness_centered={int(self.fairness_centered)} '
              f'reward_agg={self.team_reward_aggregation} '
              f'n_agents={self.n_agents})',
              flush=True)

    def run(self):
        self.net.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_hb = time.monotonic()
        last_update_done = time.monotonic()
        total_drained = 0

        while True:
            drained = 0
            try:
                while True:
                    exp = _exp_queue.get_nowait()
                    self.buf.push(exp)
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

            update_start = time.monotonic()
            collect_dt = update_start - last_update_done
            self._mappo_update()
            update_dt = time.monotonic() - update_start
            print(f'[learner] timing collect={collect_dt:.1f}s '
                  f'update={update_dt:.2f}s',
                  flush=True)
            last_update_done = time.monotonic()
            self.buf.clear()

            self.step += 1
            if self.step % self.save_every == 0:
                self._save()
            if self.step % self.param_broadcast_every == 0:
                self._broadcast()

    def _mappo_update(self):
        groups = self.buf.extract()
        if not groups:
            return

        # _build_joint_batch handles the reward standardisation internally —
        # it standardises the JOINT per-step reward (after fairness + agg),
        # not per-agent rewards which would mis-scale the critic target.
        r_std = float(self._ret_rms.std) if self._ret_rms is not None else 1.0

        joint = _build_joint_batch(
            groups, n_agents=self.n_agents,
            fairness_weight=self.fairness_weight,
            gamma=self.gamma, lam=self.lam,
            reward_normalizer=self._ret_rms,
            fairness_centered=self.fairness_centered,
            team_reward_aggregation=self.team_reward_aggregation,
        )
        if joint is None:
            return

        device = self.device
        s   = torch.from_numpy(joint['state']).to(device)             # (T, N, D)
        a   = torch.from_numpy(joint['action_raw']).to(device)
        lp  = torch.from_numpy(joint['log_prob']).to(device)
        adv = torch.from_numpy(joint['adv']).to(device)
        ret = torch.from_numpy(joint['ret']).to(device)
        ov  = torch.from_numpy(joint['old_value']).to(device)
        m   = torch.from_numpy(joint['agent_mask']).to(device)
        fairness_mean = float(joint['fairness_mean'])

        row_ok = (
            torch.isfinite(s).flatten(1).all(dim=1)
            & torch.isfinite(a).all(dim=1)
            & torch.isfinite(lp).all(dim=1)
            & torch.isfinite(adv).all(dim=1)
            & torch.isfinite(ret)
            & torch.isfinite(ov)
            & torch.isfinite(m).all(dim=1)
            & (m.sum(dim=1) > 0.0)
        )
        if not bool(row_ok.all()):
            dropped = int((~row_ok).sum().item())
            print(f'[learner] dropped {dropped} invalid/empty joint rows before MAPPO',
                  flush=True)
            s, a, lp = s[row_ok], a[row_ok], lp[row_ok]
            adv, ret, ov, m = adv[row_ok], ret[row_ok], ov[row_ok], m[row_ok]
        if s.shape[0] < 2 or m.sum().item() <= 0:
            print('[learner] joint batch empty after filtering; skipping update',
                  flush=True)
            return

        T = s.shape[0]
        n_mb = max(1, self.minibatches_per_epoch)
        mb_sz = max(1, T // n_mb)

        kl_running, kl_batches = 0.0, 0
        early_stop = False
        last_info = None
        n_total_mb = 0
        n_total_agents = int(m.sum().item())

        self.net.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(T, device=device)
            for i in range(0, T, mb_sz):
                idx = perm[i:i + mb_sz]
                info = mappo_loss(
                    self.net,
                    s_batch         = s[idx],
                    a_raw_batch     = a[idx],
                    old_logp_batch  = lp[idx],
                    adv_batch       = adv[idx],
                    ret_batch       = ret[idx],
                    old_value_batch = ov[idx],
                    agent_mask      = m[idx],
                    clip_eps        = self.clip_eps,
                    c_value         = self.c_value,
                    c_entropy       = self.c_entropy,
                    value_clip      = self.value_clip,
                )
                if not torch.isfinite(info.loss):
                    print(f'[learner] non-finite loss, skipping minibatch '
                          f'policy={info.policy} value={info.value} '
                          f'entropy={info.entropy}',
                          flush=True)
                    continue
                self.opt.zero_grad()
                info.loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip)
                if not torch.isfinite(grad_norm):
                    print(f'[learner] non-finite grad_norm={grad_norm}; '
                          f'skipping optimizer step',
                          flush=True)
                    self.opt.zero_grad()
                    continue
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
            n_groups = len(groups)
            lr = self.opt.param_groups[0]['lr']
            print(
                f'[learner] step={self.step:5d}  loss={last_info.loss.item():.4f}  '
                f'pol={last_info.policy:.4f} val={last_info.value:.4f} '
                f'ent={last_info.entropy:.4f} kl={last_info.kl:.4f} '
                f'clip={last_info.clip_frac:.3f} evar={last_info.explained_var:.3f}  '
                f'fairness={fairness_mean:.3f}  r_std={r_std:.3f}  '
                f'groups={n_groups} trans={n_total_agents} mb={n_total_mb}'
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
                    f'{fairness_mean:.6f}',
                    f'{r_std:.6f}',
                    n_groups, n_total_agents, self.n_agents, f'{lr:.2e}',
                ])

    def _broadcast(self):
        buf = io.BytesIO()
        torch.save({'state_dict': {k: v.cpu() for k, v in self.net.state_dict().items()},
                    'step': self.step,
                    'action_meta': model_action_meta()}, buf)
        payload = buf.getvalue()
        try:
            while True: _param_queue.get_nowait()
        except Exception:
            pass
        try: _param_queue.put_nowait(payload)
        except Exception: pass

    def _save(self):
        if not self.ckpt_path:
            return
        torch.save({
            'model': self.net.state_dict(),
            'opt':   self.opt.state_dict(),
            'step':  self.step,
            'action_meta': model_action_meta(),
        }, self.ckpt_path)
        print(f'[learner] saved ckpt step={self.step}', flush=True)

    def stop(self):
        self._save()
        if self._csv_file:
            self._csv_file.close()
        self._mgr.shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--port',   type=int, default=6401)
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
