"""Astraea multi-agent TD3 learner (centralized training).

Recreates the paper's training algorithm (section 3.4, Algorithm 1): a
MADDPG-style variant of TD3 where the parameter-shared deterministic actor
sees only the flow's stacked local state, while the twin critics additionally
receive the 12-value global state aggregated over all simultaneous flows
(paper Table 2). The global reward (Eq. 8) is assembled here — only the
learner sees every flow of a bottleneck — by regrouping worker experiences on
``(group_id, group_step)``, exactly like the other multi-agent learners.

Off-policy: workers stream per-flow transitions into a joint replay buffer;
each update samples windows of ``history_len + 1`` joint steps, builds the
stacked local states, the global states, and the windowed fairness/stability
reward, and applies a TD3 update (clipped double-Q, target policy smoothing,
delayed policy updates — the released agent used policy_delay=20, Huber loss,
tau=0.005, gamma=0.98).

Usage (started by the orchestrator):
  python olympus/algorithms/astraea/learner.py --config <resolved.yaml> --port 6301
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
import threading
import time
from collections import defaultdict, deque
from multiprocessing.managers import BaseManager

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from olympus.algorithms.astraea.model import (
    ACTION_DIM,
    ACTION_MAX,
    ACTION_MIN,
    Actor,
    Experience,
    GLOBAL_DIM,
    HISTORY_LEN_DEFAULT,
    STATE_DIM,
    TwinCritic,
    assert_checkpoint_state_compatible,
    astraea_team_reward,
    global_state_from_stats,
    model_state_meta,
    set_expected_actor_norm,
    set_expected_history,
    soft_update,
)
from olympus.rewards.astraea import reward_weights


# ── Manager IPC (same service layout as ma_dreamer) ───────────────────────────

_exp_queue = multiprocessing.Queue(maxsize=400_000)


def _push_exp(exp):
    try:
        _exp_queue.put_nowait(exp)
    except Exception:
        pass


def _push_exp_batch(exps):
    for exp in exps:
        try:
            _exp_queue.put_nowait(exp)
        except Exception:
            pass


class _IPCService:
    def __init__(self):
        self._latest_params = None
        self._action_count = 0
        self._lock = threading.Lock()

    def push_exp(self, exp):
        _push_exp(exp)

    def push_exp_batch(self, exps):
        _push_exp_batch(exps)

    def set_params(self, payload):
        with self._lock:
            self._latest_params = payload

    def pull_params(self):
        with self._lock:
            return self._latest_params

    def reserve_action_steps(self, count):
        """Reserve a global action-index range for paper-style exploration."""
        count = max(1, int(count))
        with self._lock:
            start = self._action_count
            self._action_count += count
            return start

    def action_count(self):
        with self._lock:
            return self._action_count

    def set_action_count(self, count):
        with self._lock:
            self._action_count = max(0, int(count))


_ipc_service = _IPCService()


def _get_ipc_service():
    return _ipc_service


class _QueueManager(BaseManager):
    pass


_QueueManager.register(
    'service',
    callable=_get_ipc_service,
    exposed=(
        'push_exp', 'push_exp_batch', 'set_params', 'pull_params',
        'reserve_action_steps', 'action_count', 'set_action_count',
    ),
)


def _manager_handle_error(self, c, msg):  # noqa: ARG001
    import io as _io
    import traceback as _tb

    buf = _io.StringIO()
    _tb.print_exc(file=buf)
    text = buf.getvalue()
    if not any(x in text for x in (
            'BrokenPipeError', 'ConnectionResetError', 'EOFError',
            'ConnectionRefusedError')):
        sys.stderr.write('[learner] manager server error:\n' + text)
        sys.stderr.flush()


from multiprocessing.managers import Server as _MgrServer  # noqa: E402
_MgrServer.handle_error = _manager_handle_error


def _max_flow_count(spec, default=2):
    if spec is None:
        spec = default
    if isinstance(spec, str):
        text = spec.strip()
        if '-' in text:
            lo, hi = text.split('-', 1)
            values = range(int(lo), int(hi) + 1)
        else:
            values = [int(text)]
    elif isinstance(spec, (list, tuple)):
        values = spec
    else:
        values = [spec]
    return max(max(1, min(int(v), 5)) for v in (values or [default]))


# ── Joint replay (group/agent/step indexed, like ma_dreamer) ──────────────────

_CHUNK_FIELDS = (
    'state', 'action', 'local_reward', 'throughput', 'latency_us',
    'loss_ratio', 'pacing_rate', 'cwnd', 'min_rtt_us', 'rtt_ref_us',
    'link_bw', 'buffer_bytes', 'done', 'mask',
)


class JointReplayBuffer:
    """Replay indexed by group, agent, and slot-wide control tick."""

    def __init__(self, max_transitions=400_000):
        self._groups = defaultdict(lambda: defaultdict(dict))
        self._group_steps = defaultdict(dict)
        self._candidates = set()
        self._traj_keys = defaultdict(list)
        self._traj_order = deque()
        self._known_trajs = set()
        self._total = 0
        self._capacity = int(max_transitions)

    def push(self, exp: Experience):
        traj_id = exp.traj_id
        if traj_id not in self._known_trajs:
            self._known_trajs.add(traj_id)
            self._traj_order.append(traj_id)

        key = (exp.group_id, int(exp.agent_id), int(exp.group_step))
        slot = self._groups[key[0]][key[1]]
        old = slot.get(key[2])
        slot[key[2]] = exp
        if old is None:
            self._total += 1
            self._traj_keys[traj_id].append(key)
            steps_seen = self._group_steps[key[0]]
            if key[2] in steps_seen:
                steps_seen[key[2]] += 1
            else:
                steps_seen[key[2]] = 1
                if len(steps_seen) >= 2:
                    self._candidates.add(key[0])

        while self._total > self._capacity and self._traj_order:
            self._evict(self._traj_order.popleft())

    def _evict(self, traj_id):
        for group_id, agent_id, group_step in self._traj_keys.pop(traj_id, []):
            agents = self._groups.get(group_id)
            if not agents:
                continue
            steps = agents.get(agent_id)
            if not steps:
                continue
            current = steps.get(group_step)
            if current is not None and current.traj_id == traj_id:
                del steps[group_step]
                self._total -= 1
                steps_seen = self._group_steps.get(group_id)
                if steps_seen is not None and group_step in steps_seen:
                    if steps_seen[group_step] <= 1:
                        del steps_seen[group_step]
                        if len(steps_seen) < 2:
                            self._candidates.discard(group_id)
                        if not steps_seen:
                            self._group_steps.pop(group_id, None)
                    else:
                        steps_seen[group_step] -= 1
            if not steps:
                agents.pop(agent_id, None)
            if not agents:
                self._groups.pop(group_id, None)
        self._known_trajs.discard(traj_id)

    def size(self):
        return self._total

    def n_groups(self):
        return len(self._groups)

    def sample_chunks(self, batch_size, seq_len, max_agents):
        """Sample joint windows of ``seq_len`` consecutive slot ticks.

        Returns dict of (B, T, N, ...) arrays: per-step observations aligned
        as ``state[t] = exp.next_state`` with ``action[t]`` the action that
        led INTO it, plus the raw per-flow statistics the global reward and
        global state are built from. Windows require the transition tick
        (last) and the tick before it to be populated for at least one agent.
        """
        if not self._candidates:
            return None
        candidates = list(self._candidates)

        batch_size = int(batch_size)
        seq_len = int(seq_len)
        max_agents = int(max_agents)
        sorted_steps = {}
        out = defaultdict(list)
        attempts = 0
        max_attempts = batch_size * 12
        while len(out['state']) < batch_size and attempts < max_attempts:
            attempts += 1
            group_id = random.choice(candidates)
            available_steps = sorted_steps.get(group_id)
            if available_steps is None:
                steps_seen = self._group_steps.get(group_id)
                available_steps = sorted(steps_seen) if steps_seen else []
                sorted_steps[group_id] = available_steps
            if len(available_steps) < 2:
                continue
            agents = self._groups[group_id]
            # Anchor the window on an existing tick so its last two indices
            # (the TD3 transition) are likely populated.
            anchor = available_steps[
                random.randrange(1, len(available_steps))]
            start = anchor - (seq_len - 1)

            states = np.zeros(
                (seq_len, max_agents, STATE_DIM), dtype=np.float32)
            actions = np.zeros(
                (seq_len, max_agents, ACTION_DIM), dtype=np.float32)
            local_rewards = np.zeros((seq_len, max_agents), dtype=np.float32)
            throughput = np.zeros((seq_len, max_agents), dtype=np.float32)
            latency_us = np.zeros((seq_len, max_agents), dtype=np.float32)
            loss_ratio = np.zeros((seq_len, max_agents), dtype=np.float32)
            pacing_rate = np.zeros((seq_len, max_agents), dtype=np.float32)
            cwnd = np.zeros((seq_len, max_agents), dtype=np.float32)
            min_rtt_us = np.zeros((seq_len, max_agents), dtype=np.float32)
            rtt_ref_us = np.zeros((seq_len, max_agents), dtype=np.float32)
            link_bw = np.zeros((seq_len, max_agents), dtype=np.float32)
            buffer_bytes = np.zeros((seq_len, max_agents), dtype=np.float32)
            dones = np.zeros(seq_len, dtype=np.float32)
            mask = np.zeros((seq_len, max_agents), dtype=np.float32)

            for agent_id, steps in agents.items():
                if agent_id < 0 or agent_id >= max_agents:
                    continue
                for t in range(seq_len):
                    exp = steps.get(start + t)
                    if exp is None:
                        continue
                    states[t, agent_id] = exp.next_state
                    actions[t, agent_id, 0] = float(exp.action)
                    local_rewards[t, agent_id] = float(exp.reward)
                    throughput[t, agent_id] = float(exp.throughput)
                    latency_us[t, agent_id] = float(exp.latency_us)
                    loss_ratio[t, agent_id] = float(exp.loss_ratio)
                    pacing_rate[t, agent_id] = float(exp.pacing_rate)
                    cwnd[t, agent_id] = float(exp.cwnd)
                    min_rtt_us[t, agent_id] = float(exp.min_rtt_us)
                    rtt_ref_us[t, agent_id] = float(exp.rtt_ref_us)
                    link_bw[t, agent_id] = float(exp.link_bw)
                    buffer_bytes[t, agent_id] = float(exp.buffer_bytes)
                    mask[t, agent_id] = 1.0
                    dones[t] = max(dones[t], float(exp.done))

            # The TD3 transition lives on the last two ticks of the window.
            if not ((mask[-1] * mask[-2]).sum() > 0.0):
                continue

            out['state'].append(states)
            out['action'].append(actions)
            out['local_reward'].append(local_rewards)
            out['throughput'].append(throughput)
            out['latency_us'].append(latency_us)
            out['loss_ratio'].append(loss_ratio)
            out['pacing_rate'].append(pacing_rate)
            out['cwnd'].append(cwnd)
            out['min_rtt_us'].append(min_rtt_us)
            out['rtt_ref_us'].append(rtt_ref_us)
            out['link_bw'].append(link_bw)
            out['buffer_bytes'].append(buffer_bytes)
            out['done'].append(dones)
            out['mask'].append(mask)

        if not out['state']:
            return None
        return {key: np.stack(value) for key, value in out.items()}


# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes,
                 start_manager: bool = True):
        training = cfg.get('training', cfg) or {}
        agent = cfg.get('agent', {}) or {}

        self.batch_size = int(training.get('batch_size', 64))
        self.min_replay = int(training.get('min_replay', 20_000))
        self.replay_capacity = int(training.get('replay_capacity', 400_000))
        self.updates_per_step = int(training.get('updates_per_step', 1))
        # Paper Table 4 training cadence: `model update step` gradient
        # iterations every `model update interval` seconds. 0 interval falls
        # back to continuous updates_per_step updates.
        self.model_update_interval_s = float(
            training.get('model_update_interval_s', 5.0))
        self.model_update_steps = int(training.get('model_update_steps', 20))
        self.save_every = int(training.get('save_every', 500))
        self.param_broadcast_every = int(
            training.get('param_broadcast_every', 50))
        self.grad_clip = float(training.get('grad_clip', 0.0) or 0.0)
        # Train the critic alone first.  This prevents a randomly initialized
        # critic from immediately steering the deterministic actor to an
        # action boundary before the long Astraea exploration phase has
        # collected representative transitions.
        self.actor_warmup_steps = max(
            0, int(training.get('actor_warmup_steps', 1_000)))
        monitor = training.get('collapse_monitor', {}) or {}
        self.collapse_monitor_enabled = bool(monitor.get('enabled', True))
        self.collapse_utilization = float(
            monitor.get('utilization_threshold', 0.05))
        self.collapse_floor_fraction = float(
            monitor.get('cwnd_floor_fraction', 0.90))
        self.collapse_negative_fraction = float(
            monitor.get('negative_action_fraction', 0.90))
        self.collapse_action_threshold = float(
            monitor.get('negative_action_threshold', -0.90))
        self.collapse_patience = max(1, int(monitor.get('patience', 5)))
        self._collapse_streak = 0

        # TD3 hyperparameters — defaults from the released astraea.json.
        self.gamma = float(training.get('gamma', 0.98))
        self.target_tau = float(training.get('target_tau', 0.005))
        self.lr_actor = float(training.get('lr_actor', 5e-5))
        self.lr_critic = float(training.get('lr_critic', 1e-3))
        self.policy_delay = max(1, int(training.get('policy_delay', 20)))
        self.target_noise_std = float(training.get('target_noise_std', 0.05))
        self.target_noise_clip = float(
            training.get('target_noise_clip', 0.1))

        self.history_len = int(
            agent.get('history_len', HISTORY_LEN_DEFAULT))
        self.actor_norm = str(
            agent.get('actor_norm', 'batchnorm')).strip().lower()
        set_expected_history(self.history_len)
        set_expected_actor_norm(self.actor_norm)
        self.h1 = int(agent.get('hidden', 256))
        self.h2 = int(agent.get('hidden2', 128))
        self.cwnd_min = int(agent.get('cwnd_min', 10))
        self.max_agents = _max_flow_count(
            (cfg.get('sweep', {}) or {}).get('flows', 2))
        self.reward_weights = reward_weights(cfg)

        torch_threads = int(training.get('torch_threads', 0) or 0)
        torch_interop_threads = int(
            training.get('torch_interop_threads', 0) or 0)
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
        if torch_interop_threads > 0:
            try:
                torch.set_num_interop_threads(torch_interop_threads)
            except RuntimeError:
                pass

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        self.actor = Actor(
            STATE_DIM, self.history_len, self.h1, self.h2,
            self.actor_norm).to(self.device)
        self.actor_target = Actor(
            STATE_DIM, self.history_len, self.h1, self.h2,
            self.actor_norm).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = TwinCritic(
            STATE_DIM, self.history_len, GLOBAL_DIM,
            self.h1, self.h2).to(self.device)
        self.critic_target = TwinCritic(
            STATE_DIM, self.history_len, GLOBAL_DIM,
            self.h1, self.h2).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for module in (self.actor_target, self.critic_target):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.opt_actor = optim.Adam(
            self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = optim.Adam(
            self.critic.parameters(), lr=self.lr_critic)

        self.buf = JointReplayBuffer(self.replay_capacity)
        self.step = 0
        self.action_count = 0
        self.ckpt_path = training.get(
            'checkpoint',
            os.path.join(_PKG, 'data', 'checkpoints',
                         'astraea_cwnd_model.pt'),
        )
        resume_from = training.get('resume_from', '')
        if self.ckpt_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.ckpt_path)),
                exist_ok=True,
            )
        load_path = resume_from or self.ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(
                f'[learner] resume checkpoint not found: {resume_from}')
        if load_path and os.path.exists(load_path):
            self._load(load_path, resume=bool(resume_from))

        log_path = training.get(
            'log_path',
            self.ckpt_path.replace('.pt', '_log.csv')
            if self.ckpt_path else '',
        )
        self._csv_file = None
        self._csv_writer = None
        if log_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            existed = os.path.exists(log_path)
            self._csv_file = open(log_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow([
                    'step', 'critic_loss', 'actor_loss',
                    'q1_mean', 'q2_mean',
                    'reward_mean', 'local_reward_mean',
                    'r_thr', 'r_lat', 'r_loss', 'r_fair', 'r_stab',
                    'actor_action_mean', 'actor_negative_fraction',
                    'cwnd_floor_fraction', 'jain_fairness',
                    'buffer', 'groups',
                ])

        self._mgr = None
        self._service = None
        if start_manager:
            self._mgr = _QueueManager(
                address=('0.0.0.0', port), authkey=authkey)
            self._mgr.start()
            self._service = self._mgr.service()
            self._service.set_action_count(self.action_count)
            print(
                f'[learner] manager on port {port} '
                f'(Astraea TD3 agents<={self.max_agents} '
                f'history={self.history_len} h1={self.h1} h2={self.h2} '
                f'norm={self.actor_norm} warmup={self.actor_warmup_steps} '
                f'gamma={self.gamma:g} tau={self.target_tau:g} '
                f'policy_delay={self.policy_delay} '
                f"weights={self.reward_weights})",
                flush=True,
            )
            self._broadcast()

    def _load(self, path, resume=False):
        try:
            ckpt = torch.load(
                path, map_location=self.device, weights_only=False)
            assert_checkpoint_state_compatible(ckpt, source=path)
            self.actor.load_state_dict(ckpt['actor'], strict=False)
            self.actor_target.load_state_dict(
                ckpt.get('actor_target', ckpt['actor']), strict=False)
            self.critic.load_state_dict(ckpt['critic'], strict=False)
            self.critic_target.load_state_dict(
                ckpt.get('critic_target', ckpt['critic']), strict=False)
            if 'opt_actor' in ckpt:
                self.opt_actor.load_state_dict(ckpt['opt_actor'])
            if 'opt_critic' in ckpt:
                self.opt_critic.load_state_dict(ckpt['opt_critic'])
            self.opt_actor.param_groups[0]['lr'] = self.lr_actor
            self.opt_critic.param_groups[0]['lr'] = self.lr_critic
            self.step = int(ckpt.get('step', 0))
            self.action_count = int(ckpt.get('action_count', 0))
            source = 'resume checkpoint' if resume else 'checkpoint'
            print(
                f'[learner] loaded {source} step={self.step} path={path}',
                flush=True,
            )
        except Exception as exc:
            print(
                f'[learner] checkpoint load failed ({exc}), fresh start',
                flush=True,
            )

    def run(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_heartbeat = time.monotonic()
        next_cycle = time.monotonic()
        total_drained = 0
        while True:
            drained = 0
            try:
                while True:
                    self.buf.push(_exp_queue.get_nowait())
                    drained += 1
            except Exception:
                pass
            total_drained += drained

            now = time.monotonic()
            if now - last_heartbeat >= 5.0:
                dt = now - last_heartbeat
                print(
                    f'[learner] hb buf={self.buf.size()}/'
                    f'{self.replay_capacity} recv={total_drained} '
                    f'({total_drained / max(dt, 1e-6):.0f}/s) '
                    f'groups={self.buf.n_groups()}',
                    flush=True,
                )
                last_heartbeat = now
                total_drained = 0

            if self.buf.size() < self.min_replay:
                if drained == 0:
                    time.sleep(0.05)
                continue

            # Paper Table 4 cadence: model_update_steps iterations per
            # model_update_interval_s seconds of collection.
            if self.model_update_interval_s > 0:
                if now < next_cycle:
                    time.sleep(min(0.05, next_cycle - now))
                    continue
                next_cycle = max(
                    next_cycle + self.model_update_interval_s, now)
                n_updates = self.model_update_steps
            else:
                n_updates = self.updates_per_step

            for _ in range(n_updates):
                if self._update():
                    self.step += 1
                    if self.save_every > 0 and self.step % self.save_every == 0:
                        self._save()
                    if (self.param_broadcast_every > 0
                            and self.step % self.param_broadcast_every == 0):
                        self._broadcast()

    # ── One TD3 update ────────────────────────────────────────────────────────

    def _update(self):
        window = self.history_len + 1
        batch = self.buf.sample_chunks(
            self.batch_size, window, self.max_agents)
        if batch is None:
            return False

        w = self.history_len
        states = batch['state']            # (B, T, N, D)
        mask = batch['mask']               # (B, T, N)
        B, T, N, D = states.shape

        # Global reward (paper Eq. 8) from the w MTPs ending at the
        # transition's next observation (window indices 1..T-1). d0 is each
        # flow's observed min_rtt.
        reward, components = astraea_team_reward(
            batch['throughput'][:, 1:, :],
            batch['latency_us'][:, 1:, :],
            batch['loss_ratio'][:, 1:, :],
            batch['pacing_rate'][:, 1:, :],
            mask[:, 1:, :],
            batch['link_bw'][:, 1:, :],
            batch['min_rtt_us'][:, 1:, :],
            weights=self.reward_weights,
        )

        # Global states (paper Table 2) before and after the transition.
        def _global_at(t):
            return global_state_from_stats(
                batch['throughput'][:, t, :],
                batch['latency_us'][:, t, :],
                batch['cwnd'][:, t, :],
                batch['loss_ratio'][:, t, :],
                mask[:, t, :],
                batch['link_bw'][:, t, :],
                batch['rtt_ref_us'][:, t, :],
                batch['buffer_bytes'][:, t, :],
            )

        g = _global_at(T - 2)              # (B, GLOBAL_DIM)
        g2 = _global_at(T - 1)

        # Stacked local states: zero-filled where a frame is missing — the
        # same zero-initialized shift register the worker acts with.
        masked_states = states * mask[..., None]
        s_stack = masked_states[:, :w].transpose(0, 2, 1, 3).reshape(
            B, N, w * D)
        s2_stack = masked_states[:, 1:].transpose(0, 2, 1, 3).reshape(
            B, N, w * D)

        valid = (mask[:, -1, :] * mask[:, -2, :]).reshape(-1) > 0.0
        if not valid.any():
            return False

        device = self.device
        sel = torch.from_numpy(valid.astype(np.bool_)).to(device)
        s_t = torch.from_numpy(
            s_stack.reshape(B * N, w * D)).to(device)[sel]
        s2_t = torch.from_numpy(
            s2_stack.reshape(B * N, w * D)).to(device)[sel]
        a_t = torch.from_numpy(
            batch['action'][:, -1, :, 0].reshape(-1).astype(np.float32)
        ).to(device)[sel]
        g_rep = torch.from_numpy(
            np.repeat(g[:, None, :], N, axis=1).reshape(B * N, GLOBAL_DIM)
        ).to(device)[sel]
        g2_rep = torch.from_numpy(
            np.repeat(g2[:, None, :], N, axis=1).reshape(B * N, GLOBAL_DIM)
        ).to(device)[sel]
        r_t = torch.from_numpy(
            np.repeat(reward[:, None], N, axis=1).reshape(-1)
        ).to(device)[sel]
        d_t = torch.from_numpy(
            np.repeat(batch['done'][:, -1][:, None], N, axis=1)
            .reshape(-1).astype(np.float32)
        ).to(device)[sel]

        sg_t = torch.cat([s_t, g_rep], dim=-1)
        sg2_t = torch.cat([s2_t, g2_rep], dim=-1)

        # ── Critic update (clipped double-Q + target policy smoothing) ───────
        self.critic.train()
        with torch.no_grad():
            a2 = self.actor_target(s2_t).squeeze(-1)
            noise = (torch.randn_like(a2) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip)
            a2 = (a2 + noise).clamp(ACTION_MIN, ACTION_MAX)
            q1_next, q2_next = self.critic_target(sg2_t, a2)
            y = r_t + self.gamma * (1.0 - d_t) * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(sg_t, a_t)
        critic_loss = F.huber_loss(q1, y) + F.huber_loss(q2, y)
        if not torch.isfinite(critic_loss):
            self.critic.eval()
            return False
        self.opt_critic.zero_grad()
        critic_loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()
        # The original agent soft-updates the critic targets every train step
        # and the actor target only on delayed policy steps.
        soft_update(self.critic, self.critic_target, self.target_tau)
        self.critic.eval()

        # ── Delayed policy update ─────────────────────────────────────────────
        actor_loss_value = float('nan')
        if (self.step >= self.actor_warmup_steps
                and self.step % self.policy_delay == 0):
            self.actor.train()
            a_pi = self.actor(s_t).squeeze(-1)
            actor_loss = -self.critic.q1_only(sg_t, a_pi).mean()
            if torch.isfinite(actor_loss):
                self.opt_actor.zero_grad()
                actor_loss.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.grad_clip)
                self.opt_actor.step()
                soft_update(self.actor, self.actor_target, self.target_tau)
                actor_loss_value = float(actor_loss.detach().item())
            self.actor.eval()

        if self.step % 10 == 0:
            valid_local = batch['local_reward'][:, -1, :][
                (mask[:, -1, :] * mask[:, -2, :]) > 0.0]
            critic_loss_value = float(critic_loss.detach().item())
            q1_mean = float(q1.detach().mean().item())
            q2_mean = float(q2.detach().mean().item())
            reward_mean = float(np.mean(reward))
            local_mean = float(valid_local.mean()) if valid_local.size else 0.0
            comp_means = {
                key: float(np.mean(value))
                for key, value in components.items()
            }
            with torch.no_grad():
                policy_actions = self.actor(s_t).squeeze(-1)
            actor_action_mean = float(policy_actions.mean().item())
            actor_negative_fraction = float((
                policy_actions <= self.collapse_action_threshold
            ).float().mean().item())
            valid_now = (mask[:, -1, :] * mask[:, -2, :]) > 0.0
            current_cwnd = batch['cwnd'][:, -1, :][valid_now]
            cwnd_floor = 0.0
            if current_cwnd.size:
                cwnd_floor = float(np.mean(current_cwnd <= self.cwnd_min))

            throughput_now = batch['throughput'][:, -1, :]
            present_now = mask[:, -1, :]
            flow_count = present_now.sum(axis=-1)
            throughput_sum = (throughput_now * present_now).sum(axis=-1)
            throughput_sq_sum = (
                throughput_now ** 2 * present_now).sum(axis=-1)
            jain_denom = flow_count * throughput_sq_sum
            jain = np.where(
                jain_denom > 0.0,
                throughput_sum ** 2 / np.maximum(jain_denom, 1e-9),
                0.0,
            )
            jain_fairness = float(np.mean(jain))

            collapse_signal = (
                comp_means['r_thr'] <= self.collapse_utilization
                and cwnd_floor >= self.collapse_floor_fraction
                and actor_negative_fraction >= self.collapse_negative_fraction
            )
            self._collapse_streak = (
                self._collapse_streak + 1 if collapse_signal else 0)
            if (self.collapse_monitor_enabled
                    and self._collapse_streak == self.collapse_patience):
                print(
                    '[learner] COLLAPSE WARNING: low utilization, CWND-floor '
                    'lock, and negative actor saturation persisted for '
                    f'{self.collapse_patience} checks '
                    f'(util={comp_means["r_thr"]:.3f} '
                    f'floor={cwnd_floor:.3f} '
                    f'negative={actor_negative_fraction:.3f})',
                    flush=True,
                )
            print(
                f'[learner] step={self.step:6d} '
                f'critic={critic_loss_value:.4f} '
                f'actor={actor_loss_value:.4f} '
                f'q1={q1_mean:.4f} '
                f'R={reward_mean:.4f} '
                f"thr={comp_means['r_thr']:.3f} "
                f"lat={comp_means['r_lat']:.4f} "
                f"loss={comp_means['r_loss']:.4f} "
                f"fair={comp_means['r_fair']:.4f} "
                f"stab={comp_means['r_stab']:.4f} "
                f'a={actor_action_mean:.3f} floor={cwnd_floor:.2f} '
                f'jain={jain_fairness:.3f} '
                f'buf={self.buf.size()} groups={self.buf.n_groups()}',
                flush=True,
            )
            if self._csv_writer:
                self._csv_writer.writerow([
                    self.step,
                    f'{critic_loss_value:.6f}',
                    f'{actor_loss_value:.6f}',
                    f'{q1_mean:.6f}',
                    f'{q2_mean:.6f}',
                    f'{reward_mean:.6f}',
                    f'{local_mean:.6f}',
                    f"{comp_means['r_thr']:.6f}",
                    f"{comp_means['r_lat']:.6f}",
                    f"{comp_means['r_loss']:.6f}",
                    f"{comp_means['r_fair']:.6f}",
                    f"{comp_means['r_stab']:.6f}",
                    f'{actor_action_mean:.6f}',
                    f'{actor_negative_fraction:.6f}',
                    f'{cwnd_floor:.6f}',
                    f'{jain_fairness:.6f}',
                    self.buf.size(),
                    self.buf.n_groups(),
                ])
        return True

    # ── Broadcast / save ─────────────────────────────────────────────────────

    def _broadcast(self):
        buffer = io.BytesIO()
        torch.save({
            'actor': {
                key: value.cpu()
                for key, value in self.actor.state_dict().items()
            },
            'step': self.step,
            'state_meta': model_state_meta(),
            'history_len': self.history_len,
            'actor_norm': self.actor_norm,
        }, buffer)
        payload = buffer.getvalue()
        if self._service is not None:
            self._service.set_params(payload)

    def _save(self):
        if not self.ckpt_path:
            return
        if self._service is not None:
            try:
                self.action_count = int(self._service.action_count())
            except Exception:
                pass
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'opt_actor': self.opt_actor.state_dict(),
            'opt_critic': self.opt_critic.state_dict(),
            'step': self.step,
            'action_count': self.action_count,
            'state_meta': model_state_meta(),
            'history_len': self.history_len,
            'actor_norm': self.actor_norm,
            'max_agents': self.max_agents,
            'global_dim': GLOBAL_DIM,
            'reward_weights': dict(self.reward_weights),
        }, self.ckpt_path)
        print(
            f'[learner] saved checkpoint step={self.step} '
            f'path={self.ckpt_path}',
            flush=True,
        )

    def stop(self):
        self._save()
        if self._csv_file:
            self._csv_file.close()
        if self._mgr is not None:
            self._mgr.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=6301)
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    authkey = secrets.token_bytes(16)
    learner = Learner(config, port=args.port, authkey=authkey)
    print(f'SAO_MANAGER_KEY={authkey.hex()}', flush=True)

    def _stop(_signum, _frame):
        learner.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        learner.run()
    except KeyboardInterrupt:
        learner.stop()
