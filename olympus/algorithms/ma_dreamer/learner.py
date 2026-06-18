"""MA-Dreamer learner with a training-only global world model.

Real per-flow transitions are aligned by ``(group_id, group_step)``. The local
Tempest world model is trained on each active flow independently, while the
global model is trained on the masked joint trajectory and learns to predict
the local RSSM latents. Actor and critic updates then use shared imagination:
all policies choose actions from their translated local latents and the global
RSSM advances the joint trajectory.
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

from olympus.algorithms.ma_dreamer.model import (
    ACTION_DIM,
    Actor,
    Critic,
    Experience,
    GlobalWorldModel,
    LocalWorldModel,
    ReturnEMA,
    STATE_DIM,
    TwoHot,
    actor_log_prob,
    assert_checkpoint_state_compatible,
    lambda_return,
    per_agent_team_reward,
    shared_team_reward,
    soft_update,
    state_meta,
    symlog,
    symexp,
)


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


_ipc_service = _IPCService()


def _get_ipc_service():
    return _ipc_service


class _QueueManager(BaseManager):
    pass


_QueueManager.register(
    'service',
    callable=_get_ipc_service,
    exposed=('push_exp', 'push_exp_batch', 'set_params', 'pull_params'),
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


from multiprocessing.managers import Server as _MgrServer
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
    return max(max(1, min(int(v), 4)) for v in (values or [default]))


class JointReplayBuffer:
    """Replay indexed by group, agent, and slot-wide control tick."""

    def __init__(self, max_transitions=400_000):
        self._groups = defaultdict(lambda: defaultdict(dict))
        # group_id -> {group_step: refcount}; refcount = #agents holding that
        # step. Maintained incrementally so sampling never scans the buffer.
        self._group_steps = defaultdict(dict)
        # group_ids that currently have >= 2 distinct steps (sampleable).
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
        if not self._candidates:
            return None
        candidates = list(self._candidates)

        batch_size = int(batch_size)
        seq_len = int(seq_len)
        max_agents = int(max_agents)
        # Sort each chosen group's step list at most once per call (the buffer
        # is not mutated while sampling), instead of rebuilding it per attempt.
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
            start = available_steps[random.randrange(len(available_steps) - 1)]

            states = np.zeros(
                (seq_len, max_agents, STATE_DIM), dtype=np.float32)
            actions = np.zeros(
                (seq_len, max_agents, ACTION_DIM), dtype=np.float32)
            local_rewards = np.zeros(
                (seq_len, max_agents), dtype=np.float32)
            avg_throughputs = np.zeros(
                (seq_len, max_agents), dtype=np.float32)
            dones = np.zeros(seq_len, dtype=np.float32)
            mask = np.zeros((seq_len, max_agents), dtype=np.float32)

            # Fetch each agent's step dict once, rather than re-scanning
            # agents.items() for every one of the seq_len timesteps.
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
                    avg_throughputs[t, agent_id] = float(exp.avg_throughput)
                    mask[t, agent_id] = 1.0
                    dones[t] = max(dones[t], float(exp.done))

            row_valid = mask.sum(axis=-1) > 0.0
            if int(row_valid.sum()) < 2:
                continue

            out['state'].append(states)
            out['action'].append(actions)
            out['local_reward'].append(local_rewards)
            out['avg_throughput'].append(avg_throughputs)
            out['done'].append(dones)
            out['mask'].append(mask)

        if not out['state']:
            return None
        return {key: np.stack(value) for key, value in out.items()}


class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes,
                 start_manager: bool = True):
        training = cfg.get('training', cfg) or {}
        agent = cfg.get('agent', {}) or {}
        reward_cfg = cfg.get('reward', {}) or {}

        self.batch_size = int(training.get('batch_size', 12))
        self.seq_len = int(training.get('seq_len', 48))
        self.min_replay = int(training.get('min_replay', 8_000))
        self.replay_capacity = int(training.get('replay_capacity', 400_000))
        self.updates_per_step = int(training.get('updates_per_step', 1))
        self.save_every = int(training.get('save_every', 500))
        self.param_broadcast_every = int(
            training.get('param_broadcast_every', 50))
        self.grad_clip = float(training.get('grad_clip', 100.0))

        self.kl_weight = float(training.get('kl_weight', 1.0))
        self.kl_balance = float(training.get('kl_balance', 0.8))
        self.kl_free_bits = float(training.get('kl_free_bits', 1.0))
        self.recon_weight = float(training.get('recon_weight', 1.0))
        self.reward_weight = float(training.get('reward_weight', 1.0))
        self.continue_weight = float(training.get('continue_weight', 1.0))
        self.local_world_weight = float(
            training.get('local_world_weight', 1.0))
        self.global_world_weight = float(
            training.get('global_world_weight', 1.0))
        self.latent_h_weight = float(training.get('latent_h_weight', 1.0))
        self.latent_z_weight = float(training.get('latent_z_weight', 1.0))

        self.lr_world = float(training.get('lr_world', 1e-4))
        self.lr_actor = float(training.get('lr_actor', 3e-5))
        self.lr_critic = float(training.get('lr_critic', 3e-5))
        self.horizon = int(training.get('horizon', 15))
        self.gamma = float(training.get('gamma', 0.99))
        self.lam = float(training.get('lam', 0.95))
        self.entropy_coef = float(training.get('entropy_coef', 3e-3))
        self.action_l2_coef = float(training.get('action_l2_coef', 2e-2))
        self.action_smooth_coef = float(
            training.get('action_smooth_coef', 2e-2))
        self.target_tau = float(training.get('target_tau', 0.02))
        self.actor_warmup_steps = int(
            training.get('actor_warmup_steps', 1000))

        self.value_bins = int(training.get('value_bins', 255))
        self.value_low = float(training.get('value_low', -20.0))
        self.value_high = float(training.get('value_high', 20.0))
        self.reward_bins = int(training.get('reward_bins', 255))

        self.fairness_weight = float(
            reward_cfg.get(
                'fairness_weight',
                training.get('fairness_weight', 25.0),
            )
        )
        self.fairness_cap = float(
            reward_cfg.get(
                'fairness_cap',
                training.get('fairness_cap', 1.0),
            )
        )
        # Optional per-agent credit assignment: each agent trains on its own
        # R_i instead of the (default, Astraea-style) shared team scalar.
        # Explicit opt-in via reward.per_agent_credit; changes the global
        # reward head to one output per agent.
        self.per_agent_credit = bool(reward_cfg.get('per_agent_credit', False))
        self.team_alpha = float(reward_cfg.get('team_alpha', 0.5))
        self.over_weight = float(reward_cfg.get('over_weight', 50.0))

        self.hidden = int(agent.get('hidden', 256))
        self.embed_dim = int(agent.get('embed_dim', 128))
        self.h_dim = int(agent.get('h_dim', 256))
        self.latent_groups = int(agent.get('latent_groups', 8))
        self.latent_classes = int(agent.get('latent_classes', 8))
        self.global_hidden = int(agent.get('global_hidden', self.hidden))
        self.global_embed_dim = int(
            agent.get('global_embed_dim', self.embed_dim))
        self.global_h_dim = int(agent.get('global_h_dim', self.h_dim))
        self.global_latent_groups = int(
            agent.get('global_latent_groups', self.latent_groups))
        self.global_latent_classes = int(
            agent.get('global_latent_classes', self.latent_classes))
        self.max_agents = _max_flow_count(
            (cfg.get('sweep', {}) or {}).get('flows', 2))

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

        self.local_world = LocalWorldModel(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden=self.hidden,
            embed_dim=self.embed_dim,
            h_dim=self.h_dim,
            latent_groups=self.latent_groups,
            latent_classes=self.latent_classes,
            reward_bins=self.reward_bins,
            reward_low=self.value_low,
            reward_high=self.value_high,
        ).to(self.device)
        self.local_latent_dim = self.local_world.latent_dim

        self.global_world = GlobalWorldModel(
            max_agents=self.max_agents,
            state_dim=STATE_DIM,
            hidden=self.global_hidden,
            embed_dim=self.global_embed_dim,
            h_dim=self.global_h_dim,
            latent_groups=self.global_latent_groups,
            latent_classes=self.global_latent_classes,
            local_h_dim=self.h_dim,
            local_latent_groups=self.latent_groups,
            local_latent_classes=self.latent_classes,
            reward_bins=self.reward_bins,
            reward_low=self.value_low,
            reward_high=self.value_high,
            reward_dim=self.max_agents if self.per_agent_credit else 1,
        ).to(self.device)

        self.actor = Actor(
            self.h_dim,
            self.local_latent_dim,
            self.hidden,
            log_std_min=float(
                training.get('actor_log_std_min', Actor.LOG_STD_MIN)),
            log_std_max=float(
                training.get('actor_log_std_max', Actor.LOG_STD_MAX)),
        ).to(self.device)
        self.critic = Critic(
            self.h_dim, self.local_latent_dim,
            self.value_bins, self.hidden).to(self.device)
        self.critic_target = Critic(
            self.h_dim, self.local_latent_dim,
            self.value_bins, self.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        world_parameters = (
            list(self.local_world.parameters())
            + list(self.global_world.parameters())
        )
        self.opt_world = optim.Adam(world_parameters, lr=self.lr_world)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = optim.Adam(
            self.critic.parameters(), lr=self.lr_critic)
        self.value_twohot = TwoHot(
            self.value_low, self.value_high, self.value_bins)
        self.return_ema = ReturnEMA()

        self.buf = JointReplayBuffer(self.replay_capacity)
        self.step = 0
        self.ckpt_path = training.get(
            'checkpoint',
            os.path.join(_PKG, 'data', 'checkpoints',
                         'ma_dreamer_cwnd_model.pt'),
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
            self._csv_file = open(
                log_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow([
                    'step',
                    'local_wm', 'local_recon', 'local_reward',
                    'global_wm', 'global_recon', 'global_reward',
                    'global_agent_h', 'global_agent_z',
                    'critic_loss', 'actor_loss', 'entropy',
                    'return_mean', 'r_fair_mean', 'fairness_cost_mean',
                    'return_low', 'return_high', 'return_scale',
                    'buffer', 'groups',
                ])

        self._mgr = None
        self._service = None
        if start_manager:
            self._mgr = _QueueManager(
                address=('0.0.0.0', port), authkey=authkey)
            self._mgr.start()
            self._service = self._mgr.service()
            print(
                f'[learner] manager on port {port} '
                f'(MA-Dreamer agents<={self.max_agents} '
                f'local_latent={self.latent_groups}x{self.latent_classes} '
                f'global_latent={self.global_latent_groups}x'
                f'{self.global_latent_classes} horizon={self.horizon} '
                f'R_fair_weight={self.fairness_weight:g} '
                f'credit={self.per_agent_credit} '
                f'alpha={self.team_alpha:g} '
                f'over_weight={self.over_weight:g})',
                flush=True,
            )
            self._broadcast()

    def _load(self, path, resume=False):
        try:
            ckpt = torch.load(
                path, map_location=self.device, weights_only=False)
            assert_checkpoint_state_compatible(ckpt, source=path)
            self.local_world.load_state_dict(
                ckpt['local_world_model'], strict=False)
            self.global_world.load_state_dict(
                ckpt['global_world_model'], strict=False)
            self.actor.load_state_dict(ckpt['actor'], strict=False)
            self.critic.load_state_dict(ckpt['critic'], strict=False)
            self.critic_target.load_state_dict(
                ckpt.get('critic_target', ckpt['critic']), strict=False)
            if 'opt_world' in ckpt:
                self.opt_world.load_state_dict(ckpt['opt_world'])
            if 'opt_actor' in ckpt:
                self.opt_actor.load_state_dict(ckpt['opt_actor'])
            if 'opt_critic' in ckpt:
                self.opt_critic.load_state_dict(ckpt['opt_critic'])
            self.opt_world.param_groups[0]['lr'] = self.lr_world
            self.opt_actor.param_groups[0]['lr'] = self.lr_actor
            self.opt_critic.param_groups[0]['lr'] = self.lr_critic
            self.step = int(ckpt.get('step', 0))
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
        self.local_world.eval()
        self.global_world.eval()
        self.actor.eval()
        self.critic.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_heartbeat = time.monotonic()
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

            for _ in range(self.updates_per_step):
                if self._update():
                    self.step += 1
                    if self.save_every > 0 and self.step % self.save_every == 0:
                        self._save()
                    if (self.param_broadcast_every > 0
                            and self.step % self.param_broadcast_every == 0):
                        self._broadcast()

    def _update(self):
        batch = self.buf.sample_chunks(
            self.batch_size, self.seq_len, self.max_agents)
        if batch is None:
            return False

        device = self.device
        states = torch.from_numpy(batch['state']).to(device)
        actions = torch.from_numpy(batch['action']).to(device)
        local_rewards = torch.from_numpy(batch['local_reward']).to(device)
        avg_throughput = torch.from_numpy(
            batch['avg_throughput']).to(device)
        dones = torch.from_numpy(batch['done']).to(device)
        agent_mask = torch.from_numpy(batch['mask']).to(device)
        if self.per_agent_credit:
            team_rewards, r_fair, fairness_cost = per_agent_team_reward(
                local_rewards,
                avg_throughput,
                agent_mask,
                team_alpha=self.team_alpha,
                fairness_weight=self.fairness_weight,
                fairness_cap=self.fairness_cap,
                over_weight=self.over_weight,
            )
        else:
            team_rewards, r_fair, fairness_cost = shared_team_reward(
                local_rewards,
                avg_throughput,
                agent_mask,
                fairness_weight=self.fairness_weight,
                fairness_cap=self.fairness_cap,
            )

        batch_size, seq_len, n_agents, _ = states.shape
        local_states = states.permute(0, 2, 1, 3).reshape(
            batch_size * n_agents, seq_len, STATE_DIM)
        local_actions = actions.permute(0, 2, 1, 3).reshape(
            batch_size * n_agents, seq_len, ACTION_DIM)
        local_reward_flat = local_rewards.permute(0, 2, 1).reshape(
            batch_size * n_agents, seq_len)
        local_mask = agent_mask.permute(0, 2, 1).reshape(
            batch_size * n_agents, seq_len)
        local_dones = dones.unsqueeze(1).expand(
            batch_size, n_agents, seq_len).reshape(
                batch_size * n_agents, seq_len)

        self.local_world.train()
        self.global_world.train()
        local_loss, local_info, local_h, local_z = self.local_world.loss(
            local_states,
            local_actions,
            local_reward_flat,
            local_dones,
            local_mask,
            free_bits=self.kl_free_bits,
            kl_balance=self.kl_balance,
            kl_weight=self.kl_weight,
            reward_weight=self.reward_weight,
            continue_weight=self.continue_weight,
            recon_weight=self.recon_weight,
        )
        local_h = local_h.reshape(
            batch_size, n_agents, seq_len, self.h_dim
        ).permute(0, 2, 1, 3)
        local_z = local_z.reshape(
            batch_size, n_agents, seq_len, self.local_latent_dim
        ).permute(0, 2, 1, 3)

        global_loss, global_info, global_h, global_z = self.global_world.loss(
            states,
            actions,
            team_rewards,
            dones,
            agent_mask,
            local_h,
            local_z,
            free_bits=self.kl_free_bits,
            kl_balance=self.kl_balance,
            kl_weight=self.kl_weight,
            reward_weight=self.reward_weight,
            continue_weight=self.continue_weight,
            recon_weight=self.recon_weight,
            latent_h_weight=self.latent_h_weight,
            latent_z_weight=self.latent_z_weight,
        )
        world_loss = (
            self.local_world_weight * local_loss
            + self.global_world_weight * global_loss
        )
        if not torch.isfinite(world_loss):
            print('[learner] non-finite world-model loss; skipping', flush=True)
            self.local_world.eval()
            self.global_world.eval()
            return False

        self.opt_world.zero_grad()
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.local_world.parameters())
            + list(self.global_world.parameters()),
            self.grad_clip,
        )
        self.opt_world.step()
        self.local_world.eval()
        self.global_world.eval()

        row_mask = agent_mask.sum(dim=-1) > 0.0
        start_h = global_h.reshape(-1, self.global_h_dim)[row_mask.reshape(-1)]
        start_z = global_z.reshape(
            -1, self.global_world.latent_dim)[row_mask.reshape(-1)]
        start_agent_mask = agent_mask.reshape(
            -1, self.max_agents)[row_mask.reshape(-1)]
        if start_h.shape[0] == 0:
            return False

        horizon = self.horizon
        n_starts = start_h.shape[0]
        global_h_t = start_h
        global_z_t = start_z
        local_h_steps = []
        local_z_steps = []
        action_steps = []
        log_prob_steps = []
        entropy_steps = []
        reward_steps = []
        continue_steps = []
        target_value_steps = []

        for _ in range(horizon):
            with torch.no_grad():
                agent_h, agent_z, _ = self.global_world.agent_latents(
                    global_h_t, global_z_t, sample=True)
            local_h_steps.append(agent_h)
            local_z_steps.append(agent_z)

            flat_h = agent_h.reshape(-1, self.h_dim)
            flat_z = agent_z.reshape(-1, self.local_latent_dim)
            base, _, _ = self.actor.dist(flat_h, flat_z)
            raw_action = base.rsample()
            action = torch.tanh(raw_action).reshape(
                n_starts, self.max_agents, ACTION_DIM)
            log_prob = actor_log_prob(
                base, raw_action.detach()).reshape(
                    n_starts, self.max_agents)
            entropy = base.entropy().sum(dim=-1).reshape(
                n_starts, self.max_agents)
            joint_action = (
                action.squeeze(-1) * start_agent_mask).detach()

            with torch.no_grad():
                next_h, next_z = self.global_world.rssm.imagine_step(
                    global_h_t, global_z_t, joint_action)
                next_agent_h, next_agent_z, _ = (
                    self.global_world.agent_latents(
                        next_h, next_z, sample=True))
                reward_logits = self.global_world.reward_head(next_h, next_z)
                reward_pred = symexp(
                    self.global_world.twohot_reward.expectation(reward_logits))
                continue_pred = torch.sigmoid(
                    self.global_world.continue_head(next_h, next_z))
                value_logits = self.critic_target(
                    next_agent_h, next_agent_z)
                value_pred = symexp(
                    self.value_twohot.expectation(value_logits))

            action_steps.append(action)
            log_prob_steps.append(log_prob)
            entropy_steps.append(entropy)
            reward_steps.append(reward_pred)
            continue_steps.append(continue_pred)
            target_value_steps.append(value_pred)
            global_h_t, global_z_t = next_h, next_z

        with torch.no_grad():
            final_agent_h, final_agent_z, _ = (
                self.global_world.agent_latents(
                    global_h_t, global_z_t, sample=True))
            bootstrap = symexp(self.value_twohot.expectation(
                self.critic_target(final_agent_h, final_agent_z)))

        reward_stack = torch.stack(reward_steps, dim=0)
        continue_stack = torch.stack(continue_steps, dim=0)
        value_target_stack = torch.stack(target_value_steps, dim=0)
        if self.per_agent_credit:
            # The per-agent head already yields one reward per agent.
            team_reward_stack = reward_stack
        else:
            team_reward_stack = reward_stack.unsqueeze(-1).expand(
                horizon, n_starts, self.max_agents)
        team_continue_stack = continue_stack.unsqueeze(-1).expand_as(
            team_reward_stack)
        returns = lambda_return(
            rewards=team_reward_stack,
            values=value_target_stack,
            conts=team_continue_stack,
            bootstrap=bootstrap,
            lam=self.lam,
            gamma=self.gamma,
        )
        imagine_mask = start_agent_mask.unsqueeze(0).expand_as(returns)
        valid_returns = returns[imagine_mask.bool()]
        if valid_returns.numel() == 0:
            return False
        self.return_ema.update(valid_returns)

        local_h_stack = torch.stack(local_h_steps, dim=0).detach()
        local_z_stack = torch.stack(local_z_steps, dim=0).detach()
        self.critic.train()
        value_logits = self.critic(local_h_stack, local_z_stack)
        value_target = self.value_twohot.encode(symlog(returns.detach()))
        critic_per = -(
            value_target * F.log_softmax(value_logits, dim=-1)
        ).sum(dim=-1)
        critic_loss = (
            critic_per * imagine_mask).sum() / imagine_mask.sum().clamp_min(1.0)
        if not torch.isfinite(critic_loss):
            self.critic.eval()
            return False
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()
        soft_update(self.critic, self.critic_target, self.target_tau)
        self.critic.eval()

        actor_loss_value = float('nan')
        entropy_value = float('nan')
        if self.step >= self.actor_warmup_steps:
            self.actor.train()
            log_prob_stack = torch.stack(log_prob_steps, dim=0)
            entropy_stack = torch.stack(entropy_steps, dim=0)
            action_stack = torch.stack(action_steps, dim=0).squeeze(-1)
            with torch.no_grad():
                current_value = symexp(self.value_twohot.expectation(
                    self.critic_target(local_h_stack, local_z_stack)))
                advantage = (
                    returns - current_value) / self.return_ema.scale()

            denom = imagine_mask.sum().clamp_min(1.0)
            actor_objective = (
                log_prob_stack * advantage.detach() * imagine_mask
            ).sum() / denom
            entropy_objective = (
                entropy_stack * imagine_mask).sum() / denom
            action_l2 = (
                action_stack.square() * imagine_mask).sum() / denom
            if horizon > 1:
                smooth_mask = imagine_mask[1:] * imagine_mask[:-1]
                smooth_denom = smooth_mask.sum().clamp_min(1.0)
                action_smooth = (
                    (action_stack[1:] - action_stack[:-1]).square()
                    * smooth_mask
                ).sum() / smooth_denom
            else:
                action_smooth = action_l2.new_tensor(0.0)

            actor_loss = (
                -(actor_objective + self.entropy_coef * entropy_objective)
                + self.action_l2_coef * action_l2
                + self.action_smooth_coef * action_smooth
            )
            if torch.isfinite(actor_loss):
                self.opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.grad_clip)
                self.opt_actor.step()
                actor_loss_value = float(actor_loss.detach().item())
                entropy_value = float(entropy_objective.detach().item())
            self.actor.eval()

        if self.step % 10 == 0:
            # Pull every logged scalar off the GPU here (the only place they are
            # consumed), so the hot path stays sync-free between steps.
            return_mean = float(valid_returns.detach().mean().item())
            local_loss = float(local_info.loss.item())
            local_recon = float(local_info.recon.item())
            local_reward = float(local_info.reward.item())
            global_loss = float(global_info.loss.item())
            global_recon = float(global_info.recon.item())
            global_reward = float(global_info.reward.item())
            global_agent_h = float(global_info.agent_h.item())
            global_agent_z = float(global_info.agent_z.item())
            critic_loss_value = float(critic_loss.item())
            r_fair_mean = float(
                r_fair[row_mask].mean().item()) if row_mask.any() else 0.0
            fairness_cost_mean = float(
                fairness_cost[row_mask].mean().item()) if row_mask.any() else 0.0
            print(
                f'[learner] step={self.step:6d} '
                f'local_wm={local_loss:.3f} '
                f'global_wm={global_loss:.3f} '
                f'latent=({global_agent_h:.3f},'
                f'{global_agent_z:.3f}) '
                f'critic={critic_loss_value:.3f} '
                f'actor={actor_loss_value:.3f} '
                f'R_fair={r_fair_mean:.4f} '
                f'fair_cost={fairness_cost_mean:.3f} '
                f'return={return_mean:.3f} '
                f'buf={self.buf.size()} groups={self.buf.n_groups()}',
                flush=True,
            )
            if self._csv_writer:
                self._csv_writer.writerow([
                    self.step,
                    f'{local_loss:.6f}',
                    f'{local_recon:.6f}',
                    f'{local_reward:.6f}',
                    f'{global_loss:.6f}',
                    f'{global_recon:.6f}',
                    f'{global_reward:.6f}',
                    f'{global_agent_h:.6f}',
                    f'{global_agent_z:.6f}',
                    f'{critic_loss_value:.6f}',
                    f'{actor_loss_value:.6f}',
                    f'{entropy_value:.6f}',
                    f'{return_mean:.6f}',
                    f'{r_fair_mean:.6f}',
                    f'{fairness_cost_mean:.6f}',
                    f'{self.return_ema.low:.6f}',
                    f'{self.return_ema.high:.6f}',
                    f'{self.return_ema.scale():.6f}',
                    self.buf.size(),
                    self.buf.n_groups(),
                ])
        return True

    def _broadcast(self):
        buffer = io.BytesIO()
        torch.save({
            'local_world_model': {
                key: value.cpu()
                for key, value in self.local_world.state_dict().items()
            },
            'actor': {
                key: value.cpu()
                for key, value in self.actor.state_dict().items()
            },
            'step': self.step,
            'state_meta': state_meta(),
        }, buffer)
        payload = buffer.getvalue()
        if self._service is not None:
            self._service.set_params(payload)

    def _save(self):
        if not self.ckpt_path:
            return
        torch.save({
            'local_world_model': self.local_world.state_dict(),
            'global_world_model': self.global_world.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'opt_world': self.opt_world.state_dict(),
            'opt_actor': self.opt_actor.state_dict(),
            'opt_critic': self.opt_critic.state_dict(),
            'step': self.step,
            'state_meta': state_meta(),
            'max_agents': self.max_agents,
            'fairness_metric': (
                'r_fair_avg_throughput_credit' if self.per_agent_credit
                else 'r_fair_avg_throughput'),
            'fairness_weight': self.fairness_weight,
            'per_agent_credit': self.per_agent_credit,
            'team_alpha': self.team_alpha,
            'over_weight': self.over_weight,
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
    parser.add_argument('--port', type=int, default=6401)
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
