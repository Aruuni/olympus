"""ORCA-repo-style TD3 learner.

This learner mirrors the public Soheil-ab/Orca training loop where practical in
the Olympus manager/worker architecture: flat replay transitions, CDQ twin
critics, MSE targets, actor update every train step by default, target-policy
noise std=0.1 clipped to 0.2, and Polyak target updates with tau=0.001.
"""

import argparse
import csv
import io
import multiprocessing
import os
import pickle
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

from olympus.algorithms.orca.model import (
    ACTION_DIM,
    STATE_DIM,
    Actor,
    Experience,
    TwinCritic,
    actor_arch_from_checkpoint,
    actor_loss,
    actor_model_meta,
    assert_checkpoint_state_compatible,
    critic_loss,
    model_state_meta,
    soft_update,
)


_exp_queue = multiprocessing.Queue(maxsize=200_000)
_PARAM_CACHE_BYTES = int(os.environ.get('SAO_PARAM_CACHE_BYTES', str(64 * 1024 * 1024)))
_param_lock = multiprocessing.Lock()
_param_size = multiprocessing.Value('i', 0)
_param_blob = multiprocessing.RawArray('B', _PARAM_CACHE_BYTES)


def _push_exp(exp):
    try:
        _exp_queue.put_nowait(exp)
    except Exception:
        pass


def _push_exp_batch(exps):
    for exp in exps:
        _push_exp(exp)


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
        print(f'[orca learner] param payload too large: {len(payload)} > '
              f'{_PARAM_CACHE_BYTES}', flush=True)
        return
    with _param_lock:
        _param_blob[:len(payload)] = payload
        _param_size.value = len(payload)


class _QueueManager(BaseManager):
    pass


_QueueManager.register('push_exp', callable=_push_exp)
_QueueManager.register('push_exp_batch', callable=_push_exp_batch)
_QueueManager.register('pull_params', callable=_pull_params)


def _manager_handle_error(self, c, msg):  # noqa: ARG001
    import io as _io
    import traceback as _tb
    buf = _io.StringIO()
    _tb.print_exc(file=buf)
    text = buf.getvalue()
    if not any(x in text for x in ('BrokenPipeError', 'ConnectionResetError',
                                   'EOFError', 'ConnectionRefusedError')):
        sys.stderr.write('[orca learner] manager server error:\n' + text)
        sys.stderr.flush()


from multiprocessing.managers import Server as _MgrServer
_MgrServer.handle_error = _manager_handle_error


def _episode_from_traj_id(traj_id) -> int:
    try:
        return int(str(traj_id).rsplit('_', 2)[-2])
    except (IndexError, TypeError, ValueError):
        return -1


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._buf = []
        self._next = 0
        self._traj_counts = defaultdict(int)

    def push(self, exp: Experience) -> None:
        if len(self._buf) < self._capacity:
            self._buf.append(exp)
            self._traj_counts[exp.traj_id] += 1
            return
        old = self._buf[self._next]
        self._traj_counts[old.traj_id] -= 1
        if self._traj_counts[old.traj_id] <= 0:
            del self._traj_counts[old.traj_id]
        self._buf[self._next] = exp
        self._traj_counts[exp.traj_id] += 1
        self._next = (self._next + 1) % self._capacity

    def size(self) -> int:
        return len(self._buf)

    def n_trajs(self) -> int:
        return len(self._traj_counts)

    def sample(self, batch_size: int):
        if not self._buf:
            return None
        n = min(int(batch_size), len(self._buf))
        idxes = np.random.randint(0, len(self._buf), size=n)
        batch = [self._buf[int(i)] for i in idxes]
        return {
            'state': np.stack([e.state for e in batch]).astype(np.float32),
            'action': np.asarray([e.action for e in batch], dtype=np.float32).reshape(-1, ACTION_DIM),
            'reward': np.asarray([e.reward for e in batch], dtype=np.float32),
            'next_state': np.stack([e.next_state for e in batch]).astype(np.float32),
            'done': np.asarray([float(e.done) for e in batch], dtype=np.float32),
        }


class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)
        self.batch_size = int(t.get('batch_size', 512))
        self.min_replay = int(t.get('min_replay', 200))
        self.replay_capacity = int(t.get('replay_capacity',
                                         t.get('memsize', 2_553_600)))
        self.gamma = float(t.get('gamma', 0.995))
        self.tau = float(t.get('target_tau', t.get('tau', 0.001)))
        self.lr_actor = float(t.get('lr_actor', t.get('lr_a', 1.0e-4)))
        self.lr_critic = float(t.get('lr_critic', t.get('lr_c', 1.0e-3)))
        self.actor_update_every = int(t.get('actor_update_every', 1))
        self.target_noise_std = float(t.get('target_noise_std', 0.1))
        self.target_noise_clip = float(t.get('target_noise_clip', 0.2))
        self.grad_clip = float(t.get('grad_clip', 0.0))
        self.save_every = int(t.get('save_every', 500))
        self.save_every_episodes = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 50))
        self.update_delay_ms = float(t.get('update_delay_ms',
                                           t.get('update_delay', 5.0)))
        self.updates_per_step = int(t.get('updates_per_step', 1))

        ckpt_path = t.get('checkpoint',
                          os.path.join(_PKG, 'data', 'checkpoints',
                                       'orca_cwnd_model.pt'))
        # envwrapper.py and d5.py keep replay_memory.pkl next to the model;
        # mirror that layout. SAO_ORCA_REPLAY_PATH overrides for tests.
        self.replay_pickle_path = (os.environ.get('SAO_ORCA_REPLAY_PATH') or
                                   (os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                                 'replay_memory.pkl')
                                    if ckpt_path else ''))
        self.persist_replay = bool(t.get('persist_replay_on_stop', True))
        resume_from = t.get('resume_from', '')
        agent_cfg = cfg.get('agent', {}) or {}
        hidden = int(agent_cfg.get('hidden', agent_cfg.get('h1_shape', 256)))
        head_hidden = int(agent_cfg.get('head_hidden',
                                        agent_cfg.get('h2_shape', 256)))
        load_path = resume_from or ckpt_path
        if resume_from and not os.path.exists(resume_from):
            raise SystemExit(f'[orca learner] resume checkpoint not found: {resume_from}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[orca learner] device={self.device}', flush=True)

        ckpt_to_load = None
        if load_path and os.path.exists(load_path):
            try:
                ckpt_to_load = torch.load(
                    load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_state_compatible(ckpt_to_load, source=load_path)
                hidden, head_hidden = actor_arch_from_checkpoint(
                    ckpt_to_load, hidden, head_hidden)
                print(f'[orca learner] using checkpoint actor config '
                      f'hidden={hidden} head_hidden={head_hidden}', flush=True)
            except Exception as e:
                print(f'[orca learner] ckpt load failed ({e}), fresh start', flush=True)
                ckpt_to_load = None

        self.actor = Actor(STATE_DIM, hidden, head_hidden).to(self.device)
        self.actor_target = Actor(STATE_DIM, hidden, head_hidden).to(self.device)
        self.critic = TwinCritic(STATE_DIM, hidden, head_hidden).to(self.device)
        self.critic_target = TwinCritic(STATE_DIM, hidden, head_hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.buf = ReplayBuffer(self.replay_capacity)
        if self.persist_replay and self.replay_pickle_path and os.path.exists(self.replay_pickle_path):
            try:
                with open(self.replay_pickle_path, 'rb') as fp:
                    restored = pickle.load(fp)
                if isinstance(restored, ReplayBuffer):
                    self.buf = restored
                    print(f'[orca learner] restored replay '
                          f'size={self.buf.size()} from {self.replay_pickle_path}',
                          flush=True)
            except Exception as e:
                print(f'[orca learner] replay restore failed ({e}); fresh buffer',
                      flush=True)
        self.step = 0
        self.ckpt_path = ckpt_path
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)

        if ckpt_to_load is not None:
            try:
                ckpt = ckpt_to_load
                self.actor.load_state_dict(ckpt['actor'], strict=False)
                self.critic.load_state_dict(ckpt['critic'], strict=False)
                self.actor_target.load_state_dict(
                    ckpt.get('actor_target', ckpt['actor']), strict=False)
                self.critic_target.load_state_dict(
                    ckpt.get('critic_target', ckpt['critic']), strict=False)
                if 'opt_actor' in ckpt:
                    self.opt_actor.load_state_dict(ckpt['opt_actor'])
                if 'opt_critic' in ckpt:
                    self.opt_critic.load_state_dict(ckpt['opt_critic'])
                self.opt_actor.param_groups[0]['lr'] = self.lr_actor
                self.opt_critic.param_groups[0]['lr'] = self.lr_critic
                self.step = int(ckpt.get('step', 0))
                src = 'resume checkpoint' if resume_from else 'checkpoint'
                print(f'[orca learner] loaded {src} step={self.step} path={load_path}',
                      flush=True)
            except Exception as e:
                print(f'[orca learner] ckpt load failed ({e}), fresh start', flush=True)

        log_path = t.get('log_path',
                         ckpt_path.replace('.pt', '_log.csv') if ckpt_path else '')
        self._csv_file, self._csv_writer = None, None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            existed = os.path.exists(log_path)
            self._csv_file = open(log_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow([
                    'step', 'critic_loss', 'actor_loss', 'q1_mean', 'q2_mean',
                    'td_abs', 'a_mean', 'a_abs', 'buf', 'trajs', 'lr_a', 'lr_c',
                ])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[orca learner] manager on port {port}', flush=True)

    def run(self):
        self.actor.eval()
        self.critic.eval()
        self._broadcast()
        print('[orca learner] running', flush=True)

        last_hb = time.monotonic()
        next_update = time.monotonic()
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

            now = time.monotonic()
            if now - last_hb >= 5.0:
                dt = now - last_hb
                rate = total_drained / max(dt, 1e-6)
                print(f'[orca learner] hb buf={self.buf.size()}/{self.replay_capacity}  '
                      f'recv={total_drained} ({rate:.0f}/s)  trajs={self.buf.n_trajs()}',
                      flush=True)
                last_hb = now
                total_drained = 0

            if self.buf.size() < self.min_replay:
                if drained == 0:
                    time.sleep(0.05)
                continue

            if now < next_update:
                if drained == 0:
                    time.sleep(min(0.01, next_update - now))
                continue

            for _ in range(self.updates_per_step):
                prev_step = self.step
                self._td3_update()
                if self.step != prev_step:
                    self._maybe_save_latest_checkpoint()
            next_update = time.monotonic() + self.update_delay_ms / 1000.0
            if self.step % self.param_broadcast_every == 0:
                self._broadcast()

    def _td3_update(self):
        batch = self.buf.sample(self.batch_size)
        if batch is None:
            return

        dev = self.device
        s = torch.from_numpy(batch['state']).to(dev)
        a = torch.from_numpy(batch['action']).to(dev).squeeze(-1)
        r = torch.from_numpy(batch['reward']).to(dev)
        s2 = torch.from_numpy(batch['next_state']).to(dev)
        d = torch.from_numpy(batch['done']).to(dev)

        self.actor.train()
        self.critic.train()
        self.actor_target.eval()
        self.critic_target.eval()

        c_info = critic_loss(
            self.critic,
            self.actor_target,
            self.critic_target,
            s_batch=s,
            a_batch=a,
            r_batch=r,
            s2_batch=s2,
            d_batch=d,
            gamma=self.gamma,
            target_noise_std=self.target_noise_std,
            target_noise_clip=self.target_noise_clip,
        )
        if not torch.isfinite(c_info.loss):
            print('[orca learner] non-finite critic loss; skip update', flush=True)
            self.actor.eval()
            self.critic.eval()
            return

        self.opt_critic.zero_grad()
        c_info.loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()

        self.step += 1
        a_info = None
        if self.step % max(self.actor_update_every, 1) == 0:
            a_info = actor_loss(self.actor, self.critic, s_batch=s)
            if torch.isfinite(a_info.loss):
                self.opt_actor.zero_grad()
                a_info.loss.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.opt_actor.step()
            else:
                a_info = None

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)
        self.actor.eval()
        self.critic.eval()

        if self.step % 10 == 0:
            lr_a = self.opt_actor.param_groups[0]['lr']
            lr_c = self.opt_critic.param_groups[0]['lr']
            a_loss = a_info.loss.item() if a_info is not None else float('nan')
            a_mean = a_info.a_mean if a_info is not None else float('nan')
            a_abs = a_info.a_abs if a_info is not None else float('nan')
            print(
                f'[orca learner] step={self.step:6d}  '
                f'critic={c_info.loss.item():.4f}  actor={a_loss:.4f}  '
                f'(q1={c_info.q1_mean:.3f} q2={c_info.q2_mean:.3f} '
                f'|td|={c_info.td_abs:.3f} a_mean={a_mean:.3f} '
                f'a_abs={a_abs:.3f}) buf={self.buf.size()} '
                f'trajs={self.buf.n_trajs()}',
                flush=True,
            )
            if self._csv_writer:
                self._csv_writer.writerow([
                    self.step,
                    f'{c_info.loss.item():.6f}',
                    f'{a_loss:.6f}',
                    f'{c_info.q1_mean:.6f}',
                    f'{c_info.q2_mean:.6f}',
                    f'{c_info.td_abs:.6f}',
                    f'{a_mean:.6f}',
                    f'{a_abs:.6f}',
                    self.buf.size(),
                    self.buf.n_trajs(),
                    f'{lr_a:.2e}',
                    f'{lr_c:.2e}',
                ])

    def _broadcast(self):
        buf = io.BytesIO()
        torch.save({
            'actor_state_dict': {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'step': self.step,
            'model_meta': actor_model_meta(self.actor),
            'state_meta': model_state_meta(),
        }, buf)
        _set_latest_params(buf.getvalue())

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

    def _checkpoint_payload(self) -> dict:
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'opt_actor': self.opt_actor.state_dict(),
            'opt_critic': self.opt_critic.state_dict(),
            'step': self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'model_meta': actor_model_meta(self.actor),
            'state_meta': model_state_meta(),
        }

    def _save(self, path: str = None, reason: str = 'ckpt'):
        path = path or self.ckpt_path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self._checkpoint_payload(), path)
        print(f'[orca learner] saved {reason} step={self.step} '
              f'episodes={len(self._completed_episode_ids)} path={path}', flush=True)

    def _save_replay(self) -> None:
        if not (self.persist_replay and self.replay_pickle_path):
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.replay_pickle_path)),
                        exist_ok=True)
            tmp = self.replay_pickle_path + '.tmp'
            with open(tmp, 'wb') as fp:
                pickle.dump(self.buf, fp, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.replay_pickle_path)
            print(f'[orca learner] saved replay size={self.buf.size()} '
                  f'path={self.replay_pickle_path}', flush=True)
        except Exception as e:
            print(f'[orca learner] replay save failed: {e}', flush=True)

    def stop(self):
        self._save()
        # d5.py learner_killer.handler_term pickles replay_memory.pkl on
        # SIGTERM so the next training session can resume the buffer.
        self._save_replay()
        if self._csv_file:
            self._csv_file.close()
        self._mgr.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--port', type=int, default=6301)
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
    signal.signal(signal.SIGINT, _sigterm)

    try:
        learner.run()
    except KeyboardInterrupt:
        learner.stop()
