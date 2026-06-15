"""
learner.py — single, long-lived training process template (single-agent).

The orchestrator starts exactly one of these per run. It must:
  1. Print `SAO_MANAGER_KEY=<hex>` on stdout once the manager is up — the
     orchestrator blocks reading stdout until it sees this line, then hands the
     addr+key to every worker. Nothing works until this line is printed.
  2. Serve a multiprocessing BaseManager exposing push_exp / push_exp_batch
     (workers push transitions) and pull_params (workers pull fresh weights).
  3. Drain pushed experience into a replay buffer, run gradient updates, and
     periodically (a) broadcast the latest weights and (b) save checkpoints.

The IPC plumbing, the manager handshake, the heartbeat/drain loop, and the
checkpoint/broadcast cadence below are generic — every single-agent learner
shares them. The replay buffer and the update step (`_update`) are the
algorithm-specific parts and are marked `# TODO`; see td3/learner.py for a
complete, working reference.

Usage (started by the orchestrator):
  python olympus/algorithms/<your_alg>/learner.py --config olympus/config.yaml --port 6301
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
from collections import deque
from multiprocessing.managers import BaseManager

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# TODO: import your model contract.
from olympus.algorithms.template_single_agent.model import (
    Actor, Experience, STATE_DIM, ACTION_DIM,
    assert_checkpoint_state_compatible, actor_model_meta, model_state_meta,
)


# ── Manager IPC ───────────────────────────────────────────────────────────────
# Experience flows worker→learner through a bounded queue. Latest params flow
# learner→worker through a shared-memory blob (avoids serializing on every pull).

_exp_queue = multiprocessing.Queue(maxsize=200_000)
_PARAM_CACHE_BYTES = int(os.environ.get('SAO_PARAM_CACHE_BYTES', str(64 * 1024 * 1024)))
_param_lock = multiprocessing.Lock()
_param_size = multiprocessing.Value('i', 0)
_param_blob = multiprocessing.RawArray('B', _PARAM_CACHE_BYTES)


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
            return bytes(_param_blob[:n]) if n > 0 else None
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


def _episode_from_traj_id(traj_id) -> int:
    """Workers encode traj_id as '<algorithm>_<cport>_<episode>_<flow>'."""
    try:
        return int(str(traj_id).rsplit('_', 2)[-2])
    except (IndexError, TypeError, ValueError):
        return -1


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Per-trajectory FIFO store keyed by traj_id, capped at `max_transitions`.

    TODO: implement sampling for your algorithm. The td3 reference samples random
    fixed-length chunks per trajectory (for its recurrent critic); an MLP method
    can sample flat random transitions instead.
    """
    def __init__(self, max_transitions: int = 400_000):
        self._trajs = {}
        self._order = deque()
        self._total = 0
        self._cap = int(max_transitions)

    def push(self, exp) -> None:
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

    def sample(self, batch_size: int):
        raise NotImplementedError('TODO: implement replay sampling for your algorithm')


# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)
        self.batch_size = int(t.get('batch_size', 256))
        self.min_replay = int(t.get('min_replay', 5_000))
        self.replay_capacity = int(t.get('replay_capacity', 400_000))
        self.save_every = int(t.get('save_every', 500))
        self.save_every_episodes = max(0, int(t.get('save_every_episodes', 0) or 0))
        self.param_broadcast_every = int(t.get('param_broadcast_every', 50))
        self.updates_per_step = int(t.get('updates_per_step', 1))
        agent_cfg = cfg.get('agent', {}) or {}
        hidden = int(agent_cfg.get('hidden', 128))
        head_hidden_cfg = agent_cfg.get('head_hidden')
        head_hidden = int(head_hidden_cfg) if head_hidden_cfg is not None else None
        ckpt_path = t.get('checkpoint',
                          os.path.join(_PKG, 'data', 'checkpoints', 'template_model.pt'))
        resume_from = t.get('resume_from', '')
        load_path = resume_from or ckpt_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        # TODO: construct your networks/optimizers and warm-start from load_path.
        self.actor = Actor(STATE_DIM, hidden, head_hidden).to(self.device)
        if load_path and os.path.exists(load_path):
            try:
                ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_state_compatible(ckpt, source=load_path)
                self.actor.load_state_dict(ckpt['actor'], strict=False)
                self.step = int(ckpt.get('step', 0))
                print(f'[learner] loaded checkpoint step={self.step} path={load_path}', flush=True)
            except Exception as e:
                print(f'[learner] ckpt load failed ({e}), fresh start', flush=True)
                self.step = 0
        else:
            self.step = 0

        self.buf = ReplayBuffer(max_transitions=self.replay_capacity)
        self.ckpt_path = ckpt_path
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)

        # Optional learner-side training log.
        log_path = t.get('log_path', ckpt_path.replace('.pt', '_log.csv') if ckpt_path else '')
        self._csv_file, self._csv_writer = None, None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            existed = os.path.exists(log_path)
            self._csv_file = open(log_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow(['step', 'loss', 'buf', 'trajs'])

        self._mgr = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._mgr.start()
        print(f'[learner] manager on port {port}', flush=True)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.actor.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_hb = time.monotonic()
        total_drained = 0

        while True:
            # Drain everything the workers have pushed since the last pass.
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
                rate = total_drained / max(now - last_hb, 1e-6)
                print(f'[learner] hb buf={self.buf.size()}/{self.replay_capacity}  '
                      f'recv={total_drained} ({rate:.0f}/s)  trajs={self.buf.n_trajs()}',
                      flush=True)
                last_hb = now
                total_drained = 0

            if self.buf.size() < self.min_replay:
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

    def _update(self):
        # TODO: sample a batch, compute your loss(es), step the optimizer(s),
        # increment self.step, and (optionally) log to self._csv_writer.
        raise NotImplementedError('TODO: implement the gradient update for your algorithm')

    # ── Broadcast / save ─────────────────────────────────────────────────────

    def _broadcast(self):
        """Publish the latest deployable weights for workers to pull. The key
        names here must match what the worker reads in pull_params handling
        (`actor_state_dict`, `step`)."""
        buf = io.BytesIO()
        torch.save({
            'actor_state_dict': {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'step': self.step,
            'model_meta': actor_model_meta(self.actor),
            'state_meta': model_state_meta(),
        }, buf)
        _set_latest_params(buf.getvalue())

    def _checkpoint_payload(self) -> dict:
        # TODO: include every tensor you need to resume (targets, optimizers …).
        return {
            'actor': self.actor.state_dict(),
            'step': self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'model_meta': actor_model_meta(self.actor),
            'state_meta': model_state_meta(),
        }

    def _episode_ckpt_path(self, episode_count: int) -> str:
        base, ext = os.path.splitext(self.ckpt_path)
        return f'{base}_ep{int(episode_count):06d}{ext or ".pt"}'

    def _save(self, path: str = None, reason: str = 'ckpt'):
        path = path or self.ckpt_path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self._checkpoint_payload(), path)
        print(f'[learner] saved {reason} step={self.step} '
              f'episodes={len(self._completed_episode_ids)} path={path}', flush=True)

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
        """Count finished episodes (terminal pushes) and drop periodic snapshots
        so the orchestrator's per-episode checkpoint copies have a source."""
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

    def stop(self):
        self._save()
        if self._csv_file:
            self._csv_file.close()
        self._mgr.shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--port', type=int, default=6301)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    authkey = secrets.token_bytes(16)
    # The orchestrator blocks on this exact line — print it before any slow init.
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
