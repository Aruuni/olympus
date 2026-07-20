"""
learner.py — single, long-lived training process template (multi-agent / CTDE).

Same lifecycle contract as the single-agent learner — print
`SAO_MANAGER_KEY=<hex>` first, serve a BaseManager, drain experience, update,
broadcast/save — with two multi-agent additions:

  1. push_bootstrap: workers send a terminal value estimate per trajectory for
     GAE (on-policy methods only).
  2. Joint regrouping: experiences are grouped by (group_id, group_step) so the
     simultaneous flows on one bottleneck form a single centralized training
     sample. This is also where the team/fairness reward is assembled from the
     sibling `throughput` fields.

The model size is taken from max_agents = sweep.flows (the orchestrator surfaces
sweep.flows into the resolved config) so the joint model is wide enough for the
largest episode. The buffer/extract and the update step are the algorithm-specific
parts (`# TODO`); see ma_td3/learner.py (off-policy CTDE) for a complete reference.

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
from collections import defaultdict
from multiprocessing.managers import BaseManager

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# TODO: import your model contract.
from olympus.algorithms.template_multi_agent.model import (
    Policy, Experience, STATE_DIM, ACTION_DIM,
    assert_checkpoint_action_compatible,
)


# ── Manager IPC ───────────────────────────────────────────────────────────────

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


def _max_flow_count(spec, default=2) -> int:
    """Largest flow count in a sweep spec ('2', '2-4', [2,3,4]) → max_agents."""
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
    return max(max(1, int(v)) for v in vals)


# ── Rollout buffer (per-agent trajectories, group-aware) ──────────────────────

class RolloutBuffer:
    """On-policy store, cleared after every update. Trajectories are keyed by
    traj_id (per-agent, per-episode); group_id+group_step let simultaneous flows
    be assembled into joint timesteps.

    TODO: implement extract() to (a) regroup experiences on (group_id,
    group_step) into joint (N, …) tensors, (b) add the team/fairness reward from
    the sibling throughput fields, and (c) compute GAE using the per-traj
    bootstrap values. An off-policy MARL method replaces this with a replay
    buffer and random sampling instead.
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

    def clear(self):
        self._trajs.clear()
        self._bootstrap.clear()
        self._total = 0

    def extract(self):
        raise NotImplementedError('TODO: regroup by (group_id, group_step) into joint samples')


# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, cfg: dict, port: int, authkey: bytes):
        t = cfg.get('training', cfg)
        agent_cfg = cfg.get('agent', {}) or {}
        self.rollout_steps = int(t.get('rollout_steps', 4_096))
        self.save_every = int(t.get('save_every', 50))
        self.save_every_episodes = max(0, int(t.get('save_every_episodes', 0) or 0))
        hidden = int(agent_cfg.get('hidden', 128))
        n_layers = int(agent_cfg.get('n_layers', 2))
        n_heads = int(agent_cfg.get('n_heads', 4))
        # Joint model width comes from the environment's flow sweep, not training.
        self.max_agents = _max_flow_count((cfg.get('sweep', {}) or {}).get('flows'))
        print(f'[learner] max_agents={self.max_agents} (from sweep.flows)', flush=True)

        ckpt_path = t.get('checkpoint',
                          os.path.join(_PKG, 'data', 'checkpoints', 'template_ma_model.pt'))
        resume_from = t.get('resume_from', '')
        load_path = resume_from or ckpt_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[learner] device={self.device}', flush=True)

        # TODO: construct the parameter-shared policy + centralized critic and
        # optimizer, then warm-start from load_path.
        self.net = Policy(STATE_DIM, hidden, n_layers=n_layers, n_heads=n_heads).to(self.device)
        self.step = 0
        if load_path and os.path.exists(load_path):
            try:
                ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
                assert_checkpoint_action_compatible(ckpt, source=load_path)
                self.net.load_state_dict(ckpt['model'], strict=False)
                self.step = int(ckpt.get('step', 0))
                print(f'[learner] loaded checkpoint step={self.step} path={load_path}', flush=True)
            except Exception as e:
                print(f'[learner] ckpt load failed ({e}), fresh start', flush=True)

        self.buf = RolloutBuffer()
        self.ckpt_path = ckpt_path
        self._completed_episode_ids = set()
        self._next_episode_ckpt = self.save_every_episodes
        if self.ckpt_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.ckpt_path)), exist_ok=True)

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

    # ── Main loop (on-policy: collect a rollout, update, clear, broadcast) ────

    def run(self):
        self.net.eval()
        self._broadcast()
        print('[learner] running', flush=True)

        last_hb = time.monotonic()
        while True:
            # Drain experience and bootstrap values.
            drained = 0
            try:
                while True:
                    exp = _exp_queue.get_nowait()
                    self.buf.push(exp)
                    self._note_episode_done(exp)
                    drained += 1
            except Exception:
                pass
            try:
                while True:
                    tid, v = _bootstrap_queue.get_nowait()
                    self.buf.push_bootstrap(tid, v)
            except Exception:
                pass

            now = time.monotonic()
            if now - last_hb >= 5.0:
                print(f'[learner] hb buf={self.buf.size()}  trajs={self.buf.n_trajs()}',
                      flush=True)
                last_hb = now

            if self.buf.size() < self.rollout_steps:
                if drained == 0:
                    time.sleep(0.05)
                continue

            self._update()              # consumes the rollout
            self.buf.clear()
            self._broadcast()
            self._maybe_save_latest_checkpoint()

    def _update(self):
        # TODO: batch = self.buf.extract(); compute the CTDE loss (policy +
        # centralized value + entropy), step the optimizer, increment self.step,
        # and (optionally) log to self._csv_writer.
        raise NotImplementedError('TODO: implement the multi-agent update for your algorithm')

    # ── Broadcast / save ─────────────────────────────────────────────────────

    def _broadcast(self):
        """Publish the latest policy. The key here (`state_dict`) must match what
        the worker reads when handling pull_params."""
        buf = io.BytesIO()
        torch.save({
            'state_dict': {k: v.cpu() for k, v in self.net.state_dict().items()},
            'step': self.step,
        }, buf)
        try:
            while True:
                _param_queue.get_nowait()   # keep only the freshest
        except Exception:
            pass
        try:
            _param_queue.put_nowait(buf.getvalue())
        except Exception:
            pass

    def _checkpoint_payload(self) -> dict:
        # `model` is the key the worker reads when warm-starting from a checkpoint.
        return {
            'model': self.net.state_dict(),
            'step': self.step,
            'episodes_completed': len(self._completed_episode_ids),
            'max_agents': self.max_agents,
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


def _episode_from_traj_id(traj_id) -> int:
    """Workers tag traj_id 'template_slot{i}_ep{episode}_a{agent}_f{flow}'.
    Recover the integer episode index for per-episode checkpointing."""
    try:
        for tok in str(traj_id).split('_'):
            if tok.startswith('ep') and tok[2:].isdigit():
                return int(tok[2:])
    except (TypeError, ValueError):
        pass
    return -1


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
