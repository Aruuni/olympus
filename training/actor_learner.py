"""
training/actor_learner.py — Central Option-Critic learner

Runs as a standalone process (started by orchestrator.py before any
environments).  Workers connect to this process via a multiprocessing
Manager to push experiences and pull updated weights.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Why a separate process and not threads?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Python threads all share one GIL.  If the learner and workers ran as
  threads in the same process, the learner's backward() call would block
  every worker doing inference — and vice versa.

  With separate OS processes:
    - Learner has its own GIL → backward() does not stall workers
    - Workers have their own GILs → inference does not stall the learner
    - Communication is via multiprocessing Manager (OS IPC), which
      releases the GIL on both sides during the transfer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Learner loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Drain exp_queue into a replay buffer
  2. When buffer ≥ batch_size: sample a batch, compute loss, step optimizer
  3. Every param_broadcast_every steps: put state_dict into param_queue
     so workers pull it on their next dwell boundary

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Start learner (orchestrator does this automatically):
  python training/actor_learner.py --port 5999 --key <hex> [--checkpoint path]

  # Workers find it via:
  OC_MANAGER_ADDR=localhost:5999
  OC_MANAGER_KEY=<hex>
"""

import argparse
import collections
import csv
import multiprocessing
import os
import queue
import random
import secrets
import signal
import sys
import time
from multiprocessing.managers import BaseManager
from collections import deque

import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.model import (
    OptionCriticNet, Experience, compute_loss, soft_update_target,
    STATE_DIM, N_OPTIONS, ARM_NAMES,
)

# ── Manager definition ────────────────────────────────────────────────────────
#
# Queues are multiprocessing.Queue objects created in the learner process.
# BaseManager.start() forks a server subprocess — because of fork(), both the
# learner and the server subprocess share the same underlying OS pipe fds.
# Workers put experiences via proxy → server calls q.put() on its copy →
# data travels through the shared OS pipe → learner reads directly with q.get().
# No client-proxy needed on the learner side, no resource_sharer involvement.

_exp_queue   = multiprocessing.Queue(maxsize=50_000)
_param_queue = multiprocessing.Queue(maxsize=200)

# Wrapper functions executed on the manager SERVER side.
# Workers call manager.push_exp(exp) / manager.pull_params().
# This avoids pickling Queue objects over the network, which would
# trigger resource_sharer fd-sharing with a mismatched authkey.
def _push_exp(exp):
    try:
        _exp_queue.put_nowait(exp)
    except Exception:
        pass  # queue full — learner behind; drop

def _pull_params():
    try:
        return _param_queue.get_nowait()
    except Exception:
        return None


class _QueueManager(BaseManager):
    pass

_QueueManager.register('push_exp',    callable=_push_exp)
_QueueManager.register('pull_params', callable=_pull_params)

# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Simple circular experience buffer.

    We use a deque (not a tensor buffer) because experiences arrive as
    Python objects from worker processes.  For the small batch sizes here
    the overhead is negligible.
    """
    def __init__(self, capacity: int = 100_000):
        self._buf = deque(maxlen=capacity)

    def push(self, exp: Experience) -> None:
        self._buf.append(exp)

    def sample(self, n: int) -> list:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)

# ── Learner ───────────────────────────────────────────────────────────────────

class Learner:
    def __init__(
        self,
        port:                  int   = 5999,
        authkey:               bytes = b'',
        batch_size:            int   = 256,
        lr:                    float = 3e-4,
        lr_min:                float = 1e-5,
        lr_decay_steps:        int   = 50_000,
        replay_capacity:       int   = 100_000,
        min_replay:            int   = 512,
        param_broadcast_every: int   = 50,
        checkpoint_path:       str   = '',
        save_every:            int   = 500,
        gamma:                 float = 0.99,
        target_tau:            float = 0.005,
    ):
        self.batch_size            = batch_size
        self.min_replay            = min_replay
        self.param_broadcast_every = param_broadcast_every
        self.checkpoint_path       = checkpoint_path
        self.save_every            = save_every
        self.gamma                 = gamma
        self.target_tau            = target_tau

        self.net        = OptionCriticNet(state_dim=STATE_DIM, n_options=N_OPTIONS)
        # Target network — frozen copy used for stable TD bootstrapping.
        # Kept at eval mode always; updated via Polyak averaging each train step.
        self.target_net = OptionCriticNet(state_dim=STATE_DIM, n_options=N_OPTIONS)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.opt        = optim.Adam(self.net.parameters(), lr=lr)
        # Cosine annealing: LR decays smoothly from lr → lr_min over
        # lr_decay_steps, then holds at lr_min.  This lets the model make
        # large updates early and fine-tune once returns start converging.
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=lr_decay_steps, eta_min=lr_min)

        self.buf    = ReplayBuffer(replay_capacity)
        self.step   = 0

        # Rolling stats over the last 200 training steps
        self._loss_window: deque = deque(maxlen=200)  # LossInfo per step
        self._reward_window: deque = deque(maxlen=200)  # (option_idx, reward) per exp
        self._arm_counts: list = [0] * N_OPTIONS

        # CSV log — written alongside the checkpoint
        self._csv_path = (
            os.path.splitext(checkpoint_path)[0] + '_log.csv'
            if checkpoint_path else ''
        )
        self._csv_file = None
        self._csv_writer = None
        if self._csv_path:
            existed = os.path.exists(self._csv_path)
            self._csv_file = open(self._csv_path, 'a', newline='', buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow([
                    'step', 'loss', 'loss_value', 'loss_policy',
                    'entropy', 'loss_term', 'loss_dwell',
                    'reward_mean', 'reward_min', 'reward_max', 'buf_size',
                ] + [f'arm_{n}' for n in ARM_NAMES])

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.net.load_state_dict(ckpt['model'])
            if 'target' in ckpt:
                self.target_net.load_state_dict(ckpt['target'])
            else:
                self.target_net.load_state_dict(ckpt['model'])
            self.opt.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            self.step = ckpt.get('step', 0)
            print(f'[learner] loaded checkpoint from {checkpoint_path} '
                  f'(step {self.step})', flush=True)

        self._manager = _QueueManager(address=('0.0.0.0', port), authkey=authkey)
        self._manager.start()
        # Direct references to the module-level queues — same OS pipe fds as
        # the server subprocess (fork shares them).
        self._exp_queue   = _exp_queue
        self._param_queue = _param_queue
        print(f'[learner] manager listening on port {port}', flush=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _drain_exp_queue(self) -> int:
        """Pull all available experiences into replay buffer."""
        n = 0
        try:
            while True:
                exp = self._exp_queue.get_nowait()
                self.buf.push(exp)
                self._reward_window.append((exp.option_idx, exp.reward))
                self._arm_counts[exp.option_idx] += 1
                n += 1
        except Exception:
            pass
        return n

    def _broadcast_params(self) -> None:
        """Push current model weights into param_queue for workers to pull."""
        payload = {
            'state_dict': {k: v.cpu() for k, v in self.net.state_dict().items()},
            'step': self.step,
        }
        # Drop old entries so workers always get the latest
        try:
            while True:
                self._param_queue.get_nowait()
        except Exception:
            pass
        try:
            self._param_queue.put_nowait(payload)
        except Exception:
            pass

    def _train_step(self):
        batch = self.buf.sample(self.batch_size)
        self.net.train()
        self.opt.zero_grad()
        info = compute_loss(self.net, batch, gamma=self.gamma,
                            target_net=self.target_net)
        info.total.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.opt.step()
        self.scheduler.step()
        self.net.eval()
        # Polyak-average the target network toward the online network
        soft_update_target(self.net, self.target_net, tau=self.target_tau)
        self._loss_window.append(info)
        return info, batch

    def _log(self, info, batch, new_exp: int) -> None:
        rewards = [e.reward for e in batch]
        r_mean = sum(rewards) / len(rewards)
        r_min  = min(rewards)
        r_max  = max(rewards)

        # Per-arm reward from rolling window
        arm_rewards: dict = collections.defaultdict(list)
        for opt_idx, r in self._reward_window:
            arm_rewards[opt_idx].append(r)
        arm_r_str = '  '.join(
            f'{ARM_NAMES[i]}={sum(v)/len(v):+.2f}' if (v := arm_rewards.get(i))
            else f'{ARM_NAMES[i]}=  n/a'
            for i in range(N_OPTIONS)
        )

        # Total arm selection counts → percentage
        total_sel = sum(self._arm_counts) or 1
        arm_pct = '  '.join(
            f'{ARM_NAMES[i]}={100*self._arm_counts[i]/total_sel:.0f}%'
            for i in range(N_OPTIONS)
        )

        cur_lr = self.scheduler.get_last_lr()[0]
        print(
            f'[learner] step={self.step:6d}  '
            f'loss={info.total.item():.4f}  '
            f'(val={info.value:.3f}  pol={info.policy:.3f}  '
            f'ent={info.entropy:.3f}  term={info.term:.3f}  dwell={info.dwell:.3f})  '
            f'buf={len(self.buf)}  new={new_exp}  lr={cur_lr:.2e}  '
            f'r=[{r_min:+.2f},{r_mean:+.2f},{r_max:+.2f}]',
            flush=True,
        )
        print(f'[learner]        arm_r:  {arm_r_str}', flush=True)
        print(f'[learner]        arm_%:  {arm_pct}', flush=True)

        if self._csv_writer:
            self._csv_writer.writerow([
                self.step,
                f'{info.total.item():.6f}',
                f'{info.value:.6f}',
                f'{info.policy:.6f}',
                f'{info.entropy:.6f}',
                f'{info.term:.6f}',
                f'{info.dwell:.6f}',
                f'{r_mean:.6f}',
                f'{r_min:.6f}',
                f'{r_max:.6f}',
                len(self.buf),
            ] + [self._arm_counts[i] for i in range(N_OPTIONS)])

    def _save(self) -> None:
        if not self.checkpoint_path:
            return
        torch.save({
            'model':      self.net.state_dict(),
            'target':     self.target_net.state_dict(),
            'optimizer':  self.opt.state_dict(),
            'scheduler':  self.scheduler.state_dict(),
            'step':       self.step,
        }, self.checkpoint_path)
        print(f'[learner] saved checkpoint step={self.step}', flush=True)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        print('[learner] running', flush=True)
        self.net.eval()

        # Broadcast initial weights immediately so workers don't have to wait
        self._broadcast_params()

        while True:
            new = self._drain_exp_queue()

            if len(self.buf) >= self.min_replay:
                info, batch = self._train_step()
                self.step += 1

                if self.step % 10 == 0:
                    self._log(info, batch, new)

                if self.step % self.param_broadcast_every == 0:
                    self._broadcast_params()

                if self.step % self.save_every == 0:
                    self._save()
            else:
                # Not enough data yet — just wait
                if new == 0:
                    time.sleep(0.05)

    def stop(self) -> None:
        self._save()
        if self._csv_file:
            self._csv_file.close()
        self._manager.shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    _here        = os.path.dirname(os.path.abspath(__file__))
    _default_cfg = os.path.join(_here, 'train_config.yaml')

    parser = argparse.ArgumentParser(description='Olympus Option-Critic learner')
    parser.add_argument('--config',     default=_default_cfg,
                        help='Path to train_config.yaml')
    parser.add_argument('--port',       type=int,   default=None,
                        help='Manager port (overrides config)')
    parser.add_argument('--key',        type=str,   default='',
                        help='Authkey as hex string (generated if empty)')
    args = parser.parse_args()

    cfg      = _load_config(args.config)
    t_cfg    = cfg.get('training', {})
    l_cfg    = cfg.get('learner',  {})

    port       = args.port or int(l_cfg.get('port', 5999))
    authkey    = bytes.fromhex(args.key) if args.key else secrets.token_bytes(32)
    # Print the key so the orchestrator can pass it to workers
    print(f'OC_MANAGER_KEY={authkey.hex()}', flush=True)

    learner = Learner(
        port=port,
        authkey=authkey,
        batch_size=int(t_cfg.get('batch_size', 256)),
        lr=float(t_cfg.get('lr', 3e-4)),
        lr_min=float(t_cfg.get('lr_min', 1e-5)),
        lr_decay_steps=int(t_cfg.get('lr_decay_steps', 50_000)),
        replay_capacity=int(t_cfg.get('replay_capacity', 100_000)),
        min_replay=int(t_cfg.get('min_replay', 512)),
        param_broadcast_every=int(t_cfg.get('param_broadcast_every', 50)),
        checkpoint_path=t_cfg.get('checkpoint', 'training/oc_model.pt'),
        save_every=int(t_cfg.get('save_every', 500)),
        gamma=float(t_cfg.get('gamma', 0.99)),
        target_tau=float(t_cfg.get('target_tau', 0.005)),
    )

    def _on_signal(sig, _frame):
        print(f'[learner] signal {sig}, saving and exiting', flush=True)
        learner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    learner.run()
