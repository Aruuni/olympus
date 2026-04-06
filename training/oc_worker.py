"""
training/oc_worker.py — per-flow Option-Critic worker (inline learner)

Spawned by oc_listener.c for each TCP flow.  Runs inference, collects
experiences into a local replay buffer, and trains the model in-process.
Checkpoint is loaded at start and saved periodically and on exit.

No IPC to a separate learner process — training happens right here.

Environment variables set by orchestrator / oc_listener:
  OC_STATE_FD    — pipe fd (C writes oc_state_t every scan_ms)
  OC_ACTION_FD   — pipe fd (Python writes oc_action_t)
  OC_FLOW_FD     — TCP socket fd (DeepCC cwnd control, -1 if unused)
  OC_FLOW_ID     — monotonic flow identifier
  OC_CPORT       — experiment identifier (iperf3 client source port)
  OC_CHECKPOINT  — path to checkpoint .pt file (load on start, save on exit)
  OC_BATCH_SIZE  — training batch size        (default: 32)
  OC_MIN_REPLAY  — min buffer size to start   (default: 64)
  OC_LR          — Adam learning rate         (default: 3e-4)
  OC_GAMMA       — TD discount factor         (default: 0.99)
  OC_SAVE_EVERY  — save checkpoint every N steps (default: 20)
  OC_LOG_CSV     — optional path to append CSV training log rows
  OC_STATE_LOG   — optional path for per-frame state CSV (same format as
                   oc_inference_worker; used to generate per-episode plots)
"""

import collections
import csv
import os
import random
import struct
import subprocess
import sys
import time

import numpy as np
import torch
import torch.optim as optim

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _ROOT)

from training.model import (
    OptionCriticNet, Experience, compute_loss, soft_update_target,
    normalize_state, arm_id_to_idx, N_OPTIONS, STATE_DIM, ARM_NAMES,
)
from training.reward import make_reward_calc

# ── Wire format (must match oc_listener.c) ────────────────────────────────────
STATE_FMT  = '<8IQ'
STATE_SIZE = struct.calcsize(STATE_FMT)   # 44 bytes
ACTION_FMT = '<II'                        #  8 bytes

ASTRAEA_ARM_ID = 13

# ── Astraea background service ────────────────────────────────────────────────

class AstraeaBackground:
    """
    Spawns astraea_service.py as a background subprocess for a single flow.

    The service runs TF inference every 20 ms.  A control pipe gates whether
    cwnd writes are actually applied:
      - write b'1' → astraea applies cwnd (we are on the Astraea arm)
      - write b'0' → inference continues but cwnd is not touched (other arms)
    """

    def __init__(self, flow_fd: int, flow_id: int) -> None:
        python_bin  = os.environ.get('ASTRAEA_PYTHON', sys.executable)
        script_path = os.environ.get(
            'ASTRAEA_SCRIPT',
            os.path.join(_ROOT, 'astraea_service.py'))
        config = os.environ.get(
            'ASTRAEA_CONFIG',
            os.path.join(_ROOT, 'astraea', 'astraea.json'))
        model = os.environ.get(
            'ASTRAEA_MODEL',
            os.path.join(_ROOT, 'astraea', 'models', 'exported'))

        ctrl_rd, ctrl_wr = os.pipe()
        self._ctrl_wr = ctrl_wr

        env = dict(os.environ)
        env.update({
            'ASTRAEA_FLOW_ID':   str(flow_id),
            'ASTRAEA_FLOW_FD':   str(flow_fd),
            'ASTRAEA_CONFIG':    config,
            'ASTRAEA_MODEL':     model,
            'ASTRAEA_CONTROL_FD': str(ctrl_rd),
        })

        self._proc = subprocess.Popen(
            [python_bin, script_path],
            env=env,
            pass_fds=(ctrl_rd, flow_fd),
        )
        os.close(ctrl_rd)   # child owns the read end now

        # astraea_service.__main__ calls set_control(True) after startup;
        # send '0' immediately so we start disabled (we're on CUBIC first).
        self.set_control(False)
        print(f'[astraea] background service started  pid={self._proc.pid}', flush=True)

    def set_control(self, active: bool) -> None:
        """Enable (True) or disable (False) cwnd writes in the service."""
        try:
            os.write(self._ctrl_wr, b'1' if active else b'0')
        except OSError:
            pass

    def stop(self) -> None:
        try:
            os.close(self._ctrl_wr)
        except OSError:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        print(f'[astraea] background service stopped', flush=True)


def _start_astraea(flow_fd: int, flow_id: int):
    """Start AstraeaBackground if the flow fd is valid and env is configured."""
    if flow_fd < 0:
        return None
    if not os.path.exists(os.environ.get('ASTRAEA_SCRIPT',
                          os.path.join(_ROOT, 'astraea_service.py'))):
        print('[astraea] astraea_service.py not found — skipping', flush=True)
        return None
    try:
        return AstraeaBackground(flow_fd, flow_id)
    except Exception as e:
        print(f'[astraea] failed to start background service: {e}', flush=True)
        return None

# ── I/O ───────────────────────────────────────────────────────────────────────

def _read_exact(fd: int, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            raise EOFError('state pipe closed')
        buf += chunk
    return buf

def _send_action(fd: int, arm_id: int, dwell_ms: int) -> None:
    os.write(fd, struct.pack(ACTION_FMT, arm_id, dwell_ms))

# ── Checkpoint ────────────────────────────────────────────────────────────────

def _load_or_init(path: str, lr: float):
    net = OptionCriticNet(state_dim=STATE_DIM, n_options=N_OPTIONS)
    opt = optim.Adam(net.parameters(), lr=lr)
    step = 0
    if path and os.path.exists(path):
        try:
            ckpt = torch.load(path, weights_only=False)
            net.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['optimizer'])
            step = ckpt.get('step', 0)
            print(f'[worker] loaded checkpoint  step={step}  {path}', flush=True)
        except Exception as e:
            print(f'[worker] checkpoint load failed ({e}), starting fresh', flush=True)
    return net, opt, step

def _save(path: str, net, opt, step: int) -> None:
    if not path:
        return
    try:
        tmp = path + '.tmp'
        torch.save({'model': net.state_dict(),
                    'optimizer': opt.state_dict(),
                    'step': step}, tmp)
        os.replace(tmp, path)
        print(f'[worker] checkpoint saved  step={step}', flush=True)
    except Exception as e:
        print(f'[worker] checkpoint save failed: {e}', flush=True)

# ── Training ──────────────────────────────────────────────────────────────────

def _train_step(net, opt, replay, batch_size: int, gamma: float,
                c_entropy: float = 0.01, c_term: float = 0.1, c_dwell: float = 0.1,
                target_net=None):
    # Balanced per-arm sampling: draw equal quota from each arm that has data.
    # Prevents the dominant arm (e.g. bbr3) from crowding out others in the
    # policy gradient update, which is the primary cause of policy collapse.
    by_arm = collections.defaultdict(list)
    for e in replay:
        by_arm[e.option_idx].append(e)
    arms_present = [exps for exps in by_arm.values() if exps]
    per_arm = max(1, batch_size // len(arms_present))
    batch = []
    for exps in arms_present:
        batch.extend(random.sample(exps, min(per_arm, len(exps))))
    # Top up to batch_size with random samples if needed
    if len(batch) < batch_size:
        batch += random.sample(list(replay), min(batch_size - len(batch), len(replay)))

    net.train()
    info = compute_loss(net, batch, gamma=gamma,
                        c_entropy=c_entropy, c_term=c_term, c_dwell=c_dwell,
                        target_net=target_net)
    opt.zero_grad()
    info.total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    net.eval()
    return info

# ── Astraea cwnd hint ─────────────────────────────────────────────────────────
# The real Astraea cwnd control requires the TF inference service from
# venv_astraea.  Until that service is bridged in, DeepCC is disabled for the
# Astraea arm (see oc_listener.c) so the kernel falls back to its own cwnd
# management.  The stub below is intentionally a no-op.


# ── CSV logging ───────────────────────────────────────────────────────────────

_CSV_HEADER = ['step', 'loss', 'value', 'policy', 'entropy', 'term', 'dwell',
               'cport', 'flow_id']

def _open_csv(path: str):
    if not path:
        return None, None
    # Always write header: truncate if file is empty or missing, append otherwise
    # but verify header matches so a stale file doesn't silently corrupt the log.
    write_header = True
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as _f:
            first = _f.readline().strip()
        write_header = (first != ','.join(_CSV_HEADER))
        if write_header:
            # stale header — truncate and start fresh
            open(path, 'w').close()
    f = open(path, 'a', newline='')
    w = csv.writer(f)
    if write_header:
        w.writerow(_CSV_HEADER)
    return f, w

# ── Main loop ─────────────────────────────────────────────────────────────────

_STATE_LOG_HEADER = [
    't_s', 'arm_id', 'arm_name',
    'rtt_us', 'rttvar_us', 'min_rtt_us', 'cwnd_mss',
    'lost', 'retrans', 'delivery_rate', 'reward',
]

def _open_state_log(path: str):
    if not path:
        return None, None
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(_STATE_LOG_HEADER)
    return f, w


def run(state_fd: int, action_fd: int, flow_fd: int,
        flow_id: int, cport: int,
        checkpoint_path: str,
        batch_size: int, min_replay: int, lr: float,
        gamma: float, save_every: int,
        c_entropy: float, c_term: float, c_dwell: float,
        csv_path: str, state_log_path: str) -> None:

    net, opt, step = _load_or_init(checkpoint_path, lr)
    net.eval()

    # Target network: frozen copy used for TD bootstrapping.
    # Prevents the value collapse caused by chasing a moving target.
    import copy
    target_net = copy.deepcopy(net)
    target_net.eval()

    astraea     = _start_astraea(flow_fd, flow_id)
    reward_calc = make_reward_calc()

    replay = collections.deque(maxlen=10_000)
    csv_file, csv_writer = _open_csv(csv_path)
    sl_file, sl_writer   = _open_state_log(state_log_path)

    # Current dwell period state
    cur_arm_id   = 0          # start CUBIC until first action
    dwell_ms     = 100.0      # initial dwell — matches bias init median
    dwell_end    = time.monotonic() + dwell_ms / 1000.0
    first_state  = None
    accum_reward = 0.0
    reward_steps = 0
    # Use OC_EPISODE_START so the state log's time axis is aligned with the
    # orchestrator's link-change scheduler (both measure from the same origin).
    _ep_start = os.environ.get('OC_EPISODE_START', '')
    t0 = float(_ep_start) if _ep_start else time.monotonic()

    print(f'[worker cport={cport} flow={flow_id}] started', flush=True)

    while True:
        # ── Read one state frame (blocks until C sends it) ────────────────
        try:
            raw = struct.unpack(STATE_FMT, _read_exact(state_fd, STATE_SIZE))
        except EOFError:
            break

        # Override cur_arm field with what we actually chose (C may lag)
        raw = (cur_arm_id,) + raw[1:]

        norm_state = normalize_state(raw)
        (_, rtt_us, rttvar_us, min_rtt_us,
         cwnd, lost, retrans, _, delivery_rate) = raw
        reward = reward_calc.step(rtt_us, delivery_rate, lost)

        if first_state is None:
            first_state = norm_state

        # Per-frame state log (for episode plots)
        if sl_writer is not None:
            t_s = time.monotonic() - t0
            arm_name = ARM_NAMES[arm_id_to_idx(cur_arm_id)]
            sl_writer.writerow([
                f'{t_s:.3f}', cur_arm_id, arm_name,
                rtt_us, rttvar_us, min_rtt_us,
                cwnd, lost, retrans, delivery_rate,
                f'{reward:.4f}',
            ])
            sl_file.flush()

        accum_reward += reward
        reward_steps += 1

        # ── Dwell elapsed: pick new arm, train, log ───────────────────────
        if time.monotonic() >= dwell_end:
            next_arm_id, next_dwell_ms = net.act(norm_state)
            _send_action(action_fd, next_arm_id, int(next_dwell_ms))

            avg_r = accum_reward / max(reward_steps, 1)
            print(f'[worker cport={cport} flow={flow_id}] '
                  f'{ARM_NAMES[arm_id_to_idx(cur_arm_id)]}→{ARM_NAMES[arm_id_to_idx(next_arm_id)]}'
                  f'  dwell={next_dwell_ms:.0f}ms  r={avg_r:.3f}',
                  flush=True)

            # Store experience
            exp = Experience(
                cport=cport, flow_id=flow_id,
                state=first_state,
                option_idx=arm_id_to_idx(cur_arm_id),
                dwell_ms=float(dwell_ms),
                reward=avg_r,
                next_state=norm_state,
                done=False,
            )
            replay.append(exp)

            # Train if buffer big enough
            if len(replay) >= min_replay:
                info = _train_step(net, opt, list(replay), batch_size, gamma,
                                   c_entropy=c_entropy, c_term=c_term, c_dwell=c_dwell,
                                   target_net=target_net)
                soft_update_target(net, target_net)
                step += 1
                print(f'[learner] step={step:5d}  '
                      f'loss={info.total.item():.4f}  '
                      f'val={info.value:.4f}  pol={info.policy:.4f}  '
                      f'ent={info.entropy:.4f}  buf={len(replay)}',
                      flush=True)
                if csv_writer:
                    csv_writer.writerow([step, info.total.item(),
                                         info.value, info.policy,
                                         info.entropy, info.term,
                                         info.dwell, cport, flow_id])
                    csv_file.flush()
                if step % save_every == 0:
                    _save(checkpoint_path, net, opt, step)

            # Advance dwell
            cur_arm_id   = next_arm_id
            dwell_ms     = next_dwell_ms
            dwell_end    = time.monotonic() + dwell_ms / 1000.0
            first_state  = norm_state
            accum_reward = 0.0
            reward_steps = 0

            # Tell astraea_service whether to apply cwnd for the new arm
            if astraea is not None:
                astraea.set_control(cur_arm_id == ASTRAEA_ARM_ID)

    # ── Flow ended ───────────────────────────────────────────────────────────
    if first_state is not None:
        replay.append(Experience(
            cport=cport, flow_id=flow_id,
            state=first_state,
            option_idx=arm_id_to_idx(cur_arm_id),
            dwell_ms=float(dwell_ms),
            reward=accum_reward / max(reward_steps, 1),
            next_state=np.zeros(STATE_DIM, np.float32),
            done=True,
        ))

    if csv_file:
        csv_file.close()
    if sl_file:
        sl_file.close()

    if astraea is not None:
        astraea.stop()

    _save(checkpoint_path, net, opt, step)
    print(f'[worker cport={cport} flow={flow_id}] done  total_steps={step}', flush=True)


if __name__ == '__main__':
    state_fd  = int(os.environ['OC_STATE_FD'])
    action_fd = int(os.environ['OC_ACTION_FD'])
    flow_fd   = int(os.environ.get('OC_FLOW_FD', '-1'))
    flow_id   = int(os.environ['OC_FLOW_ID'])
    cport     = int(os.environ['OC_CPORT'])

    run(
        state_fd      = state_fd,
        action_fd     = action_fd,
        flow_fd       = flow_fd,
        flow_id       = flow_id,
        cport         = cport,
        checkpoint_path = os.environ.get('OC_CHECKPOINT', ''),
        batch_size    = int(os.environ.get('OC_BATCH_SIZE',  '32')),
        min_replay    = int(os.environ.get('OC_MIN_REPLAY',  '64')),
        lr            = float(os.environ.get('OC_LR',        '3e-4')),
        gamma         = float(os.environ.get('OC_GAMMA',     '0.99')),
        save_every    = int(os.environ.get('OC_SAVE_EVERY',  '20')),
        c_entropy     = float(os.environ.get('OC_C_ENTROPY', '0.01')),
        c_term        = float(os.environ.get('OC_C_TERM',    '0.1')),
        c_dwell       = float(os.environ.get('OC_C_DWELL',   '0.1')),
        csv_path        = os.environ.get('OC_LOG_CSV',    ''),
        state_log_path  = os.environ.get('OC_STATE_LOG', ''),
    )
