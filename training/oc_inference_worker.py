"""
training/oc_inference_worker.py — inference-only OC worker

Same pipe protocol as oc_worker.py but:
  - No training, no replay buffer
  - Logs every state frame to OC_STATE_LOG CSV for post-hoc plotting
  - Prints arm switches to stdout

Environment variables (set by inference_test.py via oc_listener):
  OC_STATE_FD    — pipe fd (C writes oc_state_t every scan_ms)
  OC_ACTION_FD   — pipe fd (Python writes oc_action_t)
  OC_FLOW_FD     — TCP socket fd (-1 if unused)
  OC_FLOW_ID     — flow identifier
  OC_CPORT       — experiment identifier
  OC_CHECKPOINT  — path to model checkpoint (.pt)
  OC_STATE_LOG   — path to write per-frame state CSV
"""

import csv
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np
import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _ROOT)

from training.model import (
    OptionCriticNet, normalize_state,
    arm_id_to_idx, ARM_NAMES, N_OPTIONS, STATE_DIM,
)
from training.reward import make_reward_calc

STATE_FMT  = '<8IQ'
STATE_SIZE = struct.calcsize(STATE_FMT)
ACTION_FMT = '<II'
ASTRAEA_ARM_ID = 13

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


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


class AstraeaBackground:
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
            'ASTRAEA_FLOW_ID':    str(flow_id),
            'ASTRAEA_FLOW_FD':    str(flow_fd),
            'ASTRAEA_CONFIG':     config,
            'ASTRAEA_MODEL':      model,
            'ASTRAEA_CONTROL_FD': str(ctrl_rd),
        })

        self._proc = subprocess.Popen(
            [python_bin, script_path],
            env=env,
            pass_fds=(ctrl_rd, flow_fd),
        )
        os.close(ctrl_rd)
        self.set_control(False)   # start disabled; we're on CUBIC first
        print(f'[astraea] background service started  pid={self._proc.pid}', flush=True)

    def set_control(self, active: bool) -> None:
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
        print('[astraea] background service stopped', flush=True)


def _start_astraea(flow_fd: int, flow_id: int):
    if flow_fd < 0:
        return None
    script = os.environ.get('ASTRAEA_SCRIPT',
                            os.path.join(_ROOT, 'astraea_service.py'))
    if not os.path.exists(script):
        print('[astraea] astraea_service.py not found — skipping', flush=True)
        return None
    try:
        return AstraeaBackground(flow_fd, flow_id)
    except Exception as e:
        print(f'[astraea] failed to start background service: {e}', flush=True)
        return None


def run(state_fd: int, action_fd: int, flow_fd: int,
        flow_id: int, cport: int,
        checkpoint_path: str, state_log_path: str) -> None:

    # ── Load model ────────────────────────────────────────────────────────────
    net = OptionCriticNet(state_dim=STATE_DIM, n_options=N_OPTIONS)
    step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        net.load_state_dict(ckpt['model'])
        step = ckpt.get('step', 0)
        print(f'[inference] loaded checkpoint  step={step}  {checkpoint_path}', flush=True)
    else:
        print(f'[inference] no checkpoint — using random policy', flush=True)
    net.eval()

    # Epsilon decay: starts at 0.20, decays exponentially to 0.03 over 30k steps.
    # At step 0    → eps ≈ 0.20 (explore heavily)
    # At step 10k  → eps ≈ 0.10
    # At step 30k+ → eps = 0.03 (mostly exploit)
    _eps_start = float(os.environ.get('OC_EPS_START', '0.20'))
    _eps_end   = float(os.environ.get('OC_EPS_END',   '0.03'))
    _eps_decay = float(os.environ.get('OC_EPS_DECAY', '30000'))
    epsilon = max(_eps_end,
                  _eps_end + (_eps_start - _eps_end) * math.exp(-step / _eps_decay))
    print(f'[inference] epsilon={epsilon:.4f}  (step={step})', flush=True)

    # ── State log CSV ─────────────────────────────────────────────────────────
    log_file, log_writer = None, None
    if state_log_path:
        log_file = open(state_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_s',           # seconds since flow start
            'arm_id',        # raw mutant arm id
            'arm_name',      # human-readable
            'rtt_us',        # smoothed RTT µs
            'rttvar_us',     # RTT variance µs
            'min_rtt_us',    # min RTT µs
            'cwnd_mss',      # congestion window in MSS units
            'lost',          # unrecovered lost packets
            'retrans',       # retransmitted packets
            'delivery_rate', # bytes/s
            'reward',        # instantaneous reward
        ])

    astraea     = _start_astraea(flow_fd, flow_id)
    reward_calc = make_reward_calc()

    cur_arm_id  = 0
    dwell_ms    = 2000.0
    dwell_end   = time.monotonic() + dwell_ms / 1000.0
    _ep_start = os.environ.get('OC_EPISODE_START', '')
    t0 = float(_ep_start) if _ep_start else time.monotonic()

    print(f'[inference cport={cport} flow={flow_id}] started', flush=True)

    while True:
        try:
            raw = struct.unpack(STATE_FMT, _read_exact(state_fd, STATE_SIZE))
        except EOFError:
            break

        raw = (cur_arm_id,) + raw[1:]   # override C's cur_arm with our choice
        t_s = time.monotonic() - t0

        (_, rtt_us, rttvar_us, min_rtt_us,
         cwnd, lost, retrans, delivered, delivery_rate) = raw

        reward = reward_calc.step(rtt_us, delivery_rate, lost)

        # Log every frame
        if log_writer:
            arm_name = ARM_NAMES[arm_id_to_idx(cur_arm_id)] if cur_arm_id in [0,2,5,12,13] else '?'
            log_writer.writerow([
                f'{t_s:.3f}', cur_arm_id, arm_name,
                rtt_us, rttvar_us, min_rtt_us,
                cwnd, lost, retrans, delivery_rate,
                f'{reward:.4f}',
            ])
            log_file.flush()

        # ── Dwell expired: pick next arm ──────────────────────────────────────
        if time.monotonic() >= dwell_end:
            norm_state = normalize_state(raw)
            next_arm_id, next_dwell_ms = net.act(norm_state, epsilon=epsilon)
            _send_action(action_fd, next_arm_id, int(next_dwell_ms))

            arm_from = ARM_NAMES[arm_id_to_idx(cur_arm_id)] if cur_arm_id in [0,2,5,12,13] else '?'
            arm_to   = ARM_NAMES[arm_id_to_idx(next_arm_id)]
            tput_mbps = delivery_rate * 8 / 1e6
            rtt_ms    = rtt_us / 1000.0
            print(f'[inference] t={t_s:6.1f}s  {arm_from}→{arm_to}'
                  f'  dwell={next_dwell_ms:.0f}ms'
                  f'  tput={tput_mbps:.2f}Mbps  rtt={rtt_ms:.1f}ms  r={reward:.3f}',
                  flush=True)

            cur_arm_id = next_arm_id
            dwell_ms   = next_dwell_ms
            dwell_end  = time.monotonic() + dwell_ms / 1000.0

            if astraea is not None:
                astraea.set_control(cur_arm_id == ASTRAEA_ARM_ID)

    if log_file:
        log_file.close()

    if astraea is not None:
        astraea.stop()

    print(f'[inference cport={cport} flow={flow_id}] done', flush=True)


if __name__ == '__main__':
    run(
        state_fd        = int(os.environ['OC_STATE_FD']),
        action_fd       = int(os.environ['OC_ACTION_FD']),
        flow_fd         = int(os.environ.get('OC_FLOW_FD', '-1')),
        flow_id         = int(os.environ['OC_FLOW_ID']),
        cport           = int(os.environ['OC_CPORT']),
        checkpoint_path = os.environ.get('OC_CHECKPOINT', ''),
        state_log_path  = os.environ.get('OC_STATE_LOG', ''),
    )
