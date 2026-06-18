"""
worker.py — per-flow Option-Critic worker for direct CWND control.

Spawned by oc_bridge (same binary as training/) for each TCP flow.

Protocol with oc_bridge:
  OC_STATE_FD   — optional C state pipe. In the single-agent fast path this is
                  -1 because the worker reads TCP state directly via tcp_sockopt.
  OC_ACTION_FD  — legacy field; the single-agent listener sets this to -1.
  OC_FLOW_FD    — duplicated TCP socket fd; we call get_tcp_deepcc_info() on it
                  and set_cwnd() on it every INTERVAL_MS.

The worker connects to the central learner via multiprocessing Manager IPC
(OC_MANAGER_ADDR / OC_MANAGER_KEY) to push experiences and pull weights.

Environment variables (set by orchestrator → oc_bridge → inherited here):
  OC_FLOW_FD          TCP socket fd (deepcc already enabled by oc_bridge)
  OC_FLOW_ID          monotonic flow id
  OC_CPORT            iperf3 client source port (for logging)
  OC_STATE_FD         optional pipe to drain, or -1 when disabled
  OC_CHECKPOINT       .pt file path
  OC_MANAGER_ADDR     host:port of learner manager
  OC_MANAGER_KEY      hex authkey
  OC_LINK_BW          link BW in Mbps (for reward)
  Reward settings are provided by the shared olympus reward module.
  OC_INTERVAL_MS      decision interval (default 20)
  OC_CWND_MIN         min cwnd (packets, default 2)
  OC_CWND_MAX         max cwnd (packets, default 1000)
  OC_ACTION_INIT      action-mean initialisation mode
  OC_ACTION_INIT_SCALE small-spread log-multiplier half-width
  OC_Q_VALUE_CLIP     absolute bound for Q values/TD targets
  OC_EPS_START/END/DECAY epsilon schedule
  OC_DWELL_MIN_STEPS  minimum option dwell before termination is allowed
  OC_DWELL_MAX_STEPS  maximum option dwell before termination is forced (0 disables)
  OC_PUSH_EVERY       batch this many experiences per learner IPC call
  OC_STATE_LOG        optional path for per-frame CSV log
"""

import csv
import math
import os
import struct
import sys
import threading
import time

import numpy as np
import torch
from multiprocessing.managers import BaseManager

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO = os.path.dirname(_PKG)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tcp_sockopt
from olympus.algorithms.option_critic.model import (
    Experience,
    N_OPTIONS,
    STATE_DIM,
    OptionCriticNet,
    assert_checkpoint_state_compatible,
    normalize_oc_arch,
    normalize_state,
    option_critic_meta_from_checkpoint,
)
from olympus.common.action_plugins import load_action_module
from olympus.common.bbr_probe import BbrProbe
from olympus.common.registry import reward_module

_ACTION_PLUGIN = load_action_module()


def make_reward_calc():
    reward_name = os.environ.get('SAO_REWARD', 'tempest')
    return reward_module(reward_name).make_reward_calc()


def _srtt_ms(raw: dict) -> float:
    try:
        srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
        fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1e3


# ── IPC to learner ────────────────────────────────────────────────────────────

class _Mgr(BaseManager): pass
_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('pull_params')

# Number of CSV rows between explicit flushes. With 30 workers at 50 Hz,
# per-row flushing was 1500 fsync-style flushes/sec; batching to 50 rows
# (~1 s wall clock) is roughly equivalent to natural block buffering.
_LOG_FLUSH_EVERY = 50

def _connect_manager():
    addr_raw = os.environ.get('OC_MANAGER_ADDR', '')
    key_hex  = os.environ.get('OC_MANAGER_KEY',  '')
    if not addr_raw or not key_hex:
        return None
    host, port = addr_raw.rsplit(':', 1)
    mgr = _Mgr(address=(host, int(port)), authkey=bytes.fromhex(key_hex))
    try:
        mgr.connect()
        return mgr
    except Exception as e:
        print(f'[worker] manager connect failed: {e}', flush=True)
        return None

# ── Drain thread ──────────────────────────────────────────────────────────────

def _drain_state_fd(fd: int, stop: threading.Event):
    """Read and discard legacy OC_STATE_FD frames so the C write never blocks."""
    try:
        import select
        while not stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                os.read(fd, 4096)
    except Exception:
        pass

# ── Main worker loop ──────────────────────────────────────────────────────────

def run():
    flow_fd   = int(os.environ['OC_FLOW_FD'])
    flow_id   = int(os.environ['OC_FLOW_ID'])
    cport     = int(os.environ.get('OC_CPORT', '0'))
    episode   = int(os.environ.get('OC_EPISODE', '0'))
    state_fd  = int(os.environ.get('OC_STATE_FD', '-1'))
    ckpt_path = os.environ.get('OC_CHECKPOINT', '')

    interval_ms = float(os.environ.get('OC_INTERVAL_MS', '20'))
    cwnd_min    = int(os.environ.get('OC_CWND_MIN', '2'))
    cwnd_max    = int(os.environ.get('OC_CWND_MAX', '1000'))
    action_init = os.environ.get('OC_ACTION_INIT', 'small_spread')
    action_init_scale = float(os.environ.get('OC_ACTION_INIT_SCALE', '0.05'))
    q_value_clip = float(os.environ.get('OC_Q_VALUE_CLIP', '1500.0'))
    action_logsig_min = float(os.environ.get('OC_ACTION_LOGSIG_MIN', '-1.0'))
    arch        = normalize_oc_arch(os.environ.get('OC_ARCH', 'gru'))
    beta_temperature = float(os.environ.get('OC_BETA_TEMPERATURE', '1.0'))
    beta_floor  = float(os.environ.get('OC_BETA_FLOOR', '0.0'))

    eps_start      = float(os.environ.get('OC_EPS_START',       '0.20'))
    eps_end        = float(os.environ.get('OC_EPS_END',         '0.03'))
    eps_decay      = float(os.environ.get('OC_EPS_DECAY',       '30000'))
    dwell_min_steps = max(1, int(os.environ.get('OC_DWELL_MIN_STEPS', '1')))
    dwell_max_steps = int(os.environ.get('OC_DWELL_MAX_STEPS', '0'))
    dwell_max_steps = max(dwell_max_steps, dwell_min_steps) if dwell_max_steps > 0 else 0
    deterministic  = os.environ.get('OC_DETERMINISTIC', '0') == '1'
    term_threshold = float(os.environ.get('OC_TERM_THRESHOLD', '0.5'))
    mgr_addr_raw   = os.environ.get('OC_MANAGER_ADDR', '')

    state_log_path = os.environ.get('OC_STATE_LOG', '')
    if state_log_path and os.environ.get('OC_STATE_LOG_SUFFIX_BY_FLOW', '0') == '1':
        root, ext = os.path.splitext(state_log_path)
        state_log_path = f'{root}_flow{flow_id}{ext or ".csv"}'

    # ── Load model ────────────────────────────────────────────────────────────
    n_options = int(os.environ.get('OC_N_OPTIONS', N_OPTIONS))
    hidden    = int(os.environ.get('OC_HIDDEN', 256))
    net  = OptionCriticNet(
        STATE_DIM, n_options, hidden,
        beta_temperature=beta_temperature, beta_floor=beta_floor, arch=arch,
        action_init=action_init, action_init_scale=action_init_scale,
        q_value_clip=q_value_clip, action_logsig_min=action_logsig_min)
    step = 0
    if ckpt_path:
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                assert_checkpoint_state_compatible(ckpt, source=ckpt_path)
                ckpt_meta = option_critic_meta_from_checkpoint(ckpt)
                mismatches = []
                for key in (
                    'arch', 'hidden', 'n_options', 'action_dist', 'beta_floor',
                    'mult_min', 'mult_max', 'action_init', 'action_init_scale',
                    'q_value_clip', 'state_features', 'state_dim',
                ):
                    if key in ckpt_meta and ckpt_meta[key] != net.model_meta()[key]:
                        mismatches.append(
                            f'{key}={ckpt_meta[key]} (ckpt) vs {net.model_meta()[key]} (config)')
                if mismatches:
                    raise ValueError(', '.join(mismatches))
                net.load_state_dict(ckpt['model'])
                step = ckpt.get('step', 0)
                print(f'[worker] loaded checkpoint step={step}', flush=True)
            except Exception as e:
                print(f'[worker] checkpoint load failed ({e}), fresh start', flush=True)
                if not mgr_addr_raw:
                    raise
        else:
            msg = f'[worker] checkpoint not found: {ckpt_path}'
            if mgr_addr_raw:
                print(msg + ' — training mode will wait for learner weights', flush=True)
            else:
                raise FileNotFoundError(
                    msg + ' (inference mode refuses to run with random weights)')
    elif not mgr_addr_raw:
        raise FileNotFoundError(
            '[worker] OC_CHECKPOINT is unset in inference mode; refusing random rollout')
    net.eval()

    epsilon = max(eps_end, eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay))
    print(f'[worker cport={cport} flow={flow_id}] arch={net.arch}  epsilon={epsilon:.3f}  '
          f'deterministic={int(deterministic)}  term_threshold={term_threshold:.3f}',
          flush=True)

    # ── Connect to learner ────────────────────────────────────────────────────
    mgr = _connect_manager()

    # ── Drain state pipe ──────────────────────────────────────────────────────
    stop_drain = threading.Event()
    if state_fd >= 0:
        drain_thread = threading.Thread(
            target=_drain_state_fd, args=(state_fd, stop_drain), daemon=True)
        drain_thread.start()

    # ── State log CSV ─────────────────────────────────────────────────────────
    log_file, log_writer = None, None
    if state_log_path:
        log_file   = open(state_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_s', 'option', 'option_age', 'can_terminate', 'force_terminate',
            'cwnd_mult', 'cwnd_mult_applied', 'cwnd',
            'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
            'loss_ratio', 'reward', 'kalman_rtt_ms',
            'reward_base', 'reward_close', 'reward_sparse',
            'reward_unclipped',
            'act_mu', 'act_sig',
        ] + [f's{i}' for i in range(STATE_DIM)])

    reward_calc = make_reward_calc()
    # Use episode_start from orchestrator so t_s aligns with link schedule timestamps
    t0 = float(os.environ.get('OC_EPISODE_START', '0')) or time.monotonic()

    # Unique trajectory ID: include episode so back-to-back episodes in the
    # same slot (same cport, flow_id typically resets to 0) don't collide and
    # get merged into a single trajectory in the replay buffer.
    traj_id      = f'{cport}_{episode}_{flow_id}'
    cur_option    = -1
    option_age    = 0
    prev_state    = None
    prev_option   = -1
    prev_can_term = False
    prev_force_term = False
    prev_mult     = 1.0
    prev_prev_mult = 1.0
    prev_urtt     = 0.0
    prev_cwnd     = 10           # previous CWND for delta_cwnd causal feature
    peak_thr      = 0.0          # running max observed throughput — BW reference for normalize_state
    n_options     = net.n_options
    # Recurrent architectures keep hidden state warm across the episode.
    gru_h_pol = net.initial_hidden(batch_size=1)
    gru_h_val = net.initial_hidden(batch_size=1)

    interval_s = interval_ms / 1000.0
    exp_buf    = []      # local buffer before pushing to learner
    push_every = max(1, int(os.environ.get('OC_PUSH_EVERY', '16')))

    # BBR3-style independent cwnd probe; reward- and algorithm-agnostic.
    probe = BbrProbe.from_env()

    weight_pull_counter = 0
    weight_pull_every   = 50   # pull updated weights every N decisions
    log_flush_counter   = 0

    # Stop sending experiences once flow has drained — avg_thr=0 for this many
    # consecutive steps means iperf3 has finished and the worker is just spinning
    # on a dead socket.  Sending zero-thr experiences corrupts Q-values.
    dead_flow_ms      = float(os.environ.get('OC_DEAD_FLOW_MS', '1000'))
    _dead_steps       = 0
    _dead_steps_limit = int(dead_flow_ms / interval_ms)

    print(f'[worker cport={cport} flow={flow_id}] started, interval={interval_ms}ms '
          f'push_every={push_every}', flush=True)
    # Main loop: observe state, act, push experience, repeat.
    try:
        while True:
            t_step_start = time.monotonic()

            # ── Read DeepCC state ─────────────────────────────────────────────
            try:
                raw = tcp_sockopt.get_tcp_deepcc_info(flow_fd)
            except Exception as e:
                print(f'[worker] get_tcp_deepcc_info failed: {e} — exiting', flush=True)
                break

            avg_thr  = float(raw.get('avg_thr', 0))
            peak_thr = max(peak_thr, avg_thr)   # track clean-pipe throughput peak

            # Detect flow drain — stop pushing experiences once iperf3 finishes.
            # A dead flow produces avg_thr=0 experiences that corrupt Q-values
            # with "CWND action → zero reward" data that was never a real decision.
            if avg_thr <= 0:
                _dead_steps += 1
            else:
                _dead_steps = 0
            flow_active = (_dead_steps < _dead_steps_limit)

            raw['prev_urtt']        = prev_urtt
            raw['prev_cwnd']        = prev_cwnd
            raw['peak_thr']         = peak_thr
            raw['interval_ms']      = interval_ms
            norm_s = normalize_state(raw)
            reward = reward_calc.step(raw)
            _urtt = float(raw.get('avg_urtt', 0))
            if _urtt > 0:
                prev_urtt = _urtt   # don't overwrite with zero — no ACKs in window
            prev_cwnd = int(raw.get('cwnd', prev_cwnd))

            t_s = t_step_start - t0

            # ── Act ───────────────────────────────────────────────────────────
            option_age_now = option_age
            can_terminate_now = (cur_option >= 0 and option_age >= dwell_min_steps)
            force_terminate_now = (
                cur_option >= 0 and dwell_max_steps > 0 and option_age >= dwell_max_steps
            )
            opt, mult, gru_h_pol, gru_h_val, act_mu, act_sig, terminated = net.act(
                norm_s, gru_h_pol, gru_h_val,
                epsilon=epsilon, cur_option=cur_option,
                deterministic=deterministic,
                term_threshold=term_threshold,
                can_terminate=can_terminate_now,
                force_terminate=force_terminate_now)
            cur_cwnd    = int(raw.get('cwnd', 10))
            bounded_action = float(_ACTION_PLUGIN.from_multiplier(mult))
            agent_cwnd = _ACTION_PLUGIN.apply_cwnd(
                cur_cwnd, bounded_action, cwnd_min, cwnd_max)

            # Feed kernel-smoothed srtt (srtt_us>>3) into the probe's min-RTT
            # filter, then let the probe override the cwnd write if its window
            # is currently active.
            srtt_raw_us = float(raw.get('srtt_us', 0) or 0)
            filter_rtt_us = (srtt_raw_us / 8.0) if srtt_raw_us > 0 else float(raw.get('avg_urtt', 0) or 0)
            probe.observe_rtt(t_step_start, filter_rtt_us)
            actual_cwnd, agent_locked, _ = probe.decide(
                t_step_start, cur_cwnd, agent_cwnd)
            new_cwnd    = int(np.clip(actual_cwnd, cwnd_min, cwnd_max))

            applied_mult = new_cwnd / max(cur_cwnd, 1)

            try:
                tcp_sockopt.set_cwnd(flow_fd, new_cwnd)
            except Exception as e:
                print(f'[worker] set_cwnd failed: {e} — exiting', flush=True)
                break

            # ── Experience from previous step ─────────────────────────────────
            if prev_state is not None and flow_active:
                exp = Experience(
                    state      = prev_state,
                    option_idx = prev_option,
                    can_terminate      = prev_can_term,
                    force_terminate    = prev_force_term,
                    can_terminate_next = can_terminate_now,
                    force_terminate_next = force_terminate_now,
                    prev_cwnd_mult = prev_prev_mult,
                    cwnd_mult  = prev_mult,
                    reward     = reward,
                    next_state = norm_s,
                    done       = False,
                    traj_id    = traj_id,
                )
                exp_buf.append(exp)
                if mgr and len(exp_buf) >= push_every:
                    try:
                        mgr.push_exp_batch(list(exp_buf))
                    except Exception:
                        for e in exp_buf:
                            try:
                                mgr.push_exp(e)
                            except Exception:
                                pass
                    exp_buf.clear()

            if agent_locked:
                # Probe owned this step; invalidate prev_state so the next
                # iteration's push isn't built from probe-induced state.
                prev_state  = None
            else:
                prev_state  = norm_s
                prev_option = opt
                prev_can_term = can_terminate_now
                prev_force_term = force_terminate_now
                prev_prev_mult = prev_mult
                prev_mult   = mult
            if cur_option < 0 or terminated:
                option_age = 1
            else:
                option_age += 1
            cur_option  = opt

            # ── Pull updated weights periodically ─────────────────────────────
            weight_pull_counter += 1
            if mgr and weight_pull_counter >= weight_pull_every:
                weight_pull_counter = 0
                try:
                    param_bytes = mgr.pull_params()
                    if param_bytes is not None:
                        import io as _io
                        payload = torch.load(_io.BytesIO(param_bytes), map_location='cpu',
                                             weights_only=False)
                        assert_checkpoint_state_compatible(
                            payload, source='learner payload')
                        payload_meta = option_critic_meta_from_checkpoint(payload)
                        mismatches = []
                        for key in (
                            'arch', 'hidden', 'n_options', 'action_dist', 'beta_floor',
                            'mult_min', 'mult_max', 'action_init', 'action_init_scale',
                            'q_value_clip', 'state_features', 'state_dim',
                        ):
                            if key in payload_meta and payload_meta[key] != net.model_meta()[key]:
                                mismatches.append(
                                    f'{key}={payload_meta[key]} (learner) vs {net.model_meta()[key]} (worker)')
                        if mismatches:
                            raise ValueError(', '.join(mismatches))
                        net.load_state_dict(payload['state_dict'])
                        net.eval()
                        # Do NOT reset gru_h_pol/h_val — they encode flow history.
                        new_step = payload.get('step', step)
                        if new_step != step:
                            step    = new_step
                            epsilon = max(eps_end,
                                eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay))
                except Exception:
                    pass

            # ── Log ───────────────────────────────────────────────────────────
            if log_writer:
                rc = getattr(reward_calc, 'last_components', {}) or {}
                log_writer.writerow([
                    f'{t_s:.3f}', opt, option_age_now, int(can_terminate_now),
                    int(force_terminate_now), f'{mult:.3f}',
                    f'{applied_mult:.3f}', new_cwnd,
                    f'{avg_thr * 8 / 1e6:.3f}',
                    f'{float(raw.get("avg_urtt", 0)) / 1e3:.3f}',
                    f'{_srtt_ms(raw):.3f}',
                    f'{float(raw.get("min_rtt", 0)) / 1e3:.3f}',
                    f'{float(raw.get("loss_ratio", 0)):.1f}',
                    f'{reward:.4f}',
                    f'{float(raw.get("kalman_min_rtt_us", 0.0)) / 1e3:.3f}',
                    f'{float(rc.get("base", 0.0)):.4f}',
                    f'{float(rc.get("close", 0.0)):.4f}',
                    f'{float(rc.get("sparse", 0.0)):.4f}',
                    f'{float(rc.get("unclipped", reward)):.4f}',
                    f'{act_mu:.4f}', f'{act_sig:.4f}',
                ] + [f'{float(x):.5f}' for x in norm_s])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            # ── Sleep for rest of interval ────────────────────────────────────
            elapsed = time.monotonic() - t_step_start
            sleep_t = interval_s - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        if mgr and exp_buf:
            try:
                mgr.push_exp_batch(list(exp_buf))
            except Exception:
                for e in exp_buf:
                    try:
                        mgr.push_exp(e)
                    except Exception:
                        pass
            exp_buf.clear()
        # Mark last experience as done so TD bootstrap terminates correctly
        if prev_state is not None and mgr:
            try:
                mgr.push_exp(Experience(
                    state      = prev_state,
                    option_idx = prev_option,
                    can_terminate      = prev_can_term,
                    force_terminate    = prev_force_term,
                    can_terminate_next = False,
                    force_terminate_next = False,
                    prev_cwnd_mult = prev_prev_mult,
                    cwnd_mult  = prev_mult,
                    reward     = 0.0,
                    next_state = prev_state,
                    done       = True,
                    traj_id    = traj_id,
                ))
            except Exception:
                pass
        stop_drain.set()
        if log_file:
            try:
                log_file.flush()
            except Exception:
                pass
            log_file.close()
        print(f'[worker cport={cport} flow={flow_id}] done', flush=True)


if __name__ == '__main__':
    run()
