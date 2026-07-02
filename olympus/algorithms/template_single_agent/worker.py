"""Per-flow rollout worker template (single-agent).

One process per flow. The orchestrator's oc_bridge forks this script with a
fully populated environment (see run_episode in olympus/orchestrator.py). The
worker drives one TCP flow at a fixed control interval: read kernel stats →
normalize → pick an action → set cwnd → push the transition to the learner and
periodically pull fresh weights back.

Everything in this file is generic scaffolding shared by every single-agent
worker (IPC, env-var contract, the rollout loop). The only parts that are
algorithm-specific are marked `# TODO` and live in model.py — see td3/worker.py
for a complete, working reference.
"""

import csv
import io
import os
import sys
import threading
import time

import numpy as np

# Keep each worker single-threaded — dozens run in parallel under the slots.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from multiprocessing.managers import BaseManager

_HERE = os.path.dirname(os.path.abspath(__file__))
_ALG_DIR = os.path.dirname(_HERE)
_PKG = os.path.dirname(_ALG_DIR)
_REPO = os.path.dirname(_PKG)
sys.path.insert(0, _REPO)

from olympus.algorithms.template_single_agent import model   # TODO: your package
from olympus.common import flow_backend
from olympus.common.action_plugins import load_action_module
from olympus.common.registry import reward_module

_ACTION_PLUGIN = load_action_module()
_LOG_FLUSH_EVERY = 50


# ── Learner IPC ───────────────────────────────────────────────────────────────
# Mirror exactly the methods the learner registers on its BaseManager. The
# learner owns the implementations; here we only declare the proxy names.

class _Mgr(BaseManager):
    pass


_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('pull_params')


def _connect_manager():
    """Connect to the learner's manager using the addr+key the orchestrator
    injected. Returns None when learning is disabled or the connection fails."""
    addr = os.environ.get('SAO_MANAGER_ADDR', '')
    key = os.environ.get('SAO_MANAGER_KEY', '')
    if not addr or not key:
        return None
    host, port = addr.rsplit(':', 1)
    mgr = _Mgr(address=(host, int(port)), authkey=bytes.fromhex(key))
    try:
        mgr.connect()
        return mgr
    except Exception as e:
        print(f'[worker] manager connect failed: {e}', flush=True)
        return None


def _drain_state_fd(fd: int, stop: threading.Event):
    """Drain the listener's optional state pipe so it never blocks on a full fd."""
    try:
        import select
        while not stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                os.read(fd, 4096)
    except Exception:
        pass


def _srtt_ms(raw: dict) -> float:
    try:
        srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
        fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1e3


def run():
    # ── Contract read off the environment (set in run_episode) ───────────────
    reward_name = os.environ.get('SAO_REWARD', 'tempest')
    reward_plugin = reward_module(reward_name)

    flow_fd = int(os.environ['OC_FLOW_FD'])
    flow_id = int(os.environ['OC_FLOW_ID'])
    cport = int(os.environ.get('OC_CPORT', '0'))
    state_fd = int(os.environ.get('OC_STATE_FD', '-1'))
    episode = int(os.environ.get('SAO_EPISODE', '0'))
    ckpt_path = os.environ.get('SAO_CHECKPOINT', '')

    interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20'))
    interval_s = interval_ms / 1000.0
    simulation_backend = flow_backend.is_simulation_backend()
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '10'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    hidden = int(os.environ.get('SAO_HIDDEN', '128'))
    head_hidden_env = os.environ.get('SAO_HEAD_HIDDEN', '')
    head_hidden = int(head_hidden_env) if head_hidden_env else None
    noise_std = float(os.environ.get('SAO_NOISE_STD', '0.2'))
    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'
    if deterministic:
        noise_std = 0.0
    require_checkpoint = os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1'

    state_log_path = os.environ.get('SAO_TRACE_LOG', '')
    if state_log_path and os.environ.get('SAO_TRACE_LOG_SUFFIX_BY_FLOW', '0') == '1':
        root, ext = os.path.splitext(state_log_path)
        state_log_path = f'{root}_flow{flow_id}{ext or ".csv"}'

    # ── Build the policy and (optionally) warm-start from the checkpoint ──────
    # TODO: construct your policy from model.py. The td3 reference also reconciles
    # the network width with the checkpoint via model.actor_arch_from_checkpoint.
    ckpt = None
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.assert_checkpoint_state_compatible(ckpt, source=ckpt_path)
        except Exception as e:
            print(f'[worker cport={cport}] ckpt load failed ({e}) - fresh init', flush=True)
            if require_checkpoint:
                raise
            ckpt = None
    elif require_checkpoint:
        raise FileNotFoundError(f'[worker cport={cport}] checkpoint not found: {ckpt_path}')

    actor = model.Actor(model.STATE_DIM, hidden, head_hidden)   # TODO
    if ckpt is not None:
        try:
            actor.load_state_dict(ckpt['actor'])
            step = ckpt.get('step', 0)
            print(f'[worker cport={cport}] loaded checkpoint step={step}', flush=True)
        except Exception as e:
            print(f'[worker cport={cport}] ckpt load failed ({e}) - fresh init', flush=True)
            if require_checkpoint:
                raise
    actor.eval()

    mgr = _connect_manager()

    stop_drain = threading.Event()
    if state_fd >= 0:
        threading.Thread(target=_drain_state_fd, args=(state_fd, stop_drain),
                         daemon=True).start()

    # ── Per-episode trace CSV (rendered into a plot by the orchestrator) ──────
    log_file, log_writer = None, None
    if state_log_path:
        os.makedirs(os.path.dirname(state_log_path), exist_ok=True)
        log_file = open(state_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_s', 'option', 'cwnd_mult', 'cwnd',
            'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
            'loss_ratio', 'reward', 'kalman_rtt_ms',
            'act_mu', 'act_sig',
        ] + [f's{i}' for i in range(model.STATE_DIM)])

    reward_calc = reward_plugin.make_reward_calc()
    t0 = float(os.environ.get('SAO_EPISODE_START', '0')) or time.monotonic()
    traj_id = f'template_{cport}_{episode}_{flow_id}'   # algo_cport_episode_flow

    prev_state = None
    prev_action = 0.0
    prev_urtt = 0.0
    prev_cwnd = 10
    peak_thr = 0.0
    step_in_traj = 0
    ever_alive = False
    hid = None   # carried recurrent state, or None for a feed-forward policy

    weight_pull_counter = 0
    weight_pull_every = int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50'))
    push_every = max(1, int(os.environ.get('OC_PUSH_EVERY', '16')))
    exp_buf = []
    log_flush_counter = 0

    dead_flow_ms = float(os.environ.get('SAO_DEAD_FLOW_MS', '1000'))
    dead_steps = 0
    dead_steps_limit = int(dead_flow_ms / interval_ms)

    print(f'[worker alg=template reward={reward_name} cport={cport} flow={flow_id}] '
          f'started interval={interval_ms}ms det={int(deterministic)} '
          f'noise={noise_std:.2f}', flush=True)

    # ── Rollout loop — one iteration per control tick ────────────────────────
    try:
        while True:
            t_step_start = time.monotonic()

            try:
                raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            except flow_backend.SimulationFinished:
                print('[worker] RayNet simulation finished - exiting',
                      flush=True)
                break
            except Exception as e:
                print(f'[worker] get_tcp_deepcc_info failed: {e} - exiting', flush=True)
                break

            avg_thr = float(raw.get('avg_thr', 0))
            peak_thr = max(peak_thr, avg_thr)

            # Dead-flow detection: stop pushing once a flow has been silent for
            # dead_flow_ms (it left the episode). A flow that was never alive yet
            # is still considered active so its startup transitions are kept.
            if avg_thr <= 0:
                dead_steps += 1
            else:
                dead_steps = 0
                ever_alive = True
            flow_active = (not ever_alive) or (dead_steps < dead_steps_limit)

            raw['prev_urtt'] = prev_urtt
            raw['prev_cwnd'] = prev_cwnd
            raw['peak_thr'] = peak_thr
            raw['interval_ms'] = interval_ms
            norm_s = model.normalize_state(raw)
            reward = float(reward_calc.step(raw))
            components = getattr(reward_calc, 'last_components', {}) or {}
            urtt = float(raw.get('avg_urtt', 0))
            if urtt > 0:
                prev_urtt = urtt
            prev_cwnd = int(raw.get('cwnd', prev_cwnd))
            t_s = flow_backend.episode_seconds(raw, t0, wall_now=t_step_start)

            # Push the PREVIOUS step's transition now that we have its reward and
            # next_state. Batch up to push_every before crossing the IPC boundary.
            if prev_state is not None and flow_active and mgr:
                exp_buf.append(model.Experience(
                    state=prev_state,
                    action=prev_action,
                    reward=reward,
                    next_state=norm_s,
                    done=False,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
                ))
                step_in_traj += 1
                if len(exp_buf) >= push_every:
                    try:
                        mgr.push_exp_batch(list(exp_buf))
                    except Exception:
                        for e in exp_buf:
                            try:
                                mgr.push_exp(e)
                            except Exception:
                                pass
                    exp_buf.clear()

            # TODO: select an action with your policy. act() returns the bounded
            # action stored in replay, the multiplier handed to the action plugin,
            # and the next recurrent hidden state.
            a, mult, hid = actor.act(norm_s, hid, noise_std=noise_std)
            cur_cwnd = int(raw.get('cwnd', 10))
            new_cwnd = int(np.clip(
                _ACTION_PLUGIN.apply_cwnd(cur_cwnd, a, cwnd_min, cwnd_max),
                cwnd_min, cwnd_max))
            try:
                flow_backend.set_cwnd(flow_fd, new_cwnd)
            except Exception as e:
                print(f'[worker] set_cwnd failed: {e} - exiting', flush=True)
                break

            prev_state = norm_s
            prev_action = a

            # Periodically pull the freshest weights the learner has broadcast.
            weight_pull_counter += 1
            if mgr and weight_pull_counter >= weight_pull_every:
                weight_pull_counter = 0
                try:
                    param_bytes = mgr.pull_params()
                    if param_bytes is not None:
                        payload = torch.load(io.BytesIO(param_bytes),
                                             map_location='cpu', weights_only=False)
                        model.assert_checkpoint_state_compatible(
                            payload, source='learner payload')
                        actor.load_state_dict(payload['actor_state_dict'])
                        actor.eval()
                        step = payload.get('step', step)
                except Exception:
                    pass

            if log_writer:
                log_writer.writerow([
                    f'{t_s:.3f}', 0, f'{mult:.3f}', new_cwnd,
                    f'{avg_thr * 8 / 1e6:.3f}',
                    f'{float(raw.get("avg_urtt", 0)) / 1e3:.3f}',
                    f'{_srtt_ms(raw):.3f}',
                    f'{float(raw.get("min_rtt", 0)) / 1e3:.3f}',
                    f'{float(raw.get("loss_ratio", 0)):.1f}',
                    f'{reward:.4f}',
                    f'{float(raw.get("kalman_min_rtt_us", 0.0)) / 1e3:.3f}',
                    f'{a:.4f}', f'{noise_std:.4f}',
                ] + [f'{float(x):.5f}' for x in norm_s])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            # Hold the control interval.
            sleep_t = interval_s - (time.monotonic() - t_step_start)
            if sleep_t > 0 and not simulation_backend:
                time.sleep(sleep_t)

    finally:
        # Flush whatever is buffered, then push a terminal done transition so the
        # learner can close the trajectory.
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
        if prev_state is not None and mgr:
            try:
                mgr.push_exp(model.Experience(
                    state=prev_state,
                    action=prev_action,
                    reward=0.0,
                    next_state=prev_state,
                    done=True,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
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
