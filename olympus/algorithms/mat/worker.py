"""Per-flow rollout worker for MAT-CTDE.

Decentralized execution: each flow runs its own copy of the parameter-shared
transformer policy with N=1 (its own observation). The learner reconstructs
the joint state at training time by grouping experiences by `(group_id,
step_in_traj)` so each timestep contributes one centralized critic input.

Each worker emits Experience tuples with:
  state, action_raw, log_prob, value, reward, throughput,
  agent_id   — flow index within the slot (0..n-1)
  group_id   — slot+episode identifier; experiences with the same group_id
               and step_in_traj are simultaneous on the bottleneck link
  done, traj_id, step_in_traj

The fairness component of the reward is added by the learner using the
throughput field from sibling experiences at the same group_id+step_in_traj.
"""

import csv
import io
import os
import sys
import threading
import time

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from multiprocessing.managers import BaseManager

_HERE = os.path.dirname(os.path.abspath(__file__))
_ALG_DIR = os.path.dirname(_HERE)
_PKG = os.path.dirname(_ALG_DIR)
_REPO = os.path.dirname(_PKG)
sys.path.insert(0, _REPO)

import tcp_sockopt
from olympus.algorithms.mat import model
from olympus.common.action_plugins import load_action_module
from olympus.common.registry import reward_module

_ACTION_PLUGIN = load_action_module()


class _Mgr(BaseManager):
    pass


_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('push_bootstrap')
_Mgr.register('pull_params')


_LOG_FLUSH_EVERY = 50


def _srtt_ms(raw: dict) -> float:
    try:
        srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
        fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1e3


def _connect_manager():
    addr = os.environ.get('SAO_MANAGER_ADDR', '')
    key  = os.environ.get('SAO_MANAGER_KEY',  '')
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
    try:
        import select
        while not stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                os.read(fd, 4096)
    except Exception:
        pass


def run():
    reward_name   = os.environ.get('SAO_REWARD', 'tempest_fairness_ma')
    reward_plugin = reward_module(reward_name)

    flow_fd  = int(os.environ['OC_FLOW_FD'])
    flow_id  = int(os.environ['OC_FLOW_ID'])
    cport    = int(os.environ.get('OC_CPORT', '0'))
    cport_base = int(os.environ.get('SAO_CPORT_BASE', str(cport)))
    n_agents = int(os.environ.get('SAO_N_AGENTS', '2'))
    state_fd = int(os.environ.get('OC_STATE_FD', '-1'))
    episode  = int(os.environ.get('SAO_EPISODE', '0'))
    instance_id = int(os.environ.get('SAO_INSTANCE_ID', '0'))
    ckpt_path = os.environ.get('SAO_CHECKPOINT', '')

    interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20'))
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '10'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    hidden   = int(os.environ.get('SAO_HIDDEN',   '128'))
    n_layers = int(os.environ.get('SAO_MAT_LAYERS', '2'))
    n_heads  = int(os.environ.get('SAO_MAT_HEADS',  '4'))
    require_checkpoint = os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1'
    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'

    state_log_path = os.environ.get('SAO_TRACE_LOG', '')

    # agent_id is the flow's index within the slot. New orchestrators pass it
    # directly because oc_bridge matches one exact cport; keep the cport
    # fallback so older launchers still work.
    agent_id_env = os.environ.get('SAO_AGENT_ID', '')
    if agent_id_env != '':
        agent_id = int(agent_id_env)
    else:
        agent_id = cport - cport_base
    agent_id = max(0, min(n_agents - 1, agent_id))
    # group_id ties simultaneous flows together for centralized-critic batching.
    group_id = f'slot{instance_id}_ep{episode}'

    net = model.MATPolicy(model.STATE_DIM, hidden,
                          n_layers=n_layers, n_heads=n_heads)
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.assert_checkpoint_action_compatible(
                ckpt, source=ckpt_path)
            net.load_state_dict(ckpt['model'], strict=False)
            step = ckpt.get('step', 0)
            print(f'[worker cport={cport} agent={agent_id}] loaded checkpoint step={step}',
                  flush=True)
        except Exception as e:
            print(f'[worker cport={cport} agent={agent_id}] ckpt load failed ({e}) - fresh init',
                  flush=True)
            if require_checkpoint:
                raise
    elif require_checkpoint:
        raise FileNotFoundError(f'[worker cport={cport}] checkpoint not found: {ckpt_path}')
    net.eval()

    mgr = _connect_manager()

    stop_drain = threading.Event()
    if state_fd >= 0:
        threading.Thread(target=_drain_state_fd, args=(state_fd, stop_drain),
                         daemon=True).start()

    log_file, log_writer = None, None
    if state_log_path:
        # N workers share a slot, so suffix each path with the agent id so they
        # don't trample each other writing the per-episode trace CSV.
        base, ext = os.path.splitext(state_log_path)
        state_log_path = f'{base}_a{agent_id}{ext}'
        os.makedirs(os.path.dirname(state_log_path), exist_ok=True)
        log_file = open(state_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_s', 'option', 'cwnd_mult', 'cwnd',
            'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
            'loss_ratio', 'reward', 'kalman_rtt_ms',
            'act_mu', 'act_sig', 'agent_id',
        ] + [f's{i}' for i in range(model.STATE_DIM)])

    reward_calc = reward_plugin.make_reward_calc()
    t0 = float(os.environ.get('SAO_EPISODE_START', '0')) or time.monotonic()
    traj_id = f'mat_{group_id}_a{agent_id}_f{flow_id}'

    prev_state      = None
    prev_action_raw = 0.0
    prev_log_prob   = 0.0
    prev_value      = 0.0
    prev_throughput = 0.0
    prev_urtt       = 0.0
    prev_cwnd       = 10
    peak_thr        = 0.0
    step_in_traj    = 0
    ever_alive      = False
    interval_s      = interval_ms / 1000.0

    weight_pull_counter = 0
    weight_pull_every = int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50'))

    push_every = max(1, int(os.environ.get('OC_PUSH_EVERY', '16')))
    exp_buf = []
    log_flush_counter = 0

    dead_flow_ms = float(os.environ.get('SAO_DEAD_FLOW_MS', '1000'))
    dead_steps = 0
    dead_steps_limit = int(dead_flow_ms / interval_ms)

    last_norm_s = None

    print(f'[worker alg=mat reward={reward_name} cport={cport} agent={agent_id} '
          f'group={group_id} flow={flow_id}] started interval={interval_ms}ms '
          f'det={int(deterministic)} hidden={hidden}',
          flush=True)

    try:
        while True:
            t_step_start = time.monotonic()

            try:
                raw = tcp_sockopt.get_tcp_deepcc_info(flow_fd)
            except Exception as e:
                print(f'[worker] get_tcp_deepcc_info failed: {e} - exiting', flush=True)
                break

            avg_thr = float(raw.get('avg_thr', 0))
            peak_thr = max(peak_thr, avg_thr)

            if avg_thr <= 0:
                dead_steps += 1
            else:
                dead_steps = 0
                ever_alive = True
            flow_active = (not ever_alive) or (dead_steps < dead_steps_limit)

            raw['prev_urtt'] = prev_urtt
            raw['prev_cwnd'] = prev_cwnd
            raw['peak_thr']  = peak_thr
            raw['interval_ms'] = interval_ms
            norm_s = model.normalize_state(raw)
            reward = float(reward_calc.step(raw))
            # The environment's join/leave schedule decides which flows are in
            # the episode. It must outrank the dead-flow heuristic: a starved
            # flow (zero throughput while still scheduled present) has to keep
            # pushing experiences, or it drops out of the learner's agent mask
            # and the team fairness term never sees the starvation.
            reward_components = getattr(reward_calc, 'last_components', {}) or {}
            flow_present = reward_components.get('flow_present')
            if flow_present is not None:
                flow_active = bool(flow_present)
            urtt = float(raw.get('avg_urtt', 0))
            if urtt > 0:
                prev_urtt = urtt
            prev_cwnd = int(raw.get('cwnd', prev_cwnd))

            t_s = t_step_start - t0
            # Group-relative step: which slot-wide tick this experience belongs
            # to. All flows share `t0 = SAO_EPISODE_START`, so a late-joining
            # flow's first push will have group_step ≈ delay/interval. Lets
            # the learner align siblings on the same bottleneck timeline.
            group_step = int(round(t_s / interval_s)) if interval_s > 0 else step_in_traj

            # Push the previous-step transition with the just-observed reward.
            # `throughput` is the avg_thr observed at the NEXT state — i.e. what
            # the previous action produced. The learner uses this to compute
            # the per-step fairness reward across simultaneous flows.
            if prev_state is not None and flow_active and mgr:
                exp_buf.append(model.Experience(
                    state=prev_state,
                    action_raw=prev_action_raw,
                    log_prob=prev_log_prob,
                    value=prev_value,
                    reward=reward,
                    throughput=avg_thr,
                    agent_id=agent_id,
                    group_id=group_id,
                    group_step=group_step,
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
                            try: mgr.push_exp(e)
                            except Exception: pass
                    exp_buf.clear()

            # Sample action from the parameter-shared transformer policy
            # with N=1 (own observation). The transformer self-attends on a
            # single token, which collapses to a no-op token mix.
            (action_raw, mult, log_prob, value, act_mu, act_sig) = net.act(
                norm_s, deterministic=deterministic)

            cur_cwnd = int(raw.get('cwnd', 10))
            new_cwnd = _ACTION_PLUGIN.apply_cwnd(
                cur_cwnd, math.tanh(action_raw), cwnd_min, cwnd_max)
            try:
                tcp_sockopt.set_cwnd(flow_fd, new_cwnd)
            except Exception as e:
                print(f'[worker] set_cwnd failed: {e} - exiting', flush=True)
                break

            prev_state      = norm_s
            prev_action_raw = action_raw
            prev_log_prob   = log_prob
            prev_value      = value
            prev_throughput = avg_thr
            last_norm_s     = norm_s

            weight_pull_counter += 1
            if mgr and weight_pull_counter >= weight_pull_every:
                weight_pull_counter = 0
                try:
                    param_bytes = mgr.pull_params()
                    if param_bytes is not None:
                        payload = torch.load(io.BytesIO(param_bytes),
                                             map_location='cpu', weights_only=False)
                        model.assert_checkpoint_action_compatible(
                            payload, source='learner payload')
                        net.load_state_dict(payload['state_dict'])
                        net.eval()
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
                    f'{act_mu:.4f}', f'{act_sig:.4f}', agent_id,
                ] + [f'{float(x):.5f}' for x in norm_s])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

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
                    try: mgr.push_exp(e)
                    except Exception: pass
            exp_buf.clear()
        if prev_state is not None and mgr:
            try:
                mgr.push_exp(model.Experience(
                    state=prev_state,
                    action_raw=prev_action_raw,
                    log_prob=prev_log_prob,
                    value=prev_value,
                    reward=0.0,
                    throughput=prev_throughput,
                    agent_id=agent_id,
                    group_id=group_id,
                    group_step=int(round((time.monotonic() - t0) / interval_s))
                                if interval_s > 0 else step_in_traj,
                    done=True,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
                ))
            except Exception:
                pass
        if mgr and last_norm_s is not None:
            try:
                with torch.no_grad():
                    s_t = torch.from_numpy(last_norm_s).reshape(1, 1, -1)
                    _, _, v_t, _ = net.forward_joint(s_t)
                    bootstrap_value = float(v_t.reshape(-1)[-1].item())
                mgr.push_bootstrap(traj_id, bootstrap_value)
            except Exception:
                pass

        stop_drain.set()
        if log_file:
            try: log_file.flush()
            except Exception: pass
            log_file.close()
        print(f'[worker cport={cport} agent={agent_id} flow={flow_id}] done', flush=True)


if __name__ == '__main__':
    run()
