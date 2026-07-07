"""Per-flow rollout worker template (multi-agent / CTDE).

The orchestrator's run_episode_marl launches N of these per episode — one
listener+worker per flow on a shared bottleneck — each with SAO_AGENT_ID set to
its flow index. Execution is decentralized: each worker runs its own copy of the
parameter-shared policy on its own observation (N=1). The learner re-assembles
the joint picture at training time by grouping experiences on
(group_id, group_step).

Everything here is generic multi-agent scaffolding; the algorithm-specific parts
are marked `# TODO` and live in model.py. See mat/worker.py (on-policy CTDE) for
a complete reference, or ma_dreamer/worker.py for the off-policy variant.
"""

import csv
import io
import math
import os
import sys
import threading
import time

import numpy as np

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

from olympus.algorithms.template_multi_agent import model   # TODO: your package
from olympus.common import flow_backend, runtime_config
from olympus.common.action_plugins import load_action_module
from olympus.common.registry import reward_module

_ACTION_PLUGIN = load_action_module()
_LOG_FLUSH_EVERY = 50


# ── Learner IPC ───────────────────────────────────────────────────────────────
# push_bootstrap is the multi-agent (on-policy) addition over the single-agent
# set — it carries the terminal value estimate for GAE. Drop it for off-policy.

class _Mgr(BaseManager):
    pass


_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('push_bootstrap')
_Mgr.register('pull_params')


def _connect_manager():
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
    cfg = runtime_config.load_config()
    reward_name = str(runtime_config.runtime_value(
        cfg, 'reward', env='SAO_REWARD', default='tempest_fairness_ma'))
    reward_plugin = reward_module(reward_name)

    flow_fd = int(os.environ['OC_FLOW_FD'])
    flow_id = int(os.environ['OC_FLOW_ID'])
    cport = int(os.environ.get('OC_CPORT', '0'))
    cport_base = int(os.environ.get('SAO_CPORT_BASE', str(cport)))
    n_agents = int(os.environ.get('SAO_N_AGENTS', '2'))
    state_fd = int(os.environ.get('OC_STATE_FD', '-1'))
    episode = int(os.environ.get('SAO_EPISODE', '0'))
    instance_id = int(os.environ.get('SAO_INSTANCE_ID', '0'))
    ckpt_path = os.environ.get('SAO_CHECKPOINT', '')

    interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20'))
    interval_s = interval_ms / 1000.0
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '10'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    hidden = int(runtime_config.agent_value(
        cfg, 'hidden', env='SAO_HIDDEN', default=128))
    n_layers = int(runtime_config.agent_value(
        cfg, 'n_layers', env='SAO_MAT_LAYERS', default=2))
    n_heads = int(runtime_config.agent_value(
        cfg, 'n_heads', env='SAO_MAT_HEADS', default=4))
    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'
    require_checkpoint = os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1'

    # This flow's agent index. The orchestrator sets SAO_AGENT_ID per listener;
    # the cport offset is a fallback for older launchers.
    agent_id_env = os.environ.get('SAO_AGENT_ID', '')
    agent_id = int(agent_id_env) if agent_id_env != '' else (cport - cport_base)
    agent_id = max(0, min(n_agents - 1, agent_id))
    # All simultaneous flows in this episode share group_id; the learner pairs
    # them by (group_id, group_step) to build each joint training sample.
    group_id = f'slot{instance_id}_ep{episode}'

    # TODO: build the parameter-shared policy and warm-start it.
    net = model.Policy(model.STATE_DIM, hidden, n_layers=n_layers, n_heads=n_heads)
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.assert_checkpoint_action_compatible(ckpt, source=ckpt_path)
            net.load_state_dict(ckpt['model'], strict=False)
            step = ckpt.get('step', 0)
            print(f'[worker cport={cport} agent={agent_id}] loaded checkpoint step={step}',
                  flush=True)
        except Exception as e:
            print(f'[worker cport={cport} agent={agent_id}] ckpt load failed ({e}) - fresh',
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

    # N workers share a slot — suffix the trace CSV with the agent id so they
    # don't overwrite each other. The orchestrator renders these with the
    # combined multi-flow plot.
    log_file, log_writer = None, None
    state_log_path = os.environ.get('SAO_TRACE_LOG', '')
    if state_log_path:
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
    traj_id = f'template_{group_id}_a{agent_id}_f{flow_id}'

    prev_state = None
    prev_action_raw = 0.0
    prev_log_prob = 0.0
    prev_value = 0.0
    prev_throughput = 0.0
    prev_urtt = 0.0
    prev_cwnd = 10
    peak_thr = 0.0
    step_in_traj = 0
    ever_alive = False
    last_norm_s = None

    weight_pull_counter = 0
    weight_pull_every = int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50'))
    push_every = max(1, int(runtime_config.training_value(
        cfg, 'worker_push_every', env='OC_PUSH_EVERY', default=16)))
    exp_buf = []
    log_flush_counter = 0

    dead_flow_ms = float(runtime_config.agent_value(
        cfg, 'dead_flow_ms', env='SAO_DEAD_FLOW_MS', default=1000))
    dead_steps = 0
    dead_steps_limit = int(dead_flow_ms / interval_ms)

    print(f'[worker alg=template reward={reward_name} cport={cport} agent={agent_id} '
          f'group={group_id} flow={flow_id}] started interval={interval_ms}ms '
          f'det={int(deterministic)} hidden={hidden}', flush=True)

    try:
        while True:
            t_step_start = time.monotonic()

            try:
                raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            except flow_backend.SimulationFinished:
                print('[worker] RayNet simulation finished - exiting', flush=True)
                break
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
            raw['peak_thr'] = peak_thr
            raw['interval_ms'] = interval_ms
            norm_s = model.normalize_state(raw)
            reward = float(reward_calc.step(raw))
            # Scheduled membership (flow_present) outranks the dead-flow
            # heuristic: a starved-but-scheduled flow must keep pushing or it
            # drops from the learner's agent mask and the team fairness term
            # never sees the starvation. (See the MARL mask semantics.)
            reward_components = getattr(reward_calc, 'last_components', {}) or {}
            flow_present = reward_components.get('flow_present')
            if flow_present is not None:
                flow_active = bool(flow_present)
            urtt = float(raw.get('avg_urtt', 0))
            if urtt > 0:
                prev_urtt = urtt
            prev_cwnd = int(raw.get('cwnd', prev_cwnd))

            t_s = flow_backend.episode_seconds(
                raw, t0, wall_now=t_step_start)
            group_step = flow_backend.episode_step(
                raw, t0, interval_s, step_in_traj, wall_now=t_step_start)
            flow_backend.wait_collection_step(raw, group_step)

            # Push the previous step's transition. `throughput` carries the
            # avg_thr observed at the NEXT state so the learner can build the
            # per-step fairness reward across simultaneous flows.
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

            # Once a flow leaves the episode schedule (flow_present == 0) it is
            # no longer part of the learning problem. In RayNet, keep sending a
            # frozen cwnd until the simulator stops exposing that flow; otherwise
            # the shared episode waits forever for an action from this worker.
            if not flow_active:
                prev_state = None
                new_cwnd = int(raw.get('cwnd', prev_cwnd))
                mult = act_mu = act_sig = 0.0
                if flow_backend.is_simulation_backend():
                    try:
                        flow_backend.set_cwnd(flow_fd, new_cwnd)
                    except Exception as e:
                        print(f'[worker] set_cwnd failed: {e} - exiting', flush=True)
                        break
                else:
                    break
            else:
                # TODO: decentralized action with N=1 (own observation only).
                (action_raw, mult, log_prob, value, act_mu, act_sig) = net.act(
                    norm_s, deterministic=deterministic)

                cur_cwnd = int(raw.get('cwnd', 10))
                new_cwnd = int(np.clip(
                    _ACTION_PLUGIN.apply_cwnd(cur_cwnd, math.tanh(action_raw), cwnd_min, cwnd_max),
                    cwnd_min, cwnd_max))
                try:
                    flow_backend.set_cwnd(flow_fd, new_cwnd)
                except Exception as e:
                    print(f'[worker] set_cwnd failed: {e} - exiting', flush=True)
                    break

                prev_state = norm_s
                prev_action_raw = action_raw
                prev_log_prob = log_prob
                prev_value = value
                prev_throughput = avg_thr
                last_norm_s = norm_s

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

            sleep_t = interval_s - (time.monotonic() - t_step_start)
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
        # On-policy bootstrap: terminal value estimate for GAE. Drop for
        # off-policy MARL (replay-based) algorithms.
        if mgr and last_norm_s is not None:
            try:
                with torch.no_grad():
                    s_t = torch.from_numpy(last_norm_s).reshape(1, 1, -1)
                    *_, v_t, _ = net.forward_joint(s_t)   # TODO: match your signature
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
