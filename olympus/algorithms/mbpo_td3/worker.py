"""Per-flow rollout worker for MBPO-TD3.

Identical to algorithms/orca_td3/worker.py — MBPO is a learner-side change
(forward model + imagined rollouts + mixed batches). The worker just collects
real transitions and pushes them to the learner exactly as it did for orca_td3.
"""

import csv
import io
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

from olympus.algorithms.mbpo_td3 import model
from olympus.common.action_plugins import load_action_module
from olympus.common.async_pusher import AsyncPusher
from olympus.common.pace import sleep_to_grid
from olympus.common.bbr_probe import BbrProbe
from olympus.common.registry import reward_module
from olympus.common import flow_backend, runtime_config

_ACTION_PLUGIN = load_action_module()


class _Mgr(BaseManager):
    pass


_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('pull_params')


# Number of CSV rows between explicit flushes. With 30 workers at 50 Hz,
# per-row flushing was 1500 fsync-style flushes/sec; batching to 50 rows
# (~1 s wall clock) is roughly equivalent to natural block buffering.
_LOG_FLUSH_EVERY = 50


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


def _reward_value(reward_calc, raw: dict) -> float:
    return float(reward_calc.step(raw))


def _srtt_ms(raw: dict) -> float:
    try:
        srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
        fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1e3


def _parse_int_set(raw: str) -> set:
    out = set()
    for item in str(raw or '').replace(';', ',').split(','):
        item = item.strip()
        if not item:
            continue
        try:
            out.add(int(float(item)))
        except ValueError:
            pass
    return out


def run():
    cfg = runtime_config.load_config()
    reward_name = str(runtime_config.runtime_value(
        cfg, 'reward', env='SAO_REWARD', default='tempest'))
    reward_plugin = reward_module(reward_name)

    flow_fd = int(os.environ['OC_FLOW_FD'])
    flow_id = int(os.environ['OC_FLOW_ID'])
    cport = int(os.environ.get('OC_CPORT', '0'))
    state_fd = int(os.environ.get('OC_STATE_FD', '-1'))
    episode = int(os.environ.get('SAO_EPISODE', '0'))
    ckpt_path = os.environ.get('SAO_CHECKPOINT', '')
    lagged_flow_ids = _parse_int_set(os.environ.get('SAO_LAGGED_POLICY_FLOW_IDS', ''))
    is_lagged_policy = flow_id in lagged_flow_ids
    if is_lagged_policy:
        lagged_ckpt = os.environ.get('SAO_LAGGED_POLICY_CHECKPOINT', '')
        if lagged_ckpt and os.path.exists(lagged_ckpt):
            ckpt_path = lagged_ckpt
        elif os.environ.get('SAO_LAGGED_POLICY_REQUIRE_CHECKPOINT', '0') == '1':
            raise FileNotFoundError(
                f'[worker cport={cport} flow={flow_id}] lagged checkpoint not found: '
                f'{lagged_ckpt}')
        elif lagged_ckpt:
            print(f'[worker cport={cport} flow={flow_id}] lagged checkpoint '
                  f'missing, using current checkpoint: {lagged_ckpt}', flush=True)

    interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20'))
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '10'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    hidden = int(runtime_config.agent_value(
        cfg, 'hidden', env='SAO_HIDDEN', default=128))
    noise_std = float(os.environ.get('SAO_NOISE_STD', '0.2'))
    require_checkpoint = os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1'

    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'
    if is_lagged_policy and os.environ.get('SAO_LAGGED_POLICY_DETERMINISTIC', '1') == '1':
        deterministic = True
    if deterministic:
        noise_std = 0.0
    simulation_backend = flow_backend.is_simulation_backend()

    state_log_path = os.environ.get('SAO_TRACE_LOG', '')
    if state_log_path and os.environ.get('SAO_TRACE_LOG_SUFFIX_BY_FLOW', '0') == '1':
        root, ext = os.path.splitext(state_log_path)
        state_log_path = f'{root}_flow{flow_id}{ext or ".csv"}'

    actor = model.Actor(model.STATE_DIM, hidden)
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.assert_checkpoint_state_compatible(ckpt, source=ckpt_path)
            actor.load_state_dict(ckpt['actor'])
            step = ckpt.get('step', 0)
            print(f'[worker cport={cport}] loaded checkpoint step={step}', flush=True)
        except Exception as e:
            print(f'[worker cport={cport}] ckpt load failed ({e}) - fresh init', flush=True)
            if require_checkpoint:
                raise
    elif require_checkpoint:
        raise FileNotFoundError(f'[worker cport={cport}] checkpoint not found: {ckpt_path}')
    actor.eval()

    lagged_disable_learning = (
        is_lagged_policy
        and os.environ.get('SAO_LAGGED_POLICY_DISABLE_LEARNING', '1') == '1'
    )
    mgr = None if lagged_disable_learning else _connect_manager()

    # Tag pushed experiences with their collection backend (emulation/simulation)
    # so a mixed learner routes them into the matching replay buffer.
    _exp_src = flow_backend.experience_source()

    def _push_batch(exps):
        getattr(mgr, 'push_exp_batch')(exps, _exp_src)

    def _push_one(e):
        getattr(mgr, 'push_exp')(e, _exp_src)

    # Push batches off-thread so the blocking manager RPC never stalls the
    # fixed-cadence control loop (was gapping the emitted signal every
    # `push_every` steps).
    pusher = AsyncPusher(_push_batch, _push_one) if mgr else None

    stop_drain = threading.Event()
    if state_fd >= 0:
        threading.Thread(target=_drain_state_fd, args=(state_fd, stop_drain),
                         daemon=True).start()

    log_file, log_writer = None, None
    if state_log_path:
        os.makedirs(os.path.dirname(state_log_path), exist_ok=True)
        log_file = open(state_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_s', 'option', 'cwnd_mult', 'cwnd',
            'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
            'loss_ratio', 'reward', 'kalman_rtt_ms',
            'fair_bw_mbps', 'active_flows', 'fairness_cost',
            'act_mu', 'act_sig',
        ] + [f's{i}' for i in range(model.STATE_DIM)])

    reward_calc = reward_plugin.make_reward_calc()
    t0 = float(os.environ.get('SAO_EPISODE_START', '0')) or time.monotonic()
    # Pace against the shared episode-start grid (t0 + k*interval) so co-active
    # flows sample in phase; see common/pace.sleep_to_grid.
    next_tick = t0
    traj_id = f'mbpo_td3_{cport}_{episode}_{flow_id}'

    prev_state = None
    prev_action = 0.0
    prev_urtt = 0.0
    prev_cwnd = 10
    peak_thr = 0.0
    step_in_traj = 0
    ever_alive = False
    hid = None
    interval_s = interval_ms / 1000.0

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

    # BBR3-style independent cwnd probe; reward- and algorithm-agnostic.
    probe = BbrProbe.from_env()

    role = 'lagged-bg' if is_lagged_policy else 'train'
    print(f'[worker alg=mbpo_td3 reward={reward_name} cport={cport} flow={flow_id} '
          f'role={role}] '
          f'started interval={interval_ms}ms det={int(deterministic)} noise={noise_std:.2f}',
          flush=True)

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
            reward = _reward_value(reward_calc, raw)
            components = getattr(reward_calc, 'last_components', {}) or {}
            urtt = float(raw.get('avg_urtt', 0))
            if urtt > 0:
                prev_urtt = urtt
            prev_cwnd = int(raw.get('cwnd', prev_cwnd))

            t_s = flow_backend.episode_seconds(raw, t0, wall_now=t_step_start)

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
                    pusher.submit(exp_buf)
                    exp_buf.clear()

            a, mult, hid = actor.act(norm_s, hid, noise_std=noise_std)
            cur_cwnd = int(raw.get('cwnd', 10))
            desired_cwnd = _ACTION_PLUGIN.apply_cwnd(
                cur_cwnd, a, cwnd_min, cwnd_max)

            srtt_raw_us = float(raw.get('srtt_us', 0) or 0)
            filter_rtt_us = (srtt_raw_us / 8.0) if srtt_raw_us > 0 else float(raw.get('avg_urtt', 0) or 0)
            clock_t = flow_backend.observation_clock(raw, wall_now=t_step_start)
            probe.observe_rtt(clock_t, filter_rtt_us)
            actual_cwnd, agent_locked, _ = probe.decide(
                clock_t, cur_cwnd, desired_cwnd)
            new_cwnd = int(np.clip(actual_cwnd, cwnd_min, cwnd_max))
            try:
                flow_backend.set_cwnd(flow_fd, new_cwnd)
            except Exception as e:
                print(f'[worker] set_cwnd failed: {e} - exiting', flush=True)
                break

            if agent_locked:
                prev_state = None
                prev_action = 0.0
            else:
                prev_state = norm_s
                prev_action = a

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
                    f'{float(components.get("fair_bw_mbps", 0.0)):.3f}',
                    f'{float(components.get("active_flows", 0.0)):.0f}',
                    f'{float(components.get("fairness_cost", 0.0)):.4f}',
                    f'{a:.4f}', f'{noise_std:.4f}',
                ] + [f'{float(x):.5f}' for x in norm_s])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            if not simulation_backend:
                next_tick = sleep_to_grid(next_tick, interval_s)

    finally:
        if pusher:
            if exp_buf:
                pusher.submit(exp_buf)
                exp_buf.clear()
            # Drain queued batches before the terminal done-marker below so
            # ordering to the learner is preserved.
            pusher.close()
        if prev_state is not None and mgr:
            try:
                _push_one(model.Experience(
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
