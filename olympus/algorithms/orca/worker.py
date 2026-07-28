"""Per-flow rollout worker for the ORCA-repo-style learner."""

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

from olympus.algorithms.orca import model
from olympus.common.action_plugins import load_action_module
from olympus.common.async_pusher import AsyncPusher
from olympus.common.pace import sleep_to_grid
from olympus.common.bbr_probe import BbrProbe
from olympus.common import flow_backend, runtime_config
from olympus.common.registry import reward_module

_ACTION_PLUGIN = load_action_module()


class _Mgr(BaseManager):
    pass


_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('pull_params')

_LOG_FLUSH_EVERY = 50


def _connect_manager():
    addr = os.environ.get('SAO_MANAGER_ADDR', '')
    key = os.environ.get('SAO_MANAGER_KEY', '')
    if not addr or not key:
        return None
    host, port = addr.rsplit(':', 1)
    mgr = _Mgr(address=(host, int(port)), authkey=bytes.fromhex(key))
    deadline = time.monotonic() + 10.0
    last_error = None
    while time.monotonic() < deadline:
        try:
            mgr.connect()
            return mgr
        except Exception as e:
            last_error = e
            time.sleep(0.1)
    print(f'[orca worker] manager connect failed: {last_error}', flush=True)
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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ('0', 'false', 'no', 'off', '')


def _srtt_ms(raw: dict) -> float:
    try:
        srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
        fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1e3


def _has_valid_orca_observation(raw: dict) -> bool:
    """Match Orca's original RTT-based observation validity gate."""
    try:
        avg_urtt = float(raw.get('avg_urtt', 0.0) or 0.0)
        sample_count = float(raw.get('count', raw.get('cnt', 0.0)) or 0.0)
        srtt_us = float(raw.get('srtt_us', 0.0) or 0.0)
        min_rtt = float(raw.get(
            'min_rtt', raw.get('min_rtt_us', 0.0)) or 0.0)
    except (TypeError, ValueError):
        return False
    return avg_urtt > 0.0 and sample_count > 0.0 and srtt_us > 0.0 and min_rtt > 0.0


def run():
    cfg = runtime_config.load_config()
    reward_name = str(runtime_config.runtime_value(
        cfg, 'reward', env='SAO_REWARD', default='orca'))
    flow_fd = int(os.environ['OC_FLOW_FD'])
    flow_id = int(os.environ['OC_FLOW_ID'])
    cport = int(os.environ.get('OC_CPORT', '0'))
    state_fd = int(os.environ.get('OC_STATE_FD', '-1'))
    episode = int(os.environ.get('SAO_EPISODE', '0'))
    ckpt_path = os.environ.get('SAO_CHECKPOINT', '')

    interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20'))
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '4'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    # Orca CUBIC->agent handoff: Mininet retains the timing used by the deployed
    # models; RayNet uses a flow-relative simulation-time timeout below.
    cubic_warmup = runtime_config.bool_value(runtime_config.agent_value(
        cfg, 'cubic_warmup', env='SAO_ORCA_CUBIC_WARMUP', default=True), True)
    cubic_warmup_max_s = float(runtime_config.agent_value(
        cfg, 'cubic_warmup_max_s', env='SAO_ORCA_CUBIC_WARMUP_MAX_S',
        default=10.0))
    hidden = int(runtime_config.agent_value(
        cfg, 'hidden', env='SAO_HIDDEN', default=256))
    head_hidden = int(runtime_config.agent_value(
        cfg, 'head_hidden', env='SAO_HEAD_HIDDEN', default=hidden))
    noise_std = float(os.environ.get('SAO_NOISE_STD', '0.2'))
    require_checkpoint = os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1'
    # delay_margin_coef (OC_ORCA_DELAY_MARGIN_COEF) and orca_target_ms
    # (SAO_ORCA_TARGET_MS) are read directly from the environment by the state
    # and reward plugins; the orchestrator exports both.
    simulation_backend = flow_backend.is_simulation_backend()

    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'
    if deterministic:
        noise_std = 0.0

    state_log_path = os.environ.get('SAO_TRACE_LOG', '')
    if state_log_path and os.environ.get('SAO_TRACE_LOG_SUFFIX_BY_FLOW', '0') == '1':
        root, ext = os.path.splitext(state_log_path)
        state_log_path = f'{root}_flow{flow_id}{ext or ".csv"}'

    ckpt = None
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.assert_checkpoint_state_compatible(ckpt, source=ckpt_path)
            hidden, head_hidden = model.actor_arch_from_checkpoint(
                ckpt, hidden, head_hidden)
            print(f'[orca worker cport={cport}] using checkpoint actor config '
                  f'hidden={hidden} head_hidden={head_hidden}', flush=True)
        except Exception as e:
            print(f'[orca worker cport={cport}] ckpt load failed ({e}) - fresh init',
                  flush=True)
            if require_checkpoint:
                raise
            ckpt = None
    elif require_checkpoint:
        raise FileNotFoundError(f'[orca worker cport={cport}] checkpoint not found: {ckpt_path}')

    actor = model.Actor(model.STATE_DIM, hidden, head_hidden)
    if ckpt is not None:
        try:
            actor.load_state_dict(ckpt['actor'])
            step = int(ckpt.get('step', 0))
            print(f'[orca worker cport={cport}] loaded checkpoint step={step}', flush=True)
        except Exception as e:
            print(f'[orca worker cport={cport}] ckpt load failed ({e}) - fresh init',
                  flush=True)
            if require_checkpoint:
                raise
    actor.eval()

    mgr = _connect_manager()

    # Tag every pushed experience with its collection backend so a mixed
    # emulation+simulation learner routes it into the matching replay buffer.
    # For simulation, append the episode's BDP (bw*rtt, set by the orchestrator)
    # so a BDP-stratified sim buffer can route the transition into its class.
    _exp_src = flow_backend.experience_source()
    if simulation_backend:
        _sim_bdp = os.environ.get('SAO_SIM_BDP', '')
        if _sim_bdp:
            _exp_src = f'{_exp_src}|{_sim_bdp}'

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
            'act_mu', 'act_sig',
        ] + [f's{i}' for i in range(model.STATE_DIM)])

    # State and reward are both runtime plugins (runtime.state / runtime.reward,
    # defaulting to orca_repo / orca) rather than a transform baked into the
    # learner — see algorithms/orca/states/orca_repo.py and rewards/orca.py.
    # model.normalize_state is the selected state plugin (7 features + rec_dim
    # history stacking, returning the STATE_DIM policy input).
    reward_calc = reward_module(reward_name).make_reward_calc()

    t0 = float(os.environ.get('SAO_EPISODE_START', '0')) or time.monotonic()
    # Pace against the shared episode-start grid (t0 + k*interval) so co-active
    # flows sample in phase; see common/pace.sleep_to_grid.
    next_tick = t0
    traj_id = f'orca_{cport}_{episode}_{flow_id}'
    interval_s_cfg = interval_ms / 1000.0
    # A RayNet worker blocks until the flow completes its delayed broker
    # registration. Its first simulation timestamp is therefore absolute
    # episode time, not an interval measured from zero.
    last_sample_t = None if simulation_backend else time.monotonic()
    raynet_warmup_start_t = None

    prev_state = None
    prev_action = 0.0
    step_in_traj = 0
    ever_alive = False
    # False while CUBIC owns startup; flips permanently at slow-start exit.
    agent_ready = not cubic_warmup
    dead_steps = 0
    dead_flow_ms = float(runtime_config.agent_value(
        cfg, 'dead_flow_ms', env='SAO_DEAD_FLOW_MS', default=1000))
    dead_steps_limit = int(dead_flow_ms / interval_ms)

    weight_pull_counter = 0
    weight_pull_every = int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50'))
    push_every = max(1, int(runtime_config.training_value(
        cfg, 'worker_push_every', env='OC_PUSH_EVERY', default=16)))
    exp_buf = []
    log_flush_counter = 0

    # tcp_deepcc.c drains avg_urtt/cnt/avg_thr/thr_cnt on every getsockopt
    # (see deepcc_get_info). When no ACK arrived during the polling interval,
    # those fields come back 0. srtt_us and min_urtt are NOT drained, so we
    # patch the raw dict to fall back to the persistent fields rather than
    # dropping the whole sample.
    last_avg_urtt = 0.0
    last_avg_thr = 0.0
    last_cnt = 0.0

    # BBR3-style independent probe (every probe_interval_s drop cwnd to
    # probe_factor*cwnd for probe_duration_s). Disabled by default; toggle
    # via SAO_BBR_PROBE_ENABLED=1. While the probe is active the agent's
    # cwnd write is overridden and we invalidate prev_state so the replay
    # buffer never sees a probe-induced transition as if the agent caused it.
    probe = BbrProbe.from_env()
    print(f'[orca worker cport={cport}] bbr_probe enabled={int(probe.enabled)} '
          f'interval={probe.probe_interval_s}s duration={probe.probe_duration_s}s '
          f'factor={probe.probe_factor} min_rtt_window={probe.min_rtt_window_s}s',
          flush=True)

    print(f'[worker alg=orca reward={reward_name} state={model.STATE_NAME} '
          f'cport={cport} flow={flow_id}] '
          f'started interval={interval_ms}ms det={int(deterministic)} '
          f'noise={noise_std:.2f} rec_dim={model.REC_DIM}',
          flush=True)

    try:
        while True:
            t_step_start = time.monotonic()
            try:
                raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            except flow_backend.SimulationFinished:
                print('[orca worker] RayNet simulation finished - exiting',
                      flush=True)
                break
            except Exception as e:
                print(f'[orca worker] get_tcp_deepcc_info failed: {e} - exiting',
                      flush=True)
                break

            cur_cwnd = int(raw.get('cwnd', 10))
            avg_thr = float(raw.get('avg_thr', 0) or 0)
            avg_urtt = float(raw.get('avg_urtt', 0) or 0)
            sample_count = float(raw.get('count', raw.get('cnt', 0)) or 0)
            observation_valid = _has_valid_orca_observation(raw)

            observation_t = flow_backend.observation_clock(
                raw, wall_now=t_step_start)
            if simulation_backend and raynet_warmup_start_t is None:
                raynet_warmup_start_t = observation_t

            t_s = flow_backend.episode_seconds(
                raw, t0, wall_now=t_step_start)
            flow_backend.wait_collection_step(raw)

            if not observation_valid:
                # The kernel-backed worker does not apply an action when Orca
                # has no valid RTT observation. RayNet must still acknowledge
                # the paused broker step, but an empty action preserves the
                # same hands-off TCP behavior.
                prev_state = None
                prev_action = 0.0
                if simulation_backend:
                    try:
                        flow_backend.advance_without_action(flow_fd)
                    except Exception as e:
                        print(f'[orca worker] invalid-observation advance failed: '
                              f'{e} - exiting', flush=True)
                        break
                else:
                    next_tick = sleep_to_grid(next_tick, interval_s_cfg)
                continue

            # tcp_deepcc.c resets these drain-on-read fields to 0 on every
            # getsockopt. When no ACK arrived in the polling interval we hold
            # the last known value so the actor doesn't see a phantom "zero
            # throughput / zero RTT" sample. srtt_us and min_urtt are NOT
            # drained by the kernel, so they pass through untouched.
            if avg_urtt <= 0.0:
                raw['avg_urtt'] = last_avg_urtt
                avg_urtt = last_avg_urtt
            else:
                last_avg_urtt = avg_urtt
            if avg_thr <= 0.0:
                raw['avg_thr'] = last_avg_thr
                avg_thr = last_avg_thr
            else:
                last_avg_thr = avg_thr
            if sample_count <= 0.0:
                # samples/cwnd feature uses 'count' / 'cnt'; carry forward.
                hold_cnt = last_cnt
                if 'count' in raw:
                    raw['count'] = hold_cnt
                if 'cnt' in raw:
                    raw['cnt'] = hold_cnt
            else:
                last_cnt = sample_count

            if avg_thr <= 0:
                dead_steps += 1
            else:
                dead_steps = 0
                ever_alive = True
            flow_active = (not ever_alive) or (dead_steps < dead_steps_limit)

            if last_sample_t is None:
                last_sample_t = flow_backend.observation_clock(
                    raw, wall_now=t_step_start)
                interval_s = interval_s_cfg
            else:
                interval_s, last_sample_t = flow_backend.interval_seconds(
                    raw, last_sample_t, wall_now=t_step_start)
            # Orca's envwrapper.py:240 only updates Welford stats in training mode.
            # Pass the worker's measured interval into the state plugin so its
            # delta_t feature / loss-rate fallback match the reward's.
            raw['interval_s'] = interval_s
            norm_s = model.normalize_state(raw)
            reward = reward_calc.step(raw)

            # ── Orca CUBIC -> agent handoff ────────────────────────────────
            # Preserve Mininet's deployed handoff timing. For RayNet, measure
            # the equivalent timeout from this flow's first simulation
            # observation so a flow introduced later in an episode still gets
            # its own Cubic-owned startup interval.
            if not agent_ready:
                loss_now = float(
                    raw.get('loss_bytes', raw.get('lost_bytes', 0.0)) or 0.0) > 0.0
                warmup_elapsed_s = t_s
                if simulation_backend and raynet_warmup_start_t is not None:
                    warmup_elapsed_s = max(
                        observation_t - raynet_warmup_start_t, 0.0)
                if loss_now or warmup_elapsed_s >= cubic_warmup_max_s:
                    agent_ready = True
                    print(f'[orca worker flow={flow_id}] CUBIC->agent handoff '
                          f'at t={t_s:.2f}s cwnd={cur_cwnd} '
                          f'warmup_elapsed={warmup_elapsed_s:.2f}s '
                          f'({"loss" if loss_now else "timeout"})', flush=True)

            if not agent_ready:
                # Slow start: CUBIC drives cwnd (no override) and we push no
                # experience — the agent didn't cause these transitions. RayNet
                # still pauses at every MTP, so acknowledge that step with an
                # empty action dictionary; otherwise CUBIC warmup deadlocks at
                # the first reset observation while the kernel-backed flow
                # would have continued autonomously.
                prev_state = None
                prev_action = 0.0
                a, mult, new_cwnd = 0.0, 1.0, cur_cwnd
                if simulation_backend:
                    try:
                        flow_backend.advance_without_action(flow_fd)
                    except Exception as e:
                        print(f'[orca worker] warmup advance failed: {e} - exiting',
                              flush=True)
                        break
            else:
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

                a, mult = actor.act(norm_s, noise_std=noise_std)
                raw_raynet_action = (
                    flow_backend.is_simulation_backend()
                    and getattr(_ACTION_PLUGIN, 'ACTION_OUTPUT', '') == 'raynet_action')
                if raw_raynet_action:
                    new_cwnd = float(_ACTION_PLUGIN.apply_cwnd(
                        cur_cwnd, a, cwnd_min, cwnd_max))
                    agent_locked = False
                else:
                    desired_cwnd = _ACTION_PLUGIN.apply_cwnd(
                        cur_cwnd, a, cwnd_min, cwnd_max)

                    # Feed the BBR-style min-RTT filter with the kernel's
                    # persistent srtt (srtt_us is usec<<3; divide by 8). Falls
                    # back to avg_urtt only if srtt isn't populated yet.
                    srtt_raw_us = float(raw.get('srtt_us', 0) or 0)
                    filter_rtt_us = (srtt_raw_us / 8.0) if srtt_raw_us > 0 else float(raw.get('avg_urtt', 0) or 0)
                    clock_t = flow_backend.observation_clock(raw, wall_now=t_step_start)
                    probe.observe_rtt(clock_t, filter_rtt_us)

                    actual_cwnd, agent_locked, _probe_transition = probe.decide(
                        clock_t, cur_cwnd, desired_cwnd)
                    new_cwnd = int(np.clip(actual_cwnd, cwnd_min, cwnd_max))
                try:
                    flow_backend.set_cwnd(flow_fd, new_cwnd)
                except Exception as e:
                    print(f'[orca worker] set_cwnd failed: {e} - exiting', flush=True)
                    break

                if agent_locked:
                    # Probe owned this step. Invalidate prev_state so the next
                    # iteration skips its push — the (s_pre, a_pre, r_probe,
                    # s_probe) tuple would lie about causality.
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
                        step = int(payload.get('step', step))
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
                    '0.000',
                    f'{a:.4f}', f'{noise_std:.4f}',
                ] + [f'{float(x):.5f}' for x in norm_s])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            if not simulation_backend:
                next_tick = sleep_to_grid(next_tick, interval_s_cfg)

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
        print(f'[orca worker cport={cport} flow={flow_id}] done', flush=True)


if __name__ == '__main__':
    run()
