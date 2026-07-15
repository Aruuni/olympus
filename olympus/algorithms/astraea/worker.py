"""Per-flow decentralized worker for the Astraea recreation.

Decentralized execution exactly as in the paper: each flow runs the
parameter-shared deterministic actor on its OWN stacked local state (the
zero-initialized shift register of the last ``w`` MTP observations — the
original ``s0_rec_buffer``). Global information is training-only: the worker
pushes the raw per-flow statistics (throughput, latency, loss, pacing, cwnd,
link context) with every transition so the learner can assemble the global
state (Table 2) and the global reward (Eq. 8) by regrouping simultaneous
flows on ``(group_id, group_step)``.
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
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _ROOT)

from olympus.algorithms.astraea import model
from olympus.common.action_plugins import load_action_module
from olympus.common.async_pusher import AsyncPusher
from olympus.common.bbr_probe import BbrProbe
from olympus.common.pace import sleep_to_grid
from olympus.common.registry import reward_module
from olympus.common import flow_backend, link_context, runtime_config

_ACTION_PLUGIN = load_action_module()
_LOG_FLUSH_EVERY = 50


class PaperRandomExploration:
    """Released Astraea random-action replacement with global decay.

    Workers reserve small ranges from the learner so the epsilon schedule is
    based on total actions across every parallel flow, rather than resetting
    whenever an episode launches a new worker process.

    ``decay_steps`` here is the EFFECTIVE global budget. The config value
    ``random_exploration_steps`` is per actor (the released 1.5M was per one
    of the original's ~2 actor processes); the caller multiplies it by this
    run's concurrent-actor count (see ``_exploration_actor_count``) so the
    random phase lasts as long per actor as the original schedule, instead
    of burning out in minutes when 60 flows share one counter.
    """

    def __init__(self, manager, decay_steps=1_500_000,
                 epsilon_end=0.05, allocation_block=256, rng=None):
        self.manager = manager
        self.decay_steps = max(1, int(decay_steps))
        self.epsilon_end = min(max(float(epsilon_end), 0.0), 1.0)
        self.allocation_block = max(1, int(allocation_block))
        self.rng = rng if rng is not None else np.random
        self._next = 0
        self._limit = 0
        self._fallback_count = 0

    def _index(self):
        if self._next >= self._limit:
            try:
                start = int(self.manager.reserve_action_steps(
                    self.allocation_block))
            except Exception:
                start = self._fallback_count
                self._fallback_count += self.allocation_block
            self._next = start
            self._limit = start + self.allocation_block
        index = self._next
        self._next += 1
        return index

    def epsilon(self, index):
        # This is the released Random_Noise schedule: epsilon begins at 1 and
        # loses 1/noise_exp per action until reaching the 0.05 floor.
        return max(
            self.epsilon_end,
            1.0 - float(index) / float(self.decay_steps),
        )

    def action(self, policy_action):
        index = self._index()
        epsilon = self.epsilon(index)
        if float(self.rng.uniform(0.0, 1.0)) < epsilon:
            return float(self.rng.uniform(-1.0, 1.0)), epsilon, True
        return float(policy_action), epsilon, False


def _mean_flow_count(spec, default=2.0) -> float:
    """Mean concurrent flows from a sweep spec ('2', '2-4', [2, 3, 4])."""
    if spec is None:
        return float(default)
    if isinstance(spec, str):
        text = spec.strip()
        if '-' in text:
            lo, hi = text.split('-', 1)
            values = list(range(int(lo), int(hi) + 1))
        else:
            values = [int(text)]
    elif isinstance(spec, (list, tuple)):
        values = [int(v) for v in spec]
    else:
        values = [int(spec)]
    values = [max(1, v) for v in values] or [int(default)]
    return sum(values) / float(len(values))


def _exploration_actor_count(cfg) -> float:
    """Concurrent actors sharing the global epsilon counter.

    ``agent.random_exploration_actors`` overrides explicitly; ``auto``
    estimates n_parallel episode slots x mean flows per episode from the
    resolved config.
    """
    raw = runtime_config.agent_value(
        cfg, 'random_exploration_actors', default='auto')
    text = str(raw).strip().lower()
    if text not in ('', 'auto'):
        return max(1.0, float(raw))
    orchestrator = (cfg.get('orchestrator')
                    if isinstance(cfg.get('orchestrator'), dict) else {})
    n_parallel = float(
        orchestrator.get('n_parallel', cfg.get('n_parallel', 1)) or 1)
    flows = _mean_flow_count((cfg.get('sweep') or {}).get('flows'))
    return max(1.0, n_parallel * flows)


class _Mgr(BaseManager):
    pass


_Mgr.register('service')


def _make_pusher(manager):
    """Off-thread experience sender; None when there is no learner attached."""
    if not manager:
        return None
    return AsyncPusher(
        lambda exps: manager.push_exp_batch(exps),
        lambda e: manager.push_exp(e),
    )


def _connect_manager():
    address = os.environ.get('SAO_MANAGER_ADDR', '')
    key = os.environ.get('SAO_MANAGER_KEY', '')
    if not address or not key:
        return None
    host, port = address.rsplit(':', 1)
    manager = _Mgr(
        address=(host, int(port)), authkey=bytes.fromhex(key))
    deadline = time.monotonic() + 10.0
    last_error = None
    while time.monotonic() < deadline:
        try:
            manager.connect()
            return manager.service()
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    print(f'[worker] manager connect failed: {last_error}', flush=True)
    return None


def _srtt_ms(raw: dict) -> float:
    try:
        srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
        fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    srtt_us = srtt_raw / 8.0 if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1e3


def _drain_state_fd(fd: int, stop: threading.Event):
    try:
        import select
        while not stop.is_set():
            readable, _, _ = select.select([fd], [], [], 0.1)
            if readable:
                os.read(fd, 4096)
    except Exception:
        pass


def run():
    cfg = runtime_config.load_config()
    reward_name = str(runtime_config.runtime_value(
        cfg, 'reward', env='SAO_REWARD', default='astraea'))
    reward_plugin = reward_module(reward_name)

    flow_fd = int(os.environ['OC_FLOW_FD'])
    flow_id = int(os.environ['OC_FLOW_ID'])
    cport = int(os.environ.get('OC_CPORT', '0'))
    cport_base = int(os.environ.get('SAO_CPORT_BASE', str(cport)))
    n_agents = int(os.environ.get('SAO_N_AGENTS', '2'))
    state_fd = int(os.environ.get('OC_STATE_FD', '-1'))
    episode = int(os.environ.get('SAO_EPISODE', '0'))
    instance_id = int(os.environ.get('SAO_INSTANCE_ID', '0'))
    checkpoint = os.environ.get('SAO_CHECKPOINT', '')

    interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20'))
    interval_s = interval_ms / 1000.0
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '4'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'
    require_checkpoint = (
        os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1')
    simulation_backend = flow_backend.is_simulation_backend()
    noise_std = float(os.environ.get('SAO_NOISE_STD', '0.2'))
    if deterministic:
        noise_std = 0.0

    exploration_mode = str(runtime_config.agent_value(
        cfg, 'exploration_mode', default='random_replacement')).strip().lower()
    random_exploration_steps = int(runtime_config.agent_value(
        cfg, 'random_exploration_steps', default=1_500_000))
    random_epsilon_end = float(runtime_config.agent_value(
        cfg, 'random_epsilon_end', default=0.05))
    exploration_allocation_block = int(runtime_config.agent_value(
        cfg, 'exploration_allocation_block', default=256))

    agent_id_raw = os.environ.get('SAO_AGENT_ID', '')
    agent_id = int(agent_id_raw) if agent_id_raw else cport - cport_base
    agent_id = max(0, min(n_agents - 1, agent_id))
    group_id = f'slot{instance_id}_ep{episode}'

    h1 = int(runtime_config.agent_value(
        cfg, 'hidden', env='SAO_HIDDEN', default=256))
    h2 = int(runtime_config.agent_value(
        cfg, 'hidden2', env='SAO_HIDDEN2', default=128))
    history_len = int(runtime_config.agent_value(
        cfg, 'history_len', env='SAO_HISTORY_LEN',
        default=model.HISTORY_LEN_DEFAULT))
    actor_norm = str(runtime_config.agent_value(
        cfg, 'actor_norm', default='batchnorm')).strip().lower()
    model.set_expected_history(history_len)
    model.set_expected_actor_norm(actor_norm)

    actor = model.Actor(model.STATE_DIM, history_len, h1, h2, actor_norm)
    learner_step = -1
    if checkpoint and os.path.exists(checkpoint):
        try:
            ckpt = torch.load(
                checkpoint, map_location='cpu', weights_only=False)
            model.assert_checkpoint_state_compatible(ckpt, source=checkpoint)
            actor.load_state_dict(ckpt['actor'], strict=False)
            learner_step = int(ckpt.get('step', 0))
            print(
                f'[worker cport={cport} agent={agent_id}] loaded '
                f'checkpoint step={learner_step}',
                flush=True,
            )
        except Exception as exc:
            print(
                f'[worker cport={cport} agent={agent_id}] checkpoint '
                f'load failed ({exc}); fresh init',
                flush=True,
            )
            if require_checkpoint:
                raise
    elif require_checkpoint:
        raise FileNotFoundError(
            f'[worker cport={cport}] checkpoint not found: {checkpoint}')
    actor.eval()

    manager = _connect_manager()
    if manager:
        try:
            parameter_bytes = manager.pull_params()
            if parameter_bytes is not None:
                payload = torch.load(
                    io.BytesIO(parameter_bytes),
                    map_location='cpu',
                    weights_only=False,
                )
                model.assert_checkpoint_state_compatible(
                    payload, source='initial learner payload')
                actor.load_state_dict(payload['actor'], strict=False)
                actor.eval()
                learner_step = int(payload.get('step', learner_step))
        except Exception as exc:
            print(
                f'[worker cport={cport} agent={agent_id}] initial parameter '
                f'pull failed ({exc})',
                flush=True,
            )

    pusher = _make_pusher(manager)
    paper_exploration = None
    if not deterministic and exploration_mode == 'random_replacement':
        exploration_actors = _exploration_actor_count(cfg)
        effective_decay_steps = int(
            random_exploration_steps * exploration_actors)
        paper_exploration = PaperRandomExploration(
            manager,
            decay_steps=effective_decay_steps,
            epsilon_end=random_epsilon_end,
            allocation_block=exploration_allocation_block,
        )
        if agent_id == 0:
            print(
                f'[worker cport={cport}] exploration budget: '
                f'{random_exploration_steps} per-actor steps x '
                f'{exploration_actors:g} actors = '
                f'{effective_decay_steps} global steps',
                flush=True,
            )

    stop_drain = threading.Event()
    if state_fd >= 0:
        threading.Thread(
            target=_drain_state_fd,
            args=(state_fd, stop_drain),
            daemon=True,
        ).start()

    state_log_path = os.environ.get('SAO_TRACE_LOG', '')
    log_file = None
    log_writer = None
    if state_log_path:
        base, extension = os.path.splitext(state_log_path)
        state_log_path = f'{base}_a{agent_id}{extension}'
        os.makedirs(os.path.dirname(state_log_path), exist_ok=True)
        log_file = open(state_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_s', 'option', 'cwnd_mult', 'cwnd',
            'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
            'loss_ratio', 'reward', 'kalman_rtt_ms',
            'act_mu', 'act_sig', 'agent_id',
            'fair_bw_mbps', 'active_flows',
        ] + [f's{i}' for i in range(model.STATE_DIM)])

    _bw_mbps, base_rtt_us, _link_schedule = link_context.context_values()
    link_buffer_bytes = float(os.environ.get('SAO_LINK_BUFFER_BYTES', '0'))
    reward_calc = reward_plugin.make_reward_calc()
    episode_start = (
        float(os.environ.get('SAO_EPISODE_START', '0'))
        or time.monotonic()
    )
    next_tick = episode_start
    traj_id = f'astraea_{group_id}_a{agent_id}_f{flow_id}'

    # Astraea recurrent local state: zero history, newest frame shifted in
    # each MTP (the original s0_rec_buffer semantics).
    rec_buffer = np.zeros(history_len * model.STATE_DIM, dtype=np.float32)

    previous_state = None
    previous_action = 0.0
    previous_stats = None
    previous_urtt = 0.0
    previous_cwnd = 10
    peak_throughput = 0.0
    max_tput = 0.0
    last_read_us = None
    step_in_traj = 0
    last_group_step = 0
    ever_alive = False

    push_every = max(1, int(runtime_config.training_value(
        cfg, 'worker_push_every', env='OC_PUSH_EVERY', default=16)))
    experience_buffer = []
    weight_pull_every = max(
        1, int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50')))
    weight_pull_counter = 0
    log_flush_counter = 0

    dead_flow_ms = float(runtime_config.agent_value(
        cfg, 'dead_flow_ms', env='SAO_DEAD_FLOW_MS', default=500))
    dead_steps = 0
    dead_steps_limit = max(1, int(dead_flow_ms / interval_ms))
    probe = BbrProbe.from_env()

    print(
        f'[worker alg=astraea reward={reward_name} cport={cport} '
        f'agent={agent_id} group={group_id} flow={flow_id}] '
        f'started interval={interval_ms:g}ms det={int(deterministic)} '
        f'exploration={exploration_mode} noise={noise_std:.2f} '
        f'history={history_len} h1={h1} h2={h2} norm={actor_norm}',
        flush=True,
    )

    try:
        while True:
            tick_start = time.monotonic()
            try:
                raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            except flow_backend.SimulationFinished:
                print('[worker] RayNet simulation finished; exiting', flush=True)
                break
            except Exception as exc:
                print(
                    f'[worker] get_tcp_deepcc_info failed: {exc}; exiting',
                    flush=True,
                )
                break

            avg_throughput = float(raw.get('avg_thr', 0.0) or 0.0)
            peak_throughput = max(peak_throughput, avg_throughput)
            max_tput = max(max_tput, avg_throughput)
            if avg_throughput <= 0.0:
                dead_steps += 1
            else:
                dead_steps = 0
                ever_alive = True
            flow_active = (not ever_alive) or dead_steps < dead_steps_limit

            # Interval loss rate the astraea_service way: lost bytes over the
            # wall time since the previous TCP_INFO read.
            now_us = int(time.monotonic() * 1_000_000)
            time_delta_us = (
                max(1, now_us - last_read_us) if last_read_us is not None else 1)
            last_read_us = now_us
            if 'loss_ratio' not in raw:
                loss_bytes = float(raw.get('loss_bytes', 0.0) or 0.0)
                raw['loss_ratio'] = loss_bytes * 1_000_000.0 / time_delta_us

            raw['prev_urtt'] = previous_urtt
            raw['prev_cwnd'] = previous_cwnd
            raw['peak_thr'] = peak_throughput
            raw['max_tput'] = max_tput
            raw['interval_ms'] = interval_ms
            normalized_state = model.normalize_state(raw)
            reward = float(reward_calc.step(raw))
            reward_components = (
                getattr(reward_calc, 'last_components', {}) or {})
            # Scheduled membership outranks the dead-flow heuristic: a starved
            # but scheduled flow must keep pushing or the learner's fairness
            # term never sees the starvation (MARL mask semantics).
            flow_present = reward_components.get('flow_present')
            if flow_present is not None:
                flow_active = bool(flow_present)
            urtt = float(raw.get('avg_urtt', 0.0) or 0.0)
            if urtt > 0.0:
                previous_urtt = urtt
            previous_cwnd = int(raw.get('cwnd', previous_cwnd) or previous_cwnd)

            link_bw_bytes = float(
                reward_components.get('link_bw_bytes', 0.0) or 0.0)
            rtt_ref_us = float(
                reward_components.get('rtt_ref_us', base_rtt_us) or base_rtt_us)
            stats_now = dict(
                throughput=avg_throughput,
                latency_us=urtt,
                loss_ratio=float(raw.get('loss_ratio', 0.0) or 0.0),
                pacing_rate=float(raw.get('pacing_rate', 0.0) or 0.0),
                cwnd=float(raw.get('cwnd', previous_cwnd) or previous_cwnd),
                min_rtt_us=float(raw.get('min_rtt', 0.0) or 0.0),
                rtt_ref_us=rtt_ref_us,
                link_bw=link_bw_bytes,
                buffer_bytes=link_buffer_bytes,
            )

            t_s = flow_backend.episode_seconds(
                raw, episode_start, wall_now=tick_start)
            group_step = flow_backend.episode_step(
                raw, episode_start, interval_s, step_in_traj,
                wall_now=tick_start)
            flow_backend.wait_collection_step(raw, group_step)
            last_group_step = group_step
            exploration_level = noise_std
            if previous_state is not None and flow_active and manager:
                experience_buffer.append(model.Experience(
                    state=previous_state,
                    action=previous_action,
                    reward=reward,
                    next_state=normalized_state,
                    agent_id=agent_id,
                    group_id=group_id,
                    group_step=group_step,
                    done=False,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
                    **stats_now,
                ))
                step_in_traj += 1
                if len(experience_buffer) >= push_every:
                    pusher.submit(experience_buffer)
                    experience_buffer.clear()

            # Once a flow leaves the episode schedule it is no longer part of
            # the learning problem. In RayNet, keep sending a frozen cwnd until
            # the simulator stops exposing the flow; otherwise the shared
            # episode waits forever for this worker's action.
            if not flow_active:
                previous_state = None
                previous_action = 0.0
                rec_buffer = np.zeros(
                    history_len * model.STATE_DIM, dtype=np.float32)
                new_cwnd = int(raw.get('cwnd', previous_cwnd) or previous_cwnd)
                multiplier = 0.0
                action = 0.0
                if simulation_backend:
                    try:
                        flow_backend.set_cwnd(flow_fd, new_cwnd)
                    except Exception as exc:
                        print(
                            f'[worker] set_cwnd failed: {exc}; exiting',
                            flush=True,
                        )
                        break
                else:
                    break
            else:
                rec_buffer = np.concatenate(
                    [rec_buffer[model.STATE_DIM:], normalized_state])
                if paper_exploration is not None:
                    policy_action, _ = actor.act(rec_buffer, noise_std=0.0)
                    action, exploration_level, _ = paper_exploration.action(
                        policy_action)
                    multiplier = float(_ACTION_PLUGIN.to_multiplier(action))
                else:
                    action, multiplier = actor.act(
                        rec_buffer, noise_std=noise_std)
                    exploration_level = noise_std

                current_cwnd = int(raw.get('cwnd', previous_cwnd) or previous_cwnd)
                desired_cwnd = _ACTION_PLUGIN.apply_cwnd(
                    current_cwnd, action, cwnd_min, cwnd_max)
                srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
                filter_rtt_us = (
                    srtt_raw / 8.0 if srtt_raw > 0.0
                    else float(raw.get('avg_urtt', 0.0) or 0.0)
                )
                clock_t = flow_backend.observation_clock(
                    raw, wall_now=tick_start)
                probe.observe_rtt(clock_t, filter_rtt_us)
                actual_cwnd, agent_locked, _ = probe.decide(
                    clock_t, current_cwnd, desired_cwnd)
                new_cwnd = int(np.clip(actual_cwnd, cwnd_min, cwnd_max))
                try:
                    flow_backend.set_cwnd(flow_fd, new_cwnd)
                except Exception as exc:
                    print(
                        f'[worker] set_cwnd failed: {exc}; exiting',
                        flush=True,
                    )
                    break

                if agent_locked:
                    previous_state = None
                    previous_action = 0.0
                else:
                    previous_state = normalized_state
                    previous_action = action
                    previous_stats = stats_now

            weight_pull_counter += 1
            if manager and weight_pull_counter >= weight_pull_every:
                weight_pull_counter = 0
                try:
                    parameter_bytes = manager.pull_params()
                    if parameter_bytes is not None:
                        payload = torch.load(
                            io.BytesIO(parameter_bytes),
                            map_location='cpu',
                            weights_only=False,
                        )
                        payload_step = int(payload.get('step', learner_step))
                        if payload_step > learner_step:
                            model.assert_checkpoint_state_compatible(
                                payload, source='learner payload')
                            actor.load_state_dict(
                                payload['actor'], strict=False)
                            actor.eval()
                            learner_step = payload_step
                except Exception:
                    pass

            if log_writer:
                active_flows = float(
                    reward_components.get('active_flows', 1.0) or 1.0)
                link_bw_mbps = link_bw_bytes * 8.0 / 1e6
                log_writer.writerow([
                    f'{t_s:.3f}',
                    0,
                    f'{multiplier:.3f}',
                    new_cwnd,
                    f'{avg_throughput * 8.0 / 1e6:.3f}',
                    f'{float(raw.get("avg_urtt", 0.0)) / 1e3:.3f}',
                    f'{_srtt_ms(raw):.3f}',
                    f'{float(raw.get("min_rtt", 0.0)) / 1e3:.3f}',
                    f'{float(raw.get("loss_ratio", 0.0)):.1f}',
                    f'{reward:.4f}',
                    '0.000',
                    f'{action:.4f}', f'{exploration_level:.4f}',
                    agent_id,
                    f'{link_bw_mbps / max(active_flows, 1.0):.3f}',
                    f'{active_flows:.0f}',
                ] + [f'{float(value):.5f}' for value in normalized_state])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            if not simulation_backend:
                next_tick = sleep_to_grid(next_tick, interval_s)
    finally:
        if pusher:
            if experience_buffer:
                pusher.submit(experience_buffer)
                experience_buffer.clear()
            # Drain queued batches before the terminal done-marker below so
            # ordering to the learner is preserved.
            pusher.close()

        if previous_state is not None and manager:
            try:
                manager.push_exp(model.Experience(
                    state=previous_state,
                    action=previous_action,
                    reward=0.0,
                    next_state=previous_state,
                    agent_id=agent_id,
                    group_id=group_id,
                    group_step=last_group_step + 1,
                    done=True,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
                    **(previous_stats or dict(
                        throughput=0.0, latency_us=0.0, loss_ratio=0.0,
                        pacing_rate=0.0, cwnd=float(previous_cwnd),
                        min_rtt_us=0.0, rtt_ref_us=base_rtt_us, link_bw=0.0,
                        buffer_bytes=link_buffer_bytes,
                    )),
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
        print(
            f'[worker cport={cport} agent={agent_id} flow={flow_id}] done',
            flush=True,
        )


if __name__ == '__main__':
    run()
