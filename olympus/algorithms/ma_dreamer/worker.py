"""Per-flow decentralized worker for MA-Dreamer.

Only the local Tempest world model and parameter-shared actor are loaded here.
The global model exists solely in the learner for shared imagination.
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

from olympus.algorithms.ma_dreamer import model
from olympus.common.action_plugins import load_action_module
from olympus.common.registry import reward_module
from olympus.common.bbr_probe import BbrProbe
from olympus.common import flow_backend

_ACTION_PLUGIN = load_action_module()


class _Mgr(BaseManager):
    pass


_Mgr.register('service')

_LOG_FLUSH_EVERY = 50


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


def _build_models(config):
    local_world = model.LocalWorldModel(
        state_dim=model.STATE_DIM,
        action_dim=model.ACTION_DIM,
        hidden=config['hidden'],
        embed_dim=config['embed_dim'],
        h_dim=config['h_dim'],
        latent_groups=config['latent_groups'],
        latent_classes=config['latent_classes'],
        reward_bins=config['reward_bins'],
    )
    actor = model.Actor(
        config['h_dim'],
        local_world.latent_dim,
        config['hidden'],
        log_std_min=config['actor_log_std_min'],
        log_std_max=config['actor_log_std_max'],
    )
    return local_world, actor


def run():
    reward_name = os.environ.get('SAO_REWARD', 'tempest_fairness_ma')
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
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '10'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'
    require_checkpoint = (
        os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1')
    simulation_backend = flow_backend.is_simulation_backend()

    agent_id_raw = os.environ.get('SAO_AGENT_ID', '')
    if agent_id_raw:
        agent_id = int(agent_id_raw)
    else:
        agent_id = cport - cport_base
    agent_id = max(0, min(n_agents - 1, agent_id))
    group_id = f'slot{instance_id}_ep{episode}'

    config = {
        'hidden': int(os.environ.get('SAO_DREAMER_HIDDEN', '256')),
        'embed_dim': int(os.environ.get('SAO_DREAMER_EMBED', '128')),
        'h_dim': int(os.environ.get('SAO_DREAMER_H_DIM', '256')),
        'latent_groups': int(
            os.environ.get('SAO_DREAMER_GROUPS', '8')),
        'latent_classes': int(
            os.environ.get('SAO_DREAMER_CLASSES', '8')),
        'reward_bins': int(
            os.environ.get('SAO_DREAMER_REWARD_BINS', '255')),
        'actor_log_std_min': float(
            os.environ.get('SAO_DREAMER_ACTOR_LOG_STD_MIN', '-2.0')),
        'actor_log_std_max': float(
            os.environ.get('SAO_DREAMER_ACTOR_LOG_STD_MAX', '0.0')),
    }
    local_world, actor = _build_models(config)

    learner_step = -1
    if checkpoint and os.path.exists(checkpoint):
        try:
            ckpt = torch.load(
                checkpoint, map_location='cpu', weights_only=False)
            model.assert_checkpoint_state_compatible(
                ckpt, source=checkpoint)
            local_world.load_state_dict(
                ckpt['local_world_model'], strict=False)
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
    local_world.eval()
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
                local_world.load_state_dict(
                    payload['local_world_model'], strict=False)
                actor.load_state_dict(payload['actor'], strict=False)
                local_world.eval()
                actor.eval()
                learner_step = int(payload.get('step', learner_step))
        except Exception as exc:
            print(
                f'[worker cport={cport} agent={agent_id}] initial parameter '
                f'pull failed ({exc})',
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

    base_rtt_us = float(os.environ.get('OC_BASE_RTT_US', '20000'))
    model.reset_tempest_kalman(base_rtt_us)
    reward_calc = reward_plugin.make_reward_calc()
    episode_start = (
        float(os.environ.get('SAO_EPISODE_START', '0'))
        or time.monotonic()
    )
    traj_id = f'ma_dreamer_{group_id}_a{agent_id}_f{flow_id}'

    previous_state = None
    previous_action = 0.0
    previous_avg_throughput = 0.0
    previous_urtt = 0.0
    previous_cwnd = 10
    peak_throughput = 0.0
    step_in_traj = 0
    ever_alive = False

    push_every = max(1, int(os.environ.get('OC_PUSH_EVERY', '16')))
    experience_buffer = []
    weight_pull_every = max(
        1, int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50')))
    weight_pull_counter = 0
    log_flush_counter = 0

    dead_flow_ms = float(os.environ.get('SAO_DEAD_FLOW_MS', '1000'))
    dead_steps = 0
    dead_steps_limit = max(1, int(dead_flow_ms / interval_ms))
    probe = BbrProbe.from_env()

    h, z = local_world.rssm.initial(1, torch.device('cpu'))
    action_previous_tensor = torch.zeros(1, model.ACTION_DIM)

    print(
        f'[worker alg=ma_dreamer reward={reward_name} cport={cport} '
        f'agent={agent_id} group={group_id} flow={flow_id}] '
        f'started interval={interval_ms:g}ms det={int(deterministic)} '
        f'latent={config["latent_groups"]}x{config["latent_classes"]}',
        flush=True,
    )

    try:
        while True:
            tick_start = time.monotonic()
            try:
                raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            except Exception as exc:
                print(
                    f'[worker] get_tcp_deepcc_info failed: {exc}; exiting',
                    flush=True,
                )
                break

            avg_throughput = float(raw.get('avg_thr', 0.0))
            peak_throughput = max(peak_throughput, avg_throughput)
            if avg_throughput <= 0.0:
                dead_steps += 1
            else:
                dead_steps = 0
                ever_alive = True
            flow_active = (not ever_alive) or dead_steps < dead_steps_limit

            raw['prev_urtt'] = previous_urtt
            raw['prev_cwnd'] = previous_cwnd
            raw['peak_thr'] = peak_throughput
            raw['interval_ms'] = interval_ms
            normalized_state = model.normalize_state(raw)
            reward = float(reward_calc.step(raw))
            reward_components = (
                getattr(reward_calc, 'last_components', {}) or {})
            # The environment's join/leave schedule decides which flows are in
            # the episode. It must outrank the dead-flow heuristic: a starved
            # flow (zero throughput while still scheduled present) has to keep
            # pushing experiences, or it drops out of the learner's agent mask
            # and the team fairness term never sees the starvation.
            flow_present = reward_components.get('flow_present')
            if flow_present is not None:
                flow_active = bool(flow_present)
            urtt = float(raw.get('avg_urtt', 0.0))
            if urtt > 0.0:
                previous_urtt = urtt
            previous_cwnd = int(raw.get('cwnd', previous_cwnd))

            t_s = flow_backend.episode_seconds(
                raw, episode_start, wall_now=tick_start)
            group_step = (
                step_in_traj if simulation_backend else int(round(t_s / interval_s))
                if interval_s > 0.0 else step_in_traj
            )
            if previous_state is not None and flow_active and manager:
                experience_buffer.append(model.Experience(
                    state=previous_state,
                    action=previous_action,
                    reward=reward,
                    next_state=normalized_state,
                    avg_throughput=avg_throughput,
                    agent_id=agent_id,
                    group_id=group_id,
                    group_step=group_step,
                    done=False,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
                ))
                step_in_traj += 1
                if len(experience_buffer) >= push_every:
                    try:
                        manager.push_exp_batch(list(experience_buffer))
                    except Exception:
                        for experience in experience_buffer:
                            try:
                                manager.push_exp(experience)
                            except Exception:
                                pass
                    experience_buffer.clear()

            with torch.no_grad():
                state_tensor = torch.from_numpy(
                    normalized_state).unsqueeze(0)
                embedding = local_world.encoder(state_tensor)
                h, z, _, _ = local_world.rssm.step(
                    h, z, action_previous_tensor, embedding)
                action_tensor, multiplier_tensor, mu, log_std = actor.act(
                    h, z, deterministic=deterministic)
            action = float(action_tensor.item())
            multiplier = float(multiplier_tensor.item())

            current_cwnd = int(raw.get('cwnd', 10))
            desired_cwnd = _ACTION_PLUGIN.apply_cwnd(
                current_cwnd, action, cwnd_min, cwnd_max)
            srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
            filter_rtt_us = (
                srtt_raw / 8.0 if srtt_raw > 0.0
                else float(raw.get('avg_urtt', 0.0) or 0.0)
            )
            clock_t = flow_backend.observation_clock(raw, wall_now=tick_start)
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
                previous_avg_throughput = avg_throughput
            action_previous_tensor = action_tensor
            if action_previous_tensor.dim() == 1:
                action_previous_tensor = action_previous_tensor.unsqueeze(0)

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
                        payload_step = int(
                            payload.get('step', learner_step))
                        if payload_step > learner_step:
                            model.assert_checkpoint_state_compatible(
                                payload, source='learner payload')
                            local_world.load_state_dict(
                                payload['local_world_model'], strict=False)
                            actor.load_state_dict(
                                payload['actor'], strict=False)
                            local_world.eval()
                            actor.eval()
                            learner_step = payload_step
                except Exception:
                    pass

            if log_writer:
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
                    f'{float(raw.get("kalman_min_rtt_us", 0.0)) / 1e3:.3f}',
                    f'{float(mu.item()):.4f}',
                    f'{float(log_std.exp().item()):.4f}',
                    agent_id,
                    f'{float(reward_components.get("fair_bw_mbps", 0.0)):.3f}',
                    f'{float(reward_components.get("active_flows", 0.0)):.0f}',
                ] + [f'{float(value):.5f}' for value in normalized_state])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            elapsed = time.monotonic() - tick_start
            if not simulation_backend and elapsed < interval_s:
                time.sleep(interval_s - elapsed)
    finally:
        if manager and experience_buffer:
            try:
                manager.push_exp_batch(list(experience_buffer))
            except Exception:
                for experience in experience_buffer:
                    try:
                        manager.push_exp(experience)
                    except Exception:
                        pass
            experience_buffer.clear()

        if previous_state is not None and manager:
            try:
                manager.push_exp(model.Experience(
                    state=previous_state,
                    action=previous_action,
                    reward=0.0,
                    next_state=previous_state,
                    avg_throughput=previous_avg_throughput,
                    agent_id=agent_id,
                    group_id=group_id,
                    group_step=(
                        step_in_traj if simulation_backend else int(round(
                            (time.monotonic() - episode_start) / interval_s))
                        if interval_s > 0.0 else step_in_traj
                    ),
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
        print(
            f'[worker cport={cport} agent={agent_id} flow={flow_id}] done',
            flush=True,
        )


if __name__ == '__main__':
    run()
