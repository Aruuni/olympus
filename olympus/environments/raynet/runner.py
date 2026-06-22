"""Olympus adapter for RayNet simulation episodes.

RayNet owns OMNeT++ and ``omnetbind``. Olympus launches the RayNet runner as a
subprocess, exchanges JSON-lines messages over a private socket, and keeps the
Olympus-side responsibilities here: Orca observation adaptation, actor
inference, reward/state transforms, replay pushes, and checkpoint pulls.
"""

import csv
import io
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing.managers import BaseManager
from pathlib import Path

import torch


IGNORED_AGENT_IDS = {'__all__', 'SIMULATION_END'}
_LOG_FLUSH_EVERY = 50


class _Mgr(BaseManager):
    pass


_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('pull_params')


class RayNetEpisodeClient:
    """JSON-lines client for one RayNet-owned simulation process."""

    def __init__(self, command, *, cwd=None, env=None):
        parent_sock, child_sock = socket.socketpair()
        self._sock = parent_sock
        self._reader = parent_sock.makefile('r', encoding='utf-8', newline='\n')
        self._writer = parent_sock.makefile('w', encoding='utf-8', newline='\n')
        command = list(command) + ['--control-fd', str(child_sock.fileno())]
        self._proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            pass_fds=(child_sock.fileno(),),
        )
        child_sock.close()

    @property
    def returncode(self):
        return self._proc.poll()

    def _send(self, message):
        self._writer.write(json.dumps(message, separators=(',', ':')) + '\n')
        self._writer.flush()

    def _recv(self):
        line = self._reader.readline()
        if not line:
            code = self._proc.poll()
            raise RuntimeError(f'RayNet runner closed IPC channel returncode={code}')
        message = json.loads(line)
        if message.get('type') == 'error':
            detail = message.get('traceback') or message.get('message') or 'unknown error'
            raise RuntimeError(f'RayNet runner error:\n{detail}')
        return message

    def start(self, episode_config):
        self._send({'type': 'start', 'episode': episode_config})
        message = self._recv()
        if message.get('type') != 'reset':
            raise RuntimeError(f'expected RayNet reset message, got {message.get("type")!r}')
        return message

    def step(self, actions):
        self._send({'type': 'step', 'actions': actions})
        message = self._recv()
        if message.get('type') != 'step':
            raise RuntimeError(f'expected RayNet step message, got {message.get("type")!r}')
        return message

    def close(self):
        if self._proc.poll() is None:
            try:
                self._send({'type': 'close'})
                self._recv()
            except Exception:
                pass
        self.terminate()

    def terminate(self):
        try:
            self._reader.close()
            self._writer.close()
            self._sock.close()
        except OSError:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=3)


def _connect_manager(addr: str, key: str):
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
        except ConnectionRefusedError as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error is not None:
        raise last_error
    mgr.connect()
    return mgr


def _observation_to_list(observation):
    if hasattr(observation, 'to_list'):
        return list(observation.to_list())
    return list(observation)


def raynet_orca_observation_to_raw(observation) -> dict:
    """Map RayNet Orca's 15-value observation to Olympus Orca raw fields.

    RayNet's ``Orca::computeObservation`` emits:
    delay_ms, throughput_Bps, samples, interval_s, target_ms, cwnd_packets,
    pacing_Bps, loss_Bps, srtt_ms, ssthresh_packets, packets_out,
    retrans_out, max_packets_out, mss_bytes, min_rtt_ms.
    """
    values = [float(v) for v in _observation_to_list(observation)]
    if len(values) != 15:
        raise ValueError(
            f'RayNet Orca observation must contain 15 values, got {len(values)}')
    return {
        'delay_ms': values[0],
        'avg_urtt': values[0] * 1000.0,
        'delay_us': values[0] * 1000.0,
        'throughput': values[1],
        'avg_thr': values[1],
        'samples': values[2],
        'cnt': values[2],
        'count': values[2],
        'delta_t': values[3],
        'interval_s': values[3],
        'target': values[4],
        'cwnd': values[5],
        'pacing_rate': values[6],
        'loss_rate': values[7],
        'lost_rate': values[7],
        'loss_bytes': values[7] * max(values[3], 0.0),
        'lost_bytes': values[7] * max(values[3], 0.0),
        # Olympus socket workers receive Linux srtt_us in usec<<3. RayNet
        # exports milliseconds, so encode the same shifted representation.
        'srtt_us': values[8] * 1000.0 * 8.0,
        'srtt_ms': values[8],
        'snd_ssthresh': values[9],
        'packets_out': values[10],
        'retrans_out': values[11],
        'max_packets_out': values[12],
        'mss': values[13],
        'mss_cache': values[13],
        'min_rtt_ms': values[14],
        'min_rtt': values[14] * 1000.0,
        'min_rtt_us': values[14] * 1000.0,
    }


def _srtt_ms(raw: dict) -> float:
    srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
    fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1000.0


def _episode_paths(outputs: dict, alg_name: str, episode: int):
    episodes_dir = os.path.abspath(outputs['episodes_dir'])
    os.makedirs(episodes_dir, exist_ok=True)
    return os.path.join(episodes_dir, f'{alg_name}_state_ep{episode:06d}.csv')


def _open_trace(path: str):
    if not path:
        return None, None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handle = open(path, 'w', newline='')
    writer = csv.writer(handle)
    writer.writerow([
        't_s', 'option', 'cwnd_mult', 'cwnd',
        'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
        'loss_ratio', 'reward', 'kalman_rtt_ms',
        'act_mu', 'act_sig',
    ])
    return handle, writer


def _configure_runtime_env(cfg: dict, ecfg: dict, episode: int,
                           mgr_addr: str, mgr_key: str, state_log: str) -> dict:
    from olympus.common.action_plugins import action_env

    runtime = cfg.get('runtime', {}) or {}
    agent = cfg.get('agent', {}) or {}
    reward = cfg.get('reward', {}) or {}
    state_options = cfg.get('state_options', {}) or {}
    training = cfg.get('training', cfg) or {}
    link_schedule = ecfg.get('link_schedule', []) or []

    env = {
        'SAO_ALGORITHM': runtime.get('algorithm', 'orca'),
        'SAO_REWARD': runtime.get('reward', 'sage'),
        'SAO_STATE': runtime.get('state', 'default_orca'),
        'OC_STATE': runtime.get('state', 'default_orca'),
        'SAO_CHECKPOINT': os.path.abspath(training['checkpoint']),
        'SAO_MANAGER_ADDR': mgr_addr,
        'SAO_MANAGER_KEY': mgr_key,
        'SAO_LINK_BW': str(float(ecfg.get('bw', 100.0))),
        'SAO_BASE_RTT_US': str(float(ecfg.get('delay', 20.0)) * 1000.0),
        'SAO_INTERVAL_MS': str(float(agent.get('interval_ms', 20.0))),
        'SAO_CWND_MIN': str(int(agent.get('cwnd_min', 4))),
        'SAO_CWND_MAX': str(int(agent.get('cwnd_max', 10000))),
        'SAO_HIDDEN': str(int(agent.get('hidden', 256))),
        'SAO_HEAD_HIDDEN': str(int(agent.get('head_hidden', agent.get('hidden', 256)))),
        'SAO_REC_DIM': str(int(agent.get('rec_dim', 10))),
        'SAO_ORCA_REC_DIM': str(int(agent.get('rec_dim', 10))),
        'SAO_ORCA_USE_NORMALIZER': '1' if agent.get('use_normalizer', False) else '0',
        'SAO_ORCA_TARGET_MS': str(float(state_options.get('orca_target_ms', 50.0))),
        'SAO_NOISE_STD': str(float(agent.get('noise_std', cfg.get('noise_std', 0.2)))),
        'SAO_TRACE_LOG': state_log,
        'SAO_EPISODE': str(int(episode)),
        'OC_LINK_BW': str(float(ecfg.get('bw', 100.0))),
        'OC_BASE_RTT_US': str(float(ecfg.get('delay', 20.0)) * 1000.0),
        'OC_INTERVAL_MS': str(float(agent.get('interval_ms', 20.0))),
        'OC_LINK_SCHEDULE': json.dumps(link_schedule),
        'OC_ORCA_DELAY_MARGIN_COEF': str(float(reward.get('delay_margin_coef', 1.25))),
        'OC_PUSH_EVERY': str(int(training.get('worker_push_every', 16))),
    }
    env.update(action_env(cfg))
    os.environ.update(env)
    return env


def _load_actor(model, ckpt_path: str, hidden: int, head_hidden: int,
                require_checkpoint: bool = False):
    ckpt = None
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.assert_checkpoint_state_compatible(ckpt, source=ckpt_path)
            hidden, head_hidden = model.actor_arch_from_checkpoint(
                ckpt, hidden, head_hidden)
        except Exception:
            if require_checkpoint:
                raise
            ckpt = None
    elif require_checkpoint:
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    actor = model.Actor(model.STATE_DIM, hidden, head_hidden)
    if ckpt is not None:
        try:
            actor.load_state_dict(ckpt['actor'])
            step = int(ckpt.get('step', 0))
        except Exception:
            if require_checkpoint:
                raise
    actor.eval()
    return actor, step


def _pull_actor_params(mgr, model, actor, step: int) -> int:
    if mgr is None:
        return step
    try:
        param_bytes = mgr.pull_params()
        if param_bytes is None:
            return step
        payload = torch.load(io.BytesIO(param_bytes), map_location='cpu',
                             weights_only=False)
        model.assert_checkpoint_state_compatible(payload, source='learner payload')
        actor.load_state_dict(payload['actor_state_dict'])
        actor.eval()
        return int(payload.get('step', step))
    except Exception:
        return step


def _runner_done(terminateds: dict, info: dict) -> bool:
    return bool(
        (terminateds or {}).get('__all__', False)
        or (info or {}).get('simDone', False)
    )


def _raynet_command(cfg: dict, ecfg: dict):
    paths_cfg = cfg.get('paths', {}) or {}
    raynet_path = Path(
        ecfg.get('raynet_path')
        or paths_cfg.get('raynet')
        or os.environ.get('RAYNET_PATH', '/home/james/raynet')
    ).expanduser()
    runner = Path(paths_cfg.get(
        'raynet_runner',
        raynet_path / 'runners' / 'olympus_runner.sh',
    )).expanduser()
    if not runner.exists():
        raise FileNotFoundError(f'RayNet runner not found: {runner}')
    return [str(runner)], raynet_path


def _episode_config(ecfg: dict, cfg: dict, env_vars: dict):
    episode = dict(ecfg)
    episode.setdefault('protocol', 'orca')
    episode.setdefault('section', ecfg.get('config_section', 'General'))
    episode.setdefault('interval_s', float(env_vars['SAO_INTERVAL_MS']) / 1000.0)
    replacements = dict(episode.get('replacements') or {})

    bw_mbps = float(ecfg.get('bw', 100.0))
    delay_ms = float(ecfg.get('delay', ecfg.get('base_rtt_ms', 20.0)))
    interval_s = float(episode['interval_s'])
    duration_s = None if episode.get('duration') is None else float(episode['duration'])
    qsize = ecfg.get('qsize')
    if qsize is None:
        bdp_mult = float(ecfg.get('bdp_mult', 1.0))
        qsize_bits = max(1.0, bw_mbps * delay_ms * 1000.0 * bdp_mult)
    else:
        qsize_bits = float(qsize) * 8.0

    replacements.setdefault('home', os.environ.get('HOME', str(Path.home())))
    replacements.setdefault('raynet_path', str(
        ecfg.get('raynet_path')
        or (cfg.get('paths', {}) or {}).get('raynet')
        or os.environ.get('RAYNET_PATH', '/home/james/raynet')
    ))
    replacements.setdefault('bw', f'{bw_mbps:.12g}Mbps')
    replacements.setdefault('delay', f'{delay_ms / 2.0:.12g}ms')
    replacements.setdefault('qsize', f'{qsize_bits:.12g}b')
    if duration_s is not None:
        replacements.setdefault(
            'max_rl_steps',
            str(max(1, int(round(duration_s / max(interval_s, 1e-9)))))
        )
    episode['replacements'] = replacements
    return episode


def _default_client_factory(command, *, cwd=None, env=None):
    return RayNetEpisodeClient(command, cwd=cwd, env=env)


def run_episode_raynet(cfg, ecfg, episode, listener_bin, python_bin,
                       mgr_addr, mgr_key, instance_id, client_factory=None):
    """Run one RayNet Orca episode through the RayNet-owned IPC runner."""
    runtime = cfg.get('runtime', {}) or {}
    alg_name = runtime.get('algorithm', 'orca')
    if alg_name != 'orca':
        raise ValueError(f'RayNet v1 supports runtime.algorithm=orca, got {alg_name!r}')
    protocol = str(ecfg.get('protocol', 'orca')).lower()
    if protocol != 'orca':
        raise ValueError(f'RayNet v1 supports protocol=orca, got {protocol!r}')

    ini_raw = str(ecfg.get('ini_path', '')).strip()
    if not ini_raw:
        raise ValueError('RayNet episode requires ini_path')
    ini_path = Path(ini_raw).expanduser()
    if not ini_path.exists():
        raise FileNotFoundError(f'RayNet ini_path not found: {ini_path}')

    outputs = cfg.get('outputs', {}) or {}
    state_log = _episode_paths(outputs, alg_name, episode)
    env_vars = _configure_runtime_env(cfg, ecfg, episode, mgr_addr, mgr_key, state_log)

    from olympus.algorithms.orca import model

    agent_cfg = cfg.get('agent', {}) or {}
    training = cfg.get('training', cfg) or {}
    hidden = int(agent_cfg.get('hidden', 256))
    head_hidden = int(agent_cfg.get('head_hidden', hidden))
    noise_std = float(env_vars['SAO_NOISE_STD'])
    if os.environ.get('SAO_DETERMINISTIC', '0') == '1':
        noise_std = 0.0
    ckpt_path = os.path.abspath(training['checkpoint'])
    actor, learner_step = _load_actor(
        model, ckpt_path, hidden, head_hidden,
        require_checkpoint=os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1')
    mgr = _connect_manager(mgr_addr, mgr_key)

    transform_by_agent = defaultdict(lambda: model.OrcaRepoTransform(
        delay_margin_coef=float(env_vars['OC_ORCA_DELAY_MARGIN_COEF']),
        use_normalizer=env_vars.get('SAO_ORCA_USE_NORMALIZER') == '1'))
    history_by_agent = defaultdict(lambda: model.HistoryStack(model.REC_DIM))
    prev_state = {}
    prev_action = {}
    step_in_traj = defaultdict(int)
    exp_buf = []
    push_every = max(1, int(env_vars.get('OC_PUSH_EVERY', '16')))
    weight_pull_every = int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50'))
    weight_pull_counter = 0
    rewards_seen = []

    log_file, log_writer = _open_trace(state_log)
    log_flush_counter = 0
    t0 = time.monotonic()
    command, raynet_path = _raynet_command(cfg, ecfg)
    proc_env = dict(os.environ)
    proc_env.update(env_vars)
    proc_env.setdefault('RAYNET_PATH', str(raynet_path))
    factory = client_factory or _default_client_factory
    client = None

    def push_buffer(force=False):
        if not mgr or not exp_buf:
            exp_buf.clear()
            return
        if force or len(exp_buf) >= push_every:
            try:
                mgr.push_exp_batch(list(exp_buf))
            except Exception:
                for exp in exp_buf:
                    try:
                        mgr.push_exp(exp)
                    except Exception:
                        pass
            exp_buf.clear()

    def process_observations(observations, episode_done=False):
        nonlocal learner_step, weight_pull_counter, log_flush_counter
        next_actions = {}
        for agent_index, (agent_id, observation) in enumerate(sorted((observations or {}).items())):
            if agent_id in IGNORED_AGENT_IDS:
                continue
            raw = raynet_orca_observation_to_raw(observation)
            interval_s = max(float(raw.get('interval_s', 0.0) or 0.0), 1e-6)
            transform = transform_by_agent[agent_id]
            base_state, reward = transform.step(
                raw, interval_s=interval_s,
                target=float(env_vars['SAO_ORCA_TARGET_MS']),
                evaluation=False)
            norm_s = history_by_agent[agent_id].push(base_state)
            rewards_seen.append(float(reward))

            if agent_id in prev_state and mgr:
                done = bool(episode_done)
                exp_buf.append(model.Experience(
                    state=prev_state[agent_id],
                    action=prev_action[agent_id],
                    reward=float(reward),
                    next_state=norm_s,
                    done=done,
                    traj_id=f'raynet_orca_{episode}_{instance_id}_{agent_id}',
                    step_in_traj=step_in_traj[agent_id],
                ))
                step_in_traj[agent_id] += 1
                push_buffer()

            action, mult = actor.act(norm_s, noise_std=noise_std)
            next_actions[agent_id] = float(action)
            prev_state[agent_id] = norm_s
            prev_action[agent_id] = float(action)

            weight_pull_counter += 1
            if weight_pull_counter >= weight_pull_every:
                weight_pull_counter = 0
                learner_step = _pull_actor_params(mgr, model, actor, learner_step)

            if log_writer and agent_index == 0:
                log_writer.writerow([
                    f'{time.monotonic() - t0:.3f}', 0, f'{mult:.3f}',
                    int(raw.get('cwnd', 0)),
                    f'{float(raw.get("avg_thr", 0.0)) * 8.0 / 1e6:.3f}',
                    f'{float(raw.get("avg_urtt", 0.0)) / 1000.0:.3f}',
                    f'{_srtt_ms(raw):.3f}',
                    f'{float(raw.get("min_rtt", 0.0)) / 1000.0:.3f}',
                    f'{float(raw.get("loss_rate", 0.0)):.3f}',
                    f'{float(reward):.4f}',
                    '0.000',
                    f'{float(action):.4f}', f'{noise_std:.4f}',
                ])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()
        return next_actions

    try:
        print(f'[slot={instance_id}] ep={episode} raynet protocol=orca '
              f'ini={ini_path} section={ecfg.get("section", "General")}', flush=True)
        client = factory(command, cwd=str(raynet_path), env=proc_env)
        reset_msg = client.start(_episode_config(ecfg, cfg, env_vars))
        actions = process_observations(reset_msg.get('observations') or {})

        while True:
            step_msg = client.step(actions)
            observations = step_msg.get('observations') or {}
            terminateds = step_msg.get('terminateds') or {}
            info = step_msg.get('info') or {}
            episode_done = _runner_done(terminateds, info)
            actions = process_observations(observations, episode_done=episode_done)
            if episode_done:
                break
            if not actions and not observations:
                actions = {}

        for agent_id, state in list(prev_state.items()):
            if mgr:
                exp_buf.append(model.Experience(
                    state=state,
                    action=prev_action.get(agent_id, 0.0),
                    reward=0.0,
                    next_state=state,
                    done=True,
                    traj_id=f'raynet_orca_{episode}_{instance_id}_{agent_id}',
                    step_in_traj=step_in_traj[agent_id],
                ))
        push_buffer(force=True)
    finally:
        if client is not None:
            client.terminate()
        if log_file:
            try:
                log_file.flush()
            except Exception:
                pass
            log_file.close()

    ep_return = float(sum(rewards_seen)) if rewards_seen else None
    return ep_return, ecfg, ecfg.get('link_schedule', []) or []
