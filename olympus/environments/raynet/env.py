"""RayNet simulation backend for Olympus."""

import json
import os
import secrets
import socket
import subprocess
import threading
import time
from multiprocessing.managers import BaseManager
from pathlib import Path

from olympus.environments.base import NetworkEnv


RAYNET_BASE_ENVIRONMENT = Path('_environments') / 'base_environment.ini'
RAYNET_DEFAULT_SECTION = 'General'
RAYNET_CC_ALGO_BY_LISTENER = {
    'astraea': 'TcpPacedNoCC',
    'orca': 'TcpCubic',
}


class RayNetEpisodeClient:
    """JSON-lines client for one RayNet-owned simulation process."""

    def __init__(self, command, *, cwd=None, env=None, log_path=None,
                 trace_path=None):
        parent_sock, child_sock = socket.socketpair()
        self._sock = parent_sock
        self._reader = parent_sock.makefile('r', encoding='utf-8', newline='\n')
        self._writer = parent_sock.makefile('w', encoding='utf-8', newline='\n')
        self._log_file = None
        self._trace_file = None
        command = list(command) + ['--control-fd', str(child_sock.fileno())]
        popen_kwargs = {}
        if log_path:
            log_path = Path(log_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, 'a', buffering=1, encoding='utf-8')
            self._log_file.write(
                f'[raynet-runner] command={" ".join(command)} cwd={cwd or ""}\n')
            self._log_file.flush()
            popen_kwargs.update({
                'stdout': self._log_file,
                'stderr': subprocess.STDOUT,
            })
        if trace_path:
            trace_path = Path(trace_path).expanduser()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = open(
                trace_path, 'a', buffering=1, encoding='utf-8')
            self._trace('runner_start', {
                'command': command,
                'cwd': cwd or '',
            })
        self._proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            pass_fds=(child_sock.fileno(),),
            **popen_kwargs,
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

    def _trace(self, event, payload=None):
        if self._trace_file is None:
            return
        record = {
            'event': event,
            'wall_time': time.time(),
        }
        if payload:
            record.update(payload)
        self._trace_file.write(
            json.dumps(record, separators=(',', ':'), default=str) + '\n')
        self._trace_file.flush()

    def start(self, episode_config):
        request = {'type': 'start', 'episode': episode_config}
        self._trace('send_start', {'message': request})
        self._send(request)
        message = self._recv()
        self._trace('recv_reset', {'message': message})
        if message.get('type') != 'reset':
            raise RuntimeError(f'expected RayNet reset message, got {message.get("type")!r}')
        return message

    def step(self, actions):
        request = {'type': 'step', 'actions': actions}
        self._trace('send_step', {'message': request})
        self._send(request)
        message = self._recv()
        self._trace('recv_step', {'message': message})
        if message.get('type') != 'step':
            raise RuntimeError(f'expected RayNet step message, got {message.get("type")!r}')
        return message

    def close(self):
        if self._proc.poll() is None:
            try:
                request = {'type': 'close'}
                self._trace('send_close', {'message': request})
                self._send(request)
                self._trace('recv_close', {'message': self._recv()})
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
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        if self._trace_file is not None:
            try:
                self._trace_file.flush()
                self._trace_file.close()
            except OSError:
                pass
            self._trace_file = None


class _FlowManager(BaseManager):
    pass


class _CollectionSyncManager(BaseManager):
    pass


class _RayNetCollectionSync:
    """Step barrier for RayNet simulations that share one learner."""

    def __init__(self, target_slots):
        self.condition = threading.Condition()
        self.target_slots = max(1, int(target_slots))
        self.slot_episode = {}
        self.slot_offset = {}
        self.slot_total = {}
        self.step_arrivals = {}
        self.step_required = {}

    def set_target_slots(self, target_slots):
        with self.condition:
            self.target_slots = max(1, int(target_slots))
            self.condition.notify_all()

    def wait_until_allowed(self, slot, episode, step, participant=None,
                           participants=1):
        slot = int(slot)
        episode = int(episode)
        step = max(0, int(step))
        participant = 'slot' if participant is None else str(participant)
        participants = max(1, int(participants or 1))
        with self.condition:
            total = self._step_total_locked(slot, episode, step)
            key = (slot, episode, total)
            self.step_required[key] = max(
                participants, int(self.step_required.get(key, 0)))
            arrivals = self.step_arrivals.setdefault(key, set())
            arrivals.add(participant)
            self._mark_complete_locked(slot, key, total)
            self.condition.notify_all()

            while (not self._step_complete_locked(key)
                   or not self._ready_locked(total)):
                self.condition.wait()
            self.condition.notify_all()
        return None

    def _step_total_locked(self, slot, episode, step):
        previous_episode = self.slot_episode.get(slot)
        if previous_episode is None:
            self.slot_episode[slot] = episode
            self.slot_offset[slot] = 0
        elif previous_episode != episode:
            self.slot_offset[slot] = self.slot_total.get(slot, 0) + 1
            self.slot_episode[slot] = episode
            self._clear_slot_steps_locked(slot)
        return int(self.slot_offset.get(slot, 0)) + step

    def _clear_slot_steps_locked(self, slot):
        for key in list(self.step_arrivals):
            if key[0] == slot:
                self.step_arrivals.pop(key, None)
                self.step_required.pop(key, None)

    def _mark_complete_locked(self, slot, key, total):
        if not self._step_complete_locked(key):
            return
        previous = int(self.slot_total.get(slot, -1))
        self.slot_total[slot] = max(previous, int(total))

    def _step_complete_locked(self, key):
        required = max(1, int(self.step_required.get(key, 1)))
        arrivals = self.step_arrivals.get(key, set())
        return len(arrivals) >= required

    def _ready_locked(self, total):
        if len(self.slot_total) < self.target_slots:
            return False
        active_totals = sorted(self.slot_total.values(), reverse=True)
        if len(active_totals) < self.target_slots:
            return False
        return min(active_totals[:self.target_slots]) >= total


class _RayNetFlowService:
    """Virtual TCP-flow service backed by one RayNet simulation episode."""

    def __init__(self, env):
        self.env = env
        self.condition = threading.Condition()
        self.client = None
        self.observations = {}
        self.consumed = set()
        self.pending_actions = {}
        self.expected_flows = set()
        self.agent_by_flow = {}
        self.flow_by_agent = {}
        self.flow_assignment_order = []
        self.last_raw = {}
        self.info = {}
        self.done = False
        self.error = None

    def get_tcp_deepcc_info(self, flow_id):
        flow_id = int(flow_id)
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError(self.error)
                if self.done and flow_id not in self.observations:
                    raise RuntimeError('RayNet simulation finished')
                if flow_id in self.observations and flow_id not in self.consumed:
                    self.consumed.add(flow_id)
                    return dict(self.observations[flow_id])
                self.condition.wait()

    def set_cwnd(self, flow_id, cwnd):
        flow_id = int(flow_id)
        with self.condition:
            if self.done:
                return None
            if flow_id not in self.expected_flows:
                return None
            self.pending_actions[flow_id] = float(cwnd)
            if self.expected_flows <= set(self.pending_actions):
                self._advance_locked()
            self.condition.notify_all()
        return None

    def start_episode(self):
        try:
            command = [str(self.env.raynet_runner)]
            self.client = RayNetEpisodeClient(
                command,
                cwd=str(self.env.raynet_path),
                env=self.env.process_env(),
                log_path=self.env.runner_log_path,
                trace_path=self.env.control_trace_path,
            )
            reset_msg = self.client.start(self.env.episode_config())
            with self.condition:
                self._publish_observations_locked(
                    reset_msg.get('observations') or {},
                    reset_msg.get('info') or {},
                )
                # Some RayNet protocols (e.g. Orca) expose no learner observation
                # during the connection/slow-start warm-up. Advance the simulation
                # through that no-observation window so the worker has a real
                # observation to act on instead of blocking forever.
                self._pump_warmup_locked()
                self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = str(exc)
                self.done = True
                self.condition.notify_all()

    def wait(self):
        with self.condition:
            while not self.done and self.error is None:
                self.condition.wait()
            if self.error is not None:
                raise RuntimeError(self.error)

    def close(self):
        with self.condition:
            self.done = True
            self.condition.notify_all()
        if self.client is not None:
            self.client.terminate()

    def _advance_locked(self):
        actions = {}
        for flow_id in sorted(self.expected_flows):
            agent_id = self.agent_by_flow.get(flow_id)
            if agent_id is not None:
                actions[agent_id] = float(self.pending_actions[flow_id])
        self.pending_actions.clear()
        self.consumed.clear()
        self._step_locked(actions)
        # Carry on through any subsequent no-observation interval (warm-up gaps)
        # so the worker is never handed an empty observation set.
        self._pump_warmup_locked()

    def _pump_warmup_locked(self, cap=20000):
        steps = 0
        while ((not self.done) and self.error is None
               and not self.observations and steps < cap):
            steps += 1
            self._step_locked({})

    def _step_locked(self, actions):
        try:
            step_msg = self.client.step(actions)
            terminateds = step_msg.get('terminateds') or {}
            info = step_msg.get('info') or {}
            episode_done = bool(
                terminateds.get('__all__', False) or info.get('simDone', False))
            if episode_done:
                self.done = True
                self.observations = {}
                self.expected_flows = set()
            else:
                self._publish_observations_locked(
                    step_msg.get('observations') or {},
                    info,
                )
        except BaseException as exc:
            self.error = str(exc)
            self.done = True

    def _publish_observations_locked(self, observations, info=None):
        self.info = dict(info or {})
        raw_by_flow = {}
        for agent_id, observation in sorted((observations or {}).items()):
            flow_id = self._flow_id_for_agent(agent_id)
            if flow_id >= self.env.n:
                continue
            raw = self.env.observation_to_raw(observation)
            if 'time_s' in self.info:
                raw['time_s'] = self.info['time_s']
            if 'step_id' in self.info:
                raw['step_id'] = self.info['step_id']
            if 'group_step' in self.info:
                raw['group_step'] = self.info['group_step']
            raw_by_flow[flow_id] = raw
            self.last_raw[flow_id] = raw
        participants = max(1, len(raw_by_flow))
        for raw in raw_by_flow.values():
            raw['collection_participants'] = participants
        self.observations = raw_by_flow
        self.expected_flows = set(raw_by_flow)
        self.consumed = set()
        self.pending_actions = {}

    def set_flow_schedule(self, start_delays=None):
        delays = list(start_delays or [])[:self.env.n]
        delays.extend([0.0] * (self.env.n - len(delays)))
        self.flow_assignment_order = [
            flow_id for flow_id, _ in sorted(
                enumerate(delays), key=lambda item: (float(item[1]), item[0]))
        ]

    def _flow_id_for_agent(self, agent_id):
        agent_id = str(agent_id)
        if agent_id not in self.flow_by_agent:
            assigned = len(self.flow_by_agent)
            if assigned < len(self.flow_assignment_order):
                flow_id = self.flow_assignment_order[assigned]
            else:
                flow_id = assigned
            self.flow_by_agent[agent_id] = flow_id
            self.agent_by_flow[flow_id] = agent_id
        return self.flow_by_agent[agent_id]


_FLOW_SERVICE = None
_COLLECTION_SYNC_SERVICE = None


def _get_flow_service():
    return _FLOW_SERVICE


def _get_collection_sync_service():
    return _COLLECTION_SYNC_SERVICE


_FlowManager.register('flow_service', callable=_get_flow_service)
_CollectionSyncManager.register(
    'collection_sync', callable=_get_collection_sync_service)


def start_collection_sync(target_slots):
    global _COLLECTION_SYNC_SERVICE
    _COLLECTION_SYNC_SERVICE = _RayNetCollectionSync(target_slots)
    key = secrets.token_hex(16)
    manager = _CollectionSyncManager(
        address=('127.0.0.1', 0),
        authkey=bytes.fromhex(key),
    )
    server = manager.get_server()
    host, port = server.address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return {
        'addr': f'{host}:{port}',
        'key': key,
        'manager': manager,
        'service': _COLLECTION_SYNC_SERVICE,
        'thread': thread,
    }


class RaynetEnv(NetworkEnv):
    """Lightweight handle for RayNet simulation episodes."""

    def __init__(self, n=1, bw=10, delay=20, qsize=None, bdp_mult=1.0,
                 loss=None, duration=60, cport=11111, cc_algo='raynet',
                 instance_id=None, unique_cports=False, per_flow_delays=None,
                 raynet_path=None, ini_path=None, section='General',
                 protocol=None, raynet_runner=None, omnet_path=None, **extra):
        environment_config = dict(extra.get('environment_config') or {})
        self.environment_config = environment_config
        self.n = int(n)
        self.bw = float(bw)
        self.delay = float(delay)
        self.qsize = qsize
        self.bdp_mult = bdp_mult
        self.loss = loss
        self.duration = duration
        self.cport = cport
        self.cc_algo = cc_algo
        self.instance_id = instance_id
        self.unique_cports = unique_cports
        self.per_flow_delays = per_flow_delays
        self.protocol = str(protocol or cc_algo or 'raynet')
        self.listener_cc = self._listener_cc_name(environment_config, cc_algo)
        self.raynet_cc_algo = self._raynet_cc_algo(self.listener_cc)
        self.raynet_path = Path(
            raynet_path
            or extra.get('raynet_root')
            or environment_config.get('raynet_path')
            or os.environ.get('RAYNET_PATH', '/home/james/raynet')
        ).expanduser()
        self.section = RAYNET_DEFAULT_SECTION
        self.ini_path = self.raynet_path / RAYNET_BASE_ENVIRONMENT
        self.raynet_runner = Path(
            raynet_runner
            or extra.get('runner')
            or environment_config.get('raynet_runner')
            or self.raynet_path / 'runners' / 'olympus_runner.sh'
        ).expanduser()
        omnet_value = (
            omnet_path
            or extra.get('omnet_root')
            or environment_config.get('omnet_path')
            or environment_config.get('omnetpp_path')
            or os.environ.get('OMNET_PATH', '')
        )
        self.omnet_path = Path(str(omnet_value)).expanduser() if omnet_value else None
        self.runner_log_path = (
            environment_config.get('runner_log_path')
            or environment_config.get('raynet_log_path')
            or environment_config.get('log_path')
        )
        self.control_trace_path = (
            environment_config.get('control_trace_path')
            or environment_config.get('raynet_trace_path')
            or environment_config.get('trace_path')
        )
        self.observation_fields = (
            environment_config.get('observation_fields')
            or environment_config.get('raw_observation_fields')
            or []
        )
        self.started = False
        self._service = None
        self._manager = None
        self._server_thread = None
        self.start_delays = None
        self.flow_durations = None
        self.flow_addr = ''
        self.flow_key = ''

    def start(self) -> None:
        if not self.raynet_path.exists():
            raise FileNotFoundError(f'RayNet path not found: {self.raynet_path}')
        build_dir = self.raynet_path / 'build'
        if not build_dir.exists():
            raise FileNotFoundError(
                f'RayNet build directory not found: {build_dir}. Build RayNet with ./build.sh.')
        if not self.raynet_runner.exists():
            raise FileNotFoundError(f'RayNet Olympus runner not found: {self.raynet_runner}')
        if not Path(self.ini_path).expanduser().exists():
            raise FileNotFoundError(f'RayNet ini_path not found: {self.ini_path}')
        if self.omnet_path is not None and not self.omnet_path.exists():
            raise FileNotFoundError(f'OMNeT++ path not found: {self.omnet_path}')
        self._start_flow_service()
        self.started = True

    def set_link(self, bw=None, delay=None, loss=None) -> None:
        if bw is not None:
            self.bw = float(bw)
        if delay is not None:
            self.delay = float(delay)
        if loss is not None:
            self.loss = loss
        print('[raynet-env] set_link ignored; RayNet scenarios are fixed by INI/scenario files',
              flush=True)

    def run_iperf(self, monitor_interval=0.1, start_delays=None,
                  flow_durations=None) -> None:
        if self._service is None:
            raise RuntimeError('RayNet environment not started')
        self.start_delays = start_delays
        self.flow_durations = flow_durations
        self._service.set_flow_schedule(start_delays)
        self._service.start_episode()
        self._service.wait()

    def stop(self) -> None:
        if self._service is not None:
            self._service.close()
        if self._manager is not None:
            try:
                self._manager.stop_event.set()
            except AttributeError:
                pass
        self.started = False

    def process_env(self):
        env = dict(os.environ)
        env['RAYNET_PATH'] = str(self.raynet_path)
        if self.omnet_path is not None:
            env['OMNET_PATH'] = str(self.omnet_path)
        return env

    def episode_config(self):
        interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20.0'))
        interval_s = interval_ms / 1000.0
        replacements = {
            'home': os.environ.get('HOME', str(Path.home())),
            'inet': os.environ.get(
                'INET_PATH',
                str(Path.home() / 'omnetpp' / 'samples' / 'inet4.5')),
            'raynet_path': str(self.raynet_path),
            'protocol': self.protocol,
            'cc_algo': self.raynet_cc_algo,
            'bw': f'{self.bw:.12g}Mbps',
            'delay': f'{self.delay / 2.0:.12g}ms',
            'qsize': self._qsize_bits(),
        }
        if self.omnet_path is not None:
            replacements['omnet_path'] = str(self.omnet_path)
        duration = None if self.duration is None else float(self.duration)
        if duration is not None:
            replacements['max_rl_steps'] = str(max(
                1, int(round(duration / max(interval_s, 1e-9)))))

        overrides = {
            '**.numberOfFlows': str(int(self.n)),
            '**.fixedIntervalDuration': f'{interval_s:.12g}',
            '**.step_duration': f'{interval_s:.12g}s',
        }
        config = {
            'protocol': self.protocol,
            'ini_path': str(Path(self.ini_path).expanduser()),
            'section': self.section,
            'bw': self.bw,
            'delay': self.delay,
            'flows': int(self.n),
            'bdp_mult': self.bdp_mult,
            'duration': duration,
            'quiet': True,
            'replacements': replacements,
            'overrides': overrides,
        }
        if self.omnet_path is not None:
            config['omnet_path'] = str(self.omnet_path)
        config['link_schedule'] = self.environment_config.get('link_schedule') or []
        if self.start_delays is not None:
            config['start_delays'] = self.start_delays
        if self.flow_durations is not None:
            config['flow_durations'] = self.flow_durations
        if self.per_flow_delays is not None:
            config['per_flow_delays'] = self.per_flow_delays
        if self.environment_config.get('episode') is not None:
            config['episode'] = self.environment_config.get('episode')
        if self.environment_config.get('slot') is not None:
            config['slot'] = self.environment_config.get('slot')
        if self.observation_fields:
            config['observation_fields'] = self.observation_fields
        return config

    def _listener_cc_name(self, environment_config, cc_algo):
        configured = (
            environment_config.get('cc_listener')
            or environment_config.get('listener_cc')
        )
        if configured:
            return str(configured)

        cc_name = str(cc_algo or '').strip()
        if cc_name and cc_name.lower() != 'raynet':
            return cc_name
        return 'astraea'

    def _raynet_cc_algo(self, listener_cc):
        key = str(listener_cc or '').strip().lower()
        return RAYNET_CC_ALGO_BY_LISTENER.get(key, str(listener_cc))

    def observation_to_raw(self, observation):
        if isinstance(observation, dict):
            return dict(observation)
        if not self.observation_fields:
            raise ValueError(
                'RayNet observations must be dictionaries. Add observation_fields '
                'to the RayNet environment YAML or update the RayNet runner to '
                'emit named raw fields.')
        values = [float(v) for v in observation]
        raw = {}
        for field in self.observation_fields:
            if isinstance(field, str):
                raw[field] = values[len(raw)]
                continue
            value = values[int(field['index'])]
            raw[str(field['name'])] = value
            for alias in field.get('aliases') or []:
                raw[str(alias)] = value
        return raw

    def _qsize_bits(self):
        if self.qsize is not None:
            return f'{float(self.qsize) * 8.0:.12g}b'
        bits = max(1.0, self.bw * self.delay * 1000.0 * float(self.bdp_mult))
        return f'{bits:.12g}b'

    def _start_flow_service(self):
        global _FLOW_SERVICE
        self._service = _RayNetFlowService(self)
        _FLOW_SERVICE = self._service
        self.flow_key = secrets.token_hex(16)
        self._manager = _FlowManager(
            address=('127.0.0.1', 0),
            authkey=bytes.fromhex(self.flow_key),
        )
        server = self._manager.get_server()
        host, port = server.address
        self.flow_addr = f'{host}:{port}'
        self._server_thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
        )
        self._server_thread.start()


ENV_CLASS = RaynetEnv
