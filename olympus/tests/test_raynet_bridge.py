import json
import os
import tempfile
import unittest
from unittest import mock

from olympus import orchestrator
from olympus.common import flow_backend
from olympus.environments.raynet import env as raynet_env


RAYNET_ORCA_OBS = [
    20.0,       # delay ms
    1_250_000,  # throughput B/s
    4.0,        # samples
    0.02,       # interval s
    50.0,       # target ms
    32.0,       # cwnd packets
    2_000_000,  # pacing B/s
    1000.0,     # loss B/s
    176_000.0,  # srtt_us shifted
    64.0,       # ssthresh packets
    10.0,       # packets out
    1.0,        # retrans out
    10.0,       # max packets out
    1460.0,     # mss
    18.0,       # min rtt ms
    20_000.0,   # avg urtt us
    18_000.0,   # min rtt us
]


RAYNET_ASTRAEA_OBS = [
    1_250_000.0,  # avg_thr B/s
    2_000_000.0,  # max_tput B/s
    20_000.0,     # avg_urtt us
    18_000.0,     # min_rtt us
    22_000.0 * 8, # srtt_us shifted
    32.0,         # cwnd packets
    1000.0,       # loss rate B/s
    10.0,         # packets_out
    2_000_000.0,  # pacing_rate B/s
    1.0,          # retrans_out
]


ORCA_FIELDS = [
    {'name': 'delay_ms', 'index': 0},
    {'name': 'avg_urtt', 'index': 15, 'aliases': ['delay_us']},
    {'name': 'avg_thr', 'index': 1, 'aliases': ['throughput']},
    {'name': 'count', 'index': 2, 'aliases': ['cnt', 'samples']},
    {'name': 'interval_s', 'index': 3, 'aliases': ['delta_t']},
    {'name': 'target', 'index': 4},
    {'name': 'cwnd', 'index': 5},
    {'name': 'pacing_rate', 'index': 6},
    {'name': 'loss_rate', 'index': 7, 'aliases': ['lost_rate']},
    {'name': 'srtt_us', 'index': 8},
    {'name': 'srtt_ms', 'index': 8},
    {'name': 'snd_ssthresh', 'index': 9},
    {'name': 'packets_out', 'index': 10},
    {'name': 'retrans_out', 'index': 11},
    {'name': 'max_packets_out', 'index': 12},
    {'name': 'mss', 'index': 13, 'aliases': ['mss_cache']},
    {'name': 'min_rtt', 'index': 16, 'aliases': ['min_rtt_us']},
    {'name': 'min_rtt_ms', 'index': 14},
]


ASTRAEA_FIELDS = [
    {'name': 'avg_thr', 'index': 0, 'aliases': ['throughput']},
    {'name': 'max_tput', 'index': 1},
    {'name': 'avg_urtt', 'index': 2, 'aliases': ['delay_us']},
    {'name': 'min_rtt', 'index': 3, 'aliases': ['min_rtt_us']},
    {'name': 'srtt_us', 'index': 4},
    {'name': 'cwnd', 'index': 5},
    {'name': 'loss_rate', 'index': 6, 'aliases': ['loss_ratio', 'loss_bytes']},
    {'name': 'packets_out', 'index': 7},
    {'name': 'pacing_rate', 'index': 8},
    {'name': 'retrans_out', 'index': 9},
]


def mapped_observation(values, fields):
    env = raynet_env.RaynetEnv(environment_config={'observation_fields': fields})
    return env.observation_to_raw(values)


class FakeRayNetClient:
    instances = []

    def __init__(self, command, *, cwd=None, env=None, log_path=None,
                 trace_path=None):
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env or {})
        self.log_path = log_path
        self.trace_path = trace_path
        self.started_episode = None
        self.step_actions = []
        self.terminated = False
        FakeRayNetClient.instances.append(self)

    def start(self, episode_config):
        self.started_episode = dict(episode_config)
        return {
            'type': 'reset',
            'observations': {'Orca1': mapped_observation(RAYNET_ORCA_OBS, ORCA_FIELDS)},
            'info': {'simDone': False, 'time_s': 0.25},
        }

    def step(self, actions):
        self.step_actions.append(dict(actions))
        return {
            'type': 'step',
            'observations': {'Orca1': mapped_observation(RAYNET_ORCA_OBS, ORCA_FIELDS)},
            'rewards': {},
            'terminateds': {'__all__': False},
            'info': {'simDone': True, 'time_s': 0.5},
        }

    def terminate(self):
        self.terminated = True


class FakeAstraeaRayNetClient:
    instances = []

    def __init__(self, command, *, cwd=None, env=None, log_path=None,
                 trace_path=None):
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env or {})
        self.log_path = log_path
        self.trace_path = trace_path
        self.started_episode = None
        self.step_actions = []
        self.terminated = False
        FakeAstraeaRayNetClient.instances.append(self)

    def start(self, episode_config):
        self.started_episode = dict(episode_config)
        return {
            'type': 'reset',
            'observations': {
                'Astraea1': mapped_observation(RAYNET_ASTRAEA_OBS, ASTRAEA_FIELDS),
                'Astraea2': mapped_observation(RAYNET_ASTRAEA_OBS, ASTRAEA_FIELDS),
            },
            'info': {'simDone': False, 'time_s': 0.25},
        }

    def step(self, actions):
        self.step_actions.append(dict(actions))
        return {
            'type': 'step',
            'observations': {
                'Astraea1': mapped_observation(RAYNET_ASTRAEA_OBS, ASTRAEA_FIELDS),
                'Astraea2': mapped_observation(RAYNET_ASTRAEA_OBS, ASTRAEA_FIELDS),
            },
            'rewards': {},
            'terminateds': {'__all__': False},
            'info': {'simDone': True, 'time_s': 0.5},
        }

    def terminate(self):
        self.terminated = True


class RayNetObservationAdapterTest(unittest.TestCase):
    def test_episode_client_redirects_runner_output_to_log_path(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, 'runner.log')
            proc = mock.Mock()
            proc.poll.return_value = 0
            with mock.patch.object(raynet_env.subprocess, 'Popen',
                                   return_value=proc) as popen:
                client = raynet_env.RayNetEpisodeClient(
                    ['/tmp/runner'], cwd='/tmp', env={},
                    log_path=log_path)
                client.terminate()

            _, kwargs = popen.call_args
            self.assertEqual(kwargs['stderr'], raynet_env.subprocess.STDOUT)
            self.assertEqual(kwargs['stdout'].name, log_path)
            self.assertTrue(kwargs['stdout'].closed)
            with open(log_path) as f:
                self.assertIn('[raynet-runner] command=', f.read())

    def test_episode_client_writes_control_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = os.path.join(directory, 'trace.jsonl')
            proc = mock.Mock()
            proc.poll.return_value = 0
            with mock.patch.object(raynet_env.subprocess, 'Popen',
                                   return_value=proc):
                client = raynet_env.RayNetEpisodeClient(
                    ['/tmp/runner'], cwd='/tmp', env={},
                    trace_path=trace_path)
                client._send = mock.Mock()
                client._recv = mock.Mock(side_effect=[
                    {'type': 'reset', 'observations': {'a0': {'cwnd': 4}},
                     'info': {'time_s': 0.0}},
                    {'type': 'step', 'observations': {'a0': {'cwnd': 8}},
                     'info': {'time_s': 0.02}},
                ])
                client.start({'episode': 7})
                client.step({'a0': 8})
                client.terminate()

            with open(trace_path) as f:
                events = [json.loads(line)['event'] for line in f]
            self.assertEqual(events, [
                'runner_start', 'send_start', 'recv_reset',
                'send_step', 'recv_step',
            ])

    def test_field_mapping_observation_maps_to_olympus_raw_fields(self):
        env = raynet_env.RaynetEnv(
            environment_config={'observation_fields': ORCA_FIELDS})
        raw = env.observation_to_raw(RAYNET_ORCA_OBS)
        self.assertEqual(raw['avg_thr'], 1_250_000)
        self.assertEqual(raw['throughput'], 1_250_000)
        self.assertEqual(raw['count'], 4.0)
        self.assertEqual(raw['cnt'], 4.0)
        self.assertEqual(raw['cwnd'], 32.0)
        self.assertEqual(raw['pacing_rate'], 2_000_000)
        self.assertEqual(raw['loss_rate'], 1000.0)
        self.assertEqual(raw['avg_urtt'], 20_000.0)
        self.assertEqual(raw['delay_us'], 20_000.0)
        self.assertEqual(raw['srtt_us'], 176_000.0)
        self.assertEqual(raw['min_rtt'], 18_000.0)
        self.assertEqual(raw['min_rtt_us'], 18_000.0)
        self.assertEqual(raw['min_rtt_ms'], 18.0)

    def test_rejects_list_observation_without_field_mapping(self):
        env = raynet_env.RaynetEnv()
        with self.assertRaisesRegex(ValueError, 'observations must be dictionaries'):
            env.observation_to_raw([1.0, 2.0])

    def test_raw_dict_observation_passes_through(self):
        env = raynet_env.RaynetEnv()
        raw = {'avg_thr': 123.0, 'cwnd': 42.0}
        self.assertEqual(env.observation_to_raw(raw), raw)

    def test_episode_config_omits_empty_observation_fields(self):
        env = raynet_env.RaynetEnv(
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
        )
        config = env.episode_config()
        self.assertNotIn('observation_fields', config)

    def test_raynet_paths_default_inside_environment(self):
        with mock.patch.dict(os.environ, {'RAYNET_PATH': '/opt/raynet-test'}, clear=False):
            env = raynet_env.RaynetEnv()

        self.assertEqual(str(env.raynet_path), '/opt/raynet-test')
        self.assertEqual(
            str(env.raynet_runner),
            '/opt/raynet-test/runners/olympus_runner.sh',
        )

    def test_astraea_field_mapping_observation_maps_to_olympus_raw_fields(self):
        env = raynet_env.RaynetEnv(
            environment_config={'observation_fields': ASTRAEA_FIELDS})
        raw = env.observation_to_raw(RAYNET_ASTRAEA_OBS)
        self.assertEqual(raw['avg_thr'], 1_250_000.0)
        self.assertEqual(raw['max_tput'], 2_000_000.0)
        self.assertEqual(raw['avg_urtt'], 20_000.0)
        self.assertEqual(raw['min_rtt'], 18_000.0)
        self.assertEqual(raw['srtt_us'], 22_000.0 * 8)
        self.assertEqual(raw['cwnd'], 32.0)
        self.assertEqual(raw['loss_rate'], 1000.0)
        self.assertEqual(raw['loss_ratio'], 1000.0)
        self.assertEqual(raw['packets_out'], 10.0)
        self.assertEqual(raw['pacing_rate'], 2_000_000.0)
        self.assertEqual(raw['retrans_out'], 1.0)

    def test_rejects_too_short_mapped_observation(self):
        env = raynet_env.RaynetEnv(
            environment_config={'observation_fields': ASTRAEA_FIELDS})
        with self.assertRaises(IndexError):
            env.observation_to_raw([1.0, 2.0])


class RayNetFlowServiceTest(unittest.TestCase):
    def test_flow_backend_translates_raynet_finished_to_normal_signal(self):
        service = mock.Mock()
        service.get_tcp_deepcc_info.side_effect = RuntimeError(
            'RayNet simulation finished')
        env = {
            'OC_FLOW_BACKEND': 'raynet',
            'OC_FLOW_ID': '0',
        }
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(flow_backend, '_raynet_service',
                                  return_value=service):
            with self.assertRaises(flow_backend.SimulationFinished):
                flow_backend.get_tcp_deepcc_info(0)

    def test_wait_collection_step_uses_raynet_sync_service_identity(self):
        sync = mock.Mock()
        raw = {'group_step': 7}
        env = {
            'OC_FLOW_BACKEND': 'raynet',
            'OC_RAYNET_SYNC_SLOT': '3',
            'OC_RAYNET_SYNC_EPISODE': '11',
        }
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(flow_backend, '_raynet_sync_service',
                                  return_value=sync):
            flow_backend.wait_collection_step(raw)

        sync.wait_until_allowed.assert_called_once_with(3, 11, 7, None, 1)

    def test_fake_omnet_episode_actions_and_cleanup(self):
        FakeRayNetClient.instances.clear()
        env = raynet_env.RaynetEnv(
            n=1,
            bw=100,
            delay=20,
            cc_algo='orca',
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
            environment_config={
                'observation_fields': ORCA_FIELDS,
                'runner_log_path': '/tmp/raynet-runner.log',
                'control_trace_path': '/tmp/raynet-trace.jsonl',
            },
        )
        service = raynet_env._RayNetFlowService(env)
        with mock.patch.object(raynet_env, 'RayNetEpisodeClient', FakeRayNetClient):
            service.start_episode()
            raw = service.get_tcp_deepcc_info(0)
            service.set_cwnd(0, 64)
            service.wait()
            service.close()

        fake = FakeRayNetClient.instances[-1]
        self.assertEqual(fake.log_path, '/tmp/raynet-runner.log')
        self.assertEqual(fake.trace_path, '/tmp/raynet-trace.jsonl')
        self.assertEqual(
            fake.started_episode['ini_path'],
            '/tmp/_environments/base_environment.ini')
        self.assertEqual(fake.started_episode['section'], 'General')
        self.assertEqual(
            fake.started_episode['replacements']['cc_algo'], 'TcpCubic')
        self.assertTrue(fake.terminated)
        self.assertEqual(raw['avg_thr'], 1_250_000)
        self.assertEqual(raw['time_s'], 0.25)
        self.assertTrue(fake.step_actions)
        self.assertIn('Orca1', fake.step_actions[0])
        self.assertEqual(fake.step_actions[0]['Orca1'], 64.0)

    def test_fake_astraea_episode_actions_and_ini_overrides(self):
        FakeAstraeaRayNetClient.instances.clear()
        with mock.patch.dict(os.environ, {'SAO_INTERVAL_MS': '30'}, clear=False):
            env = raynet_env.RaynetEnv(
                n=2,
                bw=20,
                delay=20,
                duration=1,
                raynet_path='/tmp',
                raynet_runner='/tmp/olympus_runner.sh',
                environment_config={
                    'observation_fields': ASTRAEA_FIELDS,
                },
            )
            service = raynet_env._RayNetFlowService(env)
        with mock.patch.dict(os.environ, {'SAO_INTERVAL_MS': '30'}, clear=False), \
                mock.patch.object(raynet_env, 'RayNetEpisodeClient',
                                  FakeAstraeaRayNetClient):
            service.start_episode()
            raw0 = service.get_tcp_deepcc_info(0)
            raw1 = service.get_tcp_deepcc_info(1)
            service.set_cwnd(0, 64)
            service.set_cwnd(1, 64)
            service.wait()
            service.close()

        fake = FakeAstraeaRayNetClient.instances[-1]
        self.assertEqual(fake.started_episode['observation_fields'], ASTRAEA_FIELDS)
        self.assertEqual(
            fake.started_episode['replacements']['cc_algo'], 'TcpPacedNoCC')
        self.assertEqual(fake.started_episode['overrides']['**.numberOfFlows'], '2')
        self.assertEqual(fake.started_episode['overrides']['**.fixedIntervalDuration'], '0.03')
        self.assertEqual(fake.started_episode['overrides']['**.step_duration'], '0.03s')
        self.assertTrue(fake.terminated)
        self.assertEqual(raw0['avg_thr'], 1_250_000.0)
        self.assertEqual(raw1['avg_thr'], 1_250_000.0)
        self.assertEqual(raw0['time_s'], 0.25)
        self.assertEqual(raw1['time_s'], 0.25)
        self.assertTrue(fake.step_actions)
        self.assertEqual(set(fake.step_actions[0]), {'Astraea1', 'Astraea2'})
        for action in fake.step_actions[0].values():
            self.assertEqual(action, 64.0)

    def test_raynet_env_owns_base_ini_path(self):
        with tempfile.TemporaryDirectory() as directory:
            build_dir = os.path.join(directory, 'build')
            base_dir = os.path.join(directory, '_environments')
            runner = os.path.join(directory, 'olympus_runner.sh')
            os.mkdir(build_dir)
            os.mkdir(base_dir)
            base_ini = os.path.join(base_dir, 'base_environment.ini')
            open(runner, 'w').close()
            open(base_ini, 'w').close()
            env = raynet_env.RaynetEnv(
                n=2,
                raynet_path=directory,
                raynet_runner=runner,
            )
            with mock.patch.object(env, '_start_flow_service'):
                env.start()

        self.assertEqual(str(env.ini_path), base_ini)
        self.assertEqual(env.section, 'General')

    def test_listener_cc_maps_to_raynet_child_cc_algorithm(self):
        env = raynet_env.RaynetEnv(
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
            cc_algo='orca',
        )
        self.assertEqual(env.episode_config()['replacements']['cc_algo'], 'TcpCubic')

        env = raynet_env.RaynetEnv(
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
            cc_algo='astraea',
        )
        self.assertEqual(
            env.episode_config()['replacements']['cc_algo'], 'TcpPacedNoCC')

        env = raynet_env.RaynetEnv(
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
            environment_config={'listener_cc': 'orca'},
        )
        self.assertEqual(env.episode_config()['replacements']['cc_algo'], 'TcpCubic')

        env = raynet_env.RaynetEnv(
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
            environment_config={'cc_listener': 'TcpBic'},
        )
        self.assertEqual(env.episode_config()['replacements']['cc_algo'], 'TcpBic')

    def test_raynet_env_process_env_uses_configured_paths(self):
        env = raynet_env.RaynetEnv(
            raynet_path='/tmp/raynet-from-config',
            raynet_runner='/tmp/raynet-from-config/olympus_runner.sh',
            ini_path='/tmp/raynet-from-config/scenario.ini',
            environment_config={'omnet_path': '/tmp/omnet-from-config'},
        )
        process_env = env.process_env()

        self.assertEqual(process_env['RAYNET_PATH'], '/tmp/raynet-from-config')
        self.assertEqual(process_env['OMNET_PATH'], '/tmp/omnet-from-config')

    def test_run_episode_marl_passes_opaque_environment_config_to_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            ecfg = {
                'flows': 2,
                'bw': 20,
                'delay': 20,
                'duration': 1,
            }
            cfg = {
                'environment': {'type': 'raynet', 'environment_setup': 'astraea_static'},
                'runtime': {'algorithm': 'ma_dreamer'},
                'listener_cc': 'orca',
                'paths': {
                    'raynet_path': '/tmp/raynet-from-config',
                    'omnet_path': '/tmp/omnet-from-config',
                },
                'training': {'checkpoint': os.path.join(directory, 'model.pt')},
                'agent': {},
                'outputs': {
                    'plots_dir': directory,
                    'episodes_dir': directory,
                    'plot_episodes': False,
                },
            }
            fake_env = mock.Mock()
            fake_env.flow_addr = '127.0.0.1:1'
            fake_env.flow_key = 'aa'

            with mock.patch.object(orchestrator, 'make_env',
                                   return_value=fake_env) as make_env, \
                    mock.patch.object(orchestrator, 'subprocess') as subp, \
                    mock.patch.object(orchestrator, '_terminate'), \
                    mock.patch.object(orchestrator, '_multi_episode_return',
                                      return_value=0.0):
                subp.Popen.return_value = mock.Mock()
                orchestrator.run_episode_marl(
                    cfg, ecfg, 0, 'listener', 'python', 'addr', 'key', 0)

        _, kwargs = make_env.call_args
        environment_config = kwargs['environment_config']
        for key, value in ecfg.items():
            self.assertEqual(environment_config[key], value)
        self.assertEqual(environment_config['episode'], 0)
        self.assertEqual(environment_config['slot'], 0)
        self.assertEqual(
            environment_config['raynet_path'], '/tmp/raynet-from-config')
        self.assertEqual(
            environment_config['omnet_path'], '/tmp/omnet-from-config')
        self.assertIn('runner_log_path', environment_config)
        self.assertTrue(
            environment_config['runner_log_path'].endswith(
                'raynet_runner_ep000000_slot0.log'))
        self.assertIn('control_trace_path', environment_config)
        self.assertTrue(
            environment_config['control_trace_path'].endswith(
                'raynet_trace_ep000000_slot0.jsonl'))
        self.assertEqual(kwargs['cc_algo'], 'orca')
        self.assertNotIn('ini_path', kwargs)
        self.assertNotIn('section', kwargs)

    def test_run_episode_raynet_single_agent_launches_worker_per_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            ecfg = {
                'flows': 2,
                'bw': 20,
                'delay': 20,
                'duration': 1,
                'lagged_policy_flow_ids': [2],
                'per_flow_delays': [5.0, 15.0],
            }
            cfg = {
                'environment': {'type': 'raynet', 'environment_setup': 'lagged_fairness'},
                'runtime': {'algorithm': 'td3', 'reward': 'tempest', 'state': 'tempest'},
                'listener_cc': 'orca',
                'training': {'checkpoint': os.path.join(directory, 'model.pt')},
                'agent': {},
                'outputs': {
                    'plots_dir': directory,
                    'episodes_dir': directory,
                    'plot_episodes': False,
                },
            }
            fake_env = mock.Mock()
            fake_env.flow_addr = '127.0.0.1:1'
            fake_env.flow_key = 'aa'

            with mock.patch.object(orchestrator, 'make_env',
                                   return_value=fake_env), \
                    mock.patch.object(orchestrator.subprocess, 'Popen') as popen, \
                    mock.patch.object(orchestrator, '_wait_or_terminate'):
                popen.side_effect = [mock.Mock(), mock.Mock()]
                orchestrator.run_episode(
                    cfg, ecfg, 0, 'listener', 'python', 'addr', 'key', 0)

        self.assertEqual(popen.call_count, 2)
        envs = [call.kwargs['env'] for call in popen.call_args_list]
        self.assertEqual([env['OC_FLOW_ID'] for env in envs], ['0', '1'])
        self.assertEqual([env['OC_FLOW_FD'] for env in envs], ['0', '1'])
        self.assertEqual([env['OC_CPORT'] for env in envs], ['21000', '21001'])
        self.assertEqual([env['OC_RAYNET_SYNC_SLOT'] for env in envs], ['0', '0'])
        self.assertEqual([env['OC_RAYNET_SYNC_EPISODE'] for env in envs], ['0', '0'])
        self.assertEqual(envs[0]['SAO_LAGGED_POLICY_FLOW_IDS'], '1')
        self.assertEqual(envs[1]['SAO_LAGGED_POLICY_FLOW_IDS'], '1')
        self.assertEqual(envs[0]['SAO_BASE_RTT_US'], '5000.0')
        self.assertEqual(envs[1]['SAO_BASE_RTT_US'], '15000.0')


class RayNetOrchestratorDispatchTest(unittest.TestCase):
    def test_run_episode_auto_uses_algorithm_dispatch_for_raynet_marl(self):
        cfg = {
            'environment': {'type': 'raynet', 'environment_setup': 'astraea_static'},
            'runtime': {'algorithm': 'ma_dreamer'},
        }
        with mock.patch.object(orchestrator, 'run_episode') as single, \
                mock.patch.object(
                    orchestrator, 'run_episode_marl',
                    return_value=('ret', {'x': 1}, [])) as marl:
            result = orchestrator.run_episode_auto(
                cfg, {'x': 1}, 0, '', '', 'addr', 'key', 2)
        self.assertEqual(result, ('ret', {'x': 1}, []))
        marl.assert_called_once()
        single.assert_not_called()

    def test_run_episode_auto_preserves_mininet_single_agent_dispatch(self):
        cfg = {
            'environment': {'type': 'mininet', 'environment_setup': 'static'},
            'runtime': {'algorithm': 'orca'},
        }
        with mock.patch.object(
                orchestrator, 'run_episode',
                return_value=('ret', {}, [])) as mininet:
            result = orchestrator.run_episode_auto(
                cfg, {}, 0, 'listener', 'python', 'addr', 'key', 2)
        self.assertEqual(result, ('ret', {}, []))
        mininet.assert_called_once()


if __name__ == '__main__':
    unittest.main()
