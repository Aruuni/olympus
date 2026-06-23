import os
import tempfile
import unittest
from unittest import mock

from olympus import orchestrator
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

    def __init__(self, command, *, cwd=None, env=None):
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env or {})
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

    def __init__(self, command, *, cwd=None, env=None):
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env or {})
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
            ini_path='/tmp/OrcaTraining.ini',
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
        )
        config = env.episode_config()
        self.assertNotIn('observation_fields', config)

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
    def test_fake_omnet_episode_actions_and_cleanup(self):
        FakeRayNetClient.instances.clear()
        env = raynet_env.RaynetEnv(
            n=1,
            bw=100,
            delay=20,
            raynet_path='/tmp',
            raynet_runner='/tmp/olympus_runner.sh',
            environment_config={
                'protocol': 'orca',
                'ini_path': '/tmp/OrcaTraining.ini',
                'section': 'General',
                'observation_fields': ORCA_FIELDS,
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
        self.assertEqual(fake.started_episode['ini_path'], '/tmp/OrcaTraining.ini')
        self.assertEqual(fake.started_episode['section'], 'General')
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
                    'protocol': 'astraea',
                    'ini_path': '/tmp/AstraeaTraining.ini',
                    'section': 'General',
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
        self.assertEqual(fake.started_episode['protocol'], 'astraea')
        self.assertEqual(fake.started_episode['observation_fields'], ASTRAEA_FIELDS)
        self.assertEqual(fake.started_episode['overrides']['**.numberOfFlows'], '2')
        self.assertEqual(fake.started_episode['overrides']['**.fixedIntervalDuration'], '0.03')
        self.assertTrue(fake.terminated)
        self.assertEqual(raw0['avg_thr'], 1_250_000.0)
        self.assertEqual(raw1['avg_thr'], 1_250_000.0)
        self.assertEqual(raw0['time_s'], 0.25)
        self.assertEqual(raw1['time_s'], 0.25)
        self.assertTrue(fake.step_actions)
        self.assertEqual(set(fake.step_actions[0]), {'Astraea1', 'Astraea2'})
        for action in fake.step_actions[0].values():
            self.assertEqual(action, 64.0)

    def test_raynet_env_reads_ini_from_environment_config(self):
        with tempfile.TemporaryDirectory() as directory:
            build_dir = os.path.join(directory, 'build')
            runner = os.path.join(directory, 'olympus_runner.sh')
            ini = os.path.join(directory, 'AstraeaTraining.ini')
            os.mkdir(build_dir)
            open(runner, 'w').close()
            open(ini, 'w').close()
            env = raynet_env.RaynetEnv(
                n=2,
                raynet_path=directory,
                raynet_runner=runner,
                environment_config={
                    'protocol': 'astraea',
                    'ini_path': ini,
                    'section': 'General',
                },
            )
            with mock.patch.object(env, '_start_flow_service'):
                env.start()

        self.assertEqual(env.ini_path, ini)
        self.assertEqual(env.protocol, 'astraea')

    def test_run_episode_marl_passes_opaque_environment_config_to_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            ecfg = {
                'protocol': 'astraea',
                'ini_path': os.path.join(directory, 'AstraeaTraining.ini'),
                'section': 'General',
                'flows': 2,
                'bw': 20,
                'delay': 20,
                'duration': 1,
            }
            cfg = {
                'environment': {'type': 'raynet', 'environment_setup': 'astraea_smoke'},
                'runtime': {'algorithm': 'ma_dreamer'},
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
        self.assertEqual(kwargs['environment_config'], ecfg)
        self.assertNotIn('ini_path', kwargs)
        self.assertNotIn('section', kwargs)


class RayNetOrchestratorDispatchTest(unittest.TestCase):
    def test_run_episode_auto_uses_algorithm_dispatch_for_raynet_marl(self):
        cfg = {
            'environment': {'type': 'raynet', 'environment_setup': 'astraea_smoke'},
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
