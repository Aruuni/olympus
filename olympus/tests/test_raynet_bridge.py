import os
import tempfile
import unittest
from unittest import mock

from olympus import orchestrator
from olympus.environments.raynet import runner as raynet_runner


RAYNET_ORCA_OBS = [
    20.0,       # delay ms
    1_250_000,  # throughput B/s
    4.0,        # samples
    0.02,       # interval s
    50.0,       # target ms
    32.0,       # cwnd packets
    2_000_000,  # pacing B/s
    1000.0,     # loss B/s
    22.0,       # srtt ms
    64.0,       # ssthresh packets
    10.0,       # packets out
    1.0,        # retrans out
    10.0,       # max packets out
    1460.0,     # mss
    18.0,       # min rtt ms
]


RAYNET_ASTRAEA_OBS = [
    1_250_000.0,  # avg_thr B/s
    2_000_000.0,  # max_tput B/s
    20_000.0,     # avg_urtt us
    18_000.0,     # min_rtt us
    22_000.0 * 8, # srtt_us shifted
    32.0,         # cwnd packets
    1000.0,       # loss ratio/rate
    10.0,         # packets_out
    2_000_000.0,  # pacing_rate B/s
    1.0,          # retrans_out
]


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
            'observations': {'Orca1': list(RAYNET_ORCA_OBS)},
        }

    def step(self, actions):
        self.step_actions.append(dict(actions))
        return {
            'type': 'step',
            'observations': {'Orca1': list(RAYNET_ORCA_OBS)},
            'rewards': {},
            'terminateds': {'__all__': False},
            'info': {'simDone': True},
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
                'Astraea1': list(RAYNET_ASTRAEA_OBS),
                'Astraea2': list(RAYNET_ASTRAEA_OBS),
            },
        }

    def step(self, actions):
        self.step_actions.append(dict(actions))
        return {
            'type': 'step',
            'observations': {
                'Astraea1': list(RAYNET_ASTRAEA_OBS),
                'Astraea2': list(RAYNET_ASTRAEA_OBS),
            },
            'rewards': {},
            'terminateds': {'__all__': False},
            'info': {'simDone': True},
        }

    def terminate(self):
        self.terminated = True


class RayNetObservationAdapterTest(unittest.TestCase):
    def test_orca_observation_maps_to_olympus_raw_fields(self):
        raw = raynet_runner.raynet_orca_observation_to_raw(RAYNET_ORCA_OBS)
        self.assertEqual(raw['avg_thr'], 1_250_000)
        self.assertEqual(raw['throughput'], 1_250_000)
        self.assertEqual(raw['count'], 4.0)
        self.assertEqual(raw['cnt'], 4.0)
        self.assertEqual(raw['cwnd'], 32.0)
        self.assertEqual(raw['pacing_rate'], 2_000_000)
        self.assertEqual(raw['loss_rate'], 1000.0)
        self.assertEqual(raw['srtt_us'], 22.0 * 1000.0 * 8.0)
        self.assertEqual(raw['min_rtt'], 18.0 * 1000.0)

    def test_rejects_wrong_observation_length(self):
        with self.assertRaisesRegex(ValueError, '15 values'):
            raynet_runner.raynet_orca_observation_to_raw([1.0, 2.0])

    def test_astraea_observation_maps_to_olympus_raw_fields(self):
        raw = raynet_runner.raynet_astraea_observation_to_raw(RAYNET_ASTRAEA_OBS)
        self.assertEqual(raw['avg_thr'], 1_250_000.0)
        self.assertEqual(raw['max_tput'], 2_000_000.0)
        self.assertEqual(raw['avg_urtt'], 20_000.0)
        self.assertEqual(raw['min_rtt'], 18_000.0)
        self.assertEqual(raw['srtt_us'], 22_000.0 * 8)
        self.assertEqual(raw['cwnd'], 32.0)
        self.assertEqual(raw['loss_ratio'], 1000.0)
        self.assertEqual(raw['packets_out'], 10.0)
        self.assertEqual(raw['pacing_rate'], 2_000_000.0)
        self.assertEqual(raw['retrans_out'], 1.0)

    def test_rejects_wrong_astraea_observation_length(self):
        with self.assertRaisesRegex(ValueError, '10 values'):
            raynet_runner.raynet_astraea_observation_to_raw([1.0, 2.0])


class RayNetEpisodeRunnerTest(unittest.TestCase):
    def test_fake_omnet_episode_actions_and_cleanup(self):
        FakeRayNetClient.instances.clear()
        with tempfile.TemporaryDirectory() as directory:
            ini = os.path.join(directory, 'OrcaTraining.ini')
            open(ini, 'w').close()
            cfg = {
                'runtime': {
                    'algorithm': 'orca',
                    'reward': 'sage',
                    'state': 'default_orca',
                    'action': 'astraea',
                },
                'actions': {'astraea': {}},
                'training': {
                    'checkpoint': os.path.join(directory, 'orca.pt'),
                    'worker_push_every': 1,
                },
                'agent': {
                    'hidden': 16,
                    'head_hidden': 16,
                    'rec_dim': 2,
                    'noise_std': 0.0,
                },
                'reward': {'delay_margin_coef': 1.25},
                'state_options': {'orca_target_ms': 50.0},
                'outputs': {'episodes_dir': directory},
                'paths': {'raynet': '/home/james/raynet'},
            }
            ecfg = {
                'protocol': 'orca',
                'ini_path': ini,
                'section': 'General',
                'bw': 100,
                'delay': 20,
            }

            ep_return, _, link_sched = raynet_runner.run_episode_raynet(
                cfg, ecfg, 3, '', '', '', '', 0,
                client_factory=FakeRayNetClient)

        fake = FakeRayNetClient.instances[-1]
        self.assertEqual(fake.started_episode['ini_path'], ini)
        self.assertEqual(fake.started_episode['section'], 'General')
        self.assertTrue(fake.terminated)
        self.assertTrue(fake.step_actions)
        self.assertIn('Orca1', fake.step_actions[0])
        self.assertIsInstance(fake.step_actions[0]['Orca1'], float)
        self.assertTrue(fake.command[0].endswith('olympus_runner.sh'))
        self.assertEqual(link_sched, [])
        self.assertIsNotNone(ep_return)

    def test_fake_astraea_episode_actions_and_ini_overrides(self):
        FakeAstraeaRayNetClient.instances.clear()
        with tempfile.TemporaryDirectory() as directory:
            ini = os.path.join(directory, 'AstraeaTraining.ini')
            open(ini, 'w').close()
            cfg = {
                'runtime': {
                    'algorithm': 'ma_dreamer',
                    'reward': 'tempest_fairness_ma',
                    'state': 'tempest',
                    'action': 'astraea',
                },
                'actions': {
                    'astraea': {
                        'action_min': -1.0,
                        'action_max': 1.0,
                        'step': 0.025,
                    },
                },
                'training': {
                    'checkpoint': os.path.join(directory, 'ma_dreamer.pt'),
                    'worker_push_every': 1,
                    'reward_bins': 255,
                },
                'agent': {
                    'hidden': 16,
                    'embed_dim': 8,
                    'h_dim': 16,
                    'latent_groups': 2,
                    'latent_classes': 4,
                    'interval_ms': 30,
                },
                'reward': {
                    'fairness_weight': 25.0,
                    'fairness_cap': 1.0,
                },
                'outputs': {'episodes_dir': directory},
                'paths': {'raynet': '/home/james/raynet'},
            }
            ecfg = {
                'protocol': 'astraea',
                'ini_path': ini,
                'section': 'General',
                'flows': 2,
                'bw': 20,
                'delay': 20,
                'duration': 1,
            }

            ep_return, _, link_sched = raynet_runner.run_episode_raynet(
                cfg, ecfg, 4, '', '', '', '', 0,
                client_factory=FakeAstraeaRayNetClient)

        fake = FakeAstraeaRayNetClient.instances[-1]
        self.assertEqual(fake.started_episode['protocol'], 'astraea')
        self.assertEqual(fake.started_episode['overrides']['**.numberOfFlows'], '2')
        self.assertEqual(fake.started_episode['overrides']['**.fixedIntervalDuration'], '0.03')
        self.assertTrue(fake.terminated)
        self.assertTrue(fake.step_actions)
        self.assertEqual(set(fake.step_actions[0]), {'Astraea1', 'Astraea2'})
        for action in fake.step_actions[0].values():
            self.assertIsInstance(action, float)
            self.assertGreaterEqual(action, -1.0)
            self.assertLessEqual(action, 1.0)
        self.assertEqual(link_sched, [])
        self.assertIsNotNone(ep_return)


class RayNetOrchestratorDispatchTest(unittest.TestCase):
    def test_run_episode_auto_dispatches_raynet_backend(self):
        cfg = {
            'environment': {'type': 'raynet', 'environment_setup': 'orca_static'},
            'runtime': {'algorithm': 'orca'},
        }
        with mock.patch.object(
                orchestrator, 'run_episode_raynet',
                return_value=('ret', {'x': 1}, [])) as raynet, \
                mock.patch.object(orchestrator, 'run_episode') as mininet:
            result = orchestrator.run_episode_auto(
                cfg, {'x': 1}, 0, '', '', 'addr', 'key', 2)
        self.assertEqual(result, ('ret', {'x': 1}, []))
        raynet.assert_called_once()
        mininet.assert_not_called()

    def test_run_episode_auto_preserves_mininet_single_agent_dispatch(self):
        cfg = {
            'environment': {'type': 'mininet', 'environment_setup': 'static'},
            'runtime': {'algorithm': 'orca'},
        }
        with mock.patch.object(orchestrator, 'run_episode_raynet') as raynet, \
                mock.patch.object(
                    orchestrator, 'run_episode',
                    return_value=('ret', {}, [])) as mininet:
            result = orchestrator.run_episode_auto(
                cfg, {}, 0, 'listener', 'python', 'addr', 'key', 2)
        self.assertEqual(result, ('ret', {}, []))
        mininet.assert_called_once()
        raynet.assert_not_called()


if __name__ == '__main__':
    unittest.main()
