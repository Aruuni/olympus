import os
import unittest

import yaml

from olympus import orchestrator
from olympus.common.registry import reward_module
from olympus.common.state_plugins import load_state_module


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProteusEnvironmentTest(unittest.TestCase):
    def _load_environment(self, backend):
        path = os.path.join(_ROOT, 'environments', backend, 'proteus.yaml')
        with open(path) as stream:
            return yaml.safe_load(stream)

    def test_training_preset_selects_proteus_for_500_episodes(self):
        with open(os.path.join(_ROOT, 'proteus.yaml')) as stream:
            cfg = yaml.safe_load(stream)

        self.assertEqual(cfg['runtime'], {
            'algorithm': 'sac',
            'reward': 'proteus',
            'state': 'proteus',
            'action': 'cwnd_multiplier',
        })
        self.assertEqual(cfg['environment'], {
            'type': 'mininet',
            'environment_setup': 'proteus',
        })
        self.assertEqual(cfg['orchestrator']['episodes'], 500)
        self.assertEqual(reward_module('proteus').__name__,
                         'olympus.rewards.proteus')
        self.assertEqual(load_state_module('sac', 'proteus').STATE_DIM, 10)

        orchestrator._activate_runtime_blocks(cfg)
        orchestrator._validate_environment_runtime_compatibility(cfg)
        self.assertEqual(cfg['training']['batch_size'], 256)
        self.assertEqual(cfg['agent']['interval_ms'], 20)

    def test_mininet_and_raynet_use_the_requested_static_grid(self):
        mininet = self._load_environment('mininet')['sweep']
        raynet = self._load_environment('raynet')['sweep']

        for key in ('bws', 'delays', 'bdp_mult', 'flows', 'duration',
                    'link_schedules'):
            self.assertEqual(mininet[key], raynet[key])
        expected_grid = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.assertEqual(mininet['bws'], expected_grid)
        self.assertEqual(mininet['delays'], expected_grid)
        self.assertEqual(mininet['bdp_mult'], [0.5, 1, 2, 3, 6, 12])
        self.assertEqual(mininet['flows'], 1)
        self.assertEqual(mininet['link_schedules'], [[]])

    def test_named_mininet_environment_expands_to_all_combinations(self):
        cfg = {'environment': {
            'type': 'mininet',
            'environment_setup': 'proteus',
        }}
        env, meta = orchestrator._load_environment_definition(cfg)
        pool = orchestrator._build_sweep_pool(env['sweep'], meta)

        self.assertEqual(meta['name'], 'proteus')
        self.assertEqual(len(pool), 11 * 11 * 6)
        self.assertEqual({episode['bdp_mult'] for episode in pool},
                         {0.5, 1.0, 2.0, 3.0, 6.0, 12.0})

    def test_raynet_exposes_every_proteus_state_measurement(self):
        fields = self._load_environment('raynet')['sweep'][
            'observation_fields']
        names = {field['name'] for field in fields}
        self.assertEqual(names, {
            'avg_thr', 'avg_urtt', 'min_rtt', 'srtt_us', 'cwnd',
            'packets_out', 'pacing_rate', 'retrans_out',
        })


if __name__ == '__main__':
    unittest.main()
