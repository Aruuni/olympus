import os
import unittest

import yaml

from olympus import orchestrator


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DreamerEnvironmentTest(unittest.TestCase):
    def _load(self, backend):
        path = os.path.join(_ROOT, 'environments', backend, 'dreamer.yaml')
        with open(path) as stream:
            return yaml.safe_load(stream)

    def test_mininet_and_raynet_sweeps_match(self):
        mininet = self._load('mininet')['sweep']
        raynet = self._load('raynet')['sweep']

        for key in ('bws', 'delays', 'bdp_mult', 'flows', 'duration',
                    'link_schedules'):
            self.assertEqual(mininet[key], raynet[key])

    def test_episodes_have_one_optional_change_at_fifteen_seconds(self):
        for backend in ('mininet', 'raynet'):
            sweep = self._load(backend)['sweep']
            self.assertEqual(sweep['duration'], 30)
            for schedule in sweep['link_schedules']:
                self.assertLessEqual(len(schedule), 1)
                if schedule:
                    self.assertEqual(schedule[0]['t'], 15)

    def test_four_change_modes_are_equally_represented(self):
        schedules = self._load('mininet')['sweep']['link_schedules']
        counts = {'bw': 0, 'rtt': 0, 'both': 0, 'neither': 0}
        for schedule in schedules:
            if not schedule:
                counts['neither'] += 1
                continue
            event = schedule[0]
            has_bw = 'bw_frac' in event
            has_rtt = 'delay_frac' in event
            if has_bw and has_rtt:
                counts['both'] += 1
            elif has_bw:
                counts['bw'] += 1
            elif has_rtt:
                counts['rtt'] += 1

        self.assertEqual(counts, {
            'bw': 4, 'rtt': 4, 'both': 4, 'neither': 4})

    def test_raynet_exposes_dreamer_state_and_reward_measurements(self):
        fields = self._load('raynet')['sweep']['observation_fields']
        names = {field['name'] for field in fields}

        self.assertEqual(names, {
            'avg_thr', 'avg_urtt', 'min_rtt', 'srtt_us', 'cwnd',
            'packets_out', 'pacing_rate', 'retrans_out',
        })

    def test_named_scenarios_load_and_expand_for_both_backends(self):
        for backend in ('mininet', 'raynet'):
            cfg = {'environment': {
                'type': backend,
                'environment_setup': 'dreamer',
            }}
            env, meta = orchestrator._load_environment_definition(cfg)
            pool = orchestrator._build_sweep_pool(env['sweep'], meta)

            self.assertEqual(meta['name'], 'dreamer')
            self.assertEqual(len(pool), 6 * 5 * 16)
            self.assertTrue(all(ep['duration'] == 30 for ep in pool))
            self.assertTrue(all(
                not ep['link_schedule'] or ep['link_schedule'][0]['t'] == 15
                for ep in pool))


if __name__ == '__main__':
    unittest.main()
