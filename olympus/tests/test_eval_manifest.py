import tempfile
import unittest
from pathlib import Path

import yaml

from olympus.common.eval_manifest import expand_manifest, scenario_episode_count
from olympus.orchestrator import (
    _build_sweep_pool, _materialize_scenario_generators,
)


class EvalManifestTests(unittest.TestCase):
    def test_expands_checkpoint_scenario_backend_matrix(self):
        manifest = {
            'kind': 'olympus-eval', 'version': 1,
            'defaults': {'n_parallel': 4, 'repetitions': 2, 'logging': 'minimal'},
            'checkpoints': {'a': {'path': 'a.pt'}, 'b': {'path': 'b.pt'}},
            'scenarios': {'static': {'path': 'static_eval.yaml'}},
            'environments': {
                'em': {'type': 'mininet'}, 'sim': {'type': 'raynet'},
            },
            'matrix': {
                'checkpoints': ['a', 'b'], 'scenarios': ['static'],
                'environments': ['em', 'sim'],
            },
        }
        runs = expand_manifest(manifest, Path('/tmp/manifest'))
        self.assertEqual(len(runs), 4)
        self.assertEqual(
            {(r['checkpoint'], r['environment']) for r in runs},
            {('a', 'em'), ('a', 'sim'), ('b', 'em'), ('b', 'sim')},
        )
        self.assertTrue(all(r['n_parallel'] == 4 for r in runs))
        self.assertTrue(all(r['repetitions'] == 2 for r in runs))

    def test_explicit_runs_are_added_to_matrix(self):
        manifest = {
            'kind': 'olympus-eval', 'version': 1,
            'checkpoints': {'a': 'a.pt'},
            'scenarios': {'s': 's.yaml'},
            'environments': {'em': 'mininet'},
            'matrix': {'checkpoints': ['a'], 'scenarios': ['s'],
                       'environments': ['em']},
            'runs': [{'name': 'again', 'checkpoint': 'a', 'scenario': 's',
                      'environment': 'em', 'repetitions': 3}],
        }
        runs = expand_manifest(manifest, Path('/tmp'))
        self.assertEqual([r['name'] for r in runs], ['a__s__em', 'again'])
        self.assertEqual(runs[1]['repetitions'], 3)

    def test_counts_backend_neutral_sweep_points(self):
        scenario = {
            'name': 'static_eval',
            'sweep': {
                'bws': [10, 20], 'delays': [5, 10, 20], 'flows': [1, 2],
                'link_schedules': [[], [{'t': 1, 'bw': 5}]],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scenario.yaml'
            path.write_text(yaml.safe_dump(scenario))
            self.assertEqual(scenario_episode_count(str(path)), 24)

    def test_orchestrator_sweeps_flow_counts_too(self):
        pool = _build_sweep_pool(
            {'bws': [10, 20], 'delays': [5], 'flows': [1, 2],
             'link_schedules': [[]]},
            {'name': 'shared', 'path': '/tmp/static_eval.yaml'},
        )
        self.assertEqual(len(pool), 4)
        self.assertEqual({item['flows'] for item in pool}, {1, 2})

    def test_generic_flow_and_link_schedules_materialize(self):
        episode = {
            'bw': 50, 'delay': 50, 'flows': 4, 'duration': 100, 'seed': 9,
            'flow_schedule': {
                'arrival': {'evenly_spaced_over_s': 30},
                'duration': {'until_episode_end': True},
            },
            'link_schedule_generator': {
                'interval_s': 20, 'sample_initial': True,
                'bw': {'uniform': [10, 100]},
                'delay': {'uniform': [10, 100]},
            },
        }
        first = _materialize_scenario_generators(episode, episode=3)
        second = _materialize_scenario_generators(episode, episode=3)
        self.assertEqual(first, second)
        self.assertEqual(first['start_delays'], [0, 10, 20, 30])
        self.assertEqual(first['flow_durations'], [100, 90, 80, 70])
        self.assertTrue(first['per_flow_state_logs'])
        self.assertEqual([row['t'] for row in first['link_schedule']], [20, 40, 60, 80])

    def test_scenario_references_resolve_after_sweep(self):
        out = _materialize_scenario_generators({
            'bw': 40, 'delay': 80, 'flows': 2, 'duration': 10,
            'per_flow_delays': [10, '$half_delay'],
        })
        self.assertEqual(out['per_flow_delays'], [10, 40])

    def test_rejects_unknown_matrix_name(self):
        manifest = {
            'kind': 'olympus-eval', 'version': 1,
            'checkpoints': {'a': 'a.pt'}, 'scenarios': {'s': 's.yaml'},
            'environments': {'em': 'mininet'},
            'matrix': {'checkpoints': ['missing'], 'scenarios': ['s'],
                       'environments': ['em']},
        }
        with self.assertRaisesRegex(ValueError, 'unknown names'):
            expand_manifest(manifest, Path('/tmp'))


if __name__ == '__main__':
    unittest.main()
