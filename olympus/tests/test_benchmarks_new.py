from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from benchmarks_new import common
from benchmarks_new.paper_efficiency import episode_efficiency


class PaperEfficiencyTests(unittest.TestCase):
    def test_missing_listener_samples_are_not_counted_as_zero_throughput(self):
        traces = [
            (0, {
                't_s': np.array([0.5, 1.5, 2.5]),
                'avg_thr_mbps': np.array([40.0, 40.0, 40.0]),
                'srtt_ms': np.array([10.0, 10.0, 10.0]),
            }),
            (1, {
                't_s': np.array([2.0, 3.5]),
                'avg_thr_mbps': np.array([60.0, 60.0]),
                'srtt_ms': np.array([10.0, 10.0]),
            }),
        ]

        delay, throughput = episode_efficiency(
            traces,
            bw_mbps=100.0,
            base_rtt_ms=10.0,
            duration_s=4.0,
            starts=[0.0, 1.0],
            flow_duration_s=3.0,
            score_window_s=3.0,
        )

        self.assertEqual(delay, 1.0)
        self.assertEqual(throughput, 0.8)


class BenchmarkLauncherTests(unittest.TestCase):
    def test_run_eval_slices_matrix_and_notifies_after_each_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {
                'defaults': {'output_root': directory},
                'checkpoints': {
                    'Astraea': {'path': '/tmp/astraea'},
                    'Dreamer': {'path': '/tmp/dreamer'},
                },
                'matrix': {
                    'checkpoints': ['Astraea', 'Dreamer'],
                    'scenarios': ['scenario'],
                    'environments': ['raynet'],
                },
                'runs': [
                    {'checkpoint': 'Astraea', 'name': 'astraea-special'},
                    {'checkpoint': 'Dreamer', 'name': 'dreamer-special'},
                ],
            }
            completed = []
            slices = []

            def run_slice(value, extra=None):
                slices.append(value)
                return 0

            with mock.patch.object(
                    common, '_canonical_manifest',
                    return_value=(manifest, [])):
                with mock.patch.object(
                        common, '_run_manifest', side_effect=run_slice):
                    code = common.run_eval(
                        Path('/tmp/benchmark.yaml'),
                        after_checkpoint=completed.append,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(completed, ['Astraea', 'Dreamer'])
        self.assertEqual(
            [value['matrix']['checkpoints'] for value in slices],
            [['Astraea'], ['Dreamer']],
        )
        self.assertEqual(
            [list(value['checkpoints']) for value in slices],
            [['Astraea'], ['Dreamer']],
        )
        self.assertEqual(
            [[run['checkpoint'] for run in value['runs']] for value in slices],
            [['Astraea'], ['Dreamer']],
        )

    def test_suite_updates_plot_after_each_checkpoint(self):
        plot_commands = []

        def run_eval(config, extra=None, debug=False, after_checkpoint=None):
            self.assertIsNotNone(after_checkpoint)
            self.assertEqual(after_checkpoint('Astraea'), 0)
            self.assertEqual(after_checkpoint('Dreamer'), 0)
            return 0

        def run_plot(command, cwd=None):
            plot_commands.append(command)
            return mock.Mock(returncode=0)

        with mock.patch.object(common, 'run_eval', side_effect=run_eval):
            with mock.patch.object(
                    common.subprocess, 'run', side_effect=run_plot):
                code = common.suite_main(
                    '/tmp/example/benchmark.py',
                    ['--config', '/tmp/example/config.yaml'],
                )

        self.assertEqual(code, 0)
        self.assertEqual(len(plot_commands), 2)
        self.assertTrue(all(command[1].endswith('/plot.py')
                            for command in plot_commands))


if __name__ == '__main__':
    unittest.main()
