import json
import os
import tempfile
import unittest
from unittest import mock

import yaml

from benchmarks import run_all
from olympus.common import bench_utils


class KernelBenchmarkTest(unittest.TestCase):
    def test_kernel_approach_needs_only_kernel_cc(self):
        bench_utils._validate_approach({
            'kind': 'kernel',
            'kernel_cc': 'cubic',
        })

    def test_run_all_selects_kernel_approaches(self):
        self.assertEqual(run_all._resolve_approaches(['cubic']), ['cubic'])
        self.assertIn('bbr', run_all._resolve_approaches())

    def test_configured_kernel_ccs_are_unique(self):
        with open(run_all._APPROACHES_CONFIG) as handle:
            cfg = yaml.safe_load(handle) or {}
        kernel_ccs = [
            item['kernel_cc']
            for item in cfg.get('approaches', [])
            if item.get('kind') == 'kernel'
        ]
        self.assertIn('cubic', kernel_ccs)
        self.assertIn('bbr', kernel_ccs)
        self.assertEqual(len(kernel_ccs), len(set(kernel_ccs)))

    def test_available_kernel_cc_does_not_run_modprobe(self):
        approach = {'kind': 'kernel', 'kernel_cc': 'cubic'}
        with mock.patch.object(
                bench_utils, '_available_kernel_ccs',
                return_value={'cubic'}), mock.patch.object(
                    bench_utils.subprocess, 'run') as run:
            bench_utils._ensure_kernel_cc_available(approach)
        run.assert_not_called()

    def test_iperf_parser_uses_positive_sender_summary_after_zero_receiver(self):
        data = {
            'intervals': [{
                'sum': {
                    'end': 1.0,
                    'bits_per_second': 40_000_000,
                },
            }],
            'end': {
                'sum_received': {'bits_per_second': 0},
                'sum_sent': {'bits_per_second': 42_000_000},
            },
            'error': 'interrupt - the client has terminated',
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'iperf.json')
            with open(path, 'w') as handle:
                json.dump(data, handle)
            parsed = bench_utils._parse_iperf_json(path)
        self.assertEqual(parsed['goodput_mbps'], 42.0)


if __name__ == '__main__':
    unittest.main()
