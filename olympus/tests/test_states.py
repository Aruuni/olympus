import unittest

import numpy as np

from olympus.common.state_plugins import load_state_module, normalize_state_name
from olympus.states import astraea_deepcc


class AstraeaDeepccStateTest(unittest.TestCase):
    def test_matches_astraea_deepcc_transform(self):
        raw = {
            'avg_thr': 1_250_000.0,
            'max_tput': 2_000_000.0,
            'avg_urtt': 20_000.0,
            'min_rtt': 18_000.0,
            'srtt_us': 22_000.0 * 8.0,
            'cwnd': 32.0,
            'loss_ratio': 1000.0,
            'packets_out': 10.0,
            'pacing_rate': 2_000_000.0,
            'retrans_out': 1.0,
        }

        expected = np.asarray([
            0.625,
            20_000.0 / 18_000.0,
            22_000.0 / 18_000.0,
            32.0 * 1460.0 * 8.0 / (18_000.0 / 1e6) / 2_000_000.0 / 10.0,
            0.2,
            0.036,
            1000.0 / 2_000_000.0,
            10.0 / 32.0,
            1.0,
            1.0 / 32.0,
        ], dtype=np.float32)

        np.testing.assert_allclose(
            astraea_deepcc.normalize_state(raw),
            expected,
            rtol=1e-6,
        )

    def test_uses_peak_thr_as_kernel_max_tput_fallback(self):
        raw = {
            'avg_thr': 1_250_000.0,
            'peak_thr': 2_000_000.0,
            'avg_urtt': 20_000.0,
            'min_rtt': 18_000.0,
            'srtt_us': 22_000.0 * 8.0,
            'cwnd': 32.0,
            'loss_bytes': 20.0,
            'interval_ms': 20.0,
            'packets_out': 10.0,
            'pacing_rate': 2_000_000.0,
            'retrans_out': 1.0,
        }

        state = astraea_deepcc.normalize_state(raw)

        self.assertAlmostEqual(float(state[0]), 0.625)
        self.assertAlmostEqual(float(state[6]), 1000.0 / 2_000_000.0)

    def test_rejects_normalized_clean_slate_only_observations(self):
        raw = {
            'throughput_norm': 0.8,
            'pacing_norm': 1.0,
            'loss_norm': 0.0,
            'acks_over_cwnd': 0.5,
            'interval_s': 0.02,
            'srtt_norm': 0.4,
            'delay_metric': 0.9,
        }

        with self.assertRaisesRegex(ValueError, 'requires raw DeepCC'):
            astraea_deepcc.normalize_state(raw)

    def test_clean_slate_astraea_alias_resolves(self):
        self.assertEqual(
            normalize_state_name('clean_slate_astraea'),
            'astraea_deepcc',
        )
        self.assertIs(
            load_state_module('dreamer_v3', 'clean_slate_astraea'),
            astraea_deepcc,
        )


if __name__ == '__main__':
    unittest.main()
