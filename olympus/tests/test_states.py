import unittest

import numpy as np

from olympus.common.state_plugins import load_state_module, normalize_state_name
from olympus.states import astraea, astraea_deepcc, dreamer, proteus, tempest


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


class ProteusStateTest(unittest.TestCase):
    def test_is_exact_astraea_state(self):
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

        self.assertEqual(proteus.STATE_DIM, astraea.STATE_DIM)
        self.assertEqual(proteus.STATE_FEATURES, astraea.STATE_FEATURES)
        self.assertEqual(
            proteus.STATE_FEATURE_VERSION,
            astraea.STATE_FEATURE_VERSION,
        )
        np.testing.assert_array_equal(proteus.STATE_LOW, astraea.STATE_LOW)
        np.testing.assert_array_equal(proteus.STATE_HIGH, astraea.STATE_HIGH)
        np.testing.assert_allclose(
            proteus.normalize_state(dict(raw)),
            astraea.normalize_state(dict(raw)),
        )
        self.assertIs(load_state_module('sac', 'proteus'), proteus)


class DreamerStateTest(unittest.TestCase):
    def test_is_tempest_without_kalman_min_rtt(self):
        raw = {
            'cwnd': 32,
            'avg_thr': 1_250_000.0,
            'avg_urtt': 20_000.0,
            'srtt_us': 22_000.0 * 8.0,
            'pacing_rate': 2_000_000.0,
            'packets_out': 10.0,
            'retrans_out': 1.0,
            'prev_urtt': 18_000.0,
            'prev_cwnd': 30.0,
            'peak_thr': 2_500_000.0,
            # Dreamer must ignore both forms of minimum RTT input.
            'min_rtt': 1.0,
            'min_rtt_us': 999_999_999.0,
        }

        dreamer_state = dreamer.normalize_state(dict(raw))
        tempest_state = tempest.normalize_state(dict(raw))

        self.assertEqual(dreamer.STATE_DIM, 10)
        self.assertNotIn('min_rtt', ' '.join(dreamer.STATE_FEATURES))
        np.testing.assert_allclose(dreamer_state, tempest_state[:10])

    def test_does_not_mutate_input_with_kalman_state(self):
        raw = {'cwnd': 10, 'avg_urtt': 20_000.0}

        dreamer.normalize_state(raw)

        self.assertNotIn('kalman_min_rtt_us', raw)
        self.assertIs(load_state_module('dreamer_v3', 'dreamer'), dreamer)


if __name__ == '__main__':
    unittest.main()
