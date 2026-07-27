import math
import unittest

from olympus.rewards.proteus import RewardCalc
from olympus.rewards.crl_network_utility import RewardCalc as LegacyRewardCalc


class ProteusRewardTest(unittest.TestCase):
    @staticmethod
    def _info(throughput_bytes_s=1_000_000.0, rtt_us=20_000.0,
              min_rtt_us=20_000.0, **extra):
        info = {
            'avg_thr': throughput_bytes_s,
            'avg_urtt': rtt_us,
            'srtt_us': rtt_us * 8.0,
            'min_rtt': min_rtt_us,
            'interval_ms': 20.0,
        }
        info.update(extra)
        return info

    def test_more_goodput_increases_reward_at_the_same_rtt(self):
        low = RewardCalc().step(self._info(throughput_bytes_s=500_000.0))
        high = RewardCalc().step(self._info(throughput_bytes_s=1_000_000.0))

        self.assertGreater(high, low)

    def test_old_reward_name_is_a_compatibility_alias(self):
        self.assertIs(LegacyRewardCalc, RewardCalc)

    def test_rtt_headroom_is_free_then_excess_is_penalized(self):
        calc = RewardCalc(rtt_headroom=0.05, delay_weight=2.0)
        within = calc.step(self._info(rtt_us=21_000.0))
        above = calc.step(self._info(rtt_us=42_000.0))

        self.assertEqual(
            calc.last_components['min_rtt_closeness'], 20_000 / 42_000)
        self.assertAlmostEqual(
            calc.last_components['delay_excess_log'], math.log(2.0))
        self.assertGreater(within, above)

    def test_loss_fraction_is_penalized(self):
        clean = RewardCalc().step(self._info(loss_fraction=0.0))
        lossy_calc = RewardCalc()
        lossy = lossy_calc.step(self._info(loss_fraction=0.01))

        self.assertAlmostEqual(
            lossy_calc.last_components['loss_cost'], math.log(2.0))
        self.assertGreater(clean, lossy)

    def test_counter_based_startup_loss_is_ignored_once(self):
        calc = RewardCalc(interval_ms=20.0)
        calc.step(self._info(loss_bytes=1_000_000.0))
        self.assertEqual(calc.last_components['loss_fraction'], 0.0)

        calc.step(self._info(loss_bytes=2_000.0))
        self.assertGreater(calc.last_components['loss_fraction'], 0.0)

    def test_zero_goodput_receives_stall_penalty(self):
        calc = RewardCalc(stall_penalty=1.0)
        reward = calc.step(self._info(throughput_bytes_s=0.0))

        self.assertEqual(calc.last_components['stalled'], 1.0)
        self.assertLess(reward, 0.0)

    def test_missing_min_rtt_is_finite_and_has_no_delay_penalty(self):
        calc = RewardCalc()
        reward = calc.step(self._info(min_rtt_us=0.0))

        self.assertTrue(math.isfinite(reward))
        self.assertEqual(calc.last_components['delay_cost'], 0.0)
        self.assertEqual(calc.last_components['min_rtt_closeness'], 0.0)
        self.assertTrue(math.isinf(calc.last_components['min_rtt_error_us']))

    def test_dreamer_compatible_measurement_properties(self):
        calc = RewardCalc()
        info = self._info(throughput_bytes_s=2_000_000.0)
        calc.step(info)

        self.assertEqual(calc.max_tput, 2_000_000.0)
        self.assertEqual(calc.srtt_us, 20_000.0)
        self.assertEqual(calc.min_rtt_us, 20_000.0)
        self.assertEqual(calc.kalman_min_rtt_us, 20_000.0)


if __name__ == '__main__':
    unittest.main()
