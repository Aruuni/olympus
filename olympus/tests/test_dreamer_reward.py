import math
import unittest

from olympus.rewards.dreamer import RewardCalc


class DreamerRewardTest(unittest.TestCase):
    def _calc(self):
        # One step per three-second proximity interval keeps tests direct.
        return RewardCalc(
            initial_bw_bytes_s=10_000_000.0,
            initial_rtt_us=20_000.0,
            interval_ms=3000.0,
            min_rtt_bonus=20.0,
        )

    @staticmethod
    def _info(min_rtt):
        return {
            'avg_thr': 10_000_000.0,
            'avg_urtt': 20_000.0,
            'srtt_us': 20_000.0 * 8.0,
            'min_rtt': min_rtt,
        }

    def test_exact_base_rtt_gets_full_proximity_bonus(self):
        calc = self._calc()

        reward = calc.step(self._info(20_000.0))

        self.assertAlmostEqual(reward, 45.0)
        self.assertAlmostEqual(calc.last_components['base'], 25.0)
        self.assertAlmostEqual(
            calc.last_components['min_rtt_proximity'], 20.0)
        self.assertAlmostEqual(calc.last_components['min_rtt_closeness'], 1.0)

    def test_proximity_bonus_uses_ratio_below_base(self):
        calc = self._calc()

        reward = calc.step(self._info(10_000.0))

        self.assertAlmostEqual(reward, 35.0)
        self.assertAlmostEqual(calc.last_components['min_rtt_closeness'], 0.5)
        self.assertAlmostEqual(calc.last_components['min_rtt_error_us'], 10_000.0)

    def test_proximity_bonus_uses_inverse_ratio_above_base(self):
        calc = self._calc()

        reward = calc.step(self._info(40_000.0))

        self.assertAlmostEqual(reward, 35.0)
        self.assertAlmostEqual(calc.last_components['min_rtt_closeness'], 0.5)

    def test_missing_min_rtt_gets_no_bonus(self):
        missing = self._calc()

        self.assertAlmostEqual(missing.step(self._info(0.0)), 25.0)
        self.assertEqual(missing.last_components['min_rtt_closeness'], 0.0)
        self.assertTrue(math.isinf(
            missing.last_components['min_rtt_error_us']))

    def test_supports_min_rtt_us_fallback_without_kalman(self):
        calc = self._calc()
        info = self._info(0.0)
        info.pop('min_rtt')
        info['min_rtt_us'] = 20_000.0

        calc.step(info)

        self.assertEqual(calc.min_rtt_us, 20_000.0)
        self.assertEqual(calc.kalman_min_rtt_us, 20_000.0)
        self.assertNotIn('kalman_min_rtt_us', info)


if __name__ == '__main__':
    unittest.main()
