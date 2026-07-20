import os
import unittest

from olympus.common import registry


class RegistryTest(unittest.TestCase):
    def test_multi_agent_flag(self):
        # Algorithms that train every flow jointly declare MULTI_AGENT = True.
        self.assertTrue(registry.is_multi_agent('ma_dreamer'))
        self.assertTrue(registry.is_multi_agent('ma_td3'))

    def test_single_agent_flag(self):
        # Everything else is single-agent (lagged self-play in multi-flow envs).
        for name in ('td3', 'mbpo_td3', 'dreamer_v3', 'orca',
                     'orca_td3', 'recurrent_ppo'):
            self.assertFalse(registry.is_multi_agent(name), name)

    def test_worker_script_resolves(self):
        for name in ('ma_dreamer', 'ma_td3', 'dreamer_v3'):
            self.assertIn(
                os.path.join('olympus', 'algorithms', name, 'worker.py'),
                registry.worker_script(name))

    def test_action_plugins_resolve(self):
        self.assertEqual(
            registry.action_module('cwnd_multiplier').ACTION_NAME,
            'cwnd_multiplier',
        )
        self.assertEqual(
            registry.action_module('astraea').ACTION_NAME,
            'astraea',
        )


if __name__ == '__main__':
    unittest.main()
