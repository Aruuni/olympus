import unittest

import numpy as np
import torch

from olympus.actions import astraea, cwnd_multiplier
from olympus.common.action_plugins import assert_action_compatible


class ActionPluginTest(unittest.TestCase):
    def test_default_multiplier_range(self):
        np.testing.assert_allclose(
            cwnd_multiplier.to_multiplier(
                np.asarray([-1.0, 0.0, 1.0])),
            [0.5, 1.0, 2.0],
        )
        self.assertEqual(
            cwnd_multiplier.apply_cwnd(100, 0.001, 4, 10_000),
            101,
        )

    def test_astraea_mapping_and_directional_rounding(self):
        np.testing.assert_allclose(
            astraea.to_multiplier(np.asarray([-1.0, 0.0, 1.0])),
            [1.0 / 1.025, 1.0, 1.025],
        )
        self.assertEqual(astraea.apply_cwnd(100, 0.1, 4, 10_000), 101)
        self.assertEqual(astraea.apply_cwnd(100, -0.1, 4, 10_000), 99)
        self.assertEqual(astraea.apply_cwnd(100, 0.0, 4, 10_000), 100)

    def test_inverse_mapping_round_trip(self):
        actions = np.linspace(-1.0, 1.0, 21)
        for plugin in (cwnd_multiplier, astraea):
            recovered = plugin.from_multiplier(plugin.to_multiplier(actions))
            np.testing.assert_allclose(recovered, actions, atol=1e-7)

    def test_tensor_mapping_is_differentiable(self):
        action = torch.tensor([-0.5, 0.5], requires_grad=True)
        value = astraea.to_multiplier(action).sum()
        value.backward()
        self.assertTrue(torch.isfinite(action.grad).all())

    def test_legacy_checkpoint_only_matches_default(self):
        default_meta = {
            'action_name': 'cwnd_multiplier',
            'action_dim': 1,
            'action_version': 'cwnd_multiplier_v1',
            'action_options': cwnd_multiplier.options(),
            'legacy_default': True,
        }
        assert_action_compatible(default_meta, {}, source='legacy')

        astraea_meta = {
            'action_name': 'astraea',
            'action_dim': 1,
            'action_version': 'astraea_relative_cwnd_v1',
            'action_options': astraea.options(),
            'legacy_default': False,
        }
        with self.assertRaisesRegex(ValueError, 'no action metadata'):
            assert_action_compatible(astraea_meta, {}, source='legacy')

    def test_checkpoint_action_mismatch_is_rejected(self):
        current = {
            'action_name': 'astraea',
            'action_dim': 1,
            'action_version': 'astraea_relative_cwnd_v1',
            'action_options': astraea.options(),
        }
        checkpoint = {
            'action_meta': {
                'action_name': 'cwnd_multiplier',
                'action_dim': 1,
                'action_version': 'cwnd_multiplier_v1',
                'action_options': cwnd_multiplier.options(),
            },
        }
        with self.assertRaisesRegex(ValueError, 'action mismatch'):
            assert_action_compatible(current, checkpoint)


if __name__ == '__main__':
    unittest.main()
