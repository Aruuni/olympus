import copy
import unittest

import numpy as np
import torch

from olympus.algorithms.sac.learner import DistillationMemory
from olympus.algorithms.sac.model import (
    ACTION_DIM,
    STATE_DIM,
    Actor,
    TwinCritic,
    actor_distillation_loss,
    critic_distillation_loss,
)


class DistillationLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.actor = Actor(STATE_DIM, hidden=16, head_hidden=8)
        self.critic = TwinCritic(STATE_DIM, hidden=16, head_hidden=8)
        self.teacher_actor = copy.deepcopy(self.actor)
        self.teacher_critic = copy.deepcopy(self.critic)
        self.states = torch.randn(3, 5, STATE_DIM)
        self.actions = torch.tanh(torch.randn(3, 5, ACTION_DIM))
        self.mask = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ], dtype=torch.float32)

    def test_identical_teacher_has_zero_retention_loss(self):
        actor_loss = actor_distillation_loss(
            self.actor, self.teacher_actor, self.states, self.mask)
        critic_loss = critic_distillation_loss(
            self.critic, self.teacher_critic,
            self.states, self.actions, self.mask)

        self.assertAlmostEqual(float(actor_loss.item()), 0.0, places=6)
        self.assertAlmostEqual(float(critic_loss.item()), 0.0, places=6)

    def test_drift_is_penalized_and_gradients_reach_students_only(self):
        with torch.no_grad():
            self.actor.mean_head.bias.add_(0.5)
            self.critic.q1.q_head.bias.add_(0.5)

        actor_loss = actor_distillation_loss(
            self.actor, self.teacher_actor, self.states, self.mask)
        critic_loss = critic_distillation_loss(
            self.critic, self.teacher_critic,
            self.states, self.actions, self.mask)
        (actor_loss + critic_loss).backward()

        self.assertGreater(float(actor_loss.item()), 0.0)
        self.assertGreater(float(critic_loss.item()), 0.0)
        self.assertIsNotNone(self.actor.mean_head.bias.grad)
        self.assertIsNotNone(self.critic.q1.q_head.bias.grad)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in self.teacher_actor.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in self.teacher_critic.parameters()))


class DistillationMemoryTest(unittest.TestCase):
    @staticmethod
    def _batch(count=12, seq_len=4):
        return {
            'state': np.arange(
                count * seq_len * STATE_DIM, dtype=np.float32
            ).reshape(count, seq_len, STATE_DIM),
            'action': np.zeros(
                (count, seq_len, ACTION_DIM), dtype=np.float32),
            'mask': np.ones((count, seq_len), dtype=np.float32),
        }

    def test_reservoir_is_bounded_and_round_trips(self):
        memory = DistillationMemory(capacity=5)
        memory.add_batch(self._batch())

        self.assertEqual(len(memory), 5)
        self.assertEqual(memory.seen, 12)
        sample = memory.sample(3)
        self.assertEqual(sample['state'].shape, (3, 4, STATE_DIM))
        self.assertEqual(sample['action'].shape, (3, 4, ACTION_DIM))
        self.assertEqual(sample['mask'].shape, (3, 4))

        restored = DistillationMemory(capacity=5)
        restored.restore(memory.to_payload())
        self.assertEqual(len(restored), 5)
        self.assertEqual(restored.seen, 12)


if __name__ == '__main__':
    unittest.main()
