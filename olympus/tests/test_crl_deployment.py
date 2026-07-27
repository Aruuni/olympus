import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from olympus.algorithms.sac.learner import (
    DistillationMemory,
    Learner,
    ReplayBuffer,
    UpdateBudget,
)
from olympus.algorithms.sac import learner as sac_learner
from olympus.algorithms.sac.model import Experience
from olympus.algorithms.sac.model import STATE_DIM
from olympus.algorithms.sac import worker as sac_worker
from olympus.deployment_crl import run
from olympus.deployment_crl import worker


def _config():
    return {
        'runtime': {
            'algorithm': 'sac',
            'reward': 'proteus',
            'state': 'proteus',
            'action': 'cwnd_multiplier',
        },
        'actions': {'cwnd_multiplier': {}},
        'reward': {'delay_weight': 2.0},
        'algorithms': {
            'sac': {
                'training': {'min_replay': 32},
                'agent': {'hidden': 64, 'interval_ms': 20},
            },
        },
        'deployment': {'continual': {'learner_port': 7654}},
    }


class UpdateBudgetTest(unittest.TestCase):
    def test_credit_is_data_driven_and_burst_limited(self):
        budget = UpdateBudget(0.25, 3)
        budget.add(3)
        self.assertFalse(budget.take())
        budget.add(1)
        self.assertTrue(budget.take())
        self.assertFalse(budget.take())

        budget.add(1000)
        self.assertEqual(budget.credit, 3.0)
        self.assertEqual(sum(budget.take() for _ in range(4)), 3)


class ServiceConfigTest(unittest.TestCase):
    def test_prepare_copies_seed_once_and_writes_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / 'seed.pt'
            seed.write_bytes(b'initial checkpoint')
            state_dir = root / 'state'

            cfg, resolved, live = run.prepare_service_config(
                _config(), str(seed), str(state_dir))

            self.assertEqual(Path(live).read_bytes(), b'initial checkpoint')
            self.assertTrue(Path(resolved).is_file())
            self.assertEqual(cfg['training']['checkpoint'], live)
            self.assertIsNone(cfg['training']['resume_from'])
            self.assertTrue(cfg['continual']['persist_replay'])
            self.assertEqual(cfg['continual']['learner_port'], 7654)
            self.assertTrue(
                cfg['continual']['distillation']['enabled'])

            seed.write_bytes(b'replacement seed')
            run.prepare_service_config(_config(), str(seed), str(state_dir))
            self.assertEqual(Path(live).read_bytes(), b'initial checkpoint')

            with open(resolved) as handle:
                written = yaml.safe_load(handle)
            self.assertEqual(written['runtime']['reward'],
                             'proteus')

    def test_prepare_rejects_non_sac_or_ground_truth_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / 'seed.pt'
            seed.write_bytes(b'x')
            cfg = _config()
            cfg['runtime']['algorithm'] = 'td3'
            with self.assertRaisesRegex(ValueError, 'algorithm'):
                run.prepare_service_config(cfg, str(seed), directory)

            cfg = _config()
            cfg['runtime']['reward'] = 'dreamer'
            with self.assertRaisesRegex(ValueError, 'reward'):
                run.prepare_service_config(cfg, str(seed), directory)

    def test_old_reward_name_normalizes_without_changing_replay_identity(self):
        legacy = _config()
        legacy['runtime']['reward'] = 'crl_network_utility'

        self.assertEqual(
            sac_learner._reward_signature(legacy),
            sac_learner._reward_signature(_config()),
        )
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / 'seed.pt'
            seed.write_bytes(b'x')
            cfg, _, _ = run.prepare_service_config(
                legacy, str(seed), str(Path(directory) / 'state'))
            self.assertEqual(cfg['runtime']['reward'], 'proteus')

    def test_listener_environment_controls_exploration(self):
        cfg = _config()
        cfg['agent'] = {'hidden': 64, 'interval_ms': 20}
        cfg['continual'] = {
            'exploration': True,
            'weight_pull_every': 25,
            'worker_push_every': 8,
            'replay_path': '/tmp/replay',
        }
        env = run._listener_environment(
            cfg, '/tmp/config', '/tmp/model', '127.0.0.1:1', 'abcd',
            '/tmp/traces')
        self.assertEqual(env['SAO_DETERMINISTIC'], '0')
        self.assertEqual(env['SAO_WEIGHT_PULL_EVERY'], '25')
        self.assertEqual(env['OC_PUSH_EVERY'], '8')
        self.assertEqual(env['SAO_REWARD'], 'proteus')


class WorkerAdapterTest(unittest.TestCase):
    def test_listener_environment_is_mapped_for_sac_worker(self):
        source = {
            'ASTRAEA_FLOW_FD': '7',
            'ASTRAEA_FLOW_ID': '42',
            'ASTRAEA_CONFIG': '/tmp/config.yaml',
            'ASTRAEA_MODEL': '/tmp/model.pt',
        }
        targets = {
            'OC_FLOW_FD', 'OC_FLOW_ID', 'SAO_CONFIG', 'SAO_CHECKPOINT',
            'OC_CPORT', 'OC_STATE_FD', 'SAO_EPISODE',
        }
        clean = {key: value for key, value in os.environ.items()
                 if key not in targets}
        clean.update(source)
        with mock.patch.dict(os.environ, clean, clear=True):
            worker.prepare_worker_environment()
            self.assertEqual(os.environ['OC_FLOW_FD'], '7')
            self.assertEqual(os.environ['OC_FLOW_ID'], '42')
            self.assertEqual(os.environ['OC_CPORT'], '42')
            self.assertEqual(os.environ['SAO_CONFIG'], '/tmp/config.yaml')
            self.assertEqual(os.environ['SAO_CHECKPOINT'], '/tmp/model.pt')

    def test_continual_worker_holds_empty_poll_metrics(self):
        history = {}
        first = {'avg_thr': 1_000_000, 'avg_urtt': 20_000,
                 'cnt': 2, 'thr_cnt': 2}
        empty = {'avg_thr': 0, 'avg_urtt': 0, 'cnt': 0, 'thr_cnt': 0}
        sac_worker._carry_forward_drained_metrics(first, history)
        sac_worker._carry_forward_drained_metrics(empty, history)
        self.assertEqual(empty, first)


class ReplayPersistenceTest(unittest.TestCase):
    @staticmethod
    def _learner_shell(path, signature='reward-v1'):
        instance = Learner.__new__(Learner)
        instance.persist_replay = True
        instance.replay_path = str(path)
        instance.reward_signature = signature
        instance._mixed_enabled = False
        instance._last_replay_save = 0.0
        instance.buf = ReplayBuffer(max_transitions=10)
        instance.distill_enabled = True
        instance.distill_memory = DistillationMemory(capacity=4)
        return instance

    def test_replay_round_trip_and_reward_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.pkl'
            source = self._learner_shell(path)
            state = np.zeros(STATE_DIM, dtype=np.float32)
            source.buf.push(Experience(
                state=state, action=0.0, reward=1.0, next_state=state,
                done=False, traj_id='flow-1', step_in_traj=0))
            source.distill_memory.add_batch({
                'state': state.reshape(1, 1, -1),
                'action': np.zeros((1, 1, 1), dtype=np.float32),
                'mask': np.ones((1, 1), dtype=np.float32),
            })
            source._save_replay()

            restored = self._learner_shell(path)
            restored._restore_replay()
            self.assertEqual(restored.buf.size(), 1)
            self.assertEqual(len(restored.distill_memory), 1)

            incompatible = self._learner_shell(path, signature='reward-v2')
            incompatible._restore_replay()
            self.assertEqual(incompatible.buf.size(), 0)


class ContinualLearnerSmokeTest(unittest.TestCase):
    def test_sac_update_and_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            cfg = _config()
            cfg['training'] = {
                'checkpoint': str(checkpoint),
                'batch_size': 4,
                'seq_len': 2,
                'min_replay': 2,
                'replay_capacity': 32,
                'save_every': 0,
                'param_broadcast_every': 1,
            }
            cfg['agent'] = {'hidden': 16, 'head_hidden': 8}
            cfg['continual'] = {
                'enabled': True,
                'updates_per_transition': 0.25,
                'max_update_burst': 2,
                'persist_replay': False,
                'distillation': {
                    'enabled': True,
                    'actor_weight': 0.05,
                    'critic_weight': 0.05,
                    'anchor_capacity': 4,
                    'anchor_min_sequences': 1,
                    'batch_sequences': 1,
                    'capture_every_updates': 1,
                    'teacher_update_every': 0,
                },
            }
            with mock.patch.object(sac_learner._QueueManager, 'start'):
                learner = Learner(cfg, port=0, authkey=b'crl-test')
            learner._mgr = mock.Mock()
            try:
                for index in range(4):
                    state = np.full(STATE_DIM, index / 10.0,
                                    dtype=np.float32)
                    learner.buf.push(Experience(
                        state=state, action=0.0, reward=1.0,
                        next_state=state, done=index == 3,
                        traj_id='smoke-flow', step_in_traj=index))
                learner._r_rms.update(np.asarray([1.0], dtype=np.float32))
                learner._sac_update()
                self.assertEqual(learner.step, 1)
                self.assertGreater(len(learner.distill_memory), 0)
            finally:
                learner.stop()

            self.assertTrue(checkpoint.is_file())
            self.assertFalse(Path(str(checkpoint) + '.tmp').exists())
            payload = torch.load(
                checkpoint, map_location='cpu', weights_only=False)
            distillation = payload['distillation_state']
            self.assertTrue(distillation['enabled'])
            self.assertIn('teacher_actor', distillation)
            self.assertIn('teacher_critic', distillation)


if __name__ == '__main__':
    unittest.main()
