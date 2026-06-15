import os
import csv
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from olympus.algorithms.ma_dreamer.learner import (
    JointReplayBuffer,
    Learner,
)
from olympus.algorithms.ma_dreamer.model import (
    Experience,
    STATE_DIM,
    per_agent_team_reward,
    r_fair_from_avg_throughput,
    shared_team_reward,
)
from olympus.common.marl_team_reward import over_share_fraction
from olympus.common.multi_flow_episode_plot import (
    _r_fair_matrix,
    plot as plot_multi_flow_episode,
)
from olympus.rewards.tempest_fairness_ma import (
    RewardCalc as FairShareTempestReward,
)
from olympus.rewards.tempest_fairness_ma_symetric import (
    RewardCalc as CreditTempestReward,
)
from olympus import orchestrator


def _experience(agent_id, group_step, throughput, reward=5.0):
    state = np.full(STATE_DIM, group_step + agent_id, dtype=np.float32)
    next_state = state + 0.25
    return Experience(
        state=state,
        action=0.1 * (agent_id + 1),
        reward=reward,
        next_state=next_state,
        avg_throughput=throughput,
        agent_id=agent_id,
        group_id='group',
        group_step=group_step,
        done=False,
        traj_id=f'group_agent{agent_id}',
        step_in_traj=group_step,
    )


class FairnessMetricTest(unittest.TestCase):
    def test_attached_metric_uses_average_throughput(self):
        self.assertAlmostEqual(
            float(r_fair_from_avg_throughput(
                np.asarray([10.0, 10.0]))),
            0.0,
        )
        self.assertAlmostEqual(
            float(r_fair_from_avg_throughput(
                np.asarray([20.0, 0.0]))),
            0.5,
        )
        self.assertAlmostEqual(
            float(r_fair_from_avg_throughput(
                np.asarray([10.0, 10.0, 100.0]),
                np.asarray([1.0, 1.0, 0.0]))),
            0.0,
        )

    def test_team_reward_subtracts_unfairness_cost(self):
        local_reward = torch.tensor([[8.0, 12.0]])
        avg_throughput = torch.tensor([[20.0, 0.0]])
        mask = torch.ones_like(local_reward)
        team_reward, fairness, cost = shared_team_reward(
            local_reward,
            avg_throughput,
            mask,
            fairness_weight=10.0,
        )
        self.assertAlmostEqual(float(fairness.item()), 0.5)
        self.assertAlmostEqual(float(cost.item()), 5.0)
        self.assertAlmostEqual(float(team_reward.item()), 5.0)

    def test_over_share_fraction_charges_only_the_greedy_flow(self):
        over = over_share_fraction(np.asarray([75.0, 25.0]))
        np.testing.assert_allclose(over, [0.25, 0.0])
        # Inactive agents are excluded from the mean and pay nothing.
        over = over_share_fraction(
            np.asarray([75.0, 25.0, 500.0]),
            np.asarray([1.0, 1.0, 0.0]))
        np.testing.assert_allclose(over, [0.25, 0.0, 0.0])
        # A single active flow can never be unfair.
        np.testing.assert_allclose(
            over_share_fraction(
                np.asarray([75.0, 0.0]), np.asarray([1.0, 0.0])),
            [0.0, 0.0])

    def test_per_agent_credit_reward_separates_greedy_and_starved(self):
        # 100 Mbps link, 75/25 split, drained queues: locals are 25 and 12.5,
        # R_fair = 0.25, over-share fraction = 0.25 for the greedy flow.
        local_reward = torch.tensor([[25.0, 12.5]])
        avg_throughput = torch.tensor([[75.0, 25.0]])
        mask = torch.ones_like(local_reward)
        rewards, fairness, cost = per_agent_team_reward(
            local_reward,
            avg_throughput,
            mask,
            team_alpha=0.5,
            fairness_weight=25.0,
            fairness_cap=1.0,
            over_weight=50.0,
        )
        self.assertAlmostEqual(float(fairness.item()), 0.25)
        self.assertAlmostEqual(float(cost.item()), 6.25)
        torch.testing.assert_close(
            rewards, torch.tensor([[3.125, 9.375]]))
        # NumPy path matches the torch path.
        np_rewards, _, _ = per_agent_team_reward(
            local_reward.numpy(),
            avg_throughput.numpy(),
            mask.numpy(),
            team_alpha=0.5,
            fairness_weight=25.0,
            fairness_cap=1.0,
            over_weight=50.0,
        )
        np.testing.assert_allclose(np_rewards, [[3.125, 9.375]])

    def test_per_agent_credit_reward_matches_shared_reward_at_fairness(self):
        local_reward = torch.tensor([[25.0, 25.0]])
        avg_throughput = torch.tensor([[50.0, 50.0]])
        mask = torch.ones_like(local_reward)
        rewards, fairness, _ = per_agent_team_reward(
            local_reward, avg_throughput, mask)
        shared, _, _ = shared_team_reward(
            local_reward, avg_throughput, mask)
        self.assertAlmostEqual(float(fairness.item()), 0.0)
        torch.testing.assert_close(
            rewards, shared.unsqueeze(-1).expand_as(rewards))

    def test_plot_metric_matches_training_metric(self):
        throughputs = np.asarray(
            [[10.0, 20.0], [10.0, 0.0]], dtype=np.float32)
        plotted = _r_fair_matrix(throughputs)
        expected = r_fair_from_avg_throughput(throughputs.T)
        np.testing.assert_allclose(plotted, expected, atol=1e-7)


class FairShareRewardTest(unittest.TestCase):
    def setUp(self):
        self.reward = FairShareTempestReward(
            initial_bw_bytes_s=80e6 / 8.0,
            initial_rtt_us=30_000.0,
            link_schedule=[
                {'t': 15, 'bw': 160.0, 'delay': 15.0},
            ],
            episode_start=100.0,
            interval_ms=20.0,
            flow_start_delays=[0.0, 0.0, 12.0, 12.0],
            flow_durations=[120.0, 120.0, 108.0, 108.0],
        )

    def test_uses_ground_truth_bw_over_current_active_flows(self):
        with mock.patch(
                'olympus.rewards.tempest_fairness_ma.time.monotonic',
                return_value=110.0):
            value = self.reward.step({
                'avg_thr': 40e6 / 8.0,
                'avg_urtt': 30_000.0,
                'srtt_us': 30_000.0 * 8.0,
            })
        self.assertAlmostEqual(value, 25.0)
        self.assertEqual(self.reward.last_components['active_flows'], 2.0)
        self.assertAlmostEqual(
            self.reward.last_components['fair_bw_mbps'], 40.0)

    def test_tracks_link_change_and_late_joiners(self):
        with mock.patch(
                'olympus.rewards.tempest_fairness_ma.time.monotonic',
                return_value=120.0):
            value = self.reward.step({
                'avg_thr': 40e6 / 8.0,
                'avg_urtt': 15_000.0,
                'srtt_us': 15_000.0 * 8.0,
            })
        self.assertAlmostEqual(value, 25.0)
        self.assertEqual(self.reward.last_components['active_flows'], 4.0)
        self.assertAlmostEqual(
            self.reward.last_components['fair_bw_mbps'], 40.0)

    def test_departed_flow_is_removed_from_fair_share(self):
        reward = FairShareTempestReward(
            initial_bw_bytes_s=80e6 / 8.0,
            initial_rtt_us=30_000.0,
            episode_start=100.0,
            interval_ms=20.0,
            flow_start_delays=[0.0, 0.0],
            flow_durations=[10.0, 30.0],
            agent_id=0,
        )
        with mock.patch(
                'olympus.rewards.tempest_fairness_ma.time.monotonic',
                return_value=120.0):
            reward.step({
                'avg_thr': 0.0,
                'avg_urtt': 30_000.0,
                'srtt_us': 30_000.0 * 8.0,
            })
        self.assertEqual(reward.last_components['active_flows'], 1.0)
        self.assertEqual(reward.last_components['flow_present'], 0.0)
        self.assertAlmostEqual(
            reward.last_components['fair_bw_mbps'], 80.0)


class CreditRewardOvershootTest(unittest.TestCase):
    """The credit reward's throughput term is symmetric around fair share."""

    def _calc(self, cls):
        return cls(
            initial_bw_bytes_s=80e6 / 8.0,
            initial_rtt_us=30_000.0,
            episode_start=100.0,
            interval_ms=20.0,
            flow_start_delays=[0.0, 0.0],
        )

    def _step(self, calc, thr_bytes_s):
        with mock.patch(
                'olympus.rewards.tempest_fairness_ma.time.monotonic',
                return_value=110.0):
            return calc.step({
                'avg_thr': thr_bytes_s,
                'avg_urtt': 30_000.0,
                'srtt_us': 30_000.0 * 8.0,
            })

    def test_fair_share_still_earns_full_base_reward(self):
        value = self._step(self._calc(CreditTempestReward), 40e6 / 8.0)
        self.assertAlmostEqual(value, 25.0)

    def test_overshoot_is_penalized_ratio_symmetrically(self):
        calc = self._calc(CreditTempestReward)
        # 2x the fair share scores like 1/2 the fair share: 25 * 0.5.
        self.assertAlmostEqual(self._step(calc, 80e6 / 8.0), 12.5)
        self.assertAlmostEqual(self._step(calc, 20e6 / 8.0), 12.5)

    def test_shared_reward_keeps_flat_cap_above_fair_share(self):
        value = self._step(self._calc(FairShareTempestReward), 80e6 / 8.0)
        self.assertAlmostEqual(value, 25.0)


class MultiFlowPlotTest(unittest.TestCase):
    def test_staggered_trace_lengths_plot(self):
        columns = [
            't_s', 'option', 'cwnd_mult', 'cwnd',
            'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
            'loss_ratio', 'reward', 'kalman_rtt_ms',
            'act_mu', 'act_sig', 'agent_id',
            'fair_bw_mbps', 'active_flows',
        ]
        with tempfile.TemporaryDirectory() as directory:
            base = os.path.join(directory, 'state.csv')
            for agent_id, times in enumerate(
                    ([0.0, 0.02, 0.04, 0.06], [0.02, 0.04, 0.06])):
                path = os.path.join(directory, f'state_a{agent_id}.csv')
                with open(path, 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    for t_s in times:
                        writer.writerow({
                            't_s': t_s,
                            'option': 0,
                            'cwnd_mult': 1.0,
                            'cwnd': 20,
                            'avg_thr_mbps': 20.0,
                            'avg_urtt_ms': 20.0,
                            'srtt_ms': 20.0,
                            'min_rtt_ms': 20.0,
                            'loss_ratio': 0.0,
                            'reward': 25.0,
                            'kalman_rtt_ms': 20.0,
                            'act_mu': 0.0,
                            'act_sig': 0.1,
                            'agent_id': agent_id,
                            'fair_bw_mbps': 20.0,
                            'active_flows': 2,
                        })
            output = os.path.join(directory, 'episode.pdf')
            result = plot_multi_flow_episode(
                base, output, bw=40.0, delay=20.0, n_agents=2)
            self.assertAlmostEqual(result, 175.0)
            self.assertTrue(os.path.exists(output))


class TcpCongestionControlPreflightTest(unittest.TestCase):
    def test_available_cc_passes_without_loading_module(self):
        with mock.patch.object(
                orchestrator, '_available_tcp_cc',
                return_value={'cubic', 'astraea'}), mock.patch.object(
                    orchestrator.subprocess, 'run') as run:
            orchestrator._ensure_tcp_cc('astraea')
        run.assert_not_called()

    def test_missing_cc_fails_before_run_for_unprivileged_user(self):
        with mock.patch.object(
                orchestrator, '_available_tcp_cc',
                return_value={'cubic'}), mock.patch.object(
                    orchestrator.os, 'geteuid', return_value=1000):
            with self.assertRaisesRegex(
                    SystemExit, "TCP congestion control 'astraea'"):
                orchestrator._ensure_tcp_cc('astraea')

    def test_root_autoloads_astraea_module(self):
        completed = mock.Mock(returncode=0, stdout='')
        with mock.patch.object(
                orchestrator, '_available_tcp_cc',
                side_effect=[{'cubic'}, {'cubic', 'astraea'}]), \
                mock.patch.object(
                    orchestrator.os, 'geteuid', return_value=0), \
                mock.patch.object(
                    orchestrator.os.path, 'exists', return_value=True), \
                mock.patch.object(
                    orchestrator.subprocess, 'run',
                    return_value=completed) as run:
            orchestrator._ensure_tcp_cc('astraea')
        self.assertEqual(run.call_args.args[0][0], 'insmod')


class FlowCountTest(unittest.TestCase):
    def test_training_flow_counts_remain_capped_at_four(self):
        self.assertEqual(orchestrator._flow_values([2, 4, 5, 7]), [2, 4])

    def test_evaluation_flow_count_is_not_capped(self):
        self.assertEqual(orchestrator._execution_flow_value(5), 5)
        self.assertEqual(orchestrator._execution_flow_value(7), 7)


class FlowTimingTest(unittest.TestCase):
    def test_random_lifetimes_create_departures_and_keep_one_flow(self):
        duration = 120.0
        starts, lifetimes = orchestrator._flow_timings(
            4,
            duration,
            {
                'agent_join_pattern': 'random',
                'agent_join_max_frac': 0.2,
                'flow_lifetime_pattern': 'random',
                'flow_lifetime_min_frac': 0.25,
                'flow_lifetime_max_frac': 0.8,
                'flow_lifetime_keep_one': True,
            },
            {},
            random.Random(7),
        )
        available = [duration - start for start in starts]
        self.assertEqual(starts[0], 0.0)
        self.assertTrue(all(0.0 <= start <= 24.0 for start in starts))
        self.assertTrue(all(
            1.0 <= lifetime <= window
            for lifetime, window in zip(lifetimes, available)
        ))
        self.assertTrue(any(
            lifetime == window
            for lifetime, window in zip(lifetimes, available)
        ))
        self.assertTrue(any(
            lifetime < window
            for lifetime, window in zip(lifetimes, available)
        ))

    def test_explicit_benchmark_lifetimes_override_random_sampling(self):
        starts, lifetimes = orchestrator._flow_timings(
            2,
            345.0,
            {
                'start_delays': [0.0, 45.0],
                'flow_durations': [345.0, 300.0],
                'flow_lifetime_pattern': 'random',
            },
            {},
            random.Random(1),
        )
        self.assertEqual(starts, [0.0, 45.0])
        self.assertEqual(lifetimes, [345.0, 300.0])


class JointReplayTest(unittest.TestCase):
    def test_late_joining_agent_is_masked(self):
        replay = JointReplayBuffer(max_transitions=100)
        for step in range(4):
            replay.push(_experience(0, step, throughput=10.0 + step))
        for step in range(2, 4):
            replay.push(_experience(1, step, throughput=20.0 + step))

        with mock.patch(
                'olympus.algorithms.ma_dreamer.learner.random.choice',
                side_effect=lambda values: values[0]):
            batch = replay.sample_chunks(
                batch_size=1, seq_len=4, max_agents=2)

        self.assertIsNotNone(batch)
        np.testing.assert_array_equal(
            batch['mask'][0, :, 1],
            np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(batch['avg_throughput'][0, 2, 1], 22.0)


class LearnerUpdateTest(unittest.TestCase):
    def _run_update(self, directory, rng, reward_extra=None):
        config = {
                'training': {
                    'batch_size': 1,
                    'seq_len': 4,
                    'min_replay': 1,
                    'replay_capacity': 100,
                    'updates_per_step': 1,
                    'save_every': 0,
                    'param_broadcast_every': 1,
                    'grad_clip': 100.0,
                    'horizon': 3,
                    'actor_warmup_steps': 0,
                    'reward_bins': 31,
                    'value_bins': 31,
                    'checkpoint': os.path.join(directory, 'model.pt'),
                    'log_path': '',
                },
                'agent': {
                    'hidden': 32,
                    'embed_dim': 16,
                    'h_dim': 32,
                    'latent_groups': 2,
                    'latent_classes': 4,
                    'global_hidden': 32,
                    'global_embed_dim': 16,
                    'global_h_dim': 32,
                    'global_latent_groups': 2,
                    'global_latent_classes': 4,
                },
                'reward': {
                    'fairness_weight': 25.0,
                    'fairness_cap': 1.0,
                },
                'sweep': {'flows': [2]},
        }
        if reward_extra:
            config['reward'].update(reward_extra)
        learner = Learner(
            config,
            port=0,
            authkey=b'0123456789abcdef',
            start_manager=False,
        )
        try:
            for step in range(8):
                for agent_id in range(2):
                    state = rng.random(STATE_DIM, dtype=np.float32)
                    next_state = rng.random(
                        STATE_DIM, dtype=np.float32)
                    learner.buf.push(Experience(
                        state=state,
                        action=float(rng.uniform(-1.0, 1.0)),
                        reward=float(rng.uniform(0.0, 10.0)),
                        next_state=next_state,
                        avg_throughput=float(
                            (agent_id + 1) * 1e6 + step * 1000),
                        agent_id=agent_id,
                        group_id='group',
                        group_step=step,
                        done=False,
                        traj_id=f'group_agent{agent_id}',
                        step_in_traj=step,
                    ))
            self.assertTrue(learner._update())
            for module in (
                    learner.local_world,
                    learner.global_world,
                    learner.actor,
                    learner.critic):
                self.assertTrue(all(
                    torch.isfinite(parameter).all()
                    for parameter in module.parameters()))
        finally:
            learner.stop()
        return learner

    def test_shared_imagination_update_is_finite(self):
        torch.manual_seed(2)
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            learner = self._run_update(directory, rng)
            self.assertFalse(learner.per_agent_credit)
            self.assertEqual(learner.global_world.reward_dim, 1)

    def test_per_agent_credit_opt_in_update_is_finite(self):
        torch.manual_seed(2)
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            learner = self._run_update(
                directory, rng, reward_extra={'per_agent_credit': True})
            self.assertTrue(learner.per_agent_credit)
            self.assertEqual(
                learner.global_world.reward_dim, learner.max_agents)


if __name__ == '__main__':
    unittest.main()
