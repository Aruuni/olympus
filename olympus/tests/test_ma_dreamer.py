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
from olympus.plots.multi_flow_episode_plot import (
    _r_fair_matrix,
    plot as plot_multi_flow_episode,
)
from olympus.common import link_context
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
                side_effect=lambda values: values[0]), \
             mock.patch(
                'olympus.algorithms.ma_dreamer.learner.random.randrange',
                return_value=0):
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


class PerFlowDelayChangeTest(unittest.TestCase):
    """change_link(delay=X) in per-flow-RTT mode scales each flow's own delay."""

    def _env(self):
        from olympus.environments.mininet.env import MininetEnv
        return MininetEnv(
            n=3, bw=50, delay=50.0, duration=10,
            per_flow_delays=[10.0, 0.0, 200.0])

    def test_scales_each_flow_from_its_own_base(self):
        env = self._env()
        changes = []
        with mock.patch(
                'olympus.environments.mininet.env._change_delay',
                side_effect=lambda node, intf, d, loss=None:
                changes.append((intf, d))), \
             mock.patch.object(
                env, '_ci_s1_intf',
                side_effect=lambda i: (object(), f'c{i}-eth0')), \
             mock.patch.object(
                env, '_s2s3_intfs',
                return_value=(object(), 's2-eth', object(), 's3-eth')), \
             mock.patch('olympus.environments.mininet.env._change_bw'):
            env.change_link(delay=100.0)  # 2x the 50ms episode base
            env.change_link(delay=25.0)   # then 0.5x of base, not compounded
        self.assertEqual(changes, [
            ('c1-eth0', 20.0), ('c3-eth0', 400.0),   # zero-delay flow skipped
            ('c1-eth0', 5.0), ('c3-eth0', 100.0),
        ])

    def test_legacy_shared_link_path_unchanged(self):
        from olympus.environments.mininet.env import MininetEnv
        env = MininetEnv(n=2, bw=50, delay=50.0, duration=10)
        with mock.patch(
                'olympus.environments.mininet.env._change_delay') as chg, \
             mock.patch.object(
                env, '_s1s2_intfs',
                return_value=(object(), 's1-eth', object(), 's2-eth')), \
             mock.patch.object(
                env, '_s2s3_intfs',
                return_value=(object(), 's2-eth', object(), 's3-eth')), \
             mock.patch('olympus.environments.mininet.env._change_bw'):
            env.change_link(delay=100.0)
        chg.assert_called_once_with(mock.ANY, 's1-eth', 100.0, None)


class PerFlowRttDelaysTest(unittest.TestCase):
    def test_min_mult_one_keeps_every_flow_at_or_above_base(self):
        ecfg = {'per_flow_rtt_mult': True,
                'per_flow_rtt_mult_min': 1.0,
                'per_flow_rtt_mult_max': 4.0}
        delays = orchestrator._per_flow_rtt_delays(
            50, 50.0, ecfg, {}, random.Random(7))
        self.assertEqual(len(delays), 50)
        for d in delays:
            self.assertGreaterEqual(d, 50.0)
            self.assertLessEqual(d, 200.0)
        self.assertGreater(max(delays) - min(delays), 1.0)

    def test_fixed_mult(self):
        ecfg = {'per_flow_rtt_mult': True,
                'per_flow_rtt_mult_min': 2.0,
                'per_flow_rtt_mult_max': 2.0}
        delays = orchestrator._per_flow_rtt_delays(
            2, 50.0, ecfg, {}, random.Random(7))
        self.assertEqual(delays, [100.0, 100.0])

    def test_anchor_first_pins_flow0_to_base(self):
        ecfg = {'per_flow_rtt_mult': True,
                'per_flow_rtt_mult_min': 1.0,
                'per_flow_rtt_mult_max': 4.0,
                'per_flow_rtt_anchor_first': True}
        for seed in range(20):
            delays = orchestrator._per_flow_rtt_delays(
                4, 50.0, ecfg, {}, random.Random(seed))
            self.assertEqual(delays[0], 50.0)
            for d in delays[1:]:
                self.assertGreaterEqual(d, 50.0)
                self.assertLessEqual(d, 200.0)

    def test_anchor_first_single_flow(self):
        ecfg = {'per_flow_rtt_mult': True,
                'per_flow_rtt_anchor_first': True}
        self.assertEqual(orchestrator._per_flow_rtt_delays(
            1, 50.0, ecfg, {}, random.Random(7)), [50.0])

    def test_disabled_returns_none(self):
        self.assertIsNone(orchestrator._per_flow_rtt_delays(
            4, 50.0, {}, {}, random.Random(7)))


class PerFlowRewardReferenceTest(unittest.TestCase):
    """End to end: sampled per-flow RTT -> flow link context -> reward rtt_ref.

    Mirrors the orchestrator MA path: each flow's context is written with its
    own base RTT and a _flow_link_schedule-rescaled event list, and the reward
    plugin built from that context must reference the flow's own RTT before
    AND after a scheduled delay change."""

    def test_rtt_ref_tracks_each_flows_own_scheduled_rtt(self):
        from olympus.rewards.tempest import make_reward_calc
        base_delay = 50.0
        shared_schedule = [{'t': 15, 'delay': 100.0}]  # delay_frac 2.0
        episode_start = 1000.0
        cases = [
            # (flow's own delay ms, rtt_ref before t=15, rtt_ref after)
            (50.0, 50_000.0, 100_000.0),
            (200.0, 200_000.0, 400_000.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for flow_delay, ref_before_us, ref_after_us in cases:
                path = os.path.join(directory, f'ctx_{int(flow_delay)}.json')
                link_context.write_link_context(
                    path,
                    bw_mbps=50.0,
                    base_rtt_us=flow_delay * 1000.0,
                    link_schedule=orchestrator._flow_link_schedule(
                        shared_schedule, base_delay, flow_delay),
                    episode=0, slot=0, flow_id=0,
                )
                with mock.patch.dict(os.environ, {
                        'OC_LINK_CONTEXT_PATH': path,
                        'OC_EPISODE_START': str(episode_start)}):
                    calc = make_reward_calc()
                self.assertEqual(
                    calc._current_link_rtt_us({'time_s': 10.0}), ref_before_us)
                self.assertEqual(
                    calc._current_link_rtt_us({'time_s': 20.0}), ref_after_us)


class FlowLinkScheduleTest(unittest.TestCase):
    def test_delay_events_rescale_to_flow_base(self):
        shared = [
            {'t': 15, 'delay': 25.0},
            {'t': 30, 'delay': 100.0, 'bw': 40.0},
            {'t': 45, 'bw': 80.0},
        ]
        scaled = orchestrator._flow_link_schedule(shared, 50.0, 200.0)
        self.assertEqual(scaled[0], {'t': 15, 'delay': 100.0})
        self.assertEqual(scaled[1], {'t': 30, 'delay': 400.0, 'bw': 40.0})
        self.assertEqual(scaled[2], {'t': 45, 'bw': 80.0})
        # Input entries must not be mutated (schedule is shared across flows).
        self.assertEqual(shared[0]['delay'], 25.0)

    def test_invalid_base_returns_schedule_unchanged(self):
        shared = [{'t': 15, 'delay': 25.0}]
        for base, flow in ((0.0, 200.0), (50.0, 0.0), (None, 200.0)):
            self.assertEqual(
                orchestrator._flow_link_schedule(shared, base, flow), shared)

    def test_empty_schedule(self):
        self.assertEqual(orchestrator._flow_link_schedule([], 50.0, 200.0), [])
        self.assertEqual(
            orchestrator._flow_link_schedule(None, 50.0, 200.0), [])


if __name__ == '__main__':
    unittest.main()
