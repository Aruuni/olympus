import os
import tempfile
import unittest
from unittest import mock

from olympus.common import episode_plotting


class EpisodePlottingTest(unittest.TestCase):
    def test_should_plot_episode_honors_interval(self):
        outputs = {'plot_episodes': True, 'plot_every_n': 5}
        self.assertTrue(episode_plotting.should_plot_episode(outputs, 10))
        self.assertFalse(episode_plotting.should_plot_episode(outputs, 11))
        self.assertFalse(
            episode_plotting.should_plot_episode({'plot_episodes': False}, 10))

    def test_single_agent_returns_primary_state_log_without_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            state_log = os.path.join(directory, 'td3_state_ep000001.csv')
            with open(state_log, 'w') as f:
                f.write('t_s,reward\n0,1\n')

            with mock.patch.object(
                    episode_plotting, '_episode_return',
                    return_value=42.0) as episode_return, \
                    mock.patch.object(episode_plotting, '_plot_episode') as plot:
                result = episode_plotting.render_episode_plots(
                    outputs={
                        'plots_dir': directory,
                        'plot_episodes': False,
                    },
                    episode=1,
                    alg_name='td3',
                    state_log=state_log,
                    ecfg={'bw': 20, 'delay': 10},
                    backend_type='mininet',
                    env_name='static',
                    link_schedule=[],
                    n_flows=1,
                    trim_tail_s=5.0,
                    mode='single',
                )

            self.assertEqual(result, 42.0)
            episode_return.assert_called_once_with(state_log, trim_tail_s=5.0)
            plot.assert_not_called()

    def test_multi_agent_uses_multi_flow_return_without_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            state_log = os.path.join(directory, 'ma_state_ep000001.csv')

            with mock.patch.object(
                    episode_plotting, '_multi_episode_return',
                    return_value=99.0) as multi_return, \
                    mock.patch.object(episode_plotting, '_plot_multi_episode') as plot:
                result = episode_plotting.render_episode_plots(
                    outputs={
                        'plots_dir': directory,
                        'plot_episodes': False,
                    },
                    episode=1,
                    alg_name='ma_dreamer',
                    state_log=state_log,
                    ecfg={'bw': 20, 'delay': 10},
                    backend_type='raynet',
                    env_name='astraea_static',
                    link_schedule=[],
                    n_flows=2,
                    trim_tail_s=0.0,
                    mode='multi',
                )

            self.assertEqual(result, 99.0)
            multi_return.assert_called_once_with(
                state_log, n_agents=2, trim_tail_s=0.0)
            plot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
