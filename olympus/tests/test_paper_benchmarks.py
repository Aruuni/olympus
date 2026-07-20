import threading
import tempfile
import time
import unittest
from pathlib import Path

from benchmarks_paper import benchmark
from benchmarks_paper import plot
from olympus.environments.base import NetworkEnv


class PaperBenchmarkConfigTests(unittest.TestCase):
    def setUp(self):
        self.bench = benchmark._bench_cfg({})

    def test_default_matrix_matches_paper(self):
        cases = benchmark._cases(self.bench, ['inter_rtt', 'intra_rtt'])
        self.assertEqual(len(cases), 200)
        self.assertEqual(self.bench['duration_s'], 200)
        self.assertEqual(self.bench['second_flow_start_s'], 50)
        self.assertEqual(self.bench['score_window_s'], 100)

    def test_inter_rtt_queue_uses_joining_flow_bdp(self):
        qsize = benchmark._queue_bytes(100, 200, 1)
        self.assertEqual(qsize, int(100 * (2 ** 20) * 0.2 / 8))
        self.assertEqual(
            benchmark._rtt_pair('inter_rtt', 200, self.bench), (20.0, 200.0))

    def test_intra_reconfiguration_schedule_is_seeded(self):
        first = benchmark._link_schedule('intra_rtt', self.bench, 123)
        second = benchmark._link_schedule('intra_rtt', self.bench, 123)
        self.assertEqual(first, second)
        self.assertEqual([row['t'] for row in first], list(range(15, 200, 15)))
        self.assertTrue(all(45 <= row['outage_ms'] <= 120 for row in first))
        self.assertEqual(benchmark._link_schedule('inter_rtt', self.bench, 123), [])


class _ScheduleProbe(NetworkEnv):
    def __init__(self):
        self.events = []

    def start(self):
        pass

    def setup_environment(self, link_schedule=None):
        super().setup_environment(link_schedule)

    def start_episode(self, monitor_interval=0.1, start_delays=None,
                      flow_durations=None, episode_start=None):
        pass

    def wait(self):
        pass

    def stop(self):
        pass

    def change_link(self, bw=None, delay=None, loss=None):
        self.events.append(('change', bw, delay, loss))

    def interrupt_link(self, outage_ms=0.0):
        self.events.append(('outage', outage_ms))


class LinkOutageScheduleTests(unittest.TestCase):
    def test_outage_entries_use_interrupt_primitive(self):
        env = _ScheduleProbe()
        env._replay_link_schedule(
            [{'t': 0, 'outage_ms': 75}, {'t': 0, 'bw': 50}],
            time.monotonic(), threading.Event())
        self.assertEqual(env.events[0], ('outage', 75.0))
        self.assertEqual(env.events[1], ('change', 50, None, None))


class PaperPlotTests(unittest.TestCase):
    def test_paper_style_plot_writes_pdf_and_png(self):
        plot._style()
        fig, ax = plot.plt.subplots(1, 1, figsize=(2, 1))
        rows = [{
            'suite': 'intra_rtt', 'approach': 'agent',
            'bdp_multiplier': '1', 'rtt_ms': str(rtt),
            'goodput_ratio_total_mean': '0.9',
            'goodput_ratio_total_std': '0.02',
        } for rtt in (20, 40)]
        plot._draw(ax, rows, 'intra_rtt', 1.0, ['agent'], {'agent': 'Agent'})
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'figure'
            plot._save(fig, stem)
            self.assertTrue(stem.with_suffix('.pdf').is_file())
            self.assertTrue(stem.with_suffix('.png').is_file())


if __name__ == '__main__':
    unittest.main()
