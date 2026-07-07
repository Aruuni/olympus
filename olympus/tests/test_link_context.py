import os
import tempfile
import unittest
from unittest import mock

from olympus.common import link_context


class LinkContextTest(unittest.TestCase):
    def test_write_and_read_link_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'link_context.json')
            link_context.write_link_context(
                path,
                bw_mbps=50,
                base_rtt_us=30000,
                link_schedule=[
                    {'t': 1.5, 'bw': 25},
                    {'t': 2.0, 'delay': 60},
                ],
                episode=7,
                slot=3,
                flow_id=1,
            )

            bw_mbps, rtt_us, events = link_context.context_values(path)

        self.assertEqual(bw_mbps, 50.0)
        self.assertEqual(rtt_us, 30000.0)
        self.assertEqual(events[0]['bw'], 25)
        self.assertEqual(events[1]['delay'], 60)

    def test_env_prefix_fallback(self):
        with mock.patch.dict(
                os.environ,
                {
                    'OC_LINK_BW': '80',
                    'OC_BASE_RTT_US': '40000',
                    'OC_LINK_SCHEDULE': '[{"t": 1, "delay": 50}]',
                },
                clear=False):
            bw_mbps, rtt_us, events = link_context.context_values()

        self.assertEqual(bw_mbps, 80.0)
        self.assertEqual(rtt_us, 40000.0)
        self.assertEqual(events, [{'t': 1, 'delay': 50}])

    def test_step_trace_skips_intermediate_regimes_by_time(self):
        trace = link_context.build_step_trace(
            20_000.0,
            [
                {'t': 0.005, 'delay': 30},
                {'t': 0.010, 'delay': 40},
                {'t': 0.015, 'delay': 50},
                {'t': 0.020, 'delay': 60},
            ],
            100.0,
            'delay',
            transform=lambda value: float(value) * 1_000.0,
        )

        self.assertEqual(trace.at(100.000), 20_000.0)
        self.assertEqual(trace.at(100.012), 40_000.0)
        self.assertEqual(trace.at(100.025), 60_000.0)


if __name__ == '__main__':
    unittest.main()
