"""Unit tests for BDP-stratified simulation replay.

Covers StratifiedReplay routing/sampling and its integration into MixedReplay
(the raynet group stratified by BDP, the mininet group left as a single buffer).
"""

import unittest

import numpy as np

from olympus.common.mixed_replay import (
    MixedReplay,
    group_strata_edges,
    split_source,
)
from olympus.common.stratified_replay import StratifiedReplay


class _FakeBuffer:
    """Minimal learner-style ReplayBuffer with a flat dict sampler."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._items = []

    def push(self, exp):
        if len(self._items) < self.capacity:
            self._items.append(exp)
        else:  # FIFO eviction, like the real buffer
            self._items.pop(0)
            self._items.append(exp)

    def size(self):
        return len(self._items)

    def n_trajs(self):
        return len(self._items)

    def sample(self, n):
        if not self._items:
            return None
        n = min(int(n), len(self._items))
        idx = np.random.randint(0, len(self._items), size=n)
        return {'value': np.asarray([self._items[int(i)] for i in idx],
                                    dtype=np.float32)}


class SplitSourceTest(unittest.TestCase):
    def test_plain_type_tags(self):
        self.assertEqual(split_source('mininet'), ('mininet', None))
        self.assertEqual(split_source('raynet'), ('raynet', None))

    def test_tag_with_bdp(self):
        self.assertEqual(split_source('raynet|1250.0'), ('raynet', 1250.0))

    def test_missing_tag(self):
        self.assertEqual(split_source(None), (None, None))
        self.assertEqual(split_source(''), (None, None))

    def test_malformed_bdp_is_ignored(self):
        self.assertEqual(split_source('raynet|nan?'), ('raynet', None))


class StrataConfigTest(unittest.TestCase):
    def test_absent_returns_none(self):
        self.assertIsNone(group_strata_edges({}))
        self.assertIsNone(group_strata_edges(None))
        self.assertIsNone(group_strata_edges({'strata': {}}))

    def test_parsed_and_sorted(self):
        block = {'strata': {'bdp_edges': [950, 350, 1500, 550]}}
        self.assertEqual(group_strata_edges(block),
                         [350.0, 550.0, 950.0, 1500.0])


class StratifiedReplayTest(unittest.TestCase):
    def _make(self, cap=100):
        return StratifiedReplay(lambda: _FakeBuffer(cap), edges=[350, 550, 950, 1500])

    def test_bin_routing(self):
        s = self._make()
        self.assertEqual(s.bin_index(100), 0)
        self.assertEqual(s.bin_index(349), 0)
        self.assertEqual(s.bin_index(350), 1)     # bisect_right: on-edge -> upper
        self.assertEqual(s.bin_index(400), 1)
        self.assertEqual(s.bin_index(2500), 4)
        self.assertEqual(s.bin_index(None), 0)

    def test_push_lands_in_correct_class(self):
        s = self._make()
        s.push(1.0, 100)      # bin 0
        s.push(2.0, 600)      # bin 2
        s.push(3.0, 2000)     # bin 4
        self.assertEqual(s.bin_sizes(), [1, 0, 1, 0, 1])
        self.assertEqual(s.size(), 3)
        self.assertEqual(s.n_trajs(), 3)

    def test_low_bdp_flood_cannot_evict_high_bdp(self):
        # Small per-class capacity; hammer the low-BDP class. High-BDP class must
        # keep its lone transition because classes have independent storage.
        s = StratifiedReplay(lambda: _FakeBuffer(5), edges=[1000])
        s.push(999.0, 2000)                  # one high-BDP transition
        for i in range(1000):                # flood low-BDP
            s.push(float(i), 100)
        self.assertEqual(s.bins[1].size(), 1)
        self.assertEqual(s.bins[1]._items, [999.0])

    def test_sample_is_balanced_across_nonempty_classes(self):
        s = self._make()
        for _ in range(50):
            s.push(10.0, 100)      # bin 0 marker: value 10
            s.push(90.0, 2000)     # bin 4 marker: value 90
        batch = s.sample(64)
        self.assertEqual(batch['value'].shape[0], 64)
        low = int(np.sum(batch['value'] < 50.0))
        high = 64 - low
        # Even split target is 32/32; allow slack for randint sampling.
        self.assertGreater(low, 10)
        self.assertGreater(high, 10)

    def test_split_allocations_sum_to_total(self):
        s = self._make()
        for _ in range(10):
            s.push(1.0, 100)       # bin 0
            s.push(1.0, 600)       # bin 2
        alloc = s.split(65)
        self.assertEqual(sum(alloc), 65)
        # Only the two non-empty classes receive samples.
        self.assertEqual(alloc[1], 0)
        self.assertEqual(alloc[3], 0)
        self.assertEqual(alloc[4], 0)

    def test_sample_from_single_ready_class(self):
        s = self._make()
        for v in range(20):
            s.push(float(v), 2000)   # only bin 4 populated
        batch = s.sample(16)
        self.assertEqual(batch['value'].shape[0], 16)

    def test_empty_sample_is_none(self):
        s = self._make()
        self.assertEqual(s.split(10), [0, 0, 0, 0, 0])
        self.assertIsNone(s.sample(10))


class MixedReplayStratifiedTest(unittest.TestCase):
    def _make(self, frac=0.5, emu_cap=100, sim_cap=20):
        return MixedReplay({
            'mininet': {'factory': (lambda: _FakeBuffer(emu_cap)),
                        'fraction': frac},
            'raynet': {'factory': (lambda: _FakeBuffer(sim_cap)),
                       'fraction': 1.0 - frac,
                       'strata': [350, 550, 950, 1500]},
        })

    def test_only_strata_group_is_stratified(self):
        m = self._make()
        self.assertIsInstance(m.bufs['raynet'], StratifiedReplay)
        self.assertNotIsInstance(m.bufs['mininet'], StratifiedReplay)
        self.assertEqual(m.bufs['mininet'].capacity, 100)
        self.assertEqual(m.bufs['raynet'].bins[0].capacity, 20)

    def test_push_routes_by_source_and_bdp(self):
        m = self._make()
        m.push(1.0, source='mininet')
        m.push(2.0, source='raynet|100')    # raynet bin 0
        m.push(3.0, source='raynet|2000')   # raynet bin 4
        self.assertEqual(m.sizes(), {'mininet': 1, 'raynet': 2})
        self.assertEqual(m.bufs['raynet'].bin_sizes(), [1, 0, 0, 0, 1])

    def test_legacy_tags_route_into_type_groups(self):
        m = self._make()
        m.push(1.0, source='emulation')
        m.push(2.0, source='simulation|2000')   # raynet bin 4 via alias
        self.assertEqual(m.sizes(), {'mininet': 1, 'raynet': 1})
        self.assertEqual(m.bufs['raynet'].bin_sizes(), [0, 0, 0, 0, 1])

    def test_plain_tag_routes_to_class_zero(self):
        m = self._make()
        m.push(5.0, source='raynet')   # no BDP -> class 0 fallback
        self.assertEqual(m.bufs['raynet'].bin_sizes()[0], 1)

    def test_ready_and_merged_sample(self):
        m = self._make(frac=0.5)
        for _ in range(30):
            m.push(0.0, source='mininet')
            m.push(100.0, source='raynet|100')
            m.push(900.0, source='raynet|2000')
        self.assertTrue(m.ready(10))
        batch = m.sample(40)
        self.assertEqual(batch['value'].shape[0], 40)

    def test_default_construction_is_unstratified(self):
        m = MixedReplay({'mininet': {'factory': lambda: _FakeBuffer(10),
                                     'fraction': 0.5}})
        self.assertEqual(m._stratified, set())
        self.assertNotIsInstance(m.bufs['mininet'], StratifiedReplay)


if __name__ == '__main__':
    unittest.main()
