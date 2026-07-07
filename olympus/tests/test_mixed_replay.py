"""Unit tests for olympus.common.mixed_replay.MixedReplay."""

import unittest

import numpy as np

from olympus.common.mixed_replay import (
    MixedReplay,
    build_mixed_replay,
    collection_groups,
    validate_batch_fractions,
)


class _FakeBuffer:
    """Minimal stand-in for a learner ReplayBuffer (flat dict sampler)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._items = []

    def push(self, exp):
        if len(self._items) < self.capacity:
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
        vals = np.asarray([self._items[int(i)] for i in idx], dtype=np.float32)
        return {'value': vals}


def _make(fractions, cap=1000):
    return MixedReplay({
        name: {'factory': (lambda c=cap: _FakeBuffer(c)), 'fraction': frac}
        for name, frac in fractions.items()})


class CollectionGroupsTest(unittest.TestCase):
    def test_absent_block(self):
        self.assertEqual(collection_groups({}), (False, {}))

    def test_groups_are_every_key_except_enabled(self):
        cfg = {'experience_collection': {
            'enabled': True,
            'mininet': {'n_parallel': 4},
            'raynet': {'n_parallel': 2},
            'ns3': None,
        }}
        enabled, groups = collection_groups(cfg)
        self.assertTrue(enabled)
        self.assertEqual(list(groups), ['mininet', 'raynet', 'ns3'])
        self.assertEqual(groups['ns3'], {})

    def test_disabled(self):
        cfg = {'experience_collection': {'enabled': False, 'mininet': {}}}
        enabled, groups = collection_groups(cfg)
        self.assertFalse(enabled)
        self.assertEqual(list(groups), ['mininet'])


class MixedReplayTest(unittest.TestCase):
    def test_push_routing(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        for v in range(10):
            m.push(v, source='mininet')
        for v in range(20):
            m.push(v, source='raynet')
        self.assertEqual(m.sizes(), {'mininet': 10, 'raynet': 20})
        self.assertEqual(m.size(), 30)

    def test_legacy_tags_alias_to_type_groups(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        m.push(1, source='emulation')
        m.push(2, source='simulation')
        self.assertEqual(m.sizes(), {'mininet': 1, 'raynet': 1})

    def test_unknown_source_routes_to_first_group(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        m.push(1, source=None)
        m.push(2, source='weird')
        self.assertEqual(m.sizes(), {'mininet': 2, 'raynet': 0})

    def test_fractions_normalized(self):
        m = _make({'a': 1.0, 'b': 3.0})
        self.assertAlmostEqual(m.fractions['a'], 0.25)
        self.assertAlmostEqual(m.fractions['b'], 0.75)

    def test_missing_fractions_default_to_equal_shares(self):
        m = _make({'a': None, 'b': None, 'c': None})
        for f in m.fractions.values():
            self.assertAlmostEqual(f, 1.0 / 3.0)

    def test_split_honours_fractions_when_all_have_data(self):
        m = _make({'mininet': 0.25, 'raynet': 0.75})
        for v in range(100):
            m.push(v, source='mininet')
            m.push(v, source='raynet')
        self.assertEqual(m.split(100), {'mininet': 25, 'raynet': 75})

    def test_split_three_groups_sums_to_total(self):
        m = _make({'a': 0.5, 'b': 0.3, 'c': 0.2})
        for v in range(50):
            m.push(v, source='a')
            m.push(v, source='b')
            m.push(v, source='c')
        counts = m.split(64)
        self.assertEqual(sum(counts.values()), 64)
        self.assertEqual(counts['a'], 32)

    def test_split_redistributes_when_a_group_is_empty(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        for v in range(50):
            m.push(v, source='raynet')
        # Only raynet has data -> whole batch from raynet.
        self.assertEqual(m.split(40), {'mininet': 0, 'raynet': 40})
        # Now add mininet data; both present -> honour fractions.
        for v in range(50):
            m.push(v, source='mininet')
        self.assertEqual(m.split(40), {'mininet': 20, 'raynet': 20})

    def test_split_empty(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        self.assertEqual(m.split(10), {'mininet': 0, 'raynet': 0})

    def test_ready_any_buffer(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        self.assertFalse(m.ready(5))
        for v in range(5):
            m.push(v, source='raynet')
        self.assertTrue(m.ready(5))   # one group reaching min is enough

    def test_merged_sample_batch_size_and_shape(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        for v in range(200):
            m.push(float(v), source='mininet')
            m.push(float(v) + 1000.0, source='raynet')
        batch = m.sample(64)
        self.assertIn('value', batch)
        self.assertEqual(batch['value'].shape[0], 64)
        # Roughly half should come from each value range.
        emu_like = np.sum(batch['value'] < 1000.0)
        self.assertGreater(emu_like, 0)
        self.assertLess(emu_like, 64)

    def test_sample_from_single_ready_buffer(self):
        m = _make({'mininet': 0.5, 'raynet': 0.5})
        for v in range(100):
            m.push(float(v), source='raynet')
        batch = m.sample(32)
        self.assertEqual(batch['value'].shape[0], 32)


class BuildMixedReplayTest(unittest.TestCase):
    def _cfg(self, enabled=True):
        return {'experience_collection': {
            'enabled': enabled,
            'mininet': {'buffer_capacity': 100, 'batch_fraction': 0.5},
            'raynet': {'buffer_capacity': 60, 'batch_fraction': 0.5,
                       'strata': {'bdp_edges': [500, 1000]}},
        }}

    def test_disabled_returns_none(self):
        self.assertIsNone(build_mixed_replay(self._cfg(enabled=False),
                                             _FakeBuffer, 999))
        self.assertIsNone(build_mixed_replay({}, _FakeBuffer, 999))

    def test_group_capacities_and_strata_split(self):
        m = build_mixed_replay(self._cfg(), _FakeBuffer, 999)
        self.assertEqual(m.bufs['mininet'].capacity, 100)
        # Stratified group: total 60 split across 3 BDP classes.
        self.assertEqual(m.bufs['raynet'].bins[0].capacity, 20)

    def test_capacity_falls_back_to_default(self):
        cfg = {'experience_collection': {
            'enabled': True, 'mininet': {'batch_fraction': 1.0}}}
        m = build_mixed_replay(cfg, _FakeBuffer, 777)
        self.assertEqual(m.bufs['mininet'].capacity, 777)


class ValidateBatchFractionsTest(unittest.TestCase):
    def test_fractions_summing_to_one_pass(self):
        validate_batch_fractions({'mininet': {'batch_fraction': 0.5},
                                  'raynet': {'batch_fraction': 0.5}})
        # Float rounding within tolerance is accepted.
        validate_batch_fractions({'a': {'batch_fraction': 0.3},
                                  'b': {'batch_fraction': 0.3},
                                  'c': {'batch_fraction': 0.4}})
        validate_batch_fractions({'mininet': {'batch_fraction': 1.0}})

    def test_bad_sum_raises(self):
        with self.assertRaises(ValueError):
            validate_batch_fractions({'mininet': {'batch_fraction': 0.5},
                                      'raynet': {'batch_fraction': 0.6}})
        with self.assertRaises(ValueError):
            validate_batch_fractions({'mininet': {'batch_fraction': 0.2},
                                      'raynet': {'batch_fraction': 0.2}})

    def test_missing_fraction_raises(self):
        with self.assertRaises(ValueError):
            validate_batch_fractions({'mininet': {'batch_fraction': 1.0},
                                      'raynet': {}})

    def test_build_mixed_replay_stops_on_bad_fractions(self):
        cfg = {'experience_collection': {
            'enabled': True,
            'mininet': {'batch_fraction': 0.5},
            'raynet': {'batch_fraction': 0.6},
        }}
        with self.assertRaises(ValueError):
            build_mixed_replay(cfg, _FakeBuffer, 100)


if __name__ == '__main__':
    unittest.main()
