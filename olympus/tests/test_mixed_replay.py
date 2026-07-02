"""Unit tests for olympus.common.mixed_replay.MixedReplay."""

import unittest

import numpy as np

from olympus.common.mixed_replay import MixedReplay


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


def _make(frac, cap=1000):
    return MixedReplay(lambda: _FakeBuffer(cap), emulation_fraction=frac)


class MixedReplayTest(unittest.TestCase):
    def test_push_routing(self):
        m = _make(0.5)
        for v in range(10):
            m.push(v, source='emulation')
        for v in range(20):
            m.push(v, source='simulation')
        self.assertEqual(m.size_emu(), 10)
        self.assertEqual(m.size_sim(), 20)
        self.assertEqual(m.size(), 30)

    def test_unknown_source_routes_to_simulation(self):
        m = _make(0.5)
        m.push(1, source=None)
        m.push(2, source='weird')
        self.assertEqual(m.size_emu(), 0)
        self.assertEqual(m.size_sim(), 2)

    def test_same_capacity_both_buffers(self):
        m = _make(0.5, cap=7)
        self.assertEqual(m.emu.capacity, 7)
        self.assertEqual(m.sim.capacity, 7)

    def test_split_honours_fraction_when_both_have_data(self):
        m = _make(0.25)
        for v in range(100):
            m.push(v, source='emulation')
            m.push(v, source='simulation')
        n_emu, n_sim = m.split(100)
        self.assertEqual(n_emu, 25)
        self.assertEqual(n_sim, 75)

    def test_split_redistributes_when_one_side_empty(self):
        m = _make(0.5)
        for v in range(50):
            m.push(v, source='simulation')
        # Only simulation has data -> whole batch from simulation.
        self.assertEqual(m.split(40), (0, 40))
        # Now add emulation data; both present -> honour fraction.
        for v in range(50):
            m.push(v, source='emulation')
        self.assertEqual(m.split(40), (20, 20))

    def test_split_empty(self):
        m = _make(0.5)
        self.assertEqual(m.split(10), (0, 0))

    def test_ready_either_buffer(self):
        m = _make(0.5)
        self.assertFalse(m.ready(5))
        for v in range(5):
            m.push(v, source='simulation')
        self.assertTrue(m.ready(5))   # sim alone reaching min is enough

    def test_merged_sample_batch_size_and_shape(self):
        m = _make(0.5)
        for v in range(200):
            m.push(float(v), source='emulation')
            m.push(float(v) + 1000.0, source='simulation')
        batch = m.sample(64)
        self.assertIn('value', batch)
        self.assertEqual(batch['value'].shape[0], 64)
        # Roughly half should come from each value range.
        emu_like = np.sum(batch['value'] < 1000.0)
        self.assertGreater(emu_like, 0)
        self.assertLess(emu_like, 64)

    def test_sample_from_single_ready_buffer(self):
        m = _make(0.5)
        for v in range(100):
            m.push(float(v), source='simulation')
        batch = m.sample(32)
        self.assertEqual(batch['value'].shape[0], 32)


if __name__ == '__main__':
    unittest.main()
