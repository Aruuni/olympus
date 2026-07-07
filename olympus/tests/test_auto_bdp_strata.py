"""Tests for orchestrator auto-derivation of BDP strata edges from a sweep."""

import unittest

from olympus.orchestrator import _auto_bdp_edges, _resolve_collection_strata


class AutoBdpEdgesTest(unittest.TestCase):
    def test_edge_count_is_n_buckets_minus_one(self):
        bws = [10, 20, 30, 40, 50]
        delays = [10, 20, 30, 40, 50]
        self.assertEqual(len(_auto_bdp_edges(bws, delays, 5)), 4)
        self.assertEqual(len(_auto_bdp_edges(bws, delays, 3)), 2)

    def test_edges_strictly_increasing(self):
        edges = _auto_bdp_edges([10, 20, 30, 40, 50], [10, 20, 30, 40, 50], 5)
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(len(edges), len(set(edges)))

    def test_edges_span_product_range(self):
        # BDP products run 100..2500; interior edges must sit inside that range.
        edges = _auto_bdp_edges([10, 20, 30, 40, 50], [10, 20, 30, 40, 50], 5)
        self.assertGreater(edges[0], 100)
        self.assertLess(edges[-1], 2500)

    def test_single_bucket_has_no_edges(self):
        self.assertEqual(_auto_bdp_edges([10, 20], [10, 20], 1), [])

    def test_degenerate_single_product(self):
        # One distinct product -> no meaningful split possible.
        self.assertEqual(_auto_bdp_edges([10], [20], 5), [])

    def test_buckets_capped_at_distinct_products(self):
        # Only 3 distinct products but 10 buckets requested -> at most 2 edges.
        edges = _auto_bdp_edges([10, 20], [10, 30], 10)  # products: 100,300,200,600
        self.assertLessEqual(len(edges), 3)


class ResolveCollectionStrataTest(unittest.TestCase):
    def _cfg(self, strata):
        return {
            'experience_collection': {
                'enabled': True,
                'raynet': {
                    'environment_setup': 'orca_static',
                    'strata': strata,
                },
            },
        }

    def test_n_buckets_fills_bdp_edges_from_sweep(self):
        cfg = self._cfg({'n_buckets': 5})
        _resolve_collection_strata(cfg)
        edges = cfg['experience_collection']['raynet']['strata']['bdp_edges']
        self.assertEqual(len(edges), 4)

    def test_explicit_edges_are_not_overwritten(self):
        cfg = self._cfg({'n_buckets': 5, 'bdp_edges': [500, 1000]})
        _resolve_collection_strata(cfg)
        self.assertEqual(
            cfg['experience_collection']['raynet']['strata']['bdp_edges'],
            [500, 1000])

    def test_no_strata_block_is_noop(self):
        cfg = {'experience_collection': {'enabled': True, 'raynet': {}}}
        _resolve_collection_strata(cfg)  # must not raise
        self.assertNotIn('strata', cfg['experience_collection']['raynet'])

    def test_disabled_collection_is_noop(self):
        cfg = self._cfg({'n_buckets': 5})
        cfg['experience_collection']['enabled'] = False
        _resolve_collection_strata(cfg)
        self.assertNotIn(
            'bdp_edges', cfg['experience_collection']['raynet']['strata'])


if __name__ == '__main__':
    unittest.main()
