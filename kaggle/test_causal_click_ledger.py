"""Leakage boundary checks for reconstructed full click histories."""
import unittest
from causal_click_ledger import build_ledger, context_before, other_click_context


class LedgerTests(unittest.TestCase):
    def test_overlapping_histories_repeat_items_and_equal_time_boundary(self):
        ledger = build_ledger([
            ('u', 20, [('a', 5), ('b', 10)], ['a']),
            ('u', 30, [('b', 10), ('a', 20)], ['c']),
            ('v', 30, [('x', 2)], []),
        ], 40)
        self.assertEqual(ledger['u'], ((5, 'a'), (10, 'b'), (20, 'a'), (30, 'c')))
        self.assertEqual(context_before(ledger, 'u', 20), ((10, 'b'), (5, 'a')))
        self.assertEqual(context_before(ledger, 'u', 31, 2), ((30, 'c'), (20, 'a')))
        self.assertEqual(context_before(ledger, 'unknown', 31), ())
        self.assertEqual(other_click_context(ledger, 'u', (20, 'a'), 30),
                         ((5, 'a'), (10, 'b')))
        # A later fitting click is auxiliary context, never a past impression feature.
        self.assertIn((30, 'c'), other_click_context(ledger, 'u', (20, 'a'), 40))
        self.assertNotIn((30, 'c'), context_before(ledger, 'u', 25))

    def test_invalid_sources_fail_closed(self):
        for event in [('u', 40, [], []), ('u', 20, [('a', 20)], []),
                      ('u', 20, [], ['a', 'b'])]:
            with self.assertRaises(ValueError):
                build_ledger([event], 40)


if __name__ == '__main__':
    unittest.main()
