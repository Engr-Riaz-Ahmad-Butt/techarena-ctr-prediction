"""Synthetic checks for chronological event-to-feature alignment."""
import unittest

import numpy as np

import history_attention_runner as runner


class EventAlignmentTests(unittest.TestCase):
    def setUp(self):
        start = 1_000_000
        cutoff = start + 6 * runner.DAY
        self.manifest = {"start": start}
        self.timestamps = np.array(
            [start + day * runner.DAY for day in range(7)], dtype=np.int64
        )
        self.targets = np.array([0, 10, 1, 10, 1, 10, 3], dtype=np.int8)
        self.labels = np.zeros((3, 10), dtype=np.float32)
        self.labels[1, 1] = 1
        self.cutoff = cutoff

    def test_selects_contiguous_pre_cutoff_events_in_feature_order(self):
        events = {
            "timestamp": self.timestamps,
            "target": self.targets,
        }
        selected, targets = runner._aligned_events(
            events, self.manifest, 6, self.labels.ravel()
        )
        np.testing.assert_array_equal(selected, [3, 4, 5])
        np.testing.assert_array_equal(targets, [10, 1, 10])
        self.assertTrue((self.timestamps[selected] < self.cutoff).all())

    def test_rejects_misaligned_feature_labels(self):
        events = {
            "timestamp": self.timestamps,
            "target": self.targets,
        }
        labels = self.labels.copy()
        labels[0, 0] = 1
        with self.assertRaisesRegex(AssertionError, "targets differ"):
            runner._aligned_events(
                events, self.manifest, 6, labels.ravel()
            )

    def test_rejects_nonchronological_history_order(self):
        events = {
            "timestamp": self.timestamps[[0, 1, 2, 4, 3, 5, 6]],
            "target": self.targets[[0, 1, 2, 4, 3, 5, 6]],
        }
        with self.assertRaisesRegex(AssertionError, "chronologically"):
            runner._aligned_events(
                events, self.manifest, 6, self.labels.ravel()
            )


if __name__ == "__main__":
    unittest.main()
