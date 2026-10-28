"""Factor-feature invariance and empty-history checks for cross-date stacking."""
import unittest
import numpy as np
from tied_history_stack import invariant_features


class InvariantTests(unittest.TestCase):
    def test_rotation_and_candidate_permutation(self):
        rng = np.random.default_rng(7301)
        weight = rng.normal(size=(8, 4))
        bias = rng.normal(size=8)
        history = np.array([[1, 2, 1, -1], [-1, -1, -1, -1]])
        candidates = np.array([[1, 3, 4], [2, 5, 7]])
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        a = invariant_features(weight, bias, -2., history, candidates)
        b = invariant_features(weight@rotation, bias, -2., history, candidates)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        permutation = [2, 0, 1]
        c = invariant_features(weight, bias, -2., history, candidates[:, permutation])
        np.testing.assert_allclose(c.reshape(2, 3, -1), a.reshape(2, 3, -1)[:, permutation])
        self.assertTrue(np.isfinite(a).all())
        # Empty history has no cosine, profile norm, maximum or pair variance.
        np.testing.assert_array_equal(a.reshape(2, 3, -1)[1][:, [2, 3, 6, 7]], 0)


if __name__ == '__main__':
    unittest.main()
