"""Independent temporal, geometry and cross-day alignment checks before GPU fitting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tied_history_oof_tree import score_contexts, geometry, extra_features, Augmented, NAMES
from tied_history_gpu_bag import BagFactor


class OOFTests(unittest.TestCase):
    def test_scored_labels_never_enter_context(self):
        data = dict(timestamp=np.array([10, 20, 40, 50]), users=np.zeros(4, dtype=int),
                    hist=np.array([[1, -1], [1, 2], [3, 4], [4, 7]]),
                    items=np.tile(np.arange(10), (4, 1)), target=np.array([2, 3, 8, 9]))
        stamps = np.array([[1, 0], [1, 10], [20, 35], [35, 45]])
        a = score_contexts(data, stamps, 30, [2, 3])
        np.testing.assert_array_equal(a[0, :4], [4, 3, 2, 1])
        np.testing.assert_array_equal(a[1, :5], [7, 4, 3, 2, 1])
        data['target'][2:] = [0, 1]
        np.testing.assert_array_equal(a, score_contexts(data, stamps, 30, [2, 3]))
        stamps[2, 0] = 40
        with self.assertRaises(AssertionError):
            score_contexts(data, stamps, 30, [2])

    def test_geometry_against_numpy_with_empty_and_duplicate_history(self):
        model = BagFactor(12, 3).double()
        vectors = np.random.default_rng(1).normal(size=(12, 3))
        with torch.no_grad():
            model.items.weight.copy_(torch.tensor(vectors))
            model.bias.copy_(torch.arange(12))
        history = np.array([[1, 1, 2, -1], [-1, -1, -1, -1]])
        items = np.array([[3, 4], [5, 6]])
        actual = geometry(model, torch.tensor(history), torch.tensor(items)).detach().numpy()
        expected = []
        for h, candidates in zip(history, items):
            hs = vectors[h[h >= 0]]
            profile = hs.mean(0) if len(hs) else np.zeros(3)
            for c in candidates:
                dot = profile@vectors[c]
                norm = np.linalg.norm(profile)
                cn = np.linalg.norm(vectors[c])
                pairs = hs@vectors[c]
                expected.append([dot, dot/max(norm*cn, 1e-9), norm, cn, c,
                                 pairs.max() if len(pairs) else 0, pairs.std() if len(pairs) else 0])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_features_and_blocks_cross_day_boundary(self):
        p = np.linspace(.01, .9, 20)
        extra = extra_features([p, p], np.zeros((20, 7)), np.array([[1, -1], [-1, -1]]))
        self.assertEqual(extra.shape, (20, len(NAMES)))
        np.testing.assert_array_equal(extra[:, 2], 0)
        np.testing.assert_allclose(extra[:, 15].reshape(2, 10).sum(1), 1)
        class Base:
            def __len__(self): return 20
            def block(self, a, b): return np.arange(a, b)[:, None]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as tmp:
            paths = [Path(tmp)/'a.npy', Path(tmp)/'b.npy']
            np.save(paths[0], extra[:10]); np.save(paths[1], extra[10:])
            blocks = Augmented(Base(), paths)
            combined = blocks.block(7, 14)
            np.testing.assert_array_equal(combined[:, 0], np.arange(7, 14))
            np.testing.assert_array_equal(combined[:, 1:], extra[7:14])
            for part in blocks.parts:
                part._mmap.close()


if __name__ == '__main__':
    unittest.main()
