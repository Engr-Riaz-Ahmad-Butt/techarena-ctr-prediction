"""CPU preflight of the GPU implementation's math and temporal reconstruction."""
import unittest
import numpy as np
import torch
from tied_history_model import TiedHistoryModel
from tied_history_full import regularization
from tied_history_gpu_bag import BagFactor, prepare, restore_timestamps


class BagTests(unittest.TestCase):
    def test_fused_loss_gradients_match_prior_duplicate_safe_equations(self):
        torch.manual_seed(7401)
        old, new = TiedHistoryModel(12, 5).double(), BagFactor(12, 5).double()
        new.load_state_dict(old.state_dict())
        h = torch.tensor([[1, 1, 2, -1], [-1, -1, -1, -1]])
        candidates = torch.tensor([[1, 3, 1], [0, 5, 0]])
        labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
        logits, penalty = new.forward_and_penalty(h, candidates)
        self.assertTrue(torch.equal(logits, old(h, candidates)))
        a = torch.nn.functional.binary_cross_entropy_with_logits(old(h, candidates), labels)
        a = a + .002*regularization(old, h, candidates)
        b = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)+.002*penalty
        self.assertAlmostEqual(a.item(), b.item(), places=12)
        a.backward()
        b.backward()
        for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-12)

    def test_prefix_contexts_and_future_label_invariance(self):
        ts = np.array([10, 20, 20, 40, 50, 60])
        history = np.array([[1, 2], [2, 3], [5, -1], [3, 4], [5, -1], [4, -1]])
        stamps = np.array([[1, 2], [2, 10], [5, 0], [10, 20], [5, 0], [20, 0]])
        data = dict(timestamp=ts, users=np.array([0, 0, 1, 0, 1, 0]),
                    items=np.tile(np.arange(10), (6, 1)), hist=history,
                    target=np.array([3, 4, 10, 5, 10, 7], dtype=np.int8))
        data['gaps'] = np.where(history >= 0, (ts[:, None]-stamps)/86400, 0).astype('float32')
        restored = restore_timestamps(data)
        np.testing.assert_array_equal(restored[history >= 0], stamps[history >= 0])
        validation = dict(indices=np.array([3, 4]),
                          y=(data['target'][[3, 4], None] == np.arange(10)).ravel())
        first = prepare(data, restored, validation, 30, progress=False)
        np.testing.assert_array_equal(first['fit_h'][0, :3], [2, 1, -1])
        np.testing.assert_array_equal(first['fit_h'][1, :4], [3, 2, 1, -1])
        np.testing.assert_array_equal(first['val_h'][0, :5], [4, 3, 2, 1, -1])
        np.testing.assert_array_equal(first['aux_target'], [1, 2, 3, 4])
        # Held-out/later labels cannot alter the fitting prefix or negatives.
        data['target'][-1] = 2
        second = prepare(data, restored, validation, 30, progress=False)
        for name in ['fit_h', 'fit_i', 'fit_y', 'aux_h', 'aux_target', 'exposure_probability']:
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == '__main__':
    unittest.main()
