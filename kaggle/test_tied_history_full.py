"""Checks for auxiliary-loss integration and sampled regularization."""
import unittest
import numpy as np
import torch
from tied_history_model import TiedHistoryModel
from tied_history_full import pad, regularization


class FullTests(unittest.TestCase):
    def test_padding_and_unused_rows_do_not_change_penalty(self):
        torch.manual_seed(7201)
        model = TiedHistoryModel(9, 4)
        h = torch.tensor([[1, 2]])
        c = torch.tensor([[3, 3, 4]])
        expected = regularization(model, h, c)
        padded = torch.tensor([[1, 2, -1, -1]])
        self.assertTrue(torch.equal(expected, regularization(model, padded, c)))
        with torch.no_grad():
            model.items.weight[0].fill_(1e6)
            model.items.weight[8].fill_(-1e6)
        actual = regularization(model, padded, c)
        self.assertTrue(torch.equal(expected, actual))
        actual.backward()
        self.assertEqual(model.items.weight.grad[8].abs().sum().item(), 0)
        self.assertEqual(model.items.weight.grad[0].abs().sum().item(), 0)

    def test_auxiliary_collision_has_finite_gradient_and_exact_target_exclusion(self):
        torch.manual_seed(7202)
        model = TiedHistoryModel(6, 4)
        history = torch.tensor([[1, 2, -1]])
        # A negative may legitimately collide with the positive under sampling.
        candidates = torch.tensor([[3, 3, 4]])
        loss = model.auxiliary_loss(history, candidates)
        loss.backward()
        self.assertTrue(torch.isfinite(model.items.weight.grad).all())
        self.assertTrue(np.array_equal(pad([(5, 1), (10, 2)], 3), [2, 1, -1]))


if __name__ == '__main__':
    unittest.main()
