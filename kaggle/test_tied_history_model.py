"""Shared-role gradient and tiny-learning sanity checks, not AUC evidence."""
import unittest
import torch
from tied_history_model import TiedHistoryModel, tied_logits


class TiedTests(unittest.TestCase):
    def test_shared_role_duplicate_gradient(self):
        torch.manual_seed(7101)
        w = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(5, dtype=torch.float64, requires_grad=True)
        g = torch.randn((), dtype=torch.float64, requires_grad=True)
        h = torch.tensor([[1, 1, 2, -1], [-1, -1, -1, -1]])
        c = torch.tensor([[1, 3, 1], [2, 4, 2]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, y, z: tied_logits(x, y, z, h, c), (w, b, g)))
        output = tied_logits(w, b, g, h, c)
        self.assertTrue(torch.allclose(output[1], b[c[1]] + g))
        self.assertTrue(torch.equal(output[:, 0], output[:, 2]))

    def test_preferences_learn_with_balanced_item_popularity(self):
        torch.manual_seed(7102)
        torch.set_num_threads(2)
        model = TiedHistoryModel(6, 8)
        # Balanced labels: item bias cannot explain these opposite preferences.
        history = torch.tensor([[0, 1], [3, 4]] * 16)
        candidates = torch.tensor([[2, 5], [2, 5]] * 16)
        labels = torch.tensor([[1., 0.], [0., 1.]] * 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
        initial = torch.nn.functional.binary_cross_entropy_with_logits(
            model(history, candidates), labels).item()
        for _ in range(150):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(history, candidates), labels)
            loss.backward()
            optimizer.step()
        final = model(history, candidates).detach()
        self.assertLess(loss.item(), initial * 0.05)
        self.assertTrue(torch.equal(final > 0, labels.bool()))
        self.assertTrue(torch.isfinite(model.auxiliary_loss(history, candidates)))


if __name__ == '__main__':
    unittest.main()
