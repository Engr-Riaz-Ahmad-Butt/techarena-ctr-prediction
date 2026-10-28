"""Exploratory mean-history factor model with one trainable vector per item.

Independent implementation of the shared-item/full-history hypothesis described
at github.com/jancastachova/techarena2026-atu. No competitor predictions are used.
Padding is -1. Contexts must be prepared at the event's temporal boundary.
"""
import torch
from torch import nn


def tied_logits(weight, bias, global_bias, history, candidates):
    valid = history >= 0
    vectors = weight[history.clamp_min(0)] * valid.unsqueeze(-1)
    profile = vectors.sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
    return ((profile.unsqueeze(1) * weight[candidates]).sum(-1)
            + bias[candidates] + global_bias)


class TiedHistoryModel(nn.Module):
    def __init__(self, n_items, dimensions=32):
        super().__init__()
        self.items = nn.Embedding(n_items, dimensions)
        self.bias = nn.Parameter(torch.zeros(n_items))
        self.global_bias = nn.Parameter(torch.tensor(-2.2))
        nn.init.normal_(self.items.weight, std=0.005)

    def forward(self, history, candidates):
        return tied_logits(self.items.weight, self.bias, self.global_bias,
                           history, candidates)

    def auxiliary_loss(self, history, candidates):
        """First candidate is the observed click, others sampled from exposures.

        BCE keeps sampled negatives, including accidental positive collisions,
        explicit; it is not an unbiased estimator of population probability.
        """
        logits = self(history, candidates)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
