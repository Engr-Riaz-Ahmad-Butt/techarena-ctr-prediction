# Global pairwise residual: no supported improvement

Best accepted V53 main day6 history-adjusted AUC remains 0.8171690462350286. The goal 0.834786 and minimum adjusted AUC 0.82 before another upload recommendation remain unmet. No CSV was exported.

V59 changes the training objective, comparing importance-weighted 11-way likelihood against sampled pairwise logistic ranking plus 0.1 times the same likelihood. Both arms share the frozen causal context/factors, residual network, initial tensors, dropout seeds, event order, optimizer and epochs. Pair draws have a separate random generator. For each positive in a randomized event minibatch, eight negatives are sampled with replacement from all negative candidate rows in that minibatch. The scores are marginal log probabilities from the 11-way softmax. This is a stochastic ranking surrogate, not exact full-dataset pair enumeration or exact AUC optimization.

Importance ratios use unlabeled evaluation history-length proportions divided by fitting event proportions, clipped to [0.25,4]. Pair weights multiply positive and negative row weights; likelihood uses event weights. All fitting labels strictly precede the validation cutoff, and events remain intact. Settings and inputs were frozen before fitting. A central-difference gradient check has maximum error 2.81e-12 and translation-gradient sum 1.39e-17.

Six settings per arm were fixed: epochs1/2/4 and rank-residual weights0.1/0.25. The pairwise setting is chosen on day6 adjusted AUC, with an unchanged day5 confirmation allowed only after positive ordinary gain and positive adjusted lower intervals against V53 and its matched likelihood control. Dates have been repeatedly reused; intervals condition on chosen settings and exclude fitting/selection uncertainty.

| Selected epoch2, weight0.1 | Ordinary AUC | History-adjusted AUC |
|---|---:|---:|
| V53 | 0.8254233318844953 | 0.8171690462350286 |
| Weighted likelihood control | 0.8254306913434097 | 0.8171805424180930 |
| Pairwise treatment | 0.8254457140254449 | 0.8171995454290641 |

The treatment's adjusted gain against V53 is 0.0000304992, conditional interval [-0.0000453506,0.0001063490]. Against the matched likelihood control the difference is 0.0000190030, interval [-0.0000041168,0.0000421228]. Neither lower bound is positive. The experiment is rejected, and no day5 confirmation is fitted. The point estimate must not be presented as a new accepted best or official result.

Independent checks reproduce both selected checkpoints on all 351,380 validation rows with zero raw probability error and exact rank scores/AUCs. A separate NumPy forward implementation agrees within 3.88e-7 logits. Frozen hashes, context and embedding tensors, data/label/event alignment, true fitting timestamps and importance ratios pass. The recorded 3,939,008 sampled pairs match fitting positives times eight times four epochs. No validation labels enter parameter fitting.

Together V57–V59 do not provide support for promoting another small residual change. A complete causal weekday ensemble benchmark remains more informative than another small sweep on the reused weekend dates. This does not replace the main score requirement or assert that a weekday diagnostic meets the objective.
