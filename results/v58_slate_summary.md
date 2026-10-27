# Candidate-set information: no supported gain

V53 remains the best accepted local candidate at main day6 history-adjusted AUC 0.8171690462350286. The objective remains 0.834786 and the pre-upload minimum remains 0.82 adjusted AUC. Neither is achieved; no CSV was exported.

The V53 validation error decomposition reproduces overall adjusted AUC from disjoint negative sources. On day6, 59.91% of negative weight belongs to nonclicked candidates in clicked events, and 40.09% to candidates in no-click events. Their AUCs against all positives are 0.821025 and 0.811406, respectively. Conditional within-event pair AUC is 0.805309, with top1 accuracy 0.391076. Event AUC using the maximum row rank is 0.624630. These are descriptive diagnostics, not causal explanations or achievable-score bounds. A low event AUC does not establish an easy opportunity to improve it. Both item choice and cross-event ranking retain errors.

V58 tests information from the full candidate set using causal frozen item embeddings, without interpreting identifier values. An identical two-layer residual network sees either real candidate-set summaries or zero-filled set inputs. Both arms retain the same fixed causal warm factors/context, initialization tensors, event orders, dropout seeds, optimizer, losses, epochs and blending settings. Set summaries include means of the other nine candidate embeddings, candidate dispersion and candidate-by-set products. Permuting candidate order permutes logits correspondingly.

The frozen recipe selects the treatment on day6 among epochs1/2/4 and weights0.1/0.25, then requires a positive ordinary gain and positive adjusted lower confidence interval against both V53 and the matched control before unchanged day5 confirmation. Train labels are strictly before the validation cutoff; the numerical context predictions replay exactly at initialization.

| Main day6 setting | Ordinary AUC | Adjusted AUC |
|---|---:|---:|
| V53 | 0.8254233318844953 | 0.8171690462350286 |
| Matched control, epoch4 weight0.1 | 0.8254548744403862 | 0.8172092211292398 |
| Candidate-set treatment, epoch4 weight0.1 | 0.8254460453549567 | 0.8171998643957881 |

The treatment's adjusted gain against V53 is only 0.0000308182, with conditional interval [-0.0000488949, 0.0001105312]. Against its matched control, adjusted difference is -0.0000093567, interval [-0.0000325679, 0.0000138544]. This does not support the candidate-set hypothesis. The gate failed, so no day5 model was fitted and neither arm was promoted.

The independent audit replays both selected saved models on all 351,380 validation rows with zero probability difference and exact final rank-score/metric reconstruction. A separate NumPy forward implementation agrees with Torch logits within 3.88e-7. Candidate permutation errors are at most 2.39e-7. Frozen source/input hashes, context/embedding tensors, whole-event identities and causal history timestamps pass.

The first audit used chunks that split the original 1024-event inference batches. This changed some float32 rounding, which can reorder extremely close scores. The audit was repaired to preserve original batch boundaries; no model, prediction, score or tolerance was changed. Exact replay then passed. Reports and source remain reproducible; models/data stay in ignored artifacts. Reused dates and selection uncertainty remain limitations; no official score is inferred.
