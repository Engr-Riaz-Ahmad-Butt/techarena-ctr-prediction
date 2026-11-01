# Independent preference screens — 2026-10-29

Neither fixed recipe passes screening. V53 remains unchanged; no full-cohort confirmation was run.

## Reference and decision rule

Frozen 9,878-user first-event horizon screen; one event per user, ten candidates per event. Arrival and history marginals use the previously frozen IPF evaluation-mix weights. The reference is the **simplified causal V53-component horizon blend**, not a full V53 rebuild. Its weighted AUC is 0.759031195. These earlier-cutoff scores cannot be compared directly with the official 0.816623.

Screening requires +0.0002 overall AND on each of four arrival days; confirmation would require +0.0005 on the full permitted cohort. Uncertainty below is a 200-draw paired bootstrap by user with fixed event weights; it does not include refitting or recipe-selection uncertainty.

## Latent taste-group residual

Fixed 16 ALS user groups, shrunk group/item response interaction, with all scored-user counts removed. Historical-only reference transform and fixed 0.02 increment. The overall gain +0.000195975 is strictly below the gate; arrival 1 also falls short and arrival 4 is negative. Do not round this into a pass or tune the rejected recipe.

| Segment | Events | Control AUC | Candidate AUC | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| Overall | 9,878 | 0.759031195 | 0.759227170 | +0.000195975 | [+0.000017548, +0.000336645] |
| Arrival 1 | 6,919 | 0.767066917 | 0.767174708 | +0.000107791 | [-0.000081949, +0.000280952] |
| Arrival 2 | 1,794 | 0.736302752 | 0.736881645 | +0.000578893 | [+0.000190361, +0.001039507] |
| Arrival 3 | 856 | 0.738793692 | 0.739088863 | +0.000295171 | [-0.000169589, +0.000849831] |
| Arrival 4 | 309 | 0.756915154 | 0.756884125 | -0.000031029 | [-0.000901640, +0.001213046] |
| History 0 | 160 | 0.628417571 | 0.627267786 | -0.001149785 | [-0.005274021, +0.002191520] |
| History 1–5 | 1,825 | 0.675029425 | 0.675166358 | +0.000136933 | [-0.000269896, +0.000573082] |
| History 6–20 | 7,893 | 0.773689277 | 0.773915942 | +0.000226665 | [+0.000023151, +0.000444255] |

## Matched graph preference model

Own two-hop LightGCN-inspired gated user/item graph propagation against the identical warm bag model. Both use the same 16-dimensional initialization, data order, true ten-candidate/no-click loss, regularization and four fixed epochs. Each training day uses a graph from strictly before that day; validation uses the graph frozen at the three-day cutoff. Same event-time histories in both arms. Candidate = frozen reference + 0.05 × (graph probability − matched bag probability). No epoch selection or validation-fitted blend scale. The overall effect is effectively zero; reject.

| Segment | Events | Control AUC | Candidate AUC | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| Overall | 9,878 | 0.759031195 | 0.759027097 | -0.000004099 | [-0.000010819, +0.000003880] |
| Arrival 1 | 6,919 | 0.767066917 | 0.767059225 | -0.000007692 | [-0.000016958, +0.000004043] |
| Arrival 2 | 1,794 | 0.736302752 | 0.736310507 | +0.000007754 | [-0.000013093, +0.000025628] |
| Arrival 3 | 856 | 0.738793692 | 0.738806023 | +0.000012331 | [-0.000017925, +0.000037585] |
| Arrival 4 | 309 | 0.756915154 | 0.756912573 | -0.000002581 | [-0.000108067, +0.000053680] |
| History 0 | 160 | 0.628417571 | 0.628405155 | -0.000012416 | [-0.000119002, +0.000080270] |
| History 1–5 | 1,825 | 0.675029425 | 0.675055836 | +0.000026412 | [+0.000012138, +0.000041984] |
| History 6–20 | 7,893 | 0.773689277 | 0.773680606 | -0.000008671 | [-0.000019001, +0.000000108] |

## Scope and checks

Graph standalone weighted AUC: bag 0.711418694, graph 0.711085878. Both are weaker than the reference; this is evidence against this fixed graph diagnostic, not a general disproof of graph recommenders. The small positive short-history effect is below the gate and does not justify a specialist model.

Corrected graph matched run: 692.9 seconds locally, two CPU threads (11.5 minutes, versus 10.7-minute timing-pilot estimate). An earlier attempt was stopped before validation metrics after training loss exposed an Adagrad gradient-unit mismatch. The same objective was changed from mean to batch-sum backpropagation in BOTH arms; a training-only sanity check confirmed learning. Original files were preserved. No validation metrics informed that correction. See implementation_correction.json.

Automatic approval review rejected the proposed Kaggle data upload. No upload or workaround was used; the completed experiment ran locally. No repository code was copied. Research inspiration: [LightGCN, He et al. (2020)](https://arxiv.org/abs/2002.02126); the gated history-bag architecture here is an original adaptation, not a paper reproduction. Taste grouping was inspired by [Dhillon et al. (2003)](https://bigdata.oden.utexas.edu/publication/information-theoretic-co-clustering/), not an implementation of their co-clustering algorithm.

Post-fit audit: all 29 protected artifacts unchanged; input manifests unchanged; finite/aligned predictions; one event per user; at most one positive per event; target weight marginals matched; mutating labels leaves recomputed predictions identical on 128 events in both arms. Training/graph timestamps precede their permitted cutoffs. Labels are used only for the fixed validation metrics and screening gate, not fitting or tuning. No holdout or evaluation data were scored, and no submission was created.

The official target has not been reached or demonstrated. These tests add no accepted model. Other researchers’ nx/nxa/tst sources and models were treated as read-only; their results are not promoted or claimed as independently confirmed here.
