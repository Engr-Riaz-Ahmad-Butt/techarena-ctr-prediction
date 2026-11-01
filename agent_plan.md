# TechArena CTR Autonomous Research Plan

## Active Round: Round 1

### Validation Protocol & Strict Acceptance Gates
- **Selection Day 5**: baseline V53 history-adjusted AUC **0.810911**.
- **Selection Day 6**: baseline V53 history-adjusted AUC **0.817169**.
- **Fresh untouched holdout**: must be used only for candidates that pass both selection days.
- **Acceptance Rule**: improve history-adjusted AUC by at least **+0.0005** on both selection days and show a positive fresh-holdout result. Smaller gains are treated as noise and rejected.
- **Leakage Guard**: Any jump $> 0.01$ in a single step halts the pipeline for immediate audit (feature cutoffs, timestamps, label leakages).

---

## Required research order

1. Completed: V53 error analysis and data forensics, saved as `artifacts/v54_v53_error_and_forensics.json`.
2. Active: screen time-safe event-composition features on a 20% user sample, then confirm only survivors on full selection days.
3. Next: time-safe transition scores (last 1/2/3 clicks and recency-weighted item similarity).
4. Next: restricted EASE/ItemKNN, then time features and first-event distribution matching.
5. Only if two independent signals pass: stacking/blending.

The earlier neural/tuning tracks below are retained as historical notes, not active work. The current brief explicitly deprioritizes GRU/DIN/neural experiments.

## Historical tracks (inactive)

| Track | Method Focus | Working Directory | Status | Current Hypothesis / Objective |
|---|---|---|---|---|
| **Track A** | Factorization & Implicit Feedback | `track_a/` | running | Multi-seed exposure-weighted ALS with 64–128 factors and click recency decay to strengthen sparse-history profiles. |
| **Track B** | Sequential Neural (DIN / SASRec) | `track_b/` | todo | Sequence attention between candidate item and user click history with time-gap embeddings and per-event softmax loss. |
| **Track C** | Two-Tower & Item Co-occurrence Graph | `track_c/` | todo | Full-prefix sparse item-item co-occurrence graph (PPMI, cosine, slate-normalized lift) + Two-Tower embeddings. |
| **Track D** | Gradient Boosted Decision Trees & Ranking | `track_d/` | todo | Multi-seed diverse GBDT trees (LightGBM, CatBoost, XGBoost) and slate-relative pairwise rankers. |
| **Track E** | Stacking & Blending | `track_e/` | pending | Time-safe out-of-fold rank/log-odds ensembling across diverse candidates from Tracks A–D. |

---

## Experiment Queue

### Track A: Factorization
1. `exp_a01_als_multi_seed`: 5-seed exposure-weighted implicit ALS factors (dim 64, click half-life 3 days) blended with V53.
2. `exp_a02_als_sparse_prior`: Stronger prior regularization specifically for sparse-history users (0–4 clicks).
3. `exp_a03_bpr_ranking_factors`: Bayesian Personalized Ranking (BPR) factors trained strictly with pairwise loss on click vs candidate non-click.

### Track B: Sequential Neural
1. `exp_b01_din_tiny_overfit`: Overfit sanity check on small batch; verify cross-entropy loss decreases to zero.
2. `exp_b02_din_attention_model`: DIN-style candidate-to-history attention model with time-interval encodings and event-level softmax.
3. `exp_b03_sasrec_transformer`: Self-attention causal sequence transformer over user click timeline.

### Track C: Two-Tower & Retrieval
1. `exp_c01_cooccurrence_graph`: Causal item-item PPMI / cosine graph features across full prefix click history.
2. `exp_c02_two_tower_softmax`: Two-tower user/history representation vs candidate item embedding with in-batch/in-slate negatives.

### Track D: Gradient Boosting & Ranking
1. `exp_d01_lightgbm_leafwise`: High-capacity leafwise LightGBM on causal feature subset to complement CatBoost.
2. `exp_d02_pairwise_lambdarank`: XGBoost / CatBoost pairwise ranking objective grouped by event ID.
3. `exp_d03_diverse_seed_bag`: Multi-seed tree ensemble across different random feature projections.

### Track E: Stacking & Blending
1. `exp_e01_oof_meta_stack`: Out-of-fold meta-learner combining probabilities and ranks from successful Track A–D models.
2. `exp_e02_holdout_verification`: Evaluate top candidate on frozen untouched holdout.
