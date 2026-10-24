# TechArena CTR — results log

Validation is always **time-based** (train on earlier days, validate on later days / first-event-per-user). Metric: **global ROC AUC**. Keep a change only if validation AUC improves. Leakage guard: stop and audit if AUC jumps >0.03 in one step or exceeds 0.85 without a clear non-leaky reason.

## Fresh runs (this session)

| Step | Change | Val AUC (before → after) | Runtime | Kept? | Notes |
|------|--------|--------------------------|---------|-------|-------|
| 0-baseline | Re-ran original `model.py` CatBoost (13 feats, SVD32) | — → **0.702969** | ~151s | **baseline** | Day-6 first-event val; CTR-product naive 0.5408; 248 trees |
| P0 | New `pipeline.py` stage 0 (LGB, CTRs+context only) | 0.703 → 0.549750 | 28s | no | Missing history/similarity; correctly weak |
| P1 | + personalization / emb similarity / recency | 0.703 → 0.699747 | 77s | no | Matches baseline ballpark; does not beat CatBoost V1 |
| P2 | + richer Bayesian / recent CTR | 0.703 → 0.699920 | 138s | no | Tiny move; not kept |
| P4 | + within-event relative ranks/diffs | 0.703 → 0.699877 | 143s | no | No gain over P2 on this feature set |

**Conclusion for the fresh LightGBM pipeline:** it plateaus ~0.70 (same as V1). Bigger gains in this project already came from warm-pair relations, preference/ALS factors, and multi-model rank blends (see below). Those stages are retained; the weaker new pipeline stages were not kept.

## Already-completed journey in this repo (verified artifacts)

These follow the same improvement ideas as the requested plan (history personalization → popularity → user propensity → context → relatives → trees/ensembles). Scores are local January 26 / January 25 first-event AUCs unless noted.

| Version | What helped | Jan26 AUC | Jan25 AUC | Kept? |
|---------|-------------|----------:|----------:|-------|
| V1 `model.py` | Smoothed user/item CTR + SVD history similarity | 0.702969 | — | yes |
| V2 `improve.py` | Stronger factors, recency, **within-event relatives** | 0.731177 | 0.716325 | yes |
| V4 relations | Directed click-pair counts / lift | 0.742770 | 0.724951 | yes |
| V5 warm context | Warm-up history pairs + contextual CTRs | 0.758172 | 0.747077 | yes |
| V6 preference | Richer item factors + exposure-adjusted user profiles | **0.800327** | 0.786659 | yes (**largest single jump**) |
| V7–V8 | Ridge + event-choice blends | ~0.805 | ~0.794 | yes |
| Nonlinear / V11 | Log-odds ensemble + weight tweak | ~0.810 | ~0.797 | yes |
| Later ALS / LGB / rank blends (imp15–imp19) | Exposure ALS + leafwise LGB + aligned factors + rank blend | **0.821528** | **0.814600** | yes (best local) |

**Official leaderboard (user-confirmed):** V18 = **0.809189**. V20 local is higher but official score for V20 is not yet confirmed.

## V21 DIN rebuild + stacking (in progress)

**Runtime (this CPU):** DIN-style forward+backward ≈ **2.2 min/epoch** on all ~687k events (batch 256, dim 64). Expect **~25–45 min** for 12 epochs with early stopping. Free Colab/Kaggle T4 GPU is typically 5–15× faster — use `v21_din_colab.ipynb` if needed.

**Debug:** Alignment checks passed; tiny overfit loss 2.29→0.00, acc 100%. Predict path OK (8k-event train AUC→0.99). Full-data learning was initially stuck (~0.52) with ALS init + all events + lr 1e-3; switched to click-only + lr 2e-3.

| Step | Change | Day6 (before → after) | Runtime | Kept? | Notes |
|------|--------|----------------------|---------|-------|-------|
| V21-DIN-debug | DIN debug + tiny overfit | — | ~11s | YES | packing/labels/mask OK |
| V21-als-tune | ALS dim/decay/seeds vs V20 | 0.821528 → 0.821528 | 203s | no | best raw ~0.637; blend weight 0 |

## V21 DIN rebuild + stacking (in progress)

**Runtime (this CPU):** DIN-style forward+backward ≈ **2.2 min/epoch** on all ~687k events (batch 256, dim 64). Expect **~25–45 min** for 12 epochs with early stopping. Free Colab/Kaggle T4 GPU is typically 5–15× faster — use `v21_din_colab.ipynb` if needed.

**Debug:** Alignment checks passed; tiny overfit loss 2.29→0.00, acc 100%. Predict path OK (8k-event train AUC→0.99). Full-data learning was initially stuck (~0.52) with ALS init + all events + lr 1e-3; switched to click-only + lr 2e-3.

| Step | Change | Day6 (before → after) | Runtime | Kept? | Notes |
|------|--------|----------------------|---------|-------|-------|
| V21-DIN-debug | DIN debug + tiny overfit | — | ~11s | YES | packing/labels/mask OK |
| V21-als-tune | ALS dim/decay/seeds vs V20 | 0.821528 → 0.821528 | 203s | no | best raw ~0.637; blend weight 0 |

## V21 push (vs frozen V20 first-event cohorts)

Same validation as the official setup: first event per user on later training days (day 6 select, day 5 confirm). Compared every new score and rank-blend against V20 **0.821528 / 0.814600**. `submission.csv` was not modified.

| Step | Change | Day6 blend (before → after) | Runtime | Kept? | Notes |
|------|--------|-----------------------------|---------|-------|-------|
| V21-1 | Pointwise GRU over 20-click history | 0.821528 → 0.821528 | 558s | no | raw AUC 0.516; best blend weight 0 |
| V21-1b | GRU + within-event softmax | 0.821528 → 0.821528 | 230s | no | raw AUC 0.621; still no helpful blend weight |
| V21-2 | Time-safe item–item co-occurrence (histories + hist→click) | 0.821528 → 0.821528 | 42s | no | raw AUC 0.605; redundant with V4/V5 pair signal already in V20 |
| V21-3 | Time-gap + recency-weighted emb/match features | 0.821528 → 0.821528 | 79s | no | raw AUC 0.681 (strongest of the three alone); blend weight still 0 |
| V21-4 | Rank-blend all new raws with V20 | 0.821528 → 0.821528 | 9s | no | grid kept 100% V20 on both days |
| V21-final | Write `submission_v21.csv` | — | — | no | skipped — no both-day gain |
| V21-DIN-debug | DIN debug + tiny overfit sanity | 0.000000 → 0.000000 | 6s | YES | loss 2.294->0.000 |
| V21-als-tune | Preference/ALS retune (dim, decay, seeds) vs V20 | 0.821528 → 0.821528 | 248s | no | trials=18 both_day_keeps=0 |
| V21-als-tune | Preference/ALS retune (dim, decay, seeds) vs V20 | 0.821528 → 0.821528 | 198s | no | trials=18 both_day_keeps=0 |
| V21-DIN-frz-als-d6 | Frozen-emb DIN init=als day6 | 0.821528 → 0.791159 | 6647s | YES | epoch=8 |

Leakage audits on each member: stats/co-occur cutoffs before the fit window; history timestamps strictly before the event. No step gained >0.01 (all deltas were 0 vs V20), so no extra leakage deep-dive was required.

Rejected / no-gain experiments (not kept): timing-only V3, MF branch, choice64, event propensity, several neural/NCE/context-bilinear trials, first-event-only training, etc. See `results/imp*_report.json`.

## Leakage checks performed

- Statistics / CTR / embeddings built only from rows with `timestamp < cutoff` (before the fit/val window).
- Histories are the provided pre-event click lists (strictly before the event).
- Validation = later calendar days; first event per user (matches evaluation design).
- Independent audits on V20: feature replay, cold users, empty history, model replay, row-id coverage (`results/imp19_test_final_report.json`, promotion PASS).
- Suspicious-jump rule: V6’s +0.04 was audited as non-leaky (new preference factors from past data only) and confirmed on both dates.
- No evaluation labels were used. No external data.

## Position bias

Slot CTR in training is essentially flat (~0.0625–0.0632 across positions 0–9). Slot features add little alone.

## Final submission

- File: `submission.csv` (copy of audited `artifacts/submission_v20.csv`)
- Rows: 529390, order matches `evaluation.csv`, no NaNs, all in [0,1]
- SHA256: `8f3bb5e49f96e49efc4d6416e9c593f27f118aef06c14e19b85ead45291c789b`
- Local val AUC: **0.821528** (Jan26) / **0.814600** (Jan25)
- Baseline for comparison: **0.702969**
