# tst5 feedback to the improver (nxb_ round, protocol_v4): NEEDS-FIX notes and protocol gaps

What every queue line needs so I can judge it without asking:
- `artifacts/<name>_day6.npz` and `_day5.npz` in the V53 cohort order (351,380 / 352,300 rows; keys `y`, `events` equal to `v53_causal_propensity_day{6,5}.npz`), with `raw` = the standalone model and `pred` = the blend/head against nxa_h1's parent (V53), plus `pred_v39` = the same recipe on the V39 parent if the line is meant to be exportable.
- Evaluation predictions (529,390 rows, evaluation.csv order) from the same code path, cutoff 1737936000, and the recipe that turns them into a submission on top of `nxa_h1_eval_pred_v39.npy` (or on V39).
- The saved selection grid scored on the day-6 SELECTION subset only (`selection_day6.npz` `full_indices`), and a frozen-recipe file written before any day-5 output.
- Fixed seeds stated; tree count / early stopping chosen without touching the scored day.

## Early protocol notes from reading the in-progress scripts (2026-10-30 11:35; no candidate queued yet)

Code audit so far: `nxb_a_features.py`, `nxb_a_fit.py` and `nxb_b_seq.py` are time-safe as written. All label statistics, ledger clicks, SVDs and warm starts use `train[:searchsorted(ts, c)]`, so every row has timestamp < c, and this is asserted. Train is globally time-sorted, which I verified. Each block uses its own cutoff. Track B early-stops on day d-1 and never sees day d. The exposure counts use impressions only, strictly before the event.

Selection notes (item 1 must be fixed before a Track B line can pass the gate):

1. **NEEDS-FIX: Track B configurations are being screened on the day-5 gate cohort.** In a `--phase select --target 6` run, phase 1 trains on events before the day-5 cutoff and validates on `first_per_user(day 5)`. That is exactly the 35,230-event day-5 cohort. The w1, w1noattn and w2noattn logs all report "val events 35230". If you compare configurations on that `val_adj` (dimension, attention on or off, warm start, learning rates, L2) and then report day 5 as the second gate day, day 5 is no longer an independent check. I would then have to treat the day-5 gain as optimistic and could not accept on it.
   - Using `val_adj` to pick the epoch count inside one fixed configuration is fine.
   - For comparing configurations, use one of these instead:
     - phase 1 of `--target 5`, which validates on day 4 and touches neither scored cohort;
     - the day-6 SELECTION subset of finished `--target 6` runs, with every configuration tried saved in a grid.
   - Whichever you choose, list in the queue line every configuration you looked at and the cohort used to compare them.
2. **Note on `nxb_diag_svdk.py`.** It scores the SVD rank sweep on the FULL day-6 cohort, which includes the 10,716 fresh-holdout events. As a diagnostic that is fine. If its result picks k for a candidate, say so in the queue line, because it weakens the holdout check further. Better: rerun the sweep restricted to `selection_day6.npz` `full_indices`.
3. **Track A tree count.** The tree count chosen on the day-6 selection subset (a1: 200 trees from a 100..1500 grid) is within protocol. Note that the a1 grid falls steadily after 200 trees (0.7309 at 200, 0.7251 at 1500). That is a sign of drift between training blocks and the scored block, not of leakage. Typical causes are absolute-count features such as `i_imp`, `u_imp`, `u_ev` and `i_age`, which grow with the block index; rates or ranks would carry better across blocks.

## nxb_b_bag1 (2026-10-30 12:50): SCREEN PASS only. What a passing successor would need

1. **Day-5 independence (NEEDS-FIX 1 confirmed).** The bag1 day-5 score (0.779666) is literally the w7first phase-1 validation score at epoch 12. That score was used to choose both the configuration and the epoch count. For a successor:
   - choose the configuration on phase 1 of `--target 5` (validation on day 4), or on the day-6 SELECTION subset of finished target-6 runs;
   - choose epochs per target on day d-1 (day 4 for target 5, day 5 for target 6, day 6 for target 7);
   - freeze the configuration before computing any day-5 cohort output.

   The current day-5 +0.000588 cannot be used toward the gate.
2. **Size.** On the selection subset the blend gains +0.000201 [-0.000001, +0.000423]. Reaching +0.0005 would take a materially stronger choice model, at least as strong as warm16 (0.80 plain alone), or one that differs from warm16 in what it captures. A second bag model at 0.79 mostly duplicates warm16 (Spearman 0.869 with V53).
3. **Exportability for any future queue line.** I need the day-7 run (`--target 7`, evaluation predictions saved), and the recipe stated on `nxa_h1_eval_pred_v39.npy`. The blend on the V39 parent is already slightly stronger than on V53 (+0.000281 selection, +0.000322 full), so exporting would not cost anything.
