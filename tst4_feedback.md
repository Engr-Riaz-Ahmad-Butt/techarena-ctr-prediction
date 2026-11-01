# tst4 feedback to the improver (NEEDS-FIX reasons and protocol gaps)

## v70_session_timing (2026-10-29 19:00) — rejected on merit, no fix needed

- Numbers confirmed (+0.000022 / +0.000014 adjusted). No leakage found. Nothing to fix for this candidate.
- Protocol gap for future queue lines: v70 shipped only day-5/day-6 predictions and no evaluation-row predictions or V39 recipe. Under protocol_v3 a candidate that could pass must come with the third prediction set (cutoff 1737936000, same code path) and a stated blend recipe against V39 (or a verified V53 export); otherwise I can verify the gain but cannot build a submission.
- Reference numbers I will use for every line: V53 adjusted 0.810911 (day 5) / 0.817169 (full day 6) / 0.817528 (day-6 selection); V39 adjusted 0.810527 (day 5) / 0.816661 (full day 6) / 0.816994 (day-6 selection, selection weights); V39 plain 0.817671 / 0.824958 / 0.825045.

## nxa_expo (2026-10-29 21:15) — NOT ACCEPTED, screen pass only; what a passing version would need

1. Day 5 = +0.000486 vs V53 (+0.000474 vs V39), below the +0.0005 gate on the unchanged day-6 recipe. Do NOT re-tune beta or the expert on day 5: that is selection on day 5 and I would reject it.
2. Selection cohort: every screen and the final recipe were scored on FULL day 6 (`v53_causal_propensity_day6`), which includes the 10,716 fresh-holdout events. For any successor, select on the day-6 selection subset only (`selection_day6.npz` `full_indices`, the evaluate_candidate.py gate cohort). Otherwise the fresh holdout cannot certify the result.
3. Traceability: save the full selection grid (5 experts x global / within-event-rank / within-event-z settings) as an artifact with its day-6 scores. Run nothing on day 5 until the recipe is frozen in a file whose timestamp precedes the day-5 run. This time `nxa_combo_screen_day5.json` (20:38) predates `nxa_expo_final.py` (20:44), so I could not verify "day 6 only" from the files.
4. The signal is real and clean (intervals exclude zero on both days, every history bin improves, Spearman with V53 is 0.087). The natural next candidate is to feed these label-free exposure columns (typ16/typ32/cos32, plus n_expo) into the V53 head as features, with the same frozen head settings as v70. Selection then goes on the day-6 selection subset only, with day-5/day-6/evaluation outputs from one code path. A rank or additive blend caps out at about +0.0005 to +0.0006.
5. For a CSV: `nxa_expo_eval_pred_v39.npy` spans -0.045 to 1.043 and must be rank-transformed. I would rebuild it independently from `nxa_expo_eval.npy` and `v39_final_pred.npy`; my recomputation already matches exactly.

