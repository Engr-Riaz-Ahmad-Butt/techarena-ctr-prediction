# Protocol v3 — "find one new signal" round (29 Oct 2026)

Project: `C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction` (all paths relative to it). Read first:
`README.md` (top 30 lines), `results_log.md` (all 2026-10-29 entries), `agent_state.json`, `protocol_v2.md`, and the
2026-10-29 entries "Lever 1 diagnostic" and "Lever 2 screen" at the end of `results_log.md`.

## Goal and honest frame
Official score 0.8166 (V35). Leader ≈ 0.833. Local adjusted AUC has been flat at 0.8165–0.8172 across ~55 experiments; the
only +0.01-sized jump in this project's history came from a NEW SIGNAL fed into the trees (the warm choice model), never from
tuning. This round looks only for structurally new signals. Small tuning, reweighting, calibration, blending tweaks and seed
ensembles are closed (see `agent_state.json: completed_directions`, `excluded_directions`, and the Lever 1/2 entries).

## Metric and validation (what "better" means)
- Cohorts: `artifacts/selection_day5.npz` (all 35,230 Jan-25 first-per-user events), `artifacts/selection_day6.npz`
  (24,422 Jan-26 first events), `artifacts/fresh_holdout_day6.npz` (10,716 Jan-26 first events, user-disjoint; score ONLY a
  final candidate). Keys: `y`, `events`, `baseline` (= V53), `weights` (5-bin history weights), day-6 files also `full_indices`.
- Adjusted AUC = ROC AUC with `weights` (history-length bins 0 / 1–4 / 5–9 / 10–19 / 20, evaluation share ÷ cohort share;
  `weighted_auc` in `v39_regularized_als32.py`). V53 = 0.810911 (day 5), 0.817528 (day-6 selection subset), 0.817169 (full day 6).
- Gate (`evaluate_candidate.py`): ≥ +0.0005 adjusted on BOTH days vs V53, paired event-bootstrap 95% interval on day 6
  excluding zero, then non-negative on the fresh holdout. Screening threshold to continue an idea: +0.0002 on day 6.
  Any single-step gain > 0.01 is treated as leakage until the tester proves otherwise.
- Time safety: every statistic/model that uses labels must use rows with timestamp < the cutoff, cutoff ≤ start of the scored
  day (day d cutoff = `start + d*86400`, start = Jan 20 00:00 UTC = 1737331200). Histories and impressions (no labels) may be used
  from any time strictly before the event. Day-6 settings are chosen on day 6 and applied unchanged to day 5. Fixed seeds.
- Evaluation = each user's FIRST event on/after Jan 27, 67/17/7/4/2/2/1 % over Jan 27–Feb 2; 97% weekdays (no behavioural
  weekday effect exists — see Lever 1). Validation cohorts are first-per-user events, which matches.

## Exportability (mandatory for a final candidate)
V53 has NO evaluation-set predictions and cannot be exported. The last exportable chain is V39 (`export_v39.py`,
`artifacts/v39_final_pred.npy`, `submission_v39.csv`; V39's day-5/day-6 predictions are in `artifacts/v39_regularized_als32_day{5,6}.npz`
or as the `pred` key inside the V53 day files' parent — check). A new expert must therefore produce THREE prediction sets with the
same code path: day-5 cohort (cutoff day 5), day-6 cohort (cutoff day 6) and evaluation rows (cutoff day 7 = 1737936000),
plus a blend recipe against the parent. The tester verifies the blend gain on both days against V53 AND against V39, and the
final submission = recipe applied to V39's evaluation predictions (or to a V53 export if one is built and verified).

## Data and assets
- Raw arrays (`artifacts/`): `train.npy` (6,868,240 × 7: user, item, timestamp, time_bucket, day_type, clicked, event index;
  10 contiguous rows per event, row = 10·event), `train_hist.npy` / `train_times.npy` (event × 20 history items/timestamps,
  oldest→newest, −1 padding), `evaluation.npy`, `evaluation_hist.npy`, `evaluation_times.npy`, `ids.json`.
- Cached 354 causal features per day block: `artifacts/features/` (see mapping in results_log / the explorer notes: 190 stack +
  imp3_wf 4 + imp4 32 + imp5 32 + imp6 32 + imp17 32 + imp19 32); day blocks 2–6, val5, val6, eval, each with cutoff at block start.
- Expert predictions used by V53's head: `track_d/cache_data.npz` (x5_base/x6_base 190 head features, v53_pred5/6, w5/w6, fold mask).
- Kaggle (web UI only, user's signed-in browser; no CLI): existing private datasets `techarena-neural-causal-histories`,
  `techarena-frozen-event-measurement`, `techarena-history-attention-code`, and the 354-feature bundle; notebooks under `kaggle/`.
  Any job > 30 min needs the user's OK: write the plan and runtime estimate to `nx_kaggle_request.md` and hand back.
- Local: 14 cores / 16 GB RAM (2–5 GB free): one heavy fit at a time; CatBoost ~8 min / 600 trees on 2.9 M × 290.

## Known facts to build on (don't re-derive)
- CTR 6.3%; 62.9% of events contain a click; candidate in history → 30% click rate vs 5.7%; 2.6% of candidates are in history.
- Slot position carries no signal; day type carries no signal; negatives are popularity-skewed exposures, not uniform.
- Short histories are where the evaluation differs (0–4 clicks: 13.5% of evaluation events vs 7.3% locally) and where V53 is
  weakest (0-history AUC ≈ 0.52, 1–4 clicks ≈ 0.74–0.76, 20 clicks ≈ 0.83).
- Standalone strengths: ALS ≈ 0.757; warm16 choice model ≈ 0.8015; V53 ordinary 0.8254.
- Closed/failed: slate composition (+0.00003 overall), directed transitions, ItemKNN (216 configs), repeat/negative exposure,
  short-history specialists, importance weights, calibration/stackers, seed diversity, dense EASE (user-excluded), small
  DIN/GRU/attention residuals, five timing variants + session features, V60 XGBoost merge, categorical ids (−0.015).

## Rules
- Never modify or delete existing artifacts, `submission*.csv`, `baseline_backup/`, or files in
  `artifacts/v54_horizon/protected_before.json`. New files only, with the round prefix (`nx_` improver, `tst4_` tester).
- Never submit to the competition. Never make Kaggle content public, never delete datasets/notebooks, never touch billing.
- Log every experiment in `results_log.md` (hypothesis, change, runtime, adjusted AUC before/after, decision) and keep
  `nx_journal.md`. The tester alone decides ACCEPTED/REJECTED in `tst4_candidates.md`.
