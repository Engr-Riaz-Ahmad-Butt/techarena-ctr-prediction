# Mac handoff: TechArena 2026 CTR challenge (state on 1 Nov 2026)

Read this whole file before doing anything. It replaces ~10 days of conversation context. The older detailed protocols are
`protocol_v3.md` and `protocol_v4.md`; the experiment logs are `results_log.md` (append-only) and `nxb_journal.md`.

## 1. The competition

- Synthetic recommendation data. An **event** = one user shown **10 candidate items** at one moment; at most one is clicked
  (events with no click exist: 37%). Predict a click probability for every row of `evaluation.csv` (529,390 rows, 52,939 events,
  one event per user = each user's **first event at or after the cutoff 2025-01-27 00:00 UTC**, spread over Jan 27 – Feb 2).
- Training: `train.csv` (7.3 GB, 6,868,240 rows, 686,824 events, Jan 20 – Jan 26). Columns: rowid, event_id, user_id, item_id,
  timestamp, time_bucket (6-hour block), day_type (weekend flag), history_item_ids / history_timestamps (JSON, last ≤20 clicks
  strictly before the event, oldest → newest; warm-up clicks before Jan 20 appear only in histories), clicked.
- Metric: **global ROC AUC over all rows** (not per event). Public/private leaderboard = random split of the evaluation rows; the
  final ranking uses the private part. **10 submissions per day.** Full rules: `data/INSTRUCTIONS.md` (not in Git; in the bundle).

## 2. Scores so far (official public leaderboard)

| File | Public AUC |
|---|---|
| V5 | 0.7125 |
| V18 | 0.809189 |
| V35 | 0.816623 |
| **v41 = `submission_v41_candidate.csv` (nxa_h1)** | **0.817396 ← current best** |

Leaderboard on 1 Nov 2026: #1 Cybots 0.833803, #2 CSDT 0.833173, #3 TU Dublin 0.833136 … #15 MUnemployed 0.830839.
**15 teams are clustered at 0.8308–0.8338**, which suggests a common, known approach that we are missing. We are 0.016 behind #1.

## 3. Local validation (trust it: it predicted v41's leaderboard gain)

- **Adjusted AUC** = ROC AUC with row weights that reweight 5 history-length bins (0 / 1–4 / 5–9 / 10–19 / 20 clicks;
  `np.digitize(len,[0,4,9,19],right=True)`) to the evaluation mix (shares 0.01256 / 0.12207 / 0.20091 / 0.27239 / 0.39207).
  Implemented as `weighted_auc` in `nx_seq_core.py` / `v39_regularized_als32.py`. Plain single-day AUC ranked models backwards.
- Cohorts (first-per-user events of a day, labels known): `artifacts/selection_day5.npz` (35,230 events, Jan 25),
  `artifacts/selection_day6.npz` (24,422 events, Jan 26, **selection subset**), `artifacts/fresh_holdout_day6.npz`
  (10,716 user-disjoint Jan-26 events; score **only** a final candidate). Keys: y, events, baseline, weights, full_indices.
- Cutoffs (unix): start Jan 20 = 1737331200, day 5 = 1737763200, day 6 = 1737849600, evaluation = 1737936000; DAY = 86400.
  Scoring day d uses labels strictly before `start + d*DAY`. Same code path for day 5, day 6 and evaluation.
- **Gate** (vs current best nxa_h1): ≥ +0.0005 adjusted on the day-6 selection subset AND on day 5, paired event-bootstrap 95%
  interval above 0 on day 6, settings chosen on the day-6 selection subset only (grid saved), then non-negative on the fresh
  holdout. Screen threshold to keep exploring an idea: +0.0002.
- **Leakage alert**: any standalone model > 0.83 adjusted, or any single blend step > +0.003, is treated as leakage until an
  independent tester replays the features from raw arrays with pre-cutoff data only.
- Validation trap (found): a warm start built from clicks before the cutoff inflates *same-period* held-out scores
  (0.80–0.83 instead of 0.70–0.76). Only next-day (out-of-time) cohorts are honest.

Reference adjusted AUCs: nxa_h1 = 0.818331 (day-6 selection) / 0.818023 (full day 6) / 0.811431 (day 5);
V53 = 0.817528 / 0.817169 / 0.810911.

## 4. Data layout (`artifacts/`, rebuilt by `python prepare.py` or taken from the bundle)

- `train.npy` (6,868,240 × 7 int64: user, item, timestamp, time_bucket, day_type, clicked, event index; 10 contiguous rows per
  event, row = 10·event), `train_hist.npy` / `train_times.npy` (event × 20, −1 padding), `evaluation.npy`,
  `evaluation_hist.npy`, `evaluation_times.npy`, `ids.json` (id maps). 50,000 items, ~53,000 users.
- Not in the bundle (too big): `artifacts/features/` (18 GB of cached causal features for the V-chain trees, blocks 2–6, val5,
  val6, eval). Without it the V39 trees cannot be refit; work on top of the exportable predictions instead (section 5).

## 5. The model chain and how a submission is made

- **V39** = last *exportable* tree chain (`export_v39.py`); evaluation predictions `artifacts/v39_final_pred.npy`.
- **V53** = better local head (190 features), **not exportable** (no evaluation predictions). Used only as a local reference.
- **nxa_h1 (= v41)** = logistic head (C=0.01, StandardScaler, history-bin weights) on [logit(rank(parent)), E3z], where E3z is an
  "exposure-policy" expert: within-event standardised mean rank of three typicality scores from a TruncatedSVD (seed 4403) of the
  IDF-log user × item impression matrix (impressions are label-free; the recommender chose slates from its belief about the user).
  Scripts: `nxa_exposure2.py --day 5|6|7`, `nxa_h1.py`, `nxa_h1_final.py` (recipe frozen in `artifacts/nxa_h1_frozen.json`).
  Exportable output: `artifacts/nxa_h1_eval_pred_v39.npy` (head applied to V39).
- **New candidate files must be built on `nxa_h1_eval_pred_v39.npy`** (or anything else with day-5, day-6 AND evaluation
  predictions from one code path). Submission = rowid from `evaluation.csv` order + predicted_probability in [0,1];
  verify 529,390 unique rowids, header `rowid,predicted_probability`. Builder example: `tst5_build_submission.py`
  (fix its hard-coded Windows path first, see section 9).

## 6. What we learned (do not re-derive)

- CTR 6.3%; 62.9% of events have a click. Candidate already in history → 30% click rate vs 5.7%; 2.6% of candidates are in history.
- Slot position, day type, weekday, time-of-day context, arrival day (Jan 27 vs later) carry **no** signal.
- The simulator very likely uses **16-dimensional user/item factors** (history-profile AUC peaks sharply at rank 16; 12 and 20 worse).
- Tastes do not drift (oldest 10 history clicks as informative as newest 10); items do not drift.
- Histories are fully consistent with train: no organic clicks, no post-cutoff clicks. Evaluation histories add only 3,489 click
  records not visible from train (1,453 evaluation-only users) → nothing to gain from them.
- **Weakest spot = rarely clicked candidate items.** nxa_h1 day-6 plain AUC by the candidate's pre-cut clicks:
  0 → 0.745 (1.2% of rows), 1–2 → 0.784 (8.6%), 3–9 → 0.816 (46%), 10–29 → 0.842 (36%), 30–99 → 0.849 (8%).
  Evaluation mix: 0.7 / 6.2 / 42 / 40 / 11%. Exposure-based item vectors are weak even for popular items (cosine AUC 0.58 vs
  0.68 for click vectors), so impressions cannot fix rare items.
- AUC split of nxa_h1: within-event ranking 0.816; clicked rows vs candidates of no-click events 0.812; clicked vs other
  candidates of click events 0.822 — no single cheap lever.
- Standalone strengths (adjusted, day 6): clean-room LightGBM on 65 simple features 0.731; ALS ~0.757; exact MAP 16-dim choice
  model 0.777; joint click+impression model 0.776; SGD "bag" choice model (bag1) 0.789; warm16 choice model ~0.792; full chain
  (trees + experts) 0.818. The trees' strength comes from the trained choice factorisation, not from simple counts.

**Closed directions (all failed the gate; do not repeat without a genuinely new angle):** slate composition, directed
transitions, ItemKNN (216 configs), repeat/negative exposure, short-history specialists, importance weights, calibration and
stackers (incl. a conditional-logit choice head: −0.0012), seed diversity, dense EASE (user excluded it), DIN / GRU / attention /
transformer residuals and sequence models (attention overfits), timing and session features (5+ variants), weekday effects,
categorical ids, user-user CF, co-visitation, event-mass, catalogue rank, exposure in the tree, richer exposure views, HPF,
learned logging policy, graph preference, taste groups, per-arrival calibration, transductive item trend from evaluation
impressions, exact MAP factors, joint click+impression factors, Laplace posterior scoring, discrete item categories
(no cluster structure in the 16-dim item space).

## 7. Unfinished candidate: D1 (would be v42)

- D1 = conditional rank blend on top of nxa_h1: `pred = (1−Σw)·rank(nxa_h1) + Σ w_k·rank(expert_k)`, experts = bag1, C1, C2 u·v,
  weights per cell of (candidate pre-cut clicks <3 / 3–9 / ≥10) × (history <10 / ≥10), from {0, 0.05, 0.15}, chosen by coordinate
  ascent on the day-6 selection subset. Scripts `nxd_d1_blend.py`, `nxd_d1_check.py`, `nxd_d1_final.py`;
  artifacts `nxd_d1_grid.json`, `nxd_d1_check.json`, `nxd_d1_day{5,6}.npz`, `nxd_d1_eval.npy` (already applied to V39 eval preds).
- Result vs nxa_h1: day-6 selection +0.000705, full day 6 +0.000615, day 5 +0.000766 (bootstrap intervals above zero).
  Unconditional variant (one weight set): +0.000394 / +0.000401 / +0.000748 → the per-cell structure gains on the selection subset
  but not on day 5 (partly selection fit).
- **Blocker:** the experts' configurations (bag1 = `nxb_b_seq.py` w7first, 12 epochs; C1 = `nxb_c_mf.py` anc10off; C2) were chosen
  on the day-5 cohort, so day 5 is not an independent check. Fix: re-select the close configurations on day 4 (phase 1 of
  `--target 5`: train < day 4, validate on day-4 first events). `nxd_sel4.sh` lists 15 runs (~1.5 h on the Windows PC); the 6 that
  matter: w7first, w6plain, w10neg (bag family) and anc10off, anc10, c2nk (factor family). It was stopped at run 2; restart clean.
  If the same configurations win, D1's day-5 numbers stand → tester → holdout → `submission_v42_candidate.csv`.
  **Fix the hard-coded `/c/Users/...` path in `nxd_sel4.sh` first.**

## 8. Ideas not yet tried (ranked by my expectation)

1. **Find the common approach of the top-15 cluster.** 15 teams within 0.003 suggests a standard recipe giving ~0.83. Think about
   what a "solid baseline that trains in < 15 min on a laptop" (organisers' words) would be and why ours (0.73 clean-room) is far
   below it. Re-read `data/INSTRUCTIONS.md` and `data/starter_solution.py`; question our assumptions (e.g. how training rows are
   sampled, how the 16-dim structure could be exploited directly, how no-click events should be scored for a *global* AUC).
2. **Rare-item representation.** >50% of evaluation rows are items with < 10 clicks. Better shrinkage / pooling of item factors
   (hierarchical priors, co-click neighbourhoods, using non-click impressions in the choice likelihood) aimed at that bin.
3. D2 (needs the 18 GB feature cache from the Windows PC): feed bag1 / C1 / C2 scores plus candidate click/impression counts into
   the V39 trees instead of rank-blending them; historically the only +0.01 jump came from feeding a new expert into the trees.

## 9. Mac notes

- Use `python3`; create a venv; `pip install -r requirements.txt torch`. Use CPU for torch (Apple MPS has no float64 and weak
  sparse support). Check RAM and cores first (`sysctl -n hw.memsize hw.ncpu`) and decide how many heavy fits may run at once
  (each fit can use 3–6 GB).
- Some older scripts hard-code `C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction` (`tst4_*`, `tst5_*`,
  `nx_f6_trend_screen.py`, `nxd_sel4.sh`). Do not edit them in place; make copies with relative paths (`config.ARTIFACT_DIR`).
- `config.py` reads `CTR_DATA_DIR`, `CTR_TRAIN_CSV`, `CTR_EVALUATION_CSV`, `CTR_ARTIFACT_DIR` environment variables.

## 10. Rules from the owner (hard; never break)

- No leakage: every statistic uses only data before each event or cutoff; test features are built exactly like validation
  features; fixed seeds. Do not open the fresh holdout until a final candidate.
- Never overwrite or delete existing files, submissions or artifacts. New files get new names; submissions are
  `submission_v<N>_candidate.csv` (next is v42). Check hashes of protected files before and after.
- Never submit to the competition; the owner uploads manually and reports the score.
- Only an independent tester agent may accept a result. Treat any surprisingly large jump as leakage until proven otherwise.
- No external data, pretrained models, or code copied from repositories. No credentials in logs. Ask the owner before any job
  over 30 minutes on Kaggle/cloud, and before any local job expected to take over 3 hours.
- Never stop processes you did not start. Log every experiment in `results_log.md` (append only).
- Never add Claude co-author lines to git commits.
- Report to the owner in simple words (they are not a data scientist), with the numbers.
