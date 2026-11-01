# Protocol v4 — "close the gap to the top-20" round (30 Oct 2026)

Read first: `protocol_v3.md` (metric, cohorts, cutoffs, data layout, exportability, rules — all still in force),
`results_log.md` entries from 2026-10-29 onward, `tst4_candidates.md`, `nx_journal.md`.

## Why this round is different
The public leaderboard (30 Oct) has **20 teams at ≥ 0.8257** (leader 0.832869; #20 is named "SignalRank – Lean Sequential
Recommender"). Our best official score is 0.816623 (V35); our best local candidate is nxa_h1 (adjusted 0.818023 full day 6,
0.811431 day 5; `submission_v41_candidate.csv`, not yet uploaded). The organisers state a solid baseline trains in < 15 min on a
laptop without GPU. So a +0.01 gap is closable, and our 60-version pipeline is missing something. Two tracks:

### Track A — clean-room baseline (diagnose the pipeline)
Write NEW, simple, self-contained code (prefix `nxb_a_`), not importing the V-chain modules (reading the raw arrays in
`artifacts/` and the organiser's `starter_solution.py` — find it under `data/` or `C:\Users\engrr\Downloads\Task\` — is fine).
- One LightGBM (or CatBoost) on ~30–60 well-understood features, all computed per day block with statistics strictly before the
  block's cutoff: user/item smoothed CTR and counts (overall, last 1/3 days, by time bucket), candidate-in-history count and
  recency, history length and recency, item–history co-click similarity (counts within user histories, popularity-normalised),
  one factor model (ALS/SVD on pre-cutoff clicks) user·item and history-mean·item, within-event relative versions (rank,
  minus event mean/max) of the strongest columns, event-level stats.
- Train rows: days 2–5 for day-6 scoring (each block with its own cutoff), days 1–4 for day 5, days 2–6 (or 3–6) for evaluation.
  Fixed seed; trees chosen on the day-6 SELECTION subset only.
- Report standalone adjusted AUC (day 6 full & selection, day 5) and per-history-bin AUC next to V53/nxa_h1, plus its blend with
  nxa_h1. If the clean model alone is within 0.003 of nxa_h1, compare per bin to find where the chain loses, and report.

### Track B — properly trained sequence/choice model (the likely missing signal)
Prefix `nxb_b_`. A candidate-aware model over the 20-click history WITH timestamps (DIN-style target attention and/or a small
transformer encoder), user embedding, item embeddings (optionally warm-started from a pre-cutoff SVD), trained on the 10-way +
no-click softmax over ALL labelled events before the cutoff, **to convergence** (many epochs, early stopping on a time-held-out
pre-cutoff day, never on the scored day). Earlier attempts stopped after 2 CPU epochs (standalone 0.74–0.75) — undertrained.
Bar: standalone adjusted ≥ 0.80 on day 6 to be useful; report its blend with nxa_h1. Same code path for day 5 / day 6 / eval
(cutoffs 1737763200 / 1737849600 / 1737936000). torch is installed locally. If a full fit needs a GPU or > 30 min, write
`nxb_kaggle_request.md` (plan, data to upload as a PRIVATE dataset, runtime estimate, nvidia-smi check first) and hand back so the
lead can ask the user. Never start a > 30-min Kaggle job without that approval.

## Gate (tester decides; prefix tst5_)
New baseline = nxa_h1 (`artifacts/nxa_h1_day{5,6}.npz`; evaluation `artifacts/nxa_h1_eval_pred_v39.npy`). Accept = ≥ +0.0005
adjusted vs nxa_h1 on day 6 (selection subset) and day 5, day-6 paired interval above zero, selected on the selection subset only
with the grid saved, then non-negative on the fresh holdout. Any standalone model above 0.83 adjusted, or any single blend step
above +0.003, is a LEAKAGE ALERT until the tester replays features from raw arrays with pre-cutoff data only.
Queue: `tst5_queue.md` with `name | day6 sel adj | day5 adj | standalone d6/d5 | description | script | npz paths | eval npy | recipe`.

## Housekeeping
One heavy fit at a time (check python processes; other sessions may be active — never stop their processes). Log every
experiment in `results_log.md` and `nxb_journal.md`. Never modify existing files or protected artifacts. Never submit.
