# nxb_ journal — protocol_v4 round (improver "nxb"), started 2026-10-30

Read: protocol_v4.md, protocol_v3.md, results_log.md (2026-10-29 onward), tst4_candidates.md, nx_journal.md,
data/INSTRUCTIONS.md, data/starter_solution.py, choice64_core.py / imp2_warm_choice.py (warm choice model, read only).
References (adjusted AUC, project 5-bin weighting): nxa_h1 full day 6 0.818023, day-6 selection 0.818331, day 5 0.811431;
V53 0.817169 / 0.817528 / 0.810911.
Machine at start: 14 cores, 16 GB, ~2 GB free (Cursor/Chrome/ChatGPT use the rest); no other python process running.

## Track A — clean-room baseline (nxb_a_features.py, nxb_a_fit.py)
Plan: 65 causal features per day block k (events of day k; cutoff c = S + k*DAY; block 7 = evaluation rows), every label
statistic / ledger click / impression count / SVD from rows with timestamp < c; one LightGBM (lr 0.05, 63 leaves, seed 6201);
day 6 <- blocks 2..5, day 5 <- blocks 1..4, evaluation <- blocks 3..6; trees chosen on the day-6 SELECTION subset only
(grid saved to artifacts/nxb_a_<name>_grid_day6.json, frozen to artifacts/nxb_a_<name>_frozen.json before day 5 / eval).
Feature families: item (smoothed CTR overall / last 1 d / last 3 d / by time bucket, counts, age, trend), user (CTR, event
click rate overall / last day, activity, time since last event), user-item (prior exposures, prior clicks, ledger clicks),
history (length, recency, span, last-24h clicks, uniqueness; candidate count / position / recency in history), co-click
(cosine over pre-cutoff user click sets: sum / mean / max / recency-weighted / last item), SVD-32 of idf clicks (history
profile and ledger profile), exposure SVD-32 cosine (label-free), within-event rank / minus max / minus mean of the strongest,
event-level (mean item CTR, number of candidates in history, max co-click), time bucket.
Feature build: 70-80 s per block, < 1.5 GB.

### Track A result a1 (day 6; 2026-10-30 ~11:10) — clean model is FAR below the chain
- Fit: blocks 2..5, first event per user per day (1.40 M rows), 65 features, 1,500 trees grid; 346 s fit.
  Selection-subset grid peaks at 200 trees (0.730864) and declines steadily afterwards (0.7251 at 1,500).
- Day 6 standalone adjusted: full 0.730629, selection 0.730864, plain 0.736411 (nxa_h1 0.818023 / 0.818331; V53 0.817169 / 0.817528).
- Per history bin (plain AUC, day 6): bin 0 (198 ev) clean 0.5625 vs nxa_h1 0.5309 vs V53 0.5221; 1-4 0.6853 vs 0.7496;
  5-9 0.7140 vs 0.7966; 10-19 0.7344 vs 0.8242; 20 0.7484 vs 0.8429. The chain wins every bin except the tiny 0-history bin.
- Blend with nxa_h1 (selection subset grid saved in artifacts/nxb_a_a1_report.json): best rank 2% = +0.000008. Nothing.
- a2 (per-day normalisation of block-length counts, item age dropped): 0.730777 selection — the decline with trees is not
  a block-length shift; the simple features simply carry ~0.73.
- Strongest single features (day-6 cohort, plain AUC): SVD history-profile cosine 0.710, its within-event centred 0.660,
  co-click sum 0.602, exposure cosine 0.576, candidate-in-history 0.551, user event click rate 0.544, item CTR 0.525.
Reading: nothing "simple" is missing. Counts/CTRs/co-click/SVD/repeat features together reach only 0.73; the chain's
0.82 comes from the trained choice factorisation (warm16 alone = 0.8015 plain on the day-6 cohort, 0.7946 on day 5)
and the experts built around it. The organisers' "solid baseline" must be a direct user-item factor model.

### Structural diagnostics (nxb_diag_svdk.py, nxb_diag_recency.py, nxb_diag_drift.py)
- Latent rank: history-profile cosine in an idf click-SVD space, day-6 cohort, plain AUC: k=4 0.557, 8 0.631, 12 0.697,
  16 0.724, 24 0.723, 32 0.709, 64 0.695, 128 0.671, 256 0.649 -> a sharp knee at 16: the simulator very likely uses
  16-dimensional user/item factors.
- Taste is stable: all 20 history clicks 0.724 > last 10 0.715 ~ oldest 10 0.713 > last 5 0.700 > last 2 0.671.
  No recency drift; more clicks = better estimate.
- No item drift: same-day leave-one-out item CTR AUC 0.520 (day 5) / 0.522 (day 6) ~ previous day 0.519 / 0.516 ~ all
  previous days 0.528. All 50k items are shown every day.
- Event click rate does not depend on the delay of the user's first event after a cutoff (0.627-0.649 for delays 0-4 days),
  so the 7-day spread of evaluation events carries no propensity signal.
- Caution found on the way: a warm start built from clicks < cut inflates "in-period" held-out scores (0.80-0.83 in-period
  vs 0.70-0.76 next day, even for users never trained on) because the SVD contains the held-out events' own clicks.
  Only next-day (out-of-time) cohorts are valid; all numbers below are out-of-time.

## Track B — choice/sequence model (nxb_b_seq.py; nxb_b_mf.py = failed dense-Adam variant)
All selection runs: train on events < day 5, validate on the day-5 first-per-user cohort (pre-cutoff for target 6).
| run | change | best val adj (plain) | epoch |
|---|---|---|---|
| smoke | dense Adam tables, 20% users, d=32 | 0.541 | 1 (memorises: loss 2.17 -> 1.34) |
| w1 | sparse Adagrad 0.045 + idf click-SVD warm start, d=16, attention + time/repeat/exposure terms | 0.7372 (0.7433) | 6 |
| w2noattn | time embedding zero-init fix, no attention | 0.7574 (0.7645) | 4 |
| m1/m2 (nxb_b_mf.py) | dense Adam, MAP L2 to zero / to the warm start | <= 0.747; stronger L2 0.59-0.72 | Adam turns tiny L2 gradients of rare rows into full steps |
| w4aux | + warm-aux loss on all ledger clicks (10 uniform negatives, popularity log-correction) | 0.7704 (0.7778) | 6 |
| w5cooc | item-item co-click SVD warm start instead | 0.7684 | 6 |
| w6plain | no time/repeat/exposure terms | 0.7727 (0.7802) | 6 |
| w7first | aux on first-event (warm-up) histories only, as warm16 | **0.7797 (0.7874)** | 12 (flat from 9) |
| w8lr | embedding lr 0.1 | 0.7763 | 4 |
| w9attn | w7first + DIN target attention | 0.7740 | 3, then overfits |
Reference: warm16 choice model (chain) on the same day-5 cohort: plain 0.7946.
Reading: (1) the converged bag choice model plateaus ~0.78 adjusted, 0.007 plain below warm16 (per-event SGD vs
batches / init details); (2) DIN attention over the timestamped history again fits training better and generalises worse
(0.774 vs 0.780), the same verdict as nx_seq_warm; (3) time-gap, repeat and exposure terms inside the model do not help.
| w10neg | w7first + bag of the previous 5 events' unclicked items (strictly earlier events) | 0.7738 | 9 |

### Track B candidate run bag1 (= w7first, 12 epochs fixed from the day-5 selection run)
- Day 6 (refit < day 6, 331 s): adjusted full 0.789126, selection 0.788991, plain 0.797852. Bins: 0 0.550, 1-4 0.716, 5-9 0.766,
  10-19 0.795, 20 0.816. Day 5 (fit < day 5, 291 s): adjusted 0.779666, plain 0.787391.
- Blend with nxa_h1, selection-subset grid (artifacts/nxb_b_bag1_report.json): rank 10% best: selection 0.818532 (+0.000201),
  full day 6 0.818266 (+0.000242), day 5 0.812019 (+0.000588). Logistic head [logit rank nxa_h1, logit rank bag1] fitted on day 5:
  selection +0.000109 (nxb_head_screen.py, artifacts/nxb_head_screen.json). Screen pass on day 6 only at the +0.0002 line;
  below the +0.0005 gate -> NOT queued. Likely mostly "another choice-model seed" (the chain already holds warm16).
- Track A day 5 (frozen 200 trees): adjusted 0.721584; bin 0 clean 0.583 vs nxa_h1 0.521 (197 events). Using the clean model only
  inside 0-history events: +0.000004 (selection). Not useful.

## Status at hand-back (2026-10-30)
Track A diagnostic complete (no candidate). Track B: converged bag choice model ~0.79 adjusted on day 6, attention does not
help, nothing queued. Open idea not run: exact (Newton / ALS-style) MAP estimation of the 16-dim choice factors, since the
rank-16 knee says the simulator is a 16-dim factor model and all our estimators are SGD approximations.

## Track C (approved 2026-10-30 ~12:40) — exact MAP 16-dim choice factors (C1) and joint click + exposure model (C2)
Script nxb_c_mf.py (self-contained, torch CPU, float64 accumulators). Objective = event softmax (10 candidates + no-click,
s = u.v + b, s0 = c0 [+ w_u]) + optional catalogue-softmax terms (ledger clicks with u; impressions with policy belief p)
+ Gaussian priors. Alternating exact per-user Newton (batched 17x17 solves, per-user backtracking), item-block Newton with the
diag(p)/catalogue curvature bound + global backtracking, 1-D Newton for c0. Warm start: idf click-SVD rank 16.
All runs below: phase 1 = fit on events < day 5, held-out = day-5 first-per-user cohort (pre-cutoff for target 6);
best round by held-out event log-loss (coordinator rule). Reference on the same cohort: bag1 0.7797 adj, warm16 0.7946 plain.

### C1 results (held-out day 5; ll = event log-loss per event, adj = adjusted AUC)
| run | settings | best round | ll | adj (plain) |
|---|---|---|---|---|
| c1test | events only, lu=lv=1 | 1 (then overfits) | 1.9474 | 0.6904 (0.7036) |
| c1 grid | events only, lu,lv in {3,10,30} | 1 | 2.06-2.11 | 0.537-0.612 (priors shrink v to 0) |
| c1h_test | + ledger catalogue term (first-event histories), lu=lv=1 | 1 | 1.8417 | 0.7600 (0.7711); AUC 0.7655 at round 2 |
| c1h_u3v3 | same, lu=lv=3 | 3 | 1.8452 | 0.7505 |
| c1h_u10v3 / u3v10 / u10v10 | stronger priors | - | 1.99-2.10 | 0.567-0.665 |
| c1h_all | ledger = all pre-cut clicks | 1 | 1.8729 | 0.7577 (double use of event clicks overfits) |
| c1h_anc10 | prior on v centred at the SVD warm start, lv=10, lu=1 | 2 | 1.8071 | 0.7661 (0.7769); converges 0.7666 |
| c1h_anc30 | anchor lv=30 | 4 | 1.8271 | 0.7557 |
| **c1h_anc10off** | anc10 + no-click user offset (lw=3) | 2 | **1.8001** | **0.7673 (0.7781)**; 0.7678 at round 4 |
| c1h_anc10u3 | anc10, lu=3 | 5 | 1.8602 | 0.7438 |
Reading: the exact MAP optimum of the u.v choice model is NOT better than the early-stopped SGD models: best C1
0.767 adjusted vs bag1 0.780 on the same held-out day. Zero-mean priors either overfit (lu,lv <= 1) or wipe out the item
geometry (>= 3); the click-SVD geometry has to be kept through an anchored prior; choice events alone carry too little
information for 50k x 16 item factors (events-only C1 = 0.69). The bag/SGD models win through the history-item table H
(a discriminatively learned item-item association) and implicit early-stopping shrinkage toward the SVD start.
Round time ~60 s (ledger term = exact softmax over 50k items for 54k users).

### C2 results (held-out day 5, base = c1h_anc10off settings)
| run | settings | best round | ll | adj | p.v alone adj |
|---|---|---|---|---|---|
| c2_e01 | + exposure catalogue term w=0.1, user prior centred on kappa*p (kappa refitted) | 2 | 1.8191 | 0.7677 (0.7700 at r4, ll explodes: kappa 1.6 -> 5.5, |u| 2 -> 4.3) | 0.738 (p pulled toward u) |
| c2_e01nk | exposure term w=0.1, no coupling (exposures inform v only) | 2 | 1.8037 | 0.7661 (C1: 1.8001 / 0.7673) | 0.570 |
Round time ~240 s (two exact catalogue terms). Exposures do not improve the shared item factors for click prediction.

### Rank check (C1 best settings, held-out day 5): rank 12 ll 1.8033 / adj 0.7657 (6 rounds); rank 16 1.8001 / 0.7673;
rank 20 1.8138 / 0.7633 -> 16 confirmed.

### Phase-2 results (refit < cut, scored cohorts; adjusted AUC)
| model | day-6 selection | full day 6 | day 5 | bins day 6 (0 / 1-4 / 5-9 / 10-19 / 20) |
|---|---|---|---|---|
| nxa_h1 | 0.818331 | 0.818023 | 0.811431 | 0.531 / 0.750 / 0.797 / 0.824 / 0.843 |
| V53 | 0.817528 | 0.817169 | 0.810911 | 0.522 / 0.747 / 0.796 / 0.823 / 0.842 |
| warm16 (chain; epochs picked on day 6 itself) | 0.791513 | 0.792469 | 0.786732 | 0.534 / 0.712 / 0.768 / 0.799 / 0.821 |
| bag1 (Track B) | 0.788991 | 0.789126 | 0.779666 | 0.550 / 0.716 / 0.766 / 0.795 / 0.816 |
| C1 c1best (2 rounds; day 5 re-selected on day 4 = 2) | 0.776895 | 0.777413 | 0.767346 | 0.508 / 0.663 / 0.745 / 0.787 / 0.814 |
| C2 c2best (2 rounds) | 0.775655 | 0.776066 | 0.766056 | 0.509 / 0.661 / 0.743 / 0.785 / 0.813 |
C2 extras on day 6: u.v alone 0.7701, p.v alone 0.5723; u shrunk toward kappa_hat*p (kappa_hat 1.41) for 0-4-click events,
alpha 0.5: +0.00012 adj (short-history AUC 0.6525 -> 0.6539) — negligible (nxb_c_post.py).
Laplace posterior-averaged scoring of C1 (nxb_c_laplace.py, 64 samples): day 6 0.777413 -> 0.777918 (+0.0005 standalone).

### Blends with nxa_h1 (grids on the day-6 SELECTION subset; artifacts/nxb_c_*_report.json, nxb_c_combo*_grid.json)
| blend | selection | full day 6 | day 5 |
|---|---|---|---|
| C1 rank 5% | 0.818560 (+0.000229) | 0.818297 (+0.000274) | 0.811857 (+0.000426) |
| C2 rank 5% | 0.818547 (+0.000217) | 0.818281 (+0.000258) | 0.811833 (+0.000402) |
| bag1 5% + C1 5% (best of the 3-expert grid; C2 weight 0) | 0.818620 (+0.000289) | 0.818379 (+0.000355) | 0.812106 (+0.000675) |
No blend clears +0.0005 on the day-6 selection subset -> nothing queued, no eval predictions built. No leakage flags
(standalones 0.77-0.79, blend steps <= +0.0007).

### Where the AUC comes from (nxb_diag_auc_split.py, nxa_h1): positives vs negatives of click events 0.822 (day 6) /
0.815 (day 5); vs candidates of no-click events 0.812 / 0.805; within-event AUC 0.816 / 0.808. Both the within-event order
and the event propensity are about equally (im)perfect; no single cheap lever.

## Choice-level stacking head (coordinator go-ahead; last factor-model step) — DESIGN FROZEN 2026-10-30 15:40 before any score
- warm16 DROPPED: its evaluation predictions come from a different script (imp2_finalize.py) than its day-5/6 predictions
  (imp2_warm_choice.py), and its day-6 epochs were chosen on day 6 itself.
- Experts (per row): nxa_h1 (days 5/6 'pred' = V53 parent; evaluation fit uses day-6 'pred_v39' and nxa_h1_eval_pred_v39.npy),
  bag1 raw, C1 (c1best) raw, C2 (c2best) u.v. Transform x_k = logit(rank_k / (n+1)) within each scored set, then standardised with
  the fit set's mean/std.
- Conditional logit over 10 candidates + no-click: s_j = sum_k beta_k x_kj ; s_0 = sum_b gamma_b [history bin b] (5 bins,
  unpenalised) + sum_k (g_max_k * max_j x_kj + g_mean_k * mean_j x_kj). Prediction = choice probability p_j.
- Loss: eval-mix 5-bin weighted multinomial NLL (event weights) + lambda * (|beta|^2 + |g_max|^2 + |g_mean|^2);
  lambda grid (6 settings) {0.01, 0.1, 1, 10, 100, 1000}; L-BFGS float64, 500 iterations, deterministic.
- Fit on full day 5 -> score day 6; lambda chosen on the day-6 SELECTION subset only (grid saved). Day 5 by 2-fold user cross-fit
  (seed 9901, same split as nxa_h1). Evaluation: refit on full day 6 (V39-parent nxa_h1), apply to evaluation rows.
- Diagnostic only (not used for selection): control head with nxa_h1 alone (+ its event features).
- Gate: >= +0.0005 vs nxa_h1 on day-6 selection AND day 5, else nothing queued.
- Caveat: bag1 / C1 / C2 settings (epochs, priors, rounds) were chosen on the day-5 held-out day (pre-cutoff day of the day-6
  pipeline), so day-5 head scores are not fully independent.

### Stacking head result (nxb_c_head.py; grid artifacts/nxb_c_head_grid.json) — FAILS, nothing queued
- Selection-subset grid (lambda 0.01 / 0.1 / 1 / 10 / 100 / 1000): 0.817049 / 0.817049 / 0.817050 / 0.817062 / 0.817108 / 0.815823
  -> lambda 100. Head (fit day 5): day-6 selection 0.817108 (-0.001223 vs nxa_h1 0.818331), full day 6 0.816607 (-0.001417),
  day 5 (2-fold user cross-fit) 0.809585 (-0.001846). Candidate coefficients (standardised): nxa_h1 0.93, bag1 0.05, C1 -0.05,
  C2 u.v 0.30.
- Diagnostic control (same head, nxa_h1 alone + its event max/mean): selection 0.815949 (-0.002381), day 5 0.807772 (-0.003659).
  The stack is +0.0012 above its own control, but the conditional-logit form itself costs ~0.0024: renormalising the scores
  within each event (with a no-click option driven by history bin and event max/mean) ranks events against each other worse than
  nxa_h1's row-level scores do. V39-parent variant: 0.816657 vs 0.817769.
- No evaluation predictions built (rule: only before queueing). Design was frozen before scoring and was not modified afterwards.

# Round D (prefix nxd_), 2026-11-01
## D1 — conditional rank blend (nxd_d1_blend.py; grid artifacts/nxd_d1_grid.json)
- Verified the coordinator's item-frequency split (nxd_diag_itemfreq.py): nxa_h1 day-6 plain AUC by candidate pre-cut labelled
  clicks 0: 0.745 (1.2%), 1-2: 0.784 (8.6%), 3-9: 0.816 (46%), 10-29: 0.842 (36%), 30-99: 0.849 (7.7%); evaluation mix
  0.7 / 6.2 / 42 / 40 / 11%. Confirmed.
- Recipe: per cell (item clicks <3 / 3-9 / >=10 x history <10 / >=10) weights in {0, 0.05, 0.15} for bag1, C1, C2 u.v on top of
  rank(nxa_h1); coordinate ascent (2 passes) on the day-6 SELECTION subset; path saved.
- Result: selection 0.819036 (+0.000705), full day 6 0.818639 (+0.000615), day 5 0.812197 (+0.000766); V39 parent +0.000791 /
  +0.000702 / +0.000871; paired event bootstrap (400, seed 3) sel [+0.00040, +0.00097], full6 [+0.00036, +0.00086],
  day 5 [+0.00044, +0.00102]. Unconditional reference (same 27-point grid, one weight set): (bag1 0.05, C1 0, C2 0.05)
  +0.000394 / +0.000401 / +0.000748 -> the per-cell structure adds +0.0003 on the selection subset but only +0.00002 on day 5
  (mostly selection fit). No rare-item concentration of the weights; item-bin gains 0: +0.0012, 1-2: +0.0001, 3-9: +0.0012,
  10-29: +0.0005, 30+: +0.0000.
- Evaluation predictions built (bag1 / C1 / C2 at cutoff 1737936000, same code paths, fixed epochs/rounds) and the recipe
  applied to nxa_h1_eval_pred_v39.npy: artifacts/nxd_d1_eval.npy (Spearman 0.9971 vs nxa_h1, top item changed in 5.6% of events;
  day 6: 0.9968 / 6.0%).
- BLOCKER found in tst5_feedback.md (tester, 2026-10-30): the experts' CONFIGURATIONS (bag1 variants, C1/C2 priors, rank) were
  compared on the day-5 cohort (phase-1 validation of target 6), so day 5 is not an independent gate day. Fix before queueing:
  re-select every close configuration on phase 1 of --target 5 (train < day 4, validate on day-4 first events; touches neither
  scored cohort), and epochs/rounds per target on day d-1. If the same configurations win on day 4, the day-5 outputs are
  unchanged and valid; otherwise recompute them. Runner: nxd_sel4.sh (15 selection runs, ~1.5 h).
