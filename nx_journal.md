# nx_ journal — "find one new signal" round (improver), started 2026-10-29

Ownership: every section is by the Idea-A improver (this session) unless marked [NOT MINE]. From 20:30 my new files use the prefix
`nxa_`; my earlier files are nx_e_slate_short.py, nx_seq_core.py, nx_seq_warm.py, nx_userknn.py, nx_headscreen.py, nx_exposure.py
(the 19:26 run of nx_exposure.py and nx_seq2*, nx_event_mass*, nx_f6_trend* were made by another session).

Parent predictions: V53 = `artifacts/v53_causal_propensity_day{5,6}.npz` key `pred` (day 5 = 352,300 rows, day 6 = 351,380 rows);
V39 = `artifacts/v39_regularized_als32_day{5,6}.npz` key `pred`, evaluation `artifacts/v39_final_pred.npy`.
Adjusted AUC = project `weighted_auc` (5 history bins to the evaluation mix). Screen rule: +0.0002 on day 6; gate: +0.0005 both days.
Facts used for design: histories are the last 20 clicks before the event and include in-window clicks (99.6% of earlier
in-window clicks re-appear in later histories with the same timestamp), so a per-user causal click ledger (history entries +
pre-cutoff clicks, unique by user/item/time) is the full click sequence; all cohort/evaluation history stamps are < cutoff.

## Idea D — research step (delegated, done)
`nx_research.md`: 11 families, top-5 = item lifecycle/trend + same-day exposure share, full-sequence SASRec retriever with
sampled softmax, multi-anchor time-decayed covisitation, explicit logging-policy/exposure propensity model, user-user session kNN.
The last four map onto ideas A, B, C of this round; the first is a new own idea (queued as idea F if time permits).

## Idea E — short-history slate test (cheap, saved artifacts only) — REJECTED
Plan: re-score `artifacts/v58_slate_residual_day6.npz` (slate arm vs matched zero-slate control, epoch 4, 10% rank blend) per
history bin; hypothesis: the other nine candidates carry taste for cold users (0–4 clicks) even if the overall gain was +0.00003.
Result (day 6, ordinary AUC within bin): bin 0 (198 events) slate raw 0.513 vs control raw 0.501, blends 0.520 vs 0.518 vs V53
0.522; bin 1–4 (2,357 events) slate raw 0.696 vs control 0.695, blend 0.74724 = V53 0.74724; bin 0–4 blend minus control = 0.0.
Replacing V53 only on 0–4 bins by the slate blend changes the full adjusted AUC by −0.00002 (w=0.1), −0.00012 (0.25), −0.00091 (0.5).
Decision: the slate residual carries no cold-user signal beyond its control. Closed. Script `nx_e_slate_short.py`,
output `results/nx_e_slate_short.json`. Runtime 20 s.

## Idea A — full-capacity sequential recommender (SASRec-style, pretrained on all click sequences) — plan
1. Hypothesis: V53's experts see the user's clicks only as bags (SVD/ALS profiles, sqrt-scaled history sums, mean-pooled tied
   vectors); order, recency structure and item-to-item transitions beyond exact pairs are unused. A causal transformer over the
   full click ledger (not the 20-window) with a sampled-softmax next-click objective learns transition structure from ~3M click
   positions instead of 590k slates, which is the capacity/pretraining gap the earlier residual models did not close.
2. Why ≥ +0.002 is plausible: the warm16 choice model (same 10-way + no-click objective, no sequence encoder) is 0.8015 standalone
   and was the only +0.01 signal in the project; tied-history full-ledger factors gave +0.0004 with a mean pool; OTTO/KDD'23
   winners report the sequence-model recall as their largest single ablation. A standalone ≥ 0.80 that is less correlated with
   V53 than the choice model should blend for ≥ +0.002.
3. Model: item table (50k×d, tied input/output), per-element time-gap embedding (log2 minutes to the next element), reversed
   position embedding, 2 causal transformer blocks (2 heads, FFN 2d, dropout), a final QUERY token carrying the gap from the last
   click to the scoring time; state h. Scores: candidate c → h·v_c + b_c; no-click → c0 + w·h; 11-way softmax (event loss).
   Pretraining/regulariser: next-click sampled softmax at every ledger position, 256 shared popularity-sampled negatives
   with logQ correction. Single code path, cutoff parameter: ledger = events with t < cutoff; training events t < cutoff;
   scoring sequence = ledger clicks (< cutoff) ∪ the event's supplied history, last L.
4. Screen (< 20 min CPU): d=32, L=30, 1 pretrain epoch + 1 joint epoch on the day-6 cutoff; standalone adjusted AUC on the
   full day-6 cohort and rank-blend gain vs V53 at w ∈ {0.05, 0.1, 0.2}. Continue only if any blend gain ≥ +0.0002.
5. Full: d=64, L=50, epochs chosen on day 6 (fixed seed 7301), applied unchanged to day 5 and evaluation (cutoffs
   1737763200 / 1737849600 / 1737936000); outputs `nx_seq_day{5,6}.npz` (y, events, pred, raw, baseline) + `nx_seq_eval.npy`.
6. Cost: ledger build ~1 min; ~3–4 min per joint epoch at d=64 on 14 cores; RAM < 1.5 GB with mmap.
7. Kill criteria: standalone day-6 adjusted AUC < 0.76 after the screen, or blend gain < +0.0002.

### Idea A — screen 1 (nx_seq_screen_day6.log, 7.6 min): optimisation failure, not a verdict
d=32, L=30, 1 pretrain + 1 event epoch, Adam lr 1e-3 on everything: raw day-6 AUC 0.585, adjusted 0.583, next-click loss flat,
event loss 2.087 vs 2.11 for a constant model. A smoke overfit (3,000 events, 8 epochs) drove the event loss to 1.02, so the
gradient path works; the sparse embedding rows simply move ~1e-3 per occurrence at that learning rate (the choice model uses
Adagrad at 0.045). Fix: separate lr 1e-2 for the item/bias/gap/position/user tables, init std 0.1, and a per-user vector
added to the QUERY token (the choice model's user vector learns from exposed-but-unclicked items, which a click sequence
never shows). Smoke re-check: next-click loss 5.99 → 5.58 over 4 epochs on 500 windows. Screen 2 = 2 pretrain + 2 event epochs.

## [NOT MINE — written by another session; read-only for me, numbers not re-run by me] Ideas F, C, A — outcomes (2026-10-29, see results_log.md)
- F (item trend + transductive cohort exposure): rejected, matched gain +0.00024 / −0.00032.
- Slates are personalized (history items 52× over popularity), but exposure SVD adds only +0.0003 (day 6, 5% blend): screen pass, below gate.
- A (SASRec screen): standalone 0.583, killed; likely underfit/bug since repeat flag alone ≈ 0.60.
- Local↔official calibration: V35 local adjusted 0.816452 vs official 0.816623; V39 local 0.816661 (never uploaded).
- G (event-mass correction): rejected. V53 calibrated event AUC is 0.681 (the 0.563 figure was a rank-sum artefact); new event model 0.633 adds nothing (−0.00014 day 6).

### Idea A — screen 2 (nx_seq_screen2_day6.log): undertrained from scratch
Same model with the fixes, 2 pretrain + 2 event epochs: raw day-6 adjusted AUC 0.573 → 0.652 (epoch 1 → 2), best V53 blend
−0.00043 (w 0.05). The warm16 choice model reaches raw 0.787 after ONE epoch (imp2_warm16.log) because it starts from the
co-click SVD/profile initialisation. Ledger facts that weaken the premise: the ledger has 18.5 clicks per user on average
(median 14, p90 38), i.e. the supplied 20-click window already covers most of each user's sequence; "pretraining on longer
sequences" therefore adds little data. Screen 3 = matched test of the only remaining question, whether an attention encoder
adds anything over the bag: identical warm-started choice model (init32 at cutoff day−2, Adagrad 0.045 on tables) with
--blocks 0 (bag control) vs --blocks 2 (zero-initialised transformer path), 2 epochs each (`nx_seq_warm.py`).

## Idea B — user-based neighbourhood CF — plan (10 lines)
1. Hypothesis: a local user-user kernel (users who clicked what this user clicked) captures high-rank taste that the rank-16/32
   factor models smooth away; never tested (only item-KNN was).
2. Worth ≥ +0.002? Only if its standalone is ≥ 0.70 and weakly correlated with ALS; cost is minutes, so worth one screen.
3. Build: ledger(cut) click matrix, IDF-cosine; query = ledger ∪ the event's own history; self excluded; top-K=100 features
   (click, recent-2-day, decayed, neighbour count, exposure, CTR) and all-user scores with similarity powers 1/2/4
   (click, popularity lift, exposure, neighbour CTR). Screen: standalone + V53 rank blends; then a 2-fold user cross-fit
   logistic head [logit V53, features, features × logit V53] vs matched control (`nx_headscreen.py`).
Results (day 6, full cohort): top-K features non-zero on 7% of rows, raw adjusted ≤ 0.568; all-user ucf scores non-zero on
34% of rows, raw adjusted 0.606 (best: ucf1_lift); every rank blend negative (best −0.000065 at w 0.02). Head screen with 10
features: +0.000042 vs control (blend), LR alone −0.00015. Short-history (0–4) raw AUC 0.578 vs V53 0.738.
Decision: REJECT (far below +0.0002). Scripts `nx_userknn.py`, `nx_headscreen.py`; reports
`artifacts/nx_userknn_screen{,2}_day6_report.json`. Screen 1 had a bug (recent-click matrix kept explicit zeros → identical
to the click matrix); fixed in screen 2, no effect on the decision.

## Idea C — exposure-policy expert (mine, nxa_ files) — QUEUED as nxa_expo
Plan (10 lines): (1) slates are chosen by the recommender from its belief about the user → exposures are label-free taste evidence;
(2) worth ≥ +0.002? unlikely alone (raw 0.57), but it is new information for short histories and costs minutes; (3) SVD of the
IDF-log exposure matrix (< cutoff), candidate vs user exposure profile cosine; (4) screen on day 6 (my re-run reproduced the other
session's +0.0003 exactly); (5) variants on day 6 only: k 16/32/64, uniform vs decayed vs last-2-day profile, weighting idf-log /
binary / log, co-exposure space for click histories; (6) recipe forms: global rank blend, within-event centred rank or z.
Chosen on day 6: expert = mean rank(typ16_all, typ32_all, cos32), pred = rank(parent) + 0.015·z(within-event centred expert).
Day 6 +0.000641 (V39 parent +0.000622), day 5 unchanged recipe +0.000486 (V39 +0.000474); my paired 95% intervals exclude 0 on
both days; every history bin improves. Files: nxa_exposure2.py, nxa_expo_final.py, artifacts/nxa_expo_day{5,6}.npz,
artifacts/nxa_expo_eval.npy, artifacts/nxa_expo_eval_pred_v39.npy, artifacts/nxa_expo_report.json.

## F1 — time-decayed multi-anchor covisitation + exposure-conditioned pair CTR (mine, nxa_covis.py) — REJECTED
Plan: (1) closed ItemKNN used whole-prefix co-click cosine and transitions used last-item counts; untested were time-local
co-clicks (≤ 24 h, exp(−gap/6 h)), all-20-anchor recency-rank weighting, degree normalisation, and an exposure-conditioned pair
CTR (when c was shown to OTHER users whose history held h, how often was c clicked; lift over item CTR, own events removed);
(2) cost ~3 min, < 1 GB; (3) day-6 screen: global rank blend, within-event z blend, cross-fitted head.
Result (day 6): R_time non-zero on 9.4% of rows, raw adjusted 0.556; pair-CTR lift raw 0.534–0.560; every global and
within-event blend negative (best −0.000067); head screen −0.000005 vs control. Decision: reject. Runtime 181 s.

### Idea A — screen 3 (matched warm pair, nx_seq_warm.py) — CLOSED by the hard stop
Identical warm-started choice model (init32 at cutoff(day−2), Adagrad 0.045 tables, next-click aux 0.2, d=32, L=30), 2 epochs:
- bag control (--blocks 0, 518 s): raw day-6 adjusted 0.7512 → 0.7455 (ep 1 → 2); best V53 blend −0.000055 / −0.000092 (w 0.05).
- transformer (--blocks 2, zero-initialised path, 1,072 s): train event loss 1.55 → 1.34 (control 1.81 → 1.74) but raw adjusted
  0.7389 → 0.7327; best V53 blend −0.000213 / −0.000245.
The attention encoder fits the training events better and generalises worse: treatment minus control in the V53 blend
= −0.00016 (ep 1), −0.00015 (ep 2), far from the +0.0002 required. Neither arm reaches the warm16 choice model (0.8015).
Decision: Idea A closed (sequence order/attention over the click ledger adds nothing over the bag; the ledger holds only
18.5 clicks per user, so there is no extra sequence data to pretrain on). No Kaggle request.

### Idea C — steps 4–5 (day 6 only, no change to the queued recipe)
- Exposure profile in OTHER item spaces (nxa_exposure4.py): click-trained taste space (preference_stats, k 16/32/126) raw 0.58–0.59
  but within-event blends NEGATIVE (−0.00014 to −0.00083); warm16 choice-model space −0.0005 to −0.0010; popular-direction contrast
  in the exposure space ctr32 +0.000614 (≈ E3), E3+ctr16 +0.000655 (+0.000014 over E3, not worth a recipe change).
- Co-slate PMI-SVD item space (nxa_exposure5.py, 26.5 M co-slate pairs): raw 0.533–0.539, within-event +0.00009 at best, dilutes E3.
Reading: the gain is specific to the user×item exposure geometry (who is shown what), not to click geometry nor to within-slate
co-occurrence; the recommender's per-user targeting carries the information.

## C step 2 — exportable head nxa_h1 (mine) — QUEUED
Tester's NOT-ACCEPTED verdict on nxa_expo (day 5 +0.000486) + its advice (select on the day-6 selection subset only, save the grid,
freeze before day 5). Tried, all on the selection subset: new features (one-step policy MLE, exposure-space kNN CTR, choice-model
catalogue rank = research F2; artifacts/nxa_c2_feature_screen_sel6.json) — none adds cleanly to E3 (≤ +0.00006). Then a head:
LR [logit(rank(parent)), E3z] fitted on the previous day's labels: selection +0.000803 (vs +0.000581 for the z blend). F2/F3 extras
equal the no-exposure segment-recalibration control S0 (+0.000178), so F1 was frozen (21:37) before day 5 was touched.
Result: V53 parent day 6 +0.000854, day 5 +0.000520; V39 parent +0.000819 / +0.000491. Iterative policy MLE (eta 8) diverged
(norm inf) and was replaced by the closed-form one-step Newton estimate; logged here for the record.

## Phase 2 (target 0.8200 on full day 6; baseline = nxa_h1) — STOP RULE TRIGGERED
### Idea 1 — exposure columns inside the main tree (nxa_tree_feats.py, nxa_tree.py, nxa_tree_screen_eval.py)
8 label-free columns (typ16, typ32, cos32, n_expo, within-event centred typ16/typ32/cos32, within-event rank of typ32) for every
cached block (day2–6 cut at block start, val5/val6, eval cut 7); val5/val6 columns equal the nxa_h1 cohort features exactly.
Re-assembled 354-column val6 matrix replays imp19_tree_d6.txt with max diff 0.0 (layout verified). Matched screen: imp19 settings,
735 trees, days 3–5 → day 6, same fixed 20% of training events (seed 7401), control 354 vs treatment 362 columns, 4.6 min each.
Selection subset: raw tree adjusted 0.810576 → 0.811066 (+0.00049), but on top of nxa_h1: rank blends +0.00001/+0.00004/+0.00007
(w 0.1/0.2/0.3), cross-fitted head −0.00004 vs the control tree. The tree only re-learns what E3z already gives nxa_h1. FAIL (< +0.0002).
### Idea 2 — richer exposure features (nxa_expo_rich.py, nxa_rich_screen.py)
Item audience cos, exposure-minus-click contrast, click history in the exposure space, typicality percentile within the user's own
impressions, last-2-day exposure profile and count; nxa_h1 head refit on day 5, scored on the selection subset: every single feature
−0.00005 … +0.00002, all six together −0.000017. FAIL.
Ideas 1 + 2 together add ≈ 0 over nxa_h1 (needed +0.0008): stop. Idea 3 (explicit policy model) not run; its closest cheap proxy
(one-step policy MLE, pol_cos) already added only +0.00003 to E3 in the earlier selection screen.
