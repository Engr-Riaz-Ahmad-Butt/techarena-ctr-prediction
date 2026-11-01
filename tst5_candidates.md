CURRENT BEST: nxa_h1 / submission_v41_candidate.csv (adjusted AUC 0.818023 full day 6, 0.818331 day-6 selection, 0.811431 day 5; unchanged since 2026-10-29 22:15)

# tst5 candidate verdicts (tester, protocol_v4 round)

Baseline = nxa_h1 (`artifacts/nxa_h1_day{5,6}.npz` key `pred`; exportable version `pred_v39`; evaluation `artifacts/nxa_h1_eval_pred_v39.npy`).
Metric: adjusted AUC = weighted ROC AUC with history-bin weights (0 / 1-4 / 5-9 / 10-19 / 20 clicks, evaluation mix): project 5-bin code on full day 6 and day 5, selection-file weights on the day-6 selection subset (24,422 events). Plain AUC reported alongside.
Gate: ACCEPTED only if >= +0.0005 adjusted vs nxa_h1 on the day-6 selection subset AND on day 5, the day-6 selection paired 95% interval above zero, selection made on the selection subset only with the grid saved, then non-negative vs nxa_h1 on the fresh holdout (scored only after both days pass). SCREEN PASS = >= +0.0002 on day 6 selection. LEAKAGE ALERT = standalone model above 0.83 adjusted, or any blend step above +0.003.
Paired event bootstrap: 1000 resamples, seed 3, one shared set of resamples per cohort (same draw sequence as tst4). Evaluator: `tst5_candidate_eval.py`.

## References reproduced (2026-10-30 10:45, from saved arrays)

| Model | Day 6 full adj | Day 6 selection adj | Day 5 adj | Day 6 full plain | Day 5 plain |
|---|---:|---:|---:|---:|---:|
| V53 | 0.817169046 | 0.817527526 | 0.810911385 | 0.825423332 | 0.818091384 |
| nxa_h1 (`pred`, V53 parent) | 0.818023495 | 0.818330578 | 0.811430886 | 0.826186215 | 0.818595649 |
| nxa_h1 (`pred_v39`, exportable) | 0.817480929 | 0.817768856 | 0.811017168 | 0.825686507 | 0.818149046 |
| V39 | 0.816661488 | 0.816993547 | 0.810526600 | 0.824957808 | 0.817671476 |

All match the briefing references to the digit (V53 0.817169 / 0.817528 / 0.810911; nxa_h1 0.818023 / 0.811431).
Evaluator self-test: running `tst5_candidate_eval.py` on the old nxa_expo arrays reproduces tst4's stack-increment intervals exactly (sign flipped, since nxa_h1 is now the base): day 6 full -0.000213156 [-0.000364168, -0.000067321], day 5 -0.000033166 [-0.000196758, +0.000116531]. One evaluation takes about 20 minutes (1000 resamples, 7-8 arms, 3 cohorts).

## Track A diagnostic (nxb_a_a1 / a2): numbers VERIFIED; the comparison is fair as a measurement but narrower than the brief (2026-10-30 12:30)

Source: the `nxb_journal.md` entry "Track A result a1" (about 11:10 to 12:15). Recomputed from the saved arrays `artifacts/nxb_a_a1_day6.npz`, `nxb_a_a2_day6.npz`, `nxb_a_feat_k6.npy` and `features/imp3_wf_val{5,6}.npy` with `tst5_trackA_verify.py`; output in `tst5_trackA_verify.json`. Nothing was refitted.

| Claim in the journal | Tester recomputation | Match |
|---|---|---|
| a1 standalone day 6: full adjusted 0.730629, selection adjusted 0.730864, plain 0.736411 | 0.730628818 / 0.730863548 / 0.736411363 | yes |
| a2 selection adjusted 0.730777 | 0.730777404 (full 0.730573, plain 0.736361) | yes |
| Plain AUC by history bin, clean vs nxa_h1: 0 0.5625 vs 0.5309; 1-4 0.6853 vs 0.7496; 5-9 0.7140 vs 0.7966; 10-19 0.7344 vs 0.8242; 20 0.7484 vs 0.8429 | identical to 4 decimals | yes |
| Best blend with nxa_h1 = rank 2%, +0.000008 (grid on the selection subset) | grid reproduced exactly; rank 0.02 gives +0.0000083, paired 95% interval [-0.000054, +0.000069] on the selection subset, and +0.0000057 [-0.000045, +0.000057] on full day 6 | yes (no gain) |
| Single features, plain AUC on day 6: SVD history cosine 0.710, its within-event centred version 0.660, co-click sum 0.602, exposure cosine 0.576, candidate in history 0.551, user event rate 0.544, item CTR 0.525 | 0.7096, 0.6595 (`mf_h_dmean`), 0.6023, 0.5761 (NaN rows dropped; 0.5735 if NaN is set to -1), 0.5514, 0.5441, 0.5251 | yes |
| warm16 alone: plain 0.8015 on day 6, 0.7946 on day 5 | 0.801546 / 0.794615 (cached imp3_wf column 0) | yes |

Time safety of the features: see `tst5_feedback.md`. Every label statistic, ledger click, SVD and exposure count uses `train[:searchsorted(ts, c)]` with an assert that all timestamps are < c. Each block has its own cutoff. The tree count was chosen on the selection subset and frozen (`nxb_a_a1_frozen.json`, 11:05). The model is weak, not leaky: 0.73 is nowhere near the 0.83 alert level.

Is the comparison with the chain fair?
- **As a measurement, yes.** Both models are scored on the same 35,138-event cohort, with the same history-bin weights and the same selection subset.
  - Selection optimism exists on both sides but is tiny next to the gap. Track A picked one number (the tree count; 100 to 300 trees all lie within 0.0002). The chain went through about 60 versions, most selected on full day 6, which is worth perhaps 0.001 to 0.003.
  - The 0.087 gap is real.
- **As an answer to the brief, only partly.** The brief asked for a clean model that includes "one factor model (ALS/SVD on pre-cutoff clicks)". a1 has only unsupervised idf-SVD cosines. Its own diagnostic shows rank 32 sits past the rank-16 knee. a1 has no TRAINED user-item factor or choice model.
  - The chain's lead comes almost entirely from that missing ingredient: warm16 alone (0.8015 plain) beats the whole 65-feature tree (0.7364 plain) by 0.065.
  - What Track A establishes: counts, CTRs, co-click, repeat, recency and exposure features do not contain anything the chain is missing. The blend adds +0.000008 and the Spearman correlation with nxa_h1 is 0.655. The chain is not losing signal in any history bin.
  - What Track A does not establish: whether a better-trained factor model would beat the chain's warm16. That is Track B's question.
- **Smaller asymmetries, none decisive:**
  1. a1 trains on first-per-user events only (1.40 M rows). `--rows all` was not tried.
  2. No day-5 Track A number exists yet. The chain's day-5 score uses a same-day user cross-fit, so a day-5 comparison would slightly favour the chain.
  3. The 0-history bin (clean 0.5625 vs nxa_h1 0.5309) rests on 198 events. Its plain-AUC standard error is about 0.025, so that difference is noise, not evidence of a short-history weakness.
- **Supports the chain's integrity.** An independent clean-room choice model (Track B w7first, 0.7874 plain on day 5, read from its training log `artifacts/nxb_b_w7first_t6.log`; no saved predictions to recompute from) comes within 0.007 of warm16 (0.7946). That is consistent with warm16's strength being genuine, not leaked.

Verdict: Track A is a diagnostic, not a candidate. Its numbers are confirmed. Its conclusion that "nothing simple is missing" is supported. Its conclusion that "the organisers' baseline must be a direct user-item factor model" is a reasonable inference but untested. CURRENT BEST is unchanged.

## nxb_b_bag1, rank 10% blend with nxa_h1: SCREEN PASS only (not queued by the improver; recomputed for the record) (2026-10-30 12:50)

What it is: a Track B clean-room choice model. It is a bag model: 16-dimensional user and item embeddings plus a mean-pooled history embedding, no attention, 11-way softmax. It is warm-started from a pre-cutoff idf click-SVD, with an auxiliary loss on first-event warm-up clicks. It was trained for a fixed 12 epochs. Script `nxb_b_seq.py` (`--name bag1 --aux 1 --aux-src first --no-extras --no-attn --fixed-epochs 12`). Arrays: `artifacts/nxb_b_bag1_day{6,5}.npz`. The blend recipe `0.9*rank(nxa_h1) + 0.1*rank(bag1)` was picked from the improver's selection-subset grid (`artifacts/nxb_b_bag1_report.json`). I rebuilt the blend myself, on V53 (`pred`) and on V39 (`pred_v39`), in my scratchpad, and scored it with `tst5_candidate_eval.py`; output in `tst5_nxb_b_bag1_eval.json`.

| Cohort | bag1 alone, adjusted | Blend, adjusted | Gain vs nxa_h1 [paired 95%] | Gain vs V53 | Exportable: blend on V39 vs nxa_h1 on V39 |
|---|---:|---:|---|---:|---|
| Day 6 full | 0.789126 | 0.818266 | +0.000242 [+0.000066, +0.000419] | +0.001096 | +0.000322 [+0.000140, +0.000500] |
| Day 6 selection | 0.788991 | 0.818532 | **+0.000201 [-0.000001, +0.000423]** | +0.001004 | +0.000281 [+0.000074, +0.000506] |
| Day 5 | 0.779666 | 0.812019 | +0.000588 [+0.000401, +0.000777] | +0.001108 | +0.000666 [+0.000480, +0.000859] |

The improver's numbers are confirmed to the digit (standalone 0.789126 / 0.788991 / 0.779666; blend +0.000201 selection, +0.000242 full, +0.000588 day 5). Plain-AUC gains vs nxa_h1: +0.000196 (day 6 full), +0.000134 (selection), +0.000561 (day 5). Spearman correlation of the blend with nxa_h1 is 0.9987; of bag1 with V53, 0.869.

Audit:
- Time-safe. Training events, warm-start SVD, item and user biases and auxiliary pairs all come from `train[:searchsorted(ts, c)]`. The auxiliary pairs come from first-event histories of events before the cut.
- Standalone 0.789 is far below the 0.83 alert level, and no blend step exceeds +0.003.
- Caveat: `nxb_b_seq.py` was edited at 12:26, after the bag1 fits (12:20 and 12:25). The file on disk is therefore not byte-identical to the code that produced the arrays. The logged arguments match options still present in the file.
- **Day 5 is not an independent check.** The configuration (w7first, best of about 12 variants) and the epoch count (12) were both chosen on phase-1 `val_adj` of the 35,230-event day-5 cohort. That was a model trained before the day-5 cutoff. The bag1 day-5 run is the same model with the same score (0.779666 = w7first epoch 12). So the day-5 +0.000588 is in-sample for the selection. See NEEDS-FIX 1 in `tst5_feedback.md`.

**Decision: SCREEN PASS only, not accepted.**
- The day-6 selection gain (+0.000201) is 0.0005 short of the gate, and its interval touches zero.
- The day-5 number cannot certify the result because day 5 was the selection cohort.
- The fresh holdout was not scored.
- The improver's own reading is reasonable: the gain comes mostly from adding a second, weaker choice model, which acts much like another seed alongside warm16. It is real on full day 6 (interval above zero), but it is far too small to be worth a submission.
