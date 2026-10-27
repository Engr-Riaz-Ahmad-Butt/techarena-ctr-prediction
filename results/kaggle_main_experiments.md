# Kaggle GPU main experiments

The same 354 causal features, day6 validation rows, temporal training cutoffs,
and fixed 90% V53 / 10% XGBoost rank blend are used throughout. Validation
labels are used for scoring, never for training these experts. These dates
have been reused: gains remain exploratory and are not official forecasts.

## ctr-main-001 (completed)

Observed directly in the Kaggle notebook output, saved as Kaggle version 1.
XGBoost 3.4.1, CUDA, depth 6, eta 0.05, 700 rounds.

- Ordinary AUC: 0.8254490348942001
- History-adjusted AUC: 0.8172087737049849
- Baseline adjusted AUC: 0.8171690462350284
- Adjusted gain: 0.000039727469956507555
- Event-influence gain interval: [-0.000028482651526532624, 0.00010793759143954773]
- Gate failed; no day5 confirmation, promotion, or competition CSV.

## ctr-main-002-shallow (completed, rejected)

Hypothesis: a shallower, more regularized expert may generalize better and
complement V53. Fixed before scoring: depth 4, eta 0.035, child weight 150,
lambda 15, alpha 0.5, 1600 rounds. Other settings unchanged. Recipe and model
are stored in a separate output directory. Source: TechArena_CTR_Kaggle_002.ipynb.

- Ordinary AUC: 0.8254157970720297
- History-adjusted AUC: 0.8171696010974321
- Adjusted gain interval: [-0.00006404680222545267, 0.00006515652703283259]
- Gate failed; ordinary AUC declined. V53 remains accepted.

If rejected, one bounded alternative will test greater interaction capacity:
depth 8, eta 0.035, child weight 100, lambda 15, alpha 0.5, 1000 rounds,
under ctr-main-003-interactions. No tuning of blend weight on day6.

The interaction alternative was started after the shallow model was rejected.

## ctr-main-003-interactions (both dates passed; replay passed)

- Ordinary AUC: 0.8254950475978616
- History-adjusted AUC: 0.8172634369364146
- Gain over V53: 0.00009439070138617112
- Event-influence gain interval: [0.00002342238657198313, 0.0001653590162003591]
- Day6 gate passed. Unchanged day5 ordinary AUC: 0.8181984074982497.
- Day5 history-adjusted AUC: 0.81104388145966, baseline 0.8109113853236557.
- Day5 adjusted gain interval: [0.00005959869323181952, 0.00020539357877678226].
- Both dates passed. Saved-model reload reproduced every raw/blended prediction
  exactly on 351380 day6 rows and 352300 day5 rows.
- Day6 model SHA256: 3c2565c29915d21ca34d70c246afb3d305f36c463ffed68f1457c1928d2b09d4.
- Day5 model SHA256: 6fb7c7f602b5ad8c0224c394c83a09a85680ea21b8c4e73fadd1b83ed679d270.
- This is an eligible candidate with a small gain; V53 remains the stored accepted
  baseline pending broader review. Notebook code cells compile locally and the
  complete experiment and replay executed successfully on Kaggle CUDA.
- Still below the 0.82 adjusted threshold; no export or official claim.

Any passing candidate must also pass the runner's unchanged day5 confirmation.
Neither an ordinary AUC above 0.82 nor a tiny unsupported gain is sufficient to
claim the requested adjusted score or a leaderboard improvement.
