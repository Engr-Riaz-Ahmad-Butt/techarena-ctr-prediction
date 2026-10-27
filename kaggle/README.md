# Run TechArena experiments on Kaggle

The prepared package runs new CPU/GPU XGBoost experiments over the existing causal feature caches. Main mode compares against V53 on the same day6 cohort, then confirms unchanged on day5 only after a positive gate. Weekday mode is a different diagnostic benchmark. Neither mode creates a competition submission CSV or promises a score.

## Files to use

- `TechArena_CTR_Kaggle.ipynb`: import this notebook into Kaggle.
- `artifacts/kaggle_ready/techarena_kaggle_data.zip`: upload as a private Kaggle Dataset.

The ZIP contains only the required prepared features, labels, baseline predictions, metadata and portable runner. Raw training/evaluation CSVs, Git history, credentials and unrelated experiments are excluded. Checksums detect incomplete or modified inputs. The ZIP is intentionally excluded from Git.

## Steps

1. In Kaggle, create a new **private Dataset**, named for example `techarena-ctr-data`. Upload `techarena_kaggle_data.zip` and finish creating it.
2. Create a Notebook and use **Import Notebook** to open `TechArena_CTR_Kaggle.ipynb`.
3. Use **Add Input / Add Data** to attach your dataset. The notebook locates it automatically; no path editing is normally needed.
4. In notebook settings choose a GPU accelerator if one is available. Otherwise CPU works. Leave `DEVICE = "auto"`. Enable Internet only if the dependency cell needs to install XGBoost >=3.0.
5. Select **Run All**. Keep `MODE = "main"` for comparison with the accepted local candidate. Do not change the fixed 700 rounds for the intended experiment.
6. Download the final `ctr-main-001-results.zip`, or save a notebook version with its outputs. Share `summary.json` and the day6/day5 `report.json` files for review.

No Kaggle API token is needed. If accelerator access is unavailable, the notebook can use CPU; faster hardware does not itself improve accuracy. The notebook prints its actual device and installed XGBoost version. Runtime depends on the allocated machine and data loading, so no fixed runtime is promised.

`history_adjusted_auc` is the main comparison metric. Ordinary AUC above0.82 does not satisfy the user's adjusted-AUC minimum. Weekday-mode metrics have a different population and weighting and cannot replace main results. GPU/library differences may change predictions. Dates have been repeatedly reused, and conditional intervals exclude model-selection uncertainty.

The results archive contains reports, diagnostic predictions and models, **not a competition CSV**. An accepted improvement still requires a full-data refit and verified `rowid,predicted_probability` export. The confirmed official result remains V35=0.816623; the best accepted local adjusted result is V53=0.8171690462.

## Rebuild and local verification

Run `python kaggle/build_bundle.py` once with the existing canonical artifacts available. It refuses to overwrite a completed archive. Run `python kaggle/smoke_test.py` for a small synthetic CPU test covering both main dates, model replay, corrupt input rejection and result archiving. This does not execute a Kaggle session or verify GPU hardware.

Kaggle resources attach as notebook inputs; see the [official Kaggle integration documentation](https://github.com/Kaggle/kagglehub#readme). No cloud upload or notebook execution is performed by the local packaging scripts.
