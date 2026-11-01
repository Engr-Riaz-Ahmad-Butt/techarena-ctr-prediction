"""tst4: independent audit of nxa_expo (exposure-policy expert). No fitting; no holdout scoring.

Checks: (1) one code path for day 5 / day 6 / evaluation with cutoffs 1737763200 / 1737849600 / 1737936000;
(2) every scored row's timestamp >= its cutoff; impressions used by the features are exactly the training rows < cutoff
(replayed via the label-free n_expo feature with independent counting); (3) the expert and recipe recomputed with my own
code from the saved features reproduce the saved raw / pred / pred_v39 arrays on all three sets; (4) no label column is
read by the feature code (static scan); (5) provenance: file mtimes, seeds, selection trail.
"""
import json, re, sys, os, time, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"
START, DAY, NU = 1737331200, 86400, 54456
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "tst4_nxa_expo_audit.json"


def rank(x): x = np.asarray(x, np.float64); return rankdata(x, method="average") / (len(x) + 1)


def expert(f): return np.mean([rank(f[c]) for c in ("typ16_all", "typ32_all", "cos32")], axis=0)


def recipe(parent, raw):
    zc = raw - np.repeat(raw.reshape(-1, 10).mean(1), 10); zc = zc / zc.std(); return rank(parent) + 0.015 * zc


def main():
    tic = time.monotonic(); out = dict(sets={}, static={}, provenance={})
    t = np.load(R / "train.npy", mmap_mode="r"); ts_all = np.asarray(t[:, 2]); users_all = np.asarray(t[:, 0])
    heads = ts_all[::10]; out["train_event_heads_sorted"] = bool(np.all(np.diff(heads) >= 0))
    out["train_rows_within_event_same_time"] = bool(np.all(ts_all.reshape(-1, 10) == heads[:, None]))
    for day in (5, 6, 7):
        cut = START + day * DAY; s = {"cutoff": cut}
        f = np.load(R / f"nxa_exposure2_day{day}_features.npz")
        if day == 7:
            ev = np.load(R / "evaluation.npy", mmap_mode="r"); su = np.asarray(ev[::10, 0]).astype(np.int64); sts = np.asarray(ev[:, 2])
            s["scored_rows"] = int(len(sts)); s["features_have_labels"] = bool("y" in f.files)
        else:
            z = np.load(R / f"v53_causal_propensity_day{day}.npz"); ei = z["events"][::10]; rows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()])
            su = rows[::10, 0].astype(np.int64); sts = rows[:, 2]; s["scored_rows"] = int(len(sts))
            s["feature_events_equal_v53"] = bool(np.array_equal(f["events"], z["events"]) and np.array_equal(f["y"], z["y"]))
        s["scored_ts_min"] = int(sts.min()); s["scored_ts_max"] = int(sts.max()); s["all_scored_rows_at_or_after_cutoff"] = bool(sts.min() >= cut)
        # independent replay of the label-free n_expo feature: count impressions (rows) of each user with time < cutoff
        pre = ts_all < cut; cnt = np.bincount(users_all[pre], minlength=NU).astype(np.float64)
        replay = np.repeat(np.log1p(cnt[su]), 10); s["n_expo_replay_max_abs_diff"] = float(np.abs(replay - f["n_expo"]).max())
        s["impression_rows_used"] = int(pre.sum()); s["latest_impression_ts"] = int(ts_all[pre].max()); s["latest_impression_before_cutoff"] = bool(ts_all[pre].max() < cut)
        # a strict-vs-loose cutoff probe: would including rows AT the cutoff change n_expo? (shows the feature is cutoff-sensitive)
        at = ts_all == cut; s["rows_exactly_at_cutoff"] = int(at.sum())
        # recompute expert + recipe
        raw = expert(f)
        if day == 7:
            v39 = np.load(R / "v39_final_pred.npy"); sraw = np.load(R / "nxa_expo_eval.npy"); sp39 = np.load(R / "nxa_expo_eval_pred_v39.npy")
            s["raw_max_abs_diff"] = float(np.abs(raw - sraw).max()); s["pred_v39_max_abs_diff"] = float(np.abs(recipe(v39, raw) - sp39).max())
            s["pred_v39_range"] = [float(sp39.min()), float(sp39.max())]
            s["spearman_pred_v39_vs_v39"] = float(np.corrcoef(rank(sp39), rank(v39))[0, 1])
        else:
            c = np.load(R / f"nxa_expo_day{day}.npz"); v39 = np.load(R / f"v39_regularized_als32_day{day}.npz")["pred"]
            s["raw_max_abs_diff"] = float(np.abs(raw - c["raw"]).max()); s["pred_max_abs_diff"] = float(np.abs(recipe(z["pred"], raw) - c["pred"]).max())
            s["pred_v39_max_abs_diff"] = float(np.abs(recipe(v39, raw) - c["pred_v39"]).max()); s["baseline_is_v53"] = bool(np.array_equal(c["baseline"], z["pred"]))
            s["spearman_raw_vs_v53"] = float(np.corrcoef(rank(raw), rank(z["pred"]))[0, 1])
        zc = raw - np.repeat(raw.reshape(-1, 10).mean(1), 10); s["zc_std_scored_set"] = float(zc.std())
        out["sets"][f"day{day}"] = s; print(json.dumps({"day": day, **{k: v for k, v in s.items() if not isinstance(v, list)}}), flush=True)
    # static scan of the feature code: label column (index 5) must never be read
    src = (B / "nxa_exposure2.py").read_text(); fin = (B / "nxa_expo_final.py").read_text()
    out["static"] = dict(feature_slice_user_item_time_only=":3]" in src and "t[:n * 10, :3]" in src, label_column_reads_in_feature_path=[m.group(0) for m in re.finditer(r"\[[^\]]*,\s*5\]", src)],
                         label_reads_only_for_assert=("srows[:, 5]" in src and "assert np.array_equal(srows[:, 5], z['y'])" in src), svd_seed="random_state=seed" in src and "seed=4403" in src,
                         final_reads_labels_only_for_metrics=all(k in fin for k in ("weighted_auc(e, y,", "roc_auc_score(y")), beta=0.015, columns=["typ16_all", "typ32_all", "cos32"])
    mt = {p: time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(B / p))) for p in ["nxa_exposure2.py", "nxa_expo_final.py", "nxa_combo_screen.py", "artifacts/nxa_exposure2_day6_features.npz", "artifacts/nxa_exposure2_day5_features.npz",
                                                                                               "artifacts/nxa_exposure2_day7_features.npz", "artifacts/nxa_combo_screen_day6.json", "artifacts/nxa_combo_screen_day5.json", "artifacts/nxa_exposure2_day5_report.json", "artifacts/nxa_expo_day6.npz", "artifacts/nxa_expo_report.json"]}
    out["provenance"]["mtimes"] = mt
    ho = np.load(R / "fresh_holdout_day6.npz"); s6 = np.load(R / "selection_day6.npz")
    out["provenance"]["full_day6_contains_fresh_holdout_rows"] = int(len(ho["full_indices"])); out["provenance"]["selection_day6_rows"] = int(len(s6["full_indices"]))
    out["provenance"]["holdout_selection_disjoint"] = bool(len(np.intersect1d(ho["full_indices"], s6["full_indices"])) == 0)
    out["elapsed_seconds"] = time.monotonic() - tic; OUT.write_text(json.dumps(out, indent=2)); print("WROTE", OUT)


if __name__ == "__main__":
    main()
