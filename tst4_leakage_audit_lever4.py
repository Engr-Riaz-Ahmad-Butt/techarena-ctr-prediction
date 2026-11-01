"""tst4: array-level leakage audit of the Idea2 Step2 inputs. No model is fitted.

Replays, with independent code, the feature statistics that depend on labels
(user event-click rate, last-activity gap, hours since cutoff, history length)
for the frozen horizon and saved-V53 bundles, and checks every timestamp
ordering the frozen plan claims. Also replays the saved LightGBM measurement
model on the frozen validation features (scoring only).
"""
import json, sys, time
from pathlib import Path
import numpy as np

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); sys.path.insert(0, str(B))
A = B / "artifacts"; H = A / "v54_horizon"; R = H / "event_product_001"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "tst4_leakage_audit_out.json"


def rate_replay(t, users, cutoff, nu):
    prior = np.asarray(t[t[:, 2] < cutoff]); assert prior[:, 2].max() < cutoff
    ev = prior[::10]; click = prior[:, 5].reshape(-1, 10).sum(1)
    n = np.bincount(ev[:, 0], minlength=nu); c = np.bincount(ev[:, 0], weights=click, minlength=nu); g = float(click.mean())
    last = np.zeros(nu, dtype=np.int64); np.maximum.at(last, ev[:, 0], ev[:, 2])
    return (c[users] + 20 * g) / (n[users] + 20), last[users], int(prior[:, 2].max())


def main():
    tic = time.monotonic(); out = dict(horizon={}, saved_v53={}, model_replay={})
    t = np.load(A / "train.npy", mmap_mode="r"); h = np.load(A / "train_hist.npy", mmap_mode="r"); ht = np.load(A / "train_times.npy", mmap_mode="r")
    head = np.asarray(t[::10]); start = int(head[:, 2].min()); cut1 = start + 86400; cut3 = start + 3 * 86400; nu = int(t[:, 0].max()) + 1
    d = np.load(R / "upload/horizon_product.npz"); ma = np.load(H / "event_measurement_001/measurement_arrays.npz"); co = np.load(H / "cohort.npz")
    rows = np.load(H / "score_rows.npy", mmap_mode="r"); vl = np.load(H / "validation_labels.npy")
    hz = out["horizon"]
    # --- fit events: first event per user inside [cut1, cut3) ---
    fe = ma["fit_events"]; available = np.flatnonzero((head[:, 2] >= cut1) & (head[:, 2] < cut3)); _, ix = np.unique(head[available, 0], return_index=True); expected = available[np.sort(ix)]
    hz["fit_events_match_first_event_rule"] = bool(np.array_equal(fe, expected)); hz["fit_events"] = int(len(fe))
    hz["fit_ts_range"] = [int(head[fe, 2].min()), int(head[fe, 2].max())]; hz["fit_ts_inside_cut1_cut3"] = bool(head[fe, 2].min() >= cut1 and head[fe, 2].max() < cut3)
    hz["fit_users_unique"] = bool(len(np.unique(head[fe, 0])) == len(fe)); hz["fit_user_ts_match_bundle"] = bool(np.array_equal(head[fe, 0], d["fit_user"]) and np.array_equal(head[fe, 2], d["fit_ts"]))
    # fit labels: bundle yfit equals train labels of those events (labels precede cut3)
    fr = np.asarray(t[(fe[:, None] * 10 + np.arange(10)).ravel()]); hz["yfit_equals_train_labels"] = bool(np.array_equal(fr[:, 5].reshape(-1, 10), d["yfit"]))
    hz["train_y_equals_event_click"] = bool(np.array_equal(ma["train_y"], d["yfit"].sum(1)))
    # --- validation events ---
    ve = ma["validation_events"]; hz["validation_events_match_cohort"] = bool(np.array_equal(ve, co["events"]))
    hz["fit_val_events_disjoint"] = bool(len(np.intersect1d(fe, ve)) == 0); hz["val_users_unique"] = bool(len(np.unique(head[ve, 0])) == len(ve))
    hz["score_rows_label_sentinel_all_minus1"] = bool(np.all(np.asarray(rows[:, 5]) == -1)); hz["score_rows_events_match"] = bool(np.array_equal(np.asarray(rows[::10, 6]), ve))
    true_labels = np.asarray(t[(ve[:, None] * 10 + np.arange(10)).ravel(), 5]).reshape(-1, 10); hz["validation_labels_equal_train_labels"] = bool(np.array_equal(true_labels, vl)); hz["yval_equals_validation_labels"] = bool(np.array_equal(vl, d["yval"]))
    hz["val_ts_min"] = int(head[ve, 2].min()); hz["cutoff3"] = cut3; hz["fit_label_ts_max"] = int(head[fe, 2].max()); hz["fit_labels_strictly_before_cutoff_and_validation"] = bool(head[fe, 2].max() < cut3 <= head[ve, 2].min())
    # --- histories: strictly before their event; count post-cutoff history clicks ---
    hv = np.asarray(h[ve]); tv = np.asarray(ht[ve]); valid = hv >= 0
    hz["val_history_times_before_event"] = bool(np.all((tv < head[ve, 2, None]) | ~valid)); hz["val_history_clicks_at_or_after_cutoff3"] = int(((tv >= cut3) & valid).sum()); hz["val_history_clicks_total"] = int(valid.sum())
    hz["val_history_length_matches_cohort"] = bool(np.array_equal(valid.sum(1), co["history_length"])) and bool(np.array_equal(valid.sum(1), d["history_length"]))
    hf = np.asarray(h[fe]); tf = np.asarray(ht[fe]); validf = hf >= 0
    hz["fit_history_times_before_event"] = bool(np.all((tf < head[fe, 2, None]) | ~validf)); hz["fit_history_clicks_at_or_after_cutoff1"] = int(((tf >= cut1) & validf).sum()); hz["fit_history_clicks_total"] = int(validf.sum())
    # --- independent replay of label-derived features ---
    xv = d["xval"]; xf = d["xfit"]; hz["xval_equals_measurement_features"] = bool(np.array_equal(xv, ma["features"])); hz["xfit_equals_measurement_train_features"] = bool(np.array_equal(xf, ma["train_features"]))
    for name, x, events, cutoff in (("val", xv, ve, cut3), ("fit", xf, fe, cut1)):
        users = head[events, 0]; ts = head[events, 2]; rate, last, prior_max = rate_replay(t, users, cutoff, nu)
        hist_times = np.asarray(ht[events]); hvalid = np.asarray(h[events]) >= 0; known = np.maximum(last, np.where(hvalid, hist_times, 0).max(1)); missing = known == 0; gap = np.where(missing, 7., (ts - known) / 86400.)
        hz[name + "_feature_replay"] = dict(prior_events_max_ts=prior_max, cutoff=cutoff, prior_strictly_before_cutoff=bool(prior_max < cutoff),
                                             rate_max_abs_diff=float(np.abs(rate - x[:, 4]).max()), hours_since_cutoff_max_abs_diff=float(np.abs(np.log1p((ts - cutoff) / 3600.) - x[:, 2]).max()),
                                             history_length_max_abs_diff=float(np.abs(hvalid.sum(1) - x[:, 3]).max()), gap_max_abs_diff=float(np.abs(np.log1p(gap) - x[:, 0]).max()), missing_flag_match=bool(np.array_equal(missing.astype(float), x[:, 1])),
                                             time_bucket_match=bool(np.array_equal(head[events, 3].astype(float), x[:, 5])), day_type_match=bool(np.array_equal(head[events, 4].astype(float), x[:, 6])),
                                             hours_since_cutoff_min=float(np.expm1(x[:, 2]).min()), hours_since_cutoff_max=float(np.expm1(x[:, 2]).max()))
    # ALS checkpoints used for score features
    m1 = np.load(H / "imp17_als_cut1.npz"); m3 = np.load(A / "imp17_als_cut3.npz"); hz["als_cut1_cutoff"] = int(m1["cutoff"]); hz["als_cut3_cutoff"] = int(m3["cutoff"]); hz["als_cutoffs_consistent"] = bool(int(m1["cutoff"]) == cut1 and int(m3["cutoff"]) == cut3)
    # independent ALS feature replay for validation rows (model cut3)
    items = np.asarray(rows[:, 1]).reshape(-1, 10); users = head[ve, 0]; score = float(m3["g"]) + m3["bu"][users, None] + m3["bi"][items] + np.einsum("ed,ecd->ec", m3["U"][users], m3["V"][items]); top = np.partition(score, -2, axis=1)[:, -2:]
    hz["val_als_feature_replay_max_abs_diff"] = dict(ALS_max=float(np.abs(score.max(1) - xv[:, 7]).max()), ALS_mean=float(np.abs(score.mean(1) - xv[:, 8]).max()), ALS_top2_gap=float(np.abs((top.max(1) - top.min(1)) - xv[:, 9]).max()), val_als_raw_rows=float(np.abs(score - d["val_als"]).max()))
    # label invariance of the feature function (the only code that touches rows)
    from horizon_event_measurement import features
    probe = np.asarray(rows[:2560]).copy(); probe[:, 5] = 1; alt = np.asarray(rows[:2560]).copy(); alt[:, 5] = 0
    hz["feature_function_label_invariance"] = bool(np.array_equal(features(probe, cut3, dict(m3), t, h, ht, nu), xv[:256]) and np.array_equal(features(alt, cut3, dict(m3), t, h, ht, nu), xv[:256]))
    # saved measurement model replay (scoring only)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(R / "upload/measurement_event_model.txt")); pv = booster.predict(xv)
    out["model_replay"] = dict(lightgbm_version=lgb.__version__, max_abs_diff_vs_bundle=float(np.abs(pv - d["measurement_val_probability"]).max()), max_abs_diff_vs_measurement_arrays=float(np.abs(pv - ma["event_probability"]).max()), num_trees=booster.num_trees())
    # the event model's training rows are the fit events only (model file records n rows? LightGBM stores feature names/importances only) -> rely on measurement report
    mr = json.loads((H / "event_measurement_001/report.json").read_text()); out["model_replay"]["measurement_report"] = dict(fit_events=mr["fit_events"], last_fit_label_timestamp=mr["last_fit_label_timestamp"], validation_first_timestamp=mr["validation_first_timestamp"], label_invariance_check=mr["label_invariance_check"])
    print(json.dumps(dict(stage="horizon_audit_done", elapsed=time.monotonic() - tic)), flush=True)

    # ---------------- saved V53 bundle ----------------
    d = np.load(R / "upload/saved_v53_product.npz"); z5 = np.load(A / "selection_day5.npz"); z6 = np.load(A / "selection_day6.npz"); sv = out["saved_v53"]
    cut5 = start + 5 * 86400; cut6 = start + 6 * 86400; e5 = z5["events"][::10]; e6 = z6["events"][::10]
    sv["fit_events_equal_selection_day5"] = bool(np.array_equal(head[e5, 0], d["fit_user"]) and np.array_equal(head[e5, 2], d["fit_ts"])); sv["val_events_equal_selection_day6"] = bool(np.array_equal(head[e6, 0], d["val_user"]) and np.array_equal(head[e6, 2], d["val_ts"]))
    sv["fit_ts_inside_day5"] = bool(d["fit_ts"].min() >= cut5 and d["fit_ts"].max() < cut6); sv["val_ts_inside_day6"] = bool(d["val_ts"].min() >= cut6 and d["val_ts"].max() < cut6 + 86400); sv["fit_labels_before_val"] = bool(d["fit_ts"].max() < d["val_ts"].min())
    sv["fit_val_events_disjoint"] = bool(len(np.intersect1d(e5, e6)) == 0); sv["day5_day6_user_overlap"] = int(len(np.intersect1d(head[e5, 0], head[e6, 0])))
    sv["yfit_equals_day5_labels"] = bool(np.array_equal(z5["y"].reshape(-1, 10), d["yfit"])); sv["yval_equals_day6_labels"] = bool(np.array_equal(z6["y"].reshape(-1, 10), d["yval"]))
    sv["fit_score_equals_day5_baseline"] = bool(np.array_equal(z5["baseline"].reshape(-1, 10), d["fit_score"])); sv["val_score_equals_day6_baseline"] = bool(np.array_equal(z6["baseline"].reshape(-1, 10), d["val_score"]))
    cal = d["cal"]; thr = np.quantile(d["fit_ts"], .4); sv["cal_block_is_latest_60pct_of_day5"] = bool(np.array_equal(cal, d["fit_ts"] > thr)) and bool(d["fit_ts"][~cal].max() < d["fit_ts"][cal].min())
    for name, x, events, cutoff, day in (("val", d["xval"], e6, cut6, 6), ("fit", d["xfit"], e5, cut5, 5)):
        users = head[events, 0]; ts = head[events, 2]; rate, last, prior_max = rate_replay(t, users, cutoff, nu)
        hist_times = np.asarray(ht[events]); hvalid = np.asarray(h[events]) >= 0; known = np.maximum(last, np.where(hvalid, hist_times, 0).max(1)); missing = known == 0; gap = np.where(missing, 7., (ts - known) / 86400.)
        m = np.load(A / f"imp15_model_day{day}.npz")
        sv[name + "_feature_replay"] = dict(prior_events_max_ts=prior_max, cutoff=cutoff, prior_strictly_before_cutoff=bool(prior_max < cutoff), als_model_cutoff=int(m["cutoff"]), als_cutoff_matches=bool(int(m["cutoff"]) == cutoff),
                                             rate_max_abs_diff=float(np.abs(rate - x[:, 4]).max()), hours_since_cutoff_max_abs_diff=float(np.abs(np.log1p((ts - cutoff) / 3600.) - x[:, 2]).max()),
                                             history_length_max_abs_diff=float(np.abs(hvalid.sum(1) - x[:, 3]).max()), gap_max_abs_diff=float(np.abs(np.log1p(gap) - x[:, 0]).max()), missing_flag_match=bool(np.array_equal(missing.astype(float), x[:, 1])),
                                             history_times_before_event=bool(np.all((hist_times < ts[:, None]) | ~hvalid)), history_clicks_at_or_after_cutoff=int(((hist_times >= cutoff) & hvalid).sum()))
    sv["val_history_length_matches_bundle"] = bool(np.array_equal((np.asarray(h[e6]) >= 0).sum(1), d["history_length"]))
    from sklearn.metrics import roc_auc_score
    sv["day6_v53_unweighted_auc_from_selection_file"] = float(roc_auc_score(z6["y"], z6["baseline"])); sv["day5_v53_unweighted_auc_from_selection_file"] = float(roc_auc_score(z5["y"], z5["baseline"]))
    sv["selection_day6_keys"] = list(z6.files); sv["selection_day5_keys"] = list(z5.files)
    out["elapsed_seconds"] = time.monotonic() - tic; OUT.write_text(json.dumps(out, indent=2)); print("WROTE", OUT)


if __name__ == "__main__":
    main()
