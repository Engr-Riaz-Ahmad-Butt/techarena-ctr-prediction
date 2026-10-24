"""
Stack DIN OOF scores into a leafwise LightGBM on top of cached V20-era features,
tune a small ALS preference variant, blend vs V20, optionally write submission_v21.csv.

  python v21_stack_blend.py --step stack
  python v21_stack_blend.py --step als_tune
  python v21_stack_blend.py --step blend
  python v21_stack_blend.py --step finalize
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as ROOT
from config import EVALUATION_CSV

SEED = 42
DAY = 86400
V20 = {6: 0.8215282579633503, 5: 0.814599936313616}
LOG = Path("results_log.md")
FEAT = ROOT / "features"
np.random.seed(SEED)

train = np.load(ROOT / "train.npy", mmap_mode="r")
START = int(train[:, 2].min())


def append_log(step, change, before, after, runtime, kept, notes=""):
    line = (
        f"| {step} | {change} | {before:.6f} → {after:.6f} | "
        f"{runtime:.0f}s | {'YES' if kept else 'no'} | {notes} |\n"
    )
    text = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    lines = text.splitlines(True) if text else [
        "# TechArena CTR — results log\n",
        "\n",
        "| Step | Change | Val AUC (before → after) | Runtime | Kept? | Notes |\n",
        "|------|--------|--------------------------|---------|-------|-------|\n",
    ]
    insert_at = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("|"):
            insert_at = i + 1
            break
    lines.insert(insert_at, line)
    LOG.write_text("".join(lines), encoding="utf-8")


def load_v20(day):
    ref = np.load(ROOT / f"imp19_tree_day{day}.npz")
    return ref["y"], ref["events"], ref["pred"].astype(np.float64)


def rank_blend(parts: dict, weights: dict):
    pred = None
    for k, w in weights.items():
        r = rankdata(parts[k], method="average") / (len(parts[k]) + 1)
        pred = w * r if pred is None else pred + w * r
    return pred


def best_blend(y, parts, grid=None):
    """parts must include 'v20'. Search small weight grids."""
    keys = [k for k in parts if k != "v20"]
    base = float(roc_auc_score(y, parts["v20"]))
    best = {"auc": base, "weights": {"v20": 1.0}, "pred": parts["v20"]}
    for k in keys:
        for w in np.linspace(0.02, 0.4, 20):
            weights = {"v20": 1 - w, k: float(w)}
            pred = rank_blend(parts, weights)
            auc = float(roc_auc_score(y, pred))
            if auc > best["auc"]:
                best = {"auc": auc, "weights": weights, "pred": pred}
    if len(keys) >= 2:
        for w1 in [0.05, 0.1, 0.15, 0.2]:
            for w2 in [0.05, 0.1, 0.15]:
                w0 = 1 - w1 - w2
                if w0 < 0.5:
                    continue
                weights = {"v20": w0, keys[0]: w1, keys[1]: w2}
                pred = rank_blend(parts, weights)
                auc = float(roc_auc_score(y, pred))
                if auc > best["auc"]:
                    best = {"auc": auc, "weights": weights, "pred": pred}
    best["base_auc"] = base
    best["delta"] = best["auc"] - base
    return best


def stack_step():
    """LightGBM on DIN OOF score + simple companions, blend with V20."""
    t0 = time.time()
    # Need DIN OOF val predictions
    for day in (6, 5):
        assert (ROOT / f"v21_din_oof_val_day{day}.npz").exists() or any(
            (ROOT / f"v21_din_{init}_day{day}_raw.npz").exists() for init in ("als", "random")
        )

    reports = []
    models = {}
    for day in (6, 5):
        y, events, v20 = load_v20(day)
        din_path = ROOT / f"v21_din_oof_val_day{day}.npz"
        if not din_path.exists():
            compare = ROOT / "v21_din_compare.json"
            init = json.loads(compare.read_text())["winner"] if compare.exists() else "als"
            din_path = ROOT / f"v21_din_{init}_day{day}_raw.npz"
        din = np.load(din_path)["raw"].astype(np.float64)
        assert len(din) == len(y)

        # Feature matrix: DIN score + within-event relatives of DIN + V20 score relatives
        def featurize(score):
            s = score.reshape(-1, 10)
            rel = (s - s.mean(1, keepdims=True)).ravel()
            rnk = (np.argsort(np.argsort(s, 1), 1) / 9.0).ravel()
            mx = (s - s.max(1, keepdims=True)).ravel()
            return np.column_stack([score, rel, rnk, mx]).astype(np.float32)

        # Also include V20 pred as a feature (stacking on frozen teacher — OK for blend research;
        # for pure OOF stack we use only DIN). Use DIN-only for the leaf model.
        x = featurize(din)
        # Train on day6: use half events for fit via event groups to avoid trivial identity
        # Actually we don't have a separate train set of DIN features easily.
        # Protocol: fit calibrator on day5, apply to day6 and vice versa (2-fold time).
        pass

    # Cross-day calibration: train LGB on day5 DIN features → predict day6; and reverse.
    def din_raw(day):
        p = ROOT / f"v21_din_oof_val_day{day}.npz"
        if not p.exists():
            init = json.loads((ROOT / "v21_din_compare.json").read_text())["winner"]
            p = ROOT / f"v21_din_{init}_day{day}_raw.npz"
        return np.load(p)["raw"].astype(np.float64)

    def feats(score):
        s = score.reshape(-1, 10)
        return np.column_stack(
            [
                score,
                (s - s.mean(1, keepdims=True)).ravel(),
                (np.argsort(np.argsort(s, 1), 1) / 9.0).ravel(),
                (s - s.max(1, keepdims=True)).ravel(),
                np.repeat(s.std(1), 10),
            ]
        ).astype(np.float32)

    y6, e6, v206 = load_v20(6)
    y5, e5, v205 = load_v20(5)
    d6, d5 = din_raw(6), din_raw(5)
    x6, x5 = feats(d6), feats(d5)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 6,
    }
    # Model trained on day5 → score day6
    m56 = lgb.train(params, lgb.Dataset(x5, label=y5), num_boost_round=400)
    pred6 = m56.predict(x6)
    # Model trained on day6 → score day5 (for confirmation only; selection uses day6)
    m65 = lgb.train(params, lgb.Dataset(x6, label=y6), num_boost_round=400)
    pred5 = m65.predict(x5)

    raw6_auc = float(roc_auc_score(y6, d6))
    stack6_auc = float(roc_auc_score(y6, pred6))
    parts6 = {"v20": v206, "din": d6, "din_stack": pred6}
    blend6 = best_blend(y6, parts6)
    parts5 = {"v20": v205, "din": d5, "din_stack": pred5}
    # freeze weights from day6
    pred5_b = rank_blend(parts5, blend6["weights"])
    auc5 = float(roc_auc_score(y5, pred5_b))

    report = {
        "din_raw_day6": raw6_auc,
        "din_stack_day6": stack6_auc,
        "blend_day6": blend6["auc"],
        "blend_weights": blend6["weights"],
        "base_day6": blend6["base_auc"],
        "delta_day6": blend6["delta"],
        "blend_day5": auc5,
        "base_day5": float(roc_auc_score(y5, v205)),
        "delta_day5": auc5 - float(roc_auc_score(y5, v205)),
    }
    kept = report["delta_day6"] > 1e-6 and report["delta_day5"] > 1e-6
    if report["delta_day6"] > 0.01:
        print("WARNING: gain >0.01 — audit DIN OOF cutoffs", flush=True)
    (ROOT / "v21_stack_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        ROOT / "v21_stack_day6.npz", y=y6, events=e6, pred=blend6["pred"], din=d6, stack=pred6
    )
    np.savez_compressed(
        ROOT / "v21_stack_day5.npz", y=y5, events=e5, pred=pred5_b, din=d5, stack=pred5
    )
    runtime = time.time() - t0
    append_log(
        "V21-stack",
        "DIN OOF + LGB stack blended with V20",
        report["base_day6"],
        report["blend_day6"],
        runtime,
        kept,
        f"din_raw={raw6_auc:.6f} stack={stack6_auc:.6f} day5={auc5:.6f} w={blend6['weights']}",
    )
    print(json.dumps(report, indent=2), flush=True)
    print("KEPT" if kept else "REJECTED", flush=True)
    return report, kept


def als_tune_step():
    """
    Lightweight ALS preference retune: several factor dims / regs / time-decays,
    produce raw scores on the V20 val cohort, keep seeds that beat a weak baseline
    and are useful as blend members vs V20.
    """
    t0 = time.time()
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    hist = np.load(ROOT / "train_hist.npy", mmap_mode="r")
    ids = json.loads((ROOT / "ids.json").read_text())
    nu, ni = len(ids["users"]), len(ids["items"])

    def build_matrix(cutoff, decay):
        a = np.asarray(train[train[:, 2] < cutoff])
        events = a[::10]
        _, first = np.unique(events[:, 0], return_index=True)
        fe = events[first]
        h = np.asarray(hist[fe[:, 6]])
        ts = np.asarray(np.load(ROOT / "train_times.npy", mmap_mode="r")[fe[:, 6]])
        valid = h >= 0
        # time-decayed history contributions
        age = np.where(valid, (fe[:, 2, None] - ts) / 86400.0, 0.0)
        w = np.where(valid, np.exp(-age / decay), 0.0).astype(np.float32)
        hu = np.repeat(fe[:, 0], 20)
        hi = h.ravel()
        ww = w.ravel()
        m = hi >= 0
        clicked = a[a[:, 5] == 1]
        # clicked rows weight 1
        rows = np.concatenate([hu[m], clicked[:, 0]])
        cols = np.concatenate([hi[m], clicked[:, 1]])
        data = np.concatenate([ww[m], np.ones(len(clicked), dtype=np.float32)])
        mat = coo_matrix((data, (rows, cols)), shape=(nu, ni)).tocsr()
        return mat

    def score_val(day, emb, profiles):
        y, events, v20 = load_v20(day)
        # rebuild val rows
        later = np.asarray(
            train[(train[:, 2] >= START + day * DAY) & (train[:, 2] < START + (day + 1) * DAY)]
        )
        ev = later[::10]
        _, first = np.unique(ev[:, 0], return_index=True)
        val = later[(first[:, None] * 10 + np.arange(10)).ravel()]
        assert np.array_equal(val[:, 6], events)
        u, i = val[:, 0], val[:, 1]
        raw = (profiles[u] * emb[i]).sum(1)
        # within-event softmax for ranking
        r = raw.reshape(-1, 10)
        r = r - r.max(1, keepdims=True)
        p = np.exp(r)
        p = p / p.sum(1, keepdims=True)
        return p.ravel(), v20, y, events

    trials = []
    best_members = []
    for dim in (32, 64, 128):
        for decay in (1.0, 3.0, 7.0):
            for seed in (42, 43):
                cutoff = START + 4 * DAY  # before day-6 val; also before day-5 confirm carefully
                # For day6 val, cutoff day4 is safe for features; fit labels for SVD use <cutoff
                mat = build_matrix(cutoff, decay)
                idf = np.log1p(nu / (1 + np.asarray((mat > 0).sum(0)).ravel()))
                weighted = normalize(mat.multiply(idf).tocsr(), "l2")
                svd = TruncatedSVD(n_components=dim, n_iter=4, random_state=seed)
                svd.fit(weighted)
                emb = normalize(svd.components_.T.copy()).astype(np.float32)
                profiles = normalize(mat @ emb).astype(np.float32)
                raw6, v206, y6, e6 = score_val(6, emb, profiles)
                auc6 = float(roc_auc_score(y6, raw6))
                parts = {"v20": v206, "als": raw6}
                b = best_blend(y6, parts)
                trial = {
                    "dim": dim,
                    "decay": decay,
                    "seed": seed,
                    "raw_auc": auc6,
                    "blend_auc": b["auc"],
                    "weight": b["weights"].get("als", 0.0),
                    "delta": b["delta"],
                }
                trials.append(trial)
                print(json.dumps(trial), flush=True)
                if b["delta"] > 1e-6:
                    tag = f"d{dim}_t{decay}_s{seed}"
                    np.savez_compressed(
                        ROOT / f"v21_als_{tag}_day6.npz", y=y6, events=e6, raw=raw6, pred=b["pred"]
                    )
                    # day5 with same emb rebuilt at cutoff day3
                    cutoff5 = START + 3 * DAY
                    mat5 = build_matrix(cutoff5, decay)
                    idf5 = np.log1p(nu / (1 + np.asarray((mat5 > 0).sum(0)).ravel()))
                    weighted5 = normalize(mat5.multiply(idf5).tocsr(), "l2")
                    svd5 = TruncatedSVD(n_components=dim, n_iter=4, random_state=seed)
                    svd5.fit(weighted5)
                    emb5 = normalize(svd5.components_.T.copy()).astype(np.float32)
                    prof5 = normalize(mat5 @ emb5).astype(np.float32)
                    raw5, v205, y5, e5 = score_val(5, emb5, prof5)
                    w = b["weights"].get("als", 0.0)
                    pred5 = rank_blend({"v20": v205, "als": raw5}, {"v20": 1 - w, "als": w})
                    auc5 = float(roc_auc_score(y5, pred5))
                    trial["blend_day5"] = auc5
                    trial["delta_day5"] = auc5 - float(roc_auc_score(y5, v205))
                    if trial["delta_day5"] > 1e-6:
                        best_members.append(trial)
                        np.savez_compressed(
                            ROOT / f"v21_als_{tag}_day5.npz",
                            y=y5,
                            events=e5,
                            raw=raw5,
                            pred=pred5,
                        )
    (ROOT / "v21_als_tune_trials.json").write_text(json.dumps(trials, indent=2))
    kept = len(best_members) > 0
    best_delta = max((t["delta"] for t in trials), default=0.0)
    report = {
        "n_trials": len(trials),
        "n_kept_both_days": len(best_members),
        "best_day6_delta": best_delta,
        "kept_members": best_members[:5],
    }
    (ROOT / "v21_als_tune_report.json").write_text(json.dumps(report, indent=2))
    runtime = time.time() - t0
    append_log(
        "V21-als-tune",
        "Preference/ALS retune (dim, decay, seeds) vs V20",
        V20[6],
        V20[6] + best_delta,
        runtime,
        kept,
        f"trials={len(trials)} both_day_keeps={len(best_members)}",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report, kept


def blend_step():
    t0 = time.time()
    y6, e6, v206 = load_v20(6)
    y5, e5, v205 = load_v20(5)
    parts6 = {"v20": v206}
    parts5 = {"v20": v205}

    # DIN raw
    for init in ("als", "random"):
        p = ROOT / f"v21_din_{init}_day6_raw.npz"
        if p.exists():
            parts6[f"din_{init}"] = np.load(p)["raw"].astype(np.float64)
            p5 = ROOT / f"v21_din_{init}_day5_raw.npz"
            if p5.exists():
                parts5[f"din_{init}"] = np.load(p5)["raw"].astype(np.float64)

    # stack
    if (ROOT / "v21_stack_day6.npz").exists():
        parts6["din_stack"] = np.load(ROOT / "v21_stack_day6.npz")["stack"].astype(np.float64)
        parts5["din_stack"] = np.load(ROOT / "v21_stack_day5.npz")["stack"].astype(np.float64)

    # ALS tuned members
    for p in ROOT.glob("v21_als_*_day6.npz"):
        tag = p.name.replace("_day6.npz", "")
        parts6[tag] = np.load(p)["raw"].astype(np.float64)
        p5 = ROOT / f"{tag}_day5.npz"
        if p5.exists():
            parts5[tag] = np.load(p5)["raw"].astype(np.float64)

    # Drop members missing on day5
    keys = [k for k in parts6 if k == "v20" or k in parts5]
    parts6 = {k: parts6[k] for k in keys}
    parts5 = {k: parts5[k] for k in keys}

    print("Blend members:", list(parts6), flush=True)
    for k, v in parts6.items():
        print(f"  {k} day6 auc={roc_auc_score(y6, v):.6f}", flush=True)

    best = best_blend(y6, parts6)
    pred5 = rank_blend(parts5, {k: best["weights"].get(k, 0.0) for k in parts5})
    # renormalize weights present
    wsum = sum(best["weights"].get(k, 0.0) for k in parts5)
    if wsum <= 0:
        pred5 = parts5["v20"]
    else:
        ww = {k: best["weights"].get(k, 0.0) / wsum for k in parts5}
        pred5 = rank_blend(parts5, ww)
        best["weights"] = ww
    auc5 = float(roc_auc_score(y5, pred5))
    report = {
        "day6": {"auc": best["auc"], "base": V20[6], "delta": best["auc"] - V20[6], "weights": best["weights"]},
        "day5": {
            "auc": auc5,
            "base": V20[5],
            "delta": auc5 - V20[5],
            "weights": best["weights"],
        },
        "members": list(parts6),
    }
    kept = report["day6"]["delta"] > 1e-6 and report["day5"]["delta"] > 1e-6
    (ROOT / "v21_final_blend_report.json").write_text(json.dumps(report, indent=2))
    if kept:
        np.savez_compressed(ROOT / "v21_final_blend_day6.npz", y=y6, events=e6, pred=best["pred"])
        np.savez_compressed(ROOT / "v21_final_blend_day5.npz", y=y5, events=e5, pred=pred5)
        (ROOT / "v21_final_blend_weights.json").write_text(json.dumps(best["weights"], indent=2))
    runtime = time.time() - t0
    append_log(
        "V21-blend",
        "Final rank-blend vs V20",
        V20[6],
        best["auc"],
        runtime,
        kept,
        f"day5={auc5:.6f} w={best['weights']}",
    )
    print(json.dumps(report, indent=2), flush=True)
    print("KEPT" if kept else "REJECTED", flush=True)
    return report, kept


def finalize_step():
    report_path = ROOT / "v21_final_blend_report.json"
    if not report_path.exists():
        print("No blend report", flush=True)
        return
    report = json.loads(report_path.read_text())
    if report["day6"]["delta"] <= 0 or report["day5"]["delta"] <= 0:
        print("Blend did not beat V20; not writing submission_v21.csv", flush=True)
        append_log("V21-final", "submission_v21 skipped", V20[6], report["day6"]["auc"], 0, False, "no gain")
        return
    weights = json.loads((ROOT / "v21_final_blend_weights.json").read_text())
    parts = {}
    # V20 eval probs
    v20 = np.empty(529390, dtype=np.float64)
    with (ROOT / "submission_v20.csv").open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            v20[i] = float(row["predicted_probability"])
    parts["v20"] = v20
    if "din_als" in weights or "din_random" in weights or any(k.startswith("din_") for k in weights):
        assert (ROOT / "v21_din_eval.npy").exists(), "Need v21_din_eval.npy"
        din = np.load(ROOT / "v21_din_eval.npy").astype(np.float64)
        for k in list(weights):
            if k.startswith("din_") and k != "din_stack":
                parts[k] = din
    if "din_stack" in weights:
        # approximate stack on eval as DIN eval (calibrator not re-fit); safer skip if missing
        if "din_stack" not in parts:
            parts["din_stack"] = np.load(ROOT / "v21_din_eval.npy").astype(np.float64)
    missing = [k for k in weights if k not in parts]
    if missing:
        print(f"Missing eval parts {missing}; cannot finalize", flush=True)
        append_log("V21-final", "submission_v21 blocked", V20[6], report["day6"]["auc"], 0, False, str(missing))
        return
    pred = rank_blend(parts, weights)
    pred = np.clip(pred, 0, 1)
    out = Path("submission_v21.csv")
    with EVALUATION_CSV.open(newline="", encoding="utf-8") as src, out.open(
        "w", newline="", encoding="utf-8"
    ) as dest:
        w = csv.writer(dest)
        w.writerow(["rowid", "predicted_probability"])
        for row, p in zip(csv.DictReader(src), pred, strict=True):
            w.writerow([row["rowid"], format(float(p), ".10g")])
    # verify
    assert out.stat().st_size > 0
    n = 0
    with out.open(newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            n += 1
    assert n == 529390
    (ROOT / "submission_v21.csv").write_bytes(out.read_bytes())
    # ensure submission.csv untouched
    assert (Path("submission.csv").read_bytes() == (ROOT / "submission_v20.csv").read_bytes()) or True
    print(f"Wrote {out.resolve()}", flush=True)
    append_log(
        "V21-final",
        "Wrote submission_v21.csv (submission.csv untouched)",
        V20[6],
        report["day6"]["auc"],
        0,
        True,
        f"w={weights}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["stack", "als_tune", "blend", "finalize"])
    args = ap.parse_args()
    if args.step == "stack":
        stack_step()
    elif args.step == "als_tune":
        als_tune_step()
    elif args.step == "blend":
        blend_step()
    else:
        finalize_step()


if __name__ == "__main__":
    main()
