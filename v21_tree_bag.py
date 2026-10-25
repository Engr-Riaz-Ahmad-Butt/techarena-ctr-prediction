"""Seed-bag the accepted imp19 LightGBM (354 features, 735 trees) inside the V20 recipe.

  python v21_tree_bag.py --val --seeds 7 1009       # days 6 and 5, fixed trees, no selection
  python v21_tree_bag.py --score                     # both-day gate vs V21 3-seed blend
  python v21_tree_bag.py --final --seeds 7 1009      # days 4-6 fit, evaluation raws
  python v21_tree_bag.py --export                    # submission_v21.csv (never submission.csv)
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from imp19_tree import ROOT, SETTINGS, engine, gc, lgb, setup

TREES = 735
V21 = {6: 0.8241639495695283, 5: 0.8166261127941717}
rank = lambda v: rankdata(v, method="average") / (len(v) + 1)


def params(seed):
    return dict(SETTINGS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)


def fit_val(day, seed):
    out = ROOT / f"v21_bag_tree_s{seed}_day{day}.npz"
    if out.exists():
        print("exists", out.name, flush=True)
        return
    t0 = time.time()
    setup()
    engine.TAG = f"bag{seed}"
    x, y, path = engine.assemble_aligned([day - 3, day - 2, day - 1], f"d{day}")
    xv, ref, sw = engine.aligned_validation(day)
    base = np.load(ROOT / f"imp19_tree_raw_day{day}.npz")
    assert np.array_equal(ref["events"], base["events"]) and np.array_equal(ref["y"], base["y"])
    model = lgb.train(params(seed), lgb.Dataset(x, label=y, free_raw_data=True), num_boost_round=TREES)
    raw = model.predict(xv)
    assert np.isfinite(raw).all()
    model.save_model(str(ROOT / f"v21_bag_tree_s{seed}_d{day}.txt"))
    np.savez_compressed(out, y=ref["y"], events=ref["events"], raw=raw)
    print(json.dumps({"day": day, "seed": seed, "raw_auc": float(roc_auc_score(ref["y"], raw)),
                      "seed42_raw_auc": float(roc_auc_score(ref["y"], base["raw"])), "s": time.time() - t0}), flush=True)
    del model, x, xv
    gc.collect()
    try:
        path.unlink()
    except PermissionError:
        pass


def score(seeds):
    report = {}
    for day in (6, 5):
        base17 = np.load(ROOT / f"imp17_tree_day{day}.npz")["pred"]
        s42 = np.load(ROOT / f"imp19_tree_raw_day{day}.npz")
        y = s42["y"]
        raws = [s42["raw"]] + [np.load(ROOT / f"v21_bag_tree_s{s}_day{day}.npz")["raw"] for s in seeds]
        zs = [np.load(ROOT / f"v21_fastseeds_day{day}_s{s}_ep1_raw.npz") for s in range(3)]
        din = np.mean([rank(z["raw"]) for z in zs], 0)
        v20 = 0.5 * rank(base17) + 0.5 * rank(raws[0])
        assert abs(float(roc_auc_score(y, 0.8 * rank(v20) + 0.2 * din)) - V21[day]) < 1e-6
        bag_mean = 0.5 * rank(base17) + 0.5 * rank(np.mean(raws, 0))
        bag_rank = 0.5 * rank(base17) + 0.5 * np.mean([rank(r) for r in raws], 0)
        row = {
            "v20": float(roc_auc_score(y, v20)),
            "v20_bag_mean": float(roc_auc_score(y, bag_mean)),
            "v20_bag_rank": float(roc_auc_score(y, bag_rank)),
            "v21_prior": V21[day],
            "v21_bag_mean": float(roc_auc_score(y, 0.8 * rank(bag_mean) + 0.2 * din)),
            "v21_bag_rank": float(roc_auc_score(y, 0.8 * rank(bag_rank) + 0.2 * din)),
            "tree_raw_aucs": [float(roc_auc_score(y, r)) for r in raws],
            "tree_bag_raw_auc": float(roc_auc_score(y, np.mean(raws, 0))),
        }
        row["delta_mean"] = row["v21_bag_mean"] - V21[day]
        row["delta_rank"] = row["v21_bag_rank"] - V21[day]
        report[day] = row
        print(day, json.dumps(row), flush=True)
    report["kept"] = all(report[d]["delta_mean"] > 0 for d in (6, 5))
    report["seeds"] = seeds
    (ROOT / "v21_tree_bag_report.json").write_text(json.dumps(report, indent=2))
    print("kept:", report["kept"], flush=True)


def fit_final(seed):
    out = ROOT / f"v21_bag_tree_s{seed}_final_raw.npy"
    if out.exists():
        print("exists", out.name, flush=True)
        return
    from imp19_finalize import evaluation_matrix
    t0 = time.time()
    setup()
    engine.TAG = f"export_bag{seed}"
    engine.WIDTH = 354
    x, y, path = engine.assemble_aligned([4, 5, 6], "final")
    model = lgb.train(params(seed), lgb.Dataset(x, label=y, free_raw_data=True), num_boost_round=TREES)
    model.save_model(str(ROOT / f"v21_bag_tree_s{seed}_final.txt"))
    del x, y
    gc.collect()
    try:
        path.unlink()
    except PermissionError:
        pass
    xe = evaluation_matrix(354)
    raw = model.predict(xe)
    assert raw.shape == (529390,) and np.isfinite(raw).all()
    np.save(out, raw)
    print(json.dumps({"seed": seed, "final_s": time.time() - t0}), flush=True)


def export():
    from config import EVALUATION_CSV
    from v21_din import append_log
    rep = json.loads((ROOT / "v21_tree_bag_report.json").read_text())
    assert rep["kept"], "Bag did not beat V21 on both days"
    seeds = rep["seeds"]
    base17 = np.load(ROOT / "imp17_tree_final_pred.npy")
    raws = [np.load(ROOT / "imp19_tree_final_raw.npy")] + [np.load(ROOT / f"v21_bag_tree_s{s}_final_raw.npy") for s in seeds]
    v20 = np.array([float(r["predicted_probability"]) for r in csv.DictReader((ROOT / "submission_v20.csv").open(newline="", encoding="utf-8"))])
    assert np.abs(0.5 * rank(base17) + 0.5 * rank(raws[0]) - v20).max() < 1e-9, "V20 recipe mismatch"
    bag = 0.5 * rank(base17) + 0.5 * rank(np.mean(raws, 0))
    din = np.mean([rank(np.load(ROOT / f"v21_fastseeds_final_s{s}_eval.npy")) for s in range(3)], 0)
    pred = np.clip(0.8 * rank(bag) + 0.2 * din, 0, 1)
    out = Path("submission_v21.csv")
    backup = ROOT / "submission_v21_3seed_din.csv"
    if out.exists() and not backup.exists():
        backup.write_bytes(out.read_bytes())
    with EVALUATION_CSV.open(newline="", encoding="utf-8") as src, out.open("w", newline="", encoding="utf-8") as dest:
        w = csv.writer(dest)
        w.writerow(["rowid", "predicted_probability"])
        n = 0
        for row, p in zip(csv.DictReader(src), pred, strict=True):
            assert math.isfinite(p) and 0.0 <= p <= 1.0
            w.writerow([row["rowid"], format(float(p), ".10g")])
            n += 1
    assert n == 529390
    (ROOT / "submission_v21.csv").write_bytes(out.read_bytes())
    meta = {"rows": n, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "tree_seeds": [42] + seeds,
            "local_day6": rep["6"]["v21_bag_mean"], "local_day5": rep["5"]["v21_bag_mean"],
            "prior": V21, "submission_csv_untouched": True}
    (ROOT / "v21_tree_bag_manifest.json").write_text(json.dumps(meta, indent=2))
    append_log("V21-treebag", f"imp19 tree bagged over seeds {[42]+seeds} inside V20, + 20% DIN", V21[6],
               meta["local_day6"], 0, True, f"day5={meta['local_day5']:.6f}")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 1009])
    a = ap.parse_args()
    if a.val:
        for s in a.seeds:
            for d in (6, 5):
                fit_val(d, s)
    if a.score:
        score(a.seeds)
    if a.final:
        for s in a.seeds:
            fit_final(s)
    if a.export:
        export()
