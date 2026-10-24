"""
TechArena CTR — clean improvement pipeline (time-safe, global AUC).

Stages (run one at a time with --stage N; keep only if val AUC improves):
  0  baseline-like features + LightGBM
  1  + personalization / history match / recency / embedding similarity
  2  + richer time-safe item/user CTR (Bayesian priors, recent window)
  3  + context (slot, hour, day_type, time_bucket) strength
  4  + within-event relative ranks/diffs
  5  LightGBM hyperparameter tune
  6  CatBoost + XGBoost + blend on time validation
  7  final fit on train+val days, write submission.csv

Usage:
  python pipeline.py --stage 0
  python pipeline.py --stage 7 --final
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from config import ARTIFACT_DIR as ROOT
from config import EVALUATION_CSV

SEED = 42
DAY = 86400
np.random.seed(SEED)

train = np.load(ROOT / "train.npy", mmap_mode="r")
hist = np.load(ROOT / "train_hist.npy", mmap_mode="r")
htimes = np.load(ROOT / "train_times.npy", mmap_mode="r")
evaluation = np.load(ROOT / "evaluation.npy", mmap_mode="r")
ehist = np.load(ROOT / "evaluation_hist.npy", mmap_mode="r")
etimes = np.load(ROOT / "evaluation_times.npy", mmap_mode="r")
ids = json.loads((ROOT / "ids.json").read_text())
NU, NI = len(ids["users"]), len(ids["items"])
del ids
START = int(train[:, 2].min())
LOG_PATH = Path("results_log.md")
STATE_PATH = ROOT / "pipeline_state.json"


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"best_stage": -1, "best_auc": 0.0, "best_rounds": 300, "history": []}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def append_log(step, change, before, after, runtime, kept, notes=""):
    line = (
        f"| {step} | {change} | {before:.6f} → {after:.6f} | "
        f"{runtime:.1f}s | {'YES' if kept else 'no'} | {notes} |\n"
    )
    text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    if "| Step |" not in text:
        text = (
            "# TechArena CTR — results log\n\n"
            "| Step | Change | Val AUC (before → after) | Runtime | Kept? | Notes |\n"
            "|------|--------|--------------------------|---------|-------|-------|\n"
        )
    # Insert before any trailing text after the table, or append.
    if text.rstrip().endswith("|"):
        text = text.rstrip() + "\n" + line
    else:
        # find last table row
        lines = text.splitlines(True)
        insert_at = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("|"):
                insert_at = i + 1
                break
        lines.insert(insert_at, line)
        text = "".join(lines)
    LOG_PATH.write_text(text, encoding="utf-8")


def first_event_rows(block):
    """Keep all 10 candidates of each user's first event in a day-block."""
    events = block[::10]
    _, first = np.unique(events[:, 0], return_index=True)
    return block[(first[:, None] * 10 + np.arange(10)).ravel()]


def split_day(day):
    """Fit on [day-2, day), validate first-events on [day, day+1)."""
    fit = np.asarray(
        train[(train[:, 2] >= START + (day - 2) * DAY) & (train[:, 2] < START + day * DAY)]
    )
    later = np.asarray(
        train[(train[:, 2] >= START + day * DAY) & (train[:, 2] < START + (day + 1) * DAY)]
    )
    return fit, first_event_rows(later)


def build_stats(cutoff, emb_dim=64):
    """Time-safe counts + item embeddings from histories/clicks before cutoff."""
    a = np.asarray(train[train[:, 2] < cutoff])
    y = a[:, 5].astype(np.float64)
    g = float(y.mean())
    ui = np.bincount(a[:, 0], minlength=NU)
    uc = np.bincount(a[:, 0], weights=y, minlength=NU)
    ii = np.bincount(a[:, 1], minlength=NI)
    ic = np.bincount(a[:, 1], weights=y, minlength=NI)
    # Recent 1-day window
    recent = a[:, 2] >= cutoff - DAY
    ui_r = np.bincount(a[recent, 0], minlength=NU)
    uc_r = np.bincount(a[recent, 0], weights=y[recent], minlength=NU)
    ii_r = np.bincount(a[recent, 1], minlength=NI)
    ic_r = np.bincount(a[recent, 1], weights=y[recent], minlength=NI)
    bucket_impr = np.bincount(a[:, 3], minlength=4)
    bucket_click = np.bincount(a[:, 3], weights=y, minlength=4)
    bucket_ctr = (bucket_click + 100 * g) / (bucket_impr + 100)

    events = a[::10]
    _, first = np.unique(events[:, 0], return_index=True)
    firstevents = events[first]
    h = np.asarray(hist[firstevents[:, 6]])
    hu = np.repeat(firstevents[:, 0], 20)
    hi = h.ravel()
    valid = hi >= 0
    clicked = a[y == 1]
    rows = np.concatenate([hu[valid], clicked[:, 0]])
    cols = np.concatenate([hi[valid], clicked[:, 1]])
    matrix = coo_matrix(
        (np.ones(len(rows), dtype="float32"), (rows, cols)), shape=(NU, NI)
    ).tocsr()
    matrix.data = np.log1p(matrix.data)
    idf = np.log1p(NU / (1 + np.asarray((matrix > 0).sum(axis=0)).ravel()))
    weighted = normalize(matrix.multiply(idf).tocsr(), norm="l2")
    svd = TruncatedSVD(n_components=emb_dim, n_iter=5, random_state=SEED)
    svd.fit(weighted)
    emb = normalize(svd.components_.T.copy()).astype("float32")
    profiles = normalize(matrix @ emb).astype("float32")
    print(f"Stats cutoff={cutoff}: rows={len(a)} CTR={g:.6f} emb={emb_dim}", flush=True)
    return {
        "g": g,
        "ui": ui,
        "uc": uc,
        "ii": ii,
        "ic": ic,
        "ui_r": ui_r,
        "uc_r": uc_r,
        "ii_r": ii_r,
        "ic_r": ic_r,
        "bucket_ctr": bucket_ctr.astype("float32"),
        "emb": emb,
        "profiles": profiles,
    }


def _relative(cols):
    """Within-event mean-diff, max-diff, and percentile rank for each column."""
    out = []
    for v in cols:
        m = v.reshape(-1, 10)
        out.append((m - m.mean(axis=1, keepdims=True)).ravel())
        out.append((m - m.max(axis=1, keepdims=True)).ravel())
        out.append((np.argsort(np.argsort(m, axis=1), axis=1) / 9.0).ravel())
    return out


def _feature_width(stage: int) -> int:
    n = 9  # stage 0
    if stage >= 1:
        n += 10
    if stage >= 2:
        n += 11
    if stage >= 3:
        n += 4
    if stage >= 4:
        # relatives on item_ctr + last up-to-8 score cols
        n_base = n
        use_count = 1 + min(8, n_base - 1)
        n += use_count * 3
    return n


def make_features(a, histories, times, stats, stage: int, batch: int = 20000):
    """Build feature matrix in batches (embedding ops are memory-heavy)."""
    g = stats["g"]
    ui, uc, ii, ic = stats["ui"], stats["uc"], stats["ii"], stats["ic"]
    ui_r, uc_r, ii_r, ic_r = stats["ui_r"], stats["uc_r"], stats["ii_r"], stats["ic_r"]
    emb, profiles = stats["emb"], stats["profiles"]
    bucket_ctr = stats["bucket_ctr"]
    n = len(a)
    assert n % 10 == 0
    width = _feature_width(stage)
    # Precompute width without relatives, then append relatives after.
    x_core_w = 9 + (10 if stage >= 1 else 0) + (11 if stage >= 2 else 0) + (4 if stage >= 3 else 0)
    out = np.empty((n, x_core_w), dtype="float32")

    for off in range(0, n, batch):
        b = a[off : off + batch]
        bn = len(b)
        user_ctr = (uc[b[:, 0]] + 50 * g) / (ui[b[:, 0]] + 50)
        item_ctr = (ic[b[:, 1]] + 50 * g) / (ii[b[:, 1]] + 50)
        slot = (np.arange(off, off + bn) % 10).astype("float32")
        hour = ((b[:, 2] % 86400) / 86400).astype("float32")
        h = np.asarray(histories[b[:, 6]])
        valid = h >= 0
        lengths = valid.sum(axis=1).astype("float32")
        cols = [
            user_ctr.astype("float32"),
            item_ctr.astype("float32"),
            np.log1p(ui[b[:, 0]]).astype("float32"),
            np.log1p(ii[b[:, 1]]).astype("float32"),
            b[:, 3].astype("float32"),
            b[:, 4].astype("float32"),
            hour,
            slot,
            lengths,
        ]
        if stage >= 1:
            safe = np.maximum(h, 0)
            match = (h == b[:, 1, None]).astype("float32")
            match_count = match.sum(axis=1)
            pos = np.where(valid, np.arange(20)[None, :], -1)
            newest_match = np.where(match_count > 0, (pos * match).max(axis=1), -1)
            recency_pos = np.where(
                match_count > 0, (lengths - 1 - newest_match) / np.maximum(lengths, 1), 20
            )
            t = np.asarray(times[b[:, 6]])
            last_match_t = np.where(match > 0, t, 0).max(axis=1)
            gap = np.where(
                match_count > 0, (b[:, 2] - last_match_t) / 86400.0, 30.0
            ).astype("float32")
            v = emb[safe] * valid[:, :, None]
            avg = v.sum(axis=1) / np.maximum(lengths[:, None], 1)
            sim = (v * emb[b[:, 1], None, :]).sum(axis=2)
            age = lengths[:, None] - 1 - np.arange(20)[None, :]
            w = np.power(0.85, np.maximum(age, 0)) * valid
            sim_w = (sim * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)
            recent3 = valid & (age < 3)
            sim_r3 = (sim * recent3).sum(axis=1) / np.maximum(recent3.sum(axis=1), 1)
            profile_dot = (profiles[b[:, 0]] * emb[b[:, 1]]).sum(axis=1)
            avg_dot = (avg * emb[b[:, 1]]).sum(axis=1)
            cols.extend(
                [
                    match_count.astype("float32"),
                    (match_count > 0).astype("float32"),
                    recency_pos.astype("float32"),
                    gap,
                    sim.max(axis=1).astype("float32"),
                    (sim.sum(axis=1) / np.maximum(lengths, 1)).astype("float32"),
                    sim_w.astype("float32"),
                    sim_r3.astype("float32"),
                    profile_dot.astype("float32"),
                    avg_dot.astype("float32"),
                ]
            )
        if stage >= 2:
            user_ctr_tight = (uc[b[:, 0]] + 10 * g) / (ui[b[:, 0]] + 10)
            user_ctr_wide = (uc[b[:, 0]] + 150 * g) / (ui[b[:, 0]] + 150)
            item_ctr_tight = (ic[b[:, 1]] + 10 * g) / (ii[b[:, 1]] + 10)
            item_ctr_wide = (ic[b[:, 1]] + 150 * g) / (ii[b[:, 1]] + 150)
            user_r = (uc_r[b[:, 0]] + 30 * g) / (ui_r[b[:, 0]] + 30)
            item_r = (ic_r[b[:, 1]] + 30 * g) / (ii_r[b[:, 1]] + 30)
            cols.extend(
                [
                    user_ctr_tight.astype("float32"),
                    user_ctr_wide.astype("float32"),
                    item_ctr_tight.astype("float32"),
                    item_ctr_wide.astype("float32"),
                    user_r.astype("float32"),
                    item_r.astype("float32"),
                    (user_r - user_ctr).astype("float32"),
                    (item_r - item_ctr).astype("float32"),
                    np.log1p(ui_r[b[:, 0]]).astype("float32"),
                    np.log1p(ii_r[b[:, 1]]).astype("float32"),
                    bucket_ctr[b[:, 3]],
                ]
            )
        if stage >= 3:
            cols.extend(
                [
                    (slot / 9.0).astype("float32"),
                    (b[:, 3] + 0.1 * b[:, 4]).astype("float32"),
                    (hour * (1 + b[:, 4])).astype("float32"),
                    (item_ctr * user_ctr).astype("float32"),
                ]
            )
        out[off : off + bn] = np.column_stack(cols)

    if stage >= 4:
        use = sorted(set([1] + list(range(max(0, out.shape[1] - 8), out.shape[1]))))
        rel = _relative([out[:, j] for j in use])
        out = np.column_stack([out, *rel]).astype("float32")
    assert out.shape[1] == width or stage < 4
    return out


def train_lgb(xf, yf, xv, yv, rounds=800, params=None, fixed_rounds=False):
    p = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 6,
    }
    if params:
        p.update(params)
    dtrain = lgb.Dataset(xf, label=yf)
    dval = lgb.Dataset(xv, label=yv, reference=dtrain)
    callbacks = [lgb.log_evaluation(100)]
    if not fixed_rounds:
        callbacks.append(lgb.early_stopping(80, verbose=False))
    model = lgb.train(
        p,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    pred = model.predict(xv, num_iteration=model.best_iteration)
    return model, pred, int(model.best_iteration or rounds)


def run_stage(stage: int, do_final: bool = False):
    state = load_state()
    t0 = time.time()
    day = 6  # primary validation: last training day, first event per user
    cutoff = START + (day - 2) * DAY
    stats = build_stats(cutoff, emb_dim=64 if stage >= 1 else 32)
    fit, val = split_day(day)
    xf = make_features(fit, hist, htimes, stats, stage)
    xv = make_features(val, hist, htimes, stats, stage)
    print(f"Stage {stage}: features={xf.shape[1]} fit={len(fit)} val={len(val)}", flush=True)

    params = None
    if stage >= 5:
        params = {
            "learning_rate": 0.04,
            "num_leaves": 96,
            "min_data_in_leaf": 150,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "lambda_l2": 3.0,
            "max_bin": 127,
        }

    model, pred, best_rounds = train_lgb(xf, fit[:, 5], xv, val[:, 5], rounds=1200, params=params)
    auc = float(roc_auc_score(val[:, 5], pred))
    print(f"Stage {stage} LightGBM AUC={auc:.6f} rounds={best_rounds}", flush=True)

    blend_pred = pred
    if stage >= 6:
        # CatBoost + XGBoost, blend on validation
        cb = CatBoostClassifier(
            iterations=min(600, best_rounds + 50),
            depth=7,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="AUC",
            thread_count=6,
            random_seed=SEED,
            verbose=100,
            allow_writing_files=False,
        )
        cb.fit(xf, fit[:, 5], eval_set=(xv, val[:, 5]), early_stopping_rounds=60)
        cb_pred = cb.predict_proba(xv)[:, 1]
        cb_auc = float(roc_auc_score(val[:, 5], cb_pred))
        print(f"CatBoost AUC={cb_auc:.6f}", flush=True)

        try:
            import xgboost as xgb

            xd = xgb.DMatrix(xf, label=fit[:, 5])
            xv_d = xgb.DMatrix(xv, label=val[:, 5])
            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "eta": 0.05,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "lambda": 2.0,
                "seed": SEED,
                "nthread": 6,
            }
            xm = xgb.train(
                xgb_params,
                xd,
                num_boost_round=best_rounds,
                evals=[(xv_d, "val")],
                early_stopping_rounds=60,
                verbose_eval=100,
            )
            xgb_pred = xm.predict(xv_d)
            xgb_auc = float(roc_auc_score(val[:, 5], xgb_pred))
            print(f"XGBoost AUC={xgb_auc:.6f}", flush=True)
        except Exception as e:
            print("XGBoost skipped:", e, flush=True)
            xgb_pred = pred
            xgb_auc = auc
            xm = None

        # Grid blend weights on validation
        best_blend_auc = -1.0
        best_w = (1.0, 0.0, 0.0)
        for w_lgb in np.linspace(0.2, 0.8, 7):
            for w_cb in np.linspace(0.0, 0.6, 7):
                w_x = 1.0 - w_lgb - w_cb
                if w_x < -1e-9:
                    continue
                w_x = max(w_x, 0.0)
                s = w_lgb + w_cb + w_x
                w = (w_lgb / s, w_cb / s, w_x / s)
                bp = w[0] * pred + w[1] * cb_pred + w[2] * xgb_pred
                ba = float(roc_auc_score(val[:, 5], bp))
                if ba > best_blend_auc:
                    best_blend_auc = ba
                    best_w = w
                    blend_pred = bp
        auc = best_blend_auc
        print(f"Blend AUC={auc:.6f} weights={best_w}", flush=True)
        state["blend_weights"] = list(best_w)
        state["cb_rounds"] = int(cb.tree_count_)
        if xm is not None:
            state["xgb_rounds"] = int(xm.best_iteration + 1)

    # Confirm on day 5 (frozen settings)
    cutoff5 = START + (5 - 2) * DAY
    stats5 = build_stats(cutoff5, emb_dim=64 if stage >= 1 else 32)
    fit5, val5 = split_day(5)
    xf5 = make_features(fit5, hist, htimes, stats5, stage)
    xv5 = make_features(val5, hist, htimes, stats5, stage)
    model5, pred5, _ = train_lgb(
        xf5, fit5[:, 5], xv5, val5[:, 5], rounds=best_rounds, params=params, fixed_rounds=True
    )
    # Force exact rounds from day6 for fair confirm when stage<6
    pred5 = model5.predict(xv5, num_iteration=min(best_rounds, model5.best_iteration or best_rounds))
    auc5 = float(roc_auc_score(val5[:, 5], pred5))
    print(f"Stage {stage} confirm day5 LGB AUC={auc5:.6f}", flush=True)

    before = state.get("best_auc", 0.0)
    # Leakage guard
    if auc > 0.85 or (before > 0 and auc - before > 0.03 and stage > 0):
        print(
            "WARNING: suspicious AUC jump — audit features for leakage before keeping.",
            flush=True,
        )

    kept = (stage < 6 and auc > max(before, 0.8215282579633503) + 1e-5
            and auc5 > max(state.get("best_auc_day5", 0.0), 0.814599936313616))
    if kept:
        state["best_stage"] = stage
        state["best_auc"] = auc
        state["best_auc_day5"] = auc5
        state["best_rounds"] = best_rounds
        state["best_features"] = int(xf.shape[1])
    state["history"].append(
        {
            "stage": stage,
            "auc_day6": auc,
            "auc_day5": auc5,
            "rounds": best_rounds,
            "features": int(xf.shape[1]),
            "kept": kept,
        }
    )
    save_state(state)
    runtime = time.time() - t0
    append_log(
        stage,
        f"pipeline stage {stage}",
        before if before else 0.0,
        auc,
        runtime,
        kept,
        f"day5={auc5:.6f} feats={xf.shape[1]} rounds={best_rounds}",
    )

    if do_final and kept:
        print("Final fit on last training days + write separate pipeline candidate", flush=True)
        final_cutoff = START + 6 * DAY
        stats_f = build_stats(final_cutoff, emb_dim=64 if stage >= 1 else 32)
        fit_f = np.asarray(train[train[:, 2] >= START + 4 * DAY])
        xf_f = make_features(fit_f, hist, htimes, stats_f, stage)
        dtrain = lgb.Dataset(xf_f, label=fit_f[:, 5])
        p = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05 if stage < 5 else 0.04,
            "num_leaves": 63 if stage < 5 else 96,
            "min_data_in_leaf": 200 if stage < 5 else 150,
            "feature_fraction": 0.9 if stage < 5 else 0.85,
            "bagging_fraction": 0.8 if stage < 5 else 0.85,
            "bagging_freq": 1,
            "lambda_l2": 2.0 if stage < 5 else 3.0,
            "verbosity": -1,
            "seed": SEED,
            "num_threads": 6,
        }
        final = lgb.train(p, dtrain, num_boost_round=best_rounds)
        xe = make_features(evaluation, ehist, etimes, stats_f, stage)
        prob = final.predict(xe)
        if stage >= 6:
            w = state.get("blend_weights", [1, 0, 0])
            cb = CatBoostClassifier(
                iterations=state.get("cb_rounds", best_rounds),
                depth=7,
                learning_rate=0.08,
                loss_function="Logloss",
                thread_count=6,
                random_seed=SEED,
                verbose=50,
                allow_writing_files=False,
            )
            cb.fit(xf_f, fit_f[:, 5])
            cb_p = cb.predict_proba(xe)[:, 1]
            prob = w[0] * prob + w[1] * cb_p
            if w[2] > 0:
                import xgboost as xgb

                xd = xgb.DMatrix(xf_f, label=fit_f[:, 5])
                xm = xgb.train(
                    {
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "eta": 0.05,
                        "max_depth": 7,
                        "subsample": 0.8,
                        "colsample_bytree": 0.9,
                        "lambda": 2.0,
                        "seed": SEED,
                        "nthread": 6,
                    },
                    xd,
                    num_boost_round=state.get("xgb_rounds", best_rounds),
                )
                prob = prob + w[2] * xm.predict(xgb.DMatrix(xe))
                # renormalize already convex combination
        prob = np.clip(prob, 0, 1)
        assert len(prob) == 529390 and np.isfinite(prob).all()
        out = Path(f"submission_pipeline_stage{stage}_candidate.csv")
        seen = set()
        with EVALUATION_CSV.open(newline="", encoding="utf-8") as src, out.open(
            "x", newline="", encoding="utf-8"
        ) as dest:
            writer = csv.writer(dest)
            writer.writerow(["rowid", "predicted_probability"])
            for row, p in zip(csv.DictReader(src), prob, strict=True):
                assert row["rowid"] not in seen
                seen.add(row["rowid"])
                writer.writerow([row["rowid"], format(float(p), ".10g")])
        assert len(seen) == 529390
        # Also copy into artifacts
        print(f"Wrote {out} ({len(seen)} rows); independent audit required", flush=True)

    return auc, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=range(0, 8))
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    run_stage(args.stage, do_final=args.final)


if __name__ == "__main__":
    main()
