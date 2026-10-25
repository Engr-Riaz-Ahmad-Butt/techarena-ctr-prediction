"""Row-level stacker over out-of-sample V20 + DIN with event-context features.

Train on one validation day, test on the other (both directions). All inputs
are out-of-sample predictions or strictly-prior user statistics.
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as ROOT
from v21_din import DAY, START, first_event_block, hist, htimes, train

rank = lambda v: rankdata(v, method="average") / (len(v) + 1)
NU = int(train[:, 0].max()) + 1


def user_stats(cut_ts):
    prior = np.asarray(train[train[:, 2] < cut_ts])
    P = prior.reshape(len(prior) // 10, 10, 7)
    u = P[:, 0, 0]
    has = P[:, :, 5].max(1)
    cnt = np.bincount(u, minlength=NU).astype(np.float64)
    clk = np.bincount(u, weights=has, minlength=NU)
    return cnt, clk


def features(rows, base, din, cut_ts, histories, times):
    n = len(rows) // 10
    R = rows.reshape(n, 10, 7)
    head = R[:, 0, :]
    B = base.reshape(n, 10)
    D = din.reshape(n, 10)
    cnt, clk = user_stats(cut_ts)
    u = head[:, 0]
    ucnt = cnt[u]
    uctr = (clk[u] + 0.634 * 5) / (ucnt + 5)
    h = np.asarray(histories[head[:, 6]])
    hl = (h >= 0).sum(1)
    t = np.asarray(times[head[:, 6]])
    last_gap = np.where(hl > 0, (head[:, 2] - t.max(1)) / 3600.0, 1e4)
    Bs = B.sum(1, keepdims=True)
    Bn = B / Bs
    ent = -(Bn * np.log(Bn + 1e-12)).sum(1, keepdims=True)
    Dent = -(D * np.log(D + 1e-12)).sum(1, keepdims=True)
    Brank = B.argsort(1).argsort(1)
    Drank = D.argsort(1).argsort(1)
    hod = ((head[:, 2] - START) % DAY) / 3600.0
    ev = np.column_stack([
        B.max(1), Bs.ravel(), B.std(1), ent.ravel(), B.min(1),
        D.max(1), Dent.ravel(), D.std(1),
        ucnt, uctr, hl, last_gap, head[:, 3], head[:, 4], hod,
        (B.argmax(1) == D.argmax(1)).astype(float),
    ])
    X = np.column_stack([
        B.ravel(), Bn.ravel(), Brank.ravel(), (B / B.max(1, keepdims=True)).ravel(),
        D.ravel(), Drank.ravel(), (D / D.max(1, keepdims=True)).ravel(),
        (B * D).ravel(), (Bn * D).ravel(),
        np.repeat(ev, 10, axis=0),
    ])
    return X.astype(np.float32)


NAMES = ["b", "b_norm", "b_rank", "b_rel", "d", "d_rank", "d_rel", "bd", "bnd",
         "ev_bmax", "ev_bsum", "ev_bstd", "ev_bent", "ev_bmin", "ev_dmax", "ev_dent", "ev_dstd",
         "u_cnt", "u_ctr", "hist_len", "last_gap_h", "bucket", "daytype", "hour", "agree"]


def load_day(day):
    val = first_event_block(day)
    zs = [np.load(ROOT / f"v21_fastseeds_day{day}_s{s}_ep1_raw.npz") for s in range(3)]
    y = zs[0]["y"].astype(np.float64)
    base = zs[0]["base"].astype(np.float64)
    din = np.mean([z["raw"] for z in zs], 0).astype(np.float64)
    X = features(val, base, din, START + day * DAY, hist, htimes)
    return X, y, base, din


PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=15,
              min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, num_threads=12, seed=42)


def main():
    data = {d: load_day(d) for d in (6, 5)}
    out = {}
    for tr, te in ((5, 6), (6, 5)):
        Xtr, ytr, _, _ = data[tr]
        Xte, yte, bte, dte = data[te]
        ds = lgb.Dataset(Xtr, ytr, feature_name=NAMES)
        dv = lgb.Dataset(Xte, yte, reference=ds)
        m = lgb.train(PARAMS, ds, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        best_it = m.best_iteration
        # also a fixed-round variant to avoid selecting on the test day
        for rounds, tag in ((best_it, "early_stop"), (600, "fixed600"), (300, "fixed300")):
            p = m.predict(Xte, num_iteration=rounds)
            v20 = float(roc_auc_score(yte, bte))
            cur = float(roc_auc_score(yte, 0.8 * rank(bte) + 0.2 * rank(dte)))
            stk = float(roc_auc_score(yte, p))
            mix = max((float(roc_auc_score(yte, (1 - w) * rank(bte) + w * rank(p))), w) for w in (0.3, 0.5, 0.7, 1.0))
            n = len(yte) // 10
            has = yte.reshape(n, 10).max(1)
            ev_auc = float(roc_auc_score(has, p.reshape(n, 10).max(1)))
            row = dict(train=tr, test=te, rounds=int(rounds), tag=tag, v20=v20, current_blend=cur,
                       stack=stk, stack_mix=mix[0], mix_w=mix[1], delta_vs_current=stk - cur, event_auc=ev_auc)
            print(json.dumps(row), flush=True)
            out[f"{tr}->{te}:{tag}"] = row
        imp = sorted(zip(NAMES, m.feature_importance("gain")), key=lambda t: -t[1])[:10]
        print("top gain:", [(k, int(v)) for k, v in imp], flush=True)
    (ROOT / "v21_event_stack_report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
