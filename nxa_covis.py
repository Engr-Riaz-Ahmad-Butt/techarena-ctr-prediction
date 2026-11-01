"""nxa F1: time-decayed multi-anchor covisitation + exposure-conditioned pair CTR (owner: Idea-A improver session).

Differs from the closed screens (whole-prefix co-click ItemKNN cosine; last-item directed transition counts at lags 1-3):
  R_time  co-clicks of the same user within 24 h, weight exp(-gap/6 h), undirected, degree-normalised; all 20 anchors with
          recency-rank weights 1/(rank+1); score = weighted mean / max / evidence count.
  R_ctr   exposure-conditioned pair CTR: over OTHER users' training events (time < cutoff), when item c was SHOWN to a user whose
          supplied history contained h, how often was c clicked; lift = smoothed pair CTR / item CTR; anchors weighted as above.
          The scored user's own earlier events are removed from the pair counts (own repeat/negative exposure is a closed line).
ONE code path for --day 5 / 6 / 7 (cutoff start + day*86400); every count uses events/clicks with time < cutoff; scored anchors are
ledger clicks (< cutoff) merged with the event's own supplied history (stamps < event time), last 20.
"""
import argparse, json, time, gc
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import build_ledger, merged_windows, weighted_auc, rank, cutoff_of, NI, NU, PAD, DAY

HC = np.int64(NI) * NI


def features_for(day, window_h=24., tau_h=6., alpha=20., log=print):
    tic = time.monotonic(); cut = cutoff_of(day); ledger = build_ledger(cut); rows = ledger.pop('rows'); n = ledger['n_events']
    t = np.load(R / 'train.npy', mmap_mode='r'); H = np.load(R / 'train_hist.npy', mmap_mode='r')
    if day == 7:
        srows = np.asarray(np.load(R / 'evaluation.npy', mmap_mode='r')); s_h = np.asarray(np.load(R / 'evaluation_hist.npy', mmap_mode='r')); s_ht = np.asarray(np.load(R / 'evaluation_times.npy', mmap_mode='r')); z = None
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); ei = z['events'].reshape(-1, 10)[:, 0]; srows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()]); assert np.array_equal(srows[:, 5], z['y'])
        s_h = np.asarray(H[ei]); s_ht = np.asarray(np.load(R / 'train_times.npy', mmap_mode='r')[ei])
    s_u = srows[::10, 0].astype(np.int64); s_t = srows[::10, 2].astype(np.int64); s_c = srows[:, 1].reshape(-1, 10).astype(np.int64); assert s_t.min() >= cut
    A, AT, ok = merged_windows(ledger, s_u, s_t, s_h, s_ht, 20)  # anchors oldest -> newest, left padded
    rk = np.where(ok, ok[:, ::-1].cumsum(1)[:, ::-1] - 1, 0); aw = np.where(ok, 1. / (rk + 1.), 0.)  # newest anchor rank 0
    # ---- R_time: within-window co-clicks from the ledger (sorted by user, time)
    li, lt, lu = ledger['item'].astype(np.int64), ledger['time'], ledger['user']; ra, rb, rw = [], [], []
    for k in range(1, 40):
        same = (lu[k:] == lu[:-k]) & (lt[k:] - lt[:-k] <= window_h * 3600) & (li[k:] != li[:-k])
        if not same.any(): break
        g = (lt[k:] - lt[:-k])[same] / 3600.; ra += [li[:-k][same], li[k:][same]]; rb += [li[k:][same], li[:-k][same]]; w = np.exp(-g / tau_h); rw += [w, w]
    C = sp.csr_matrix((np.concatenate(rw).astype(np.float32), (np.concatenate(ra), np.concatenate(rb))), shape=(NI, NI)); C.sum_duplicates(); del ra, rb, rw
    deg = np.asarray(C.sum(1)).ravel(); Cn = sp.csr_matrix(sp.diags(1. / np.sqrt(deg + 1.)) @ C @ sp.diags(1. / np.sqrt(deg + 1.))).tocoo()
    ck = Cn.row.astype(np.int64) * NI + Cn.col; o = np.argsort(ck); ck, cv = ck[o], Cn.data[o].astype(np.float64); del C, Cn, o
    log(json.dumps(dict(stage='R_time', pairs=int(len(ck)), elapsed_s=round(time.monotonic() - tic, 1))))
    qk = A[:, None, :] * NI + s_c[:, :, None]  # (events, 10, 20) anchor-candidate keys
    def lookup(keys, k, v):
        p = np.minimum(np.searchsorted(k, keys), len(k) - 1); return np.where(k[p] == keys, v[p], 0.)
    sv = lookup(qk, ck, cv) * ok[:, None, :]; wsum = np.maximum(aw.sum(1), 1e-9)[:, None]
    out = dict(rt_mean=(sv * aw[:, None, :]).sum(2) / wsum, rt_max=sv.max(2), rt_cnt=(sv > 0).sum(2).astype(np.float64)); del sv
    # ---- R_ctr: exposure-conditioned pair CTR over training events (< cutoff), other users only
    Q = np.unique(qk[np.broadcast_to(ok[:, None, :], qk.shape)]); uq = np.repeat(s_u, 200).reshape(qk.shape)
    Q2 = np.unique((uq * HC + qk)[np.broadcast_to(ok[:, None, :], qk.shape)])
    S = np.zeros(len(Q)); K = np.zeros(len(Q)); S2 = np.zeros(len(Q2)); K2 = np.zeros(len(Q2))
    items_all = rows[:, 1].reshape(-1, 10).astype(np.int64); y_all = rows[:, 5].reshape(-1, 10).astype(np.float64); users_all = rows[::10, 0].astype(np.int64)
    for off in range(0, n, 20000):
        hh = np.asarray(H[off:min(off + 20000, n)]).astype(np.int64); m = len(hh); it = items_all[off:off + m]; yy = y_all[off:off + m]; uu = users_all[off:off + m]
        key = hh[:, None, :] * NI + it[:, :, None]; valid = np.broadcast_to((hh >= 0)[:, None, :], key.shape); kv = key[valid]; yv = np.broadcast_to(yy[:, :, None], key.shape)[valid]
        p = np.minimum(np.searchsorted(Q, kv), len(Q) - 1); hit = Q[p] == kv; S += np.bincount(p[hit], minlength=len(Q)); K += np.bincount(p[hit], weights=yv[hit], minlength=len(Q))
        k2 = np.broadcast_to(uu[:, None, None], key.shape)[valid] * HC + kv; p2 = np.minimum(np.searchsorted(Q2, k2), len(Q2) - 1); h2 = Q2[p2] == k2
        S2 += np.bincount(p2[h2], minlength=len(Q2)); K2 += np.bincount(p2[h2], weights=yv[h2], minlength=len(Q2))
    g = float(y_all.mean()); ishown = np.bincount(items_all.ravel(), minlength=NI); iclick = np.bincount(items_all.ravel(), weights=y_all.ravel(), minlength=NI); pc = (iclick + 50 * g) / (ishown + 50)
    del items_all, y_all, users_all, rows; gc.collect()
    s_all = lookup(qk, Q, S); k_all = lookup(qk, Q, K); uk = uq * HC + qk; s_own = lookup(uk, Q2, S2); k_own = lookup(uk, Q2, K2)
    so = np.maximum(s_all - s_own, 0) * ok[:, None, :]; ko = np.maximum(k_all - k_own, 0) * ok[:, None, :]; pcc = pc[s_c][:, :, None]
    lift = np.log(((ko + alpha * pcc) / (so + alpha)) / pcc); out.update(ctr_mean=(lift * aw[:, None, :]).sum(2) / wsum, ctr_max=np.where(ok[:, None, :], lift, -9).max(2),
                                                                    ctr_shown=np.log1p(so.sum(2)), ctr_clicks=np.log1p(ko.sum(2)), ctr_pooled=np.log(((ko.sum(2) + alpha * pc[s_c]) / (so.sum(2) + alpha)) / pc[s_c]))
    out = {k: np.asarray(v, np.float64).ravel() for k, v in out.items()}
    for k, v in out.items(): assert v.shape == (len(s_u) * 10,) and np.isfinite(v).all(), k
    log(json.dumps(dict(stage='done', pairs_query=int(len(Q)), own_keys=int(len(Q2)), elapsed_s=round(time.monotonic() - tic, 1))))
    return z, out, dict(day=day, cutoff=int(cut), window_h=window_h, tau_h=tau_h, alpha=alpha, elapsed_s=time.monotonic() - tic)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, required=True); a = ap.parse_args()
    z, f, info = features_for(a.day)
    np.savez_compressed(R / f'nxa_covis_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
    if z is not None:
        y, e, v53 = z['y'], z['events'], z['pred']; base = weighted_auc(e, y, v53); rv = rank(v53); rep = dict(info=info, v53_adjusted=base, features={})
        Hh = np.load(R / 'train_hist.npy', mmap_mode='r'); short = (Hh[e] >= 0).sum(1) <= 4
        for k, v in f.items():
            d = dict(nonzero=float((v != 0).mean()), raw_adjusted=weighted_auc(e, y, v), raw_auc_short=float(roc_auc_score(y[short], v[short])))
            for w in (.02, .05, .1): d[f'g{w}'] = weighted_auc(e, y, (1 - w) * rv + w * rank(v)) - base
            zc = v - np.repeat(v.reshape(-1, 10).mean(1), 10); zc = zc / max(zc.std(), 1e-12)
            for bt in (.01, .015, .02): d[f'cz{bt}'] = weighted_auc(e, y, rv + bt * zc) - base
            rep['features'][k] = d; print(k.ljust(11), json.dumps({kk: round(vv, 6) for kk, vv in d.items()}), flush=True)
        (R / f'nxa_covis_day{a.day}_report.json').write_text(json.dumps(rep, indent=1))
