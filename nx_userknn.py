"""nx idea B: user-based neighbourhood CF scores (never tested in this project; only ItemKNN was).

ONE code path, --day 5 / 6 / 7 (evaluation). For cutoff c:
  X  = user x item click matrix from ledger(c) (unique clicks with time < c), IDF-weighted, cosine-normalised;
  Xd = same with recency decay exp(-(c - t)/tau) on each click;
  E  = user x item exposure counts from impressions with time < c (no labels).
Each scored event's own vector = ledger clicks UNION its supplied history (all < event time). K most similar users
(the user itself excluded) by cosine; features for candidate i:
  knn_click   = sum_k sim_k [v_k clicked i]           knn_click_norm = knn_click / sum_k sim_k
  knn_recent  = sum_k sim_k [v_k clicked i in the last 2 days before c]
  knn_decay   = sum_k sim_k * decayed click weight     knn_n = number of neighbours that clicked i
  knn_expo    = sum_k sim_k [v_k was shown i]           knn_ctr = knn_click / (knn_expo + 1)   (neighbour CTR on the candidate)
Screening output: standalone and V53 rank-blend adjusted AUC per feature; features saved for a possible head/expert step.
"""
import argparse, json, time, gc
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import build_ledger, merged_windows, weighted_auc, rank, cutoff_of, NI, NU, DAY, PAD


ALPHAS = (1, 2, 4)


def features_for(day, K=100, tau_days=2., chunk=1000, log=print):
    cut = cutoff_of(day); tic = time.monotonic(); ledger = build_ledger(cut); rows = ledger.pop('rows'); n = ledger['n_events']
    # scored events
    if day == 7:
        ev = np.load(R / 'evaluation.npy', mmap_mode='r'); srows = np.asarray(ev); s_h = np.asarray(np.load(R / 'evaluation_hist.npy', mmap_mode='r')); s_ht = np.asarray(np.load(R / 'evaluation_times.npy', mmap_mode='r')); z = None
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); t = np.load(R / 'train.npy', mmap_mode='r'); h = np.load(R / 'train_hist.npy', mmap_mode='r'); ht = np.load(R / 'train_times.npy', mmap_mode='r')
        ei = z['events'].reshape(-1, 10)[:, 0]; srows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()]); assert np.array_equal(srows[:, 5], z['y']); s_h = np.asarray(h[ei]); s_ht = np.asarray(ht[ei])
    s_u = srows[::10, 0].astype(np.int64); s_t = srows[::10, 2].astype(np.int64); s_c = srows[:, 1].reshape(-1, 10).astype(np.int64); assert s_t.min() >= cut
    # matrices from the ledger
    lu, li, lt = ledger['user'], ledger['item'].astype(np.int64), ledger['time']
    decay = np.exp(-(cut - lt) / (tau_days * DAY)); recent = (lt >= cut - 2 * DAY).astype(np.float32)
    X = sp.csr_matrix((np.ones(len(lu), np.float32), (lu, li)), shape=(NU, NI)); X.sum_duplicates(); X.data[:] = 1.
    Xd = sp.csr_matrix((decay.astype(np.float32), (lu, li)), shape=(NU, NI)); Xd.sum_duplicates()
    Xr = sp.csr_matrix((recent, (lu, li)), shape=(NU, NI)); Xr.sum_duplicates(); Xr.eliminate_zeros(); Xr.data[:] = 1.  # (bug in screen 1: explicit zeros were set to 1)
    expo = sp.csr_matrix((np.ones(n * 10, np.float32), (rows[:, 0], rows[:, 1])), shape=(NU, NI)); expo.sum_duplicates()
    lab = rows[rows[:, 5] == 1]; XL = sp.csr_matrix((np.ones(len(lab), np.float32), (lab[:, 0], lab[:, 1])), shape=(NU, NI)); XL.sum_duplicates(); del rows, lab
    XT = X.T.tocsr(); XLT = XL.T.tocsr(); EB = expo.copy(); EB.data[:] = 1.; EBT = EB.T.tocsr(); pop_click = np.asarray(X.sum(0)).ravel() / NU; g_ctr = float(XL.sum() / max(expo.sum(), 1))
    idf = np.log((1. + NU) / (1. + np.asarray((X > 0).sum(0)).ravel())).astype(np.float32)
    Xn = sp.diags(1. / np.maximum(np.sqrt(np.asarray(X.multiply(idf ** 2).sum(1)).ravel()), 1e-8)) @ X.multiply(idf); Xn = sp.csr_matrix(Xn)
    # query vectors: merged ledger + supplied history (binary, idf, normalised)
    L = int(max(400, (ledger['end'] - ledger['start']).max() + 21)); it, st, ok = merged_windows(ledger, s_u, s_t, s_h, s_ht, L)
    qi = it[ok]; qr = np.repeat(np.arange(len(s_u)), ok.sum(1)); Q = sp.csr_matrix((np.ones(len(qi), np.float32), (qr, qi)), shape=(len(s_u), NI)); Q.sum_duplicates(); Q.data[:] = 1.
    qlen = np.asarray(Q.sum(1)).ravel(); Qn = sp.csr_matrix(sp.diags(1. / np.maximum(np.sqrt(np.asarray(Q.multiply(idf ** 2).sum(1)).ravel()), 1e-8)) @ Q.multiply(idf)); del it, st, ok
    log(json.dumps(dict(stage='matrices', ledger=int(len(lu)), scored_events=int(len(s_u)), mean_query_len=float(qlen.mean()), elapsed_s=round(time.monotonic() - tic, 1))))
    XnT = Xn.T.tocsr()
    # sorted key lookups for neighbour clicks / exposures
    def keyed(M):
        M = M.tocoo(); k = M.row.astype(np.int64) * NI + M.col; o = np.argsort(k); return k[o], M.data[o].astype(np.float32)
    kX, vX = keyed(X); kD, vD = keyed(Xd); kRr, vR = keyed(Xr); kE, vE = keyed(expo)
    def lookup(keys, k, v):
        p = np.searchsorted(k, keys); p = np.minimum(p, len(k) - 1); found = k[p] == keys; return np.where(found, v[p], 0.)
    names = ['knn_click', 'knn_click_norm', 'knn_recent', 'knn_decay', 'knn_n', 'knn_expo', 'knn_ctr', 'knn_top_sim', 'knn_sim_sum']
    for al in ALPHAS: names += [f'ucf{al}_click', f'ucf{al}_lift', f'ucf{al}_expo', f'ucf{al}_ctr', f'ucf{al}_click_vs_expo']
    out = {name: np.zeros((len(s_u), 10), np.float32) for name in names}
    def gather(MT, W, cflat):  # sum_v W[event, v] * M[v, c] for each (event, candidate), using only the candidate's nonzero users
        sub = MT[cflat]; er = np.repeat(np.arange(len(cflat)), np.diff(sub.indptr)); vals = W[er // 10, sub.indices] * sub.data
        return np.bincount(er, weights=vals, minlength=len(cflat)).reshape(-1, 10)
    for off in range(0, len(s_u), chunk):
        sl = slice(off, off + chunk); S = (Qn[sl] @ XnT).toarray(); S[np.arange(S.shape[0]), s_u[sl]] = 0.  # never the user's own row
        nb = np.argpartition(-S, K, axis=1)[:, :K]; sim = np.take_along_axis(S, nb, 1); sim = np.maximum(sim, 0.)
        keys = nb[:, :, None].astype(np.int64) * NI + s_c[sl][:, None, :]  # (B, K, 10)
        c = lookup(keys, kX, vX); d = lookup(keys, kD, vD); r = lookup(keys, kRr, vR); e = lookup(keys, kE, vE)
        ssum = sim.sum(1); out['knn_click'][sl] = (sim[:, :, None] * c).sum(1); out['knn_click_norm'][sl] = out['knn_click'][sl] / np.maximum(ssum, 1e-8)[:, None]
        out['knn_recent'][sl] = (sim[:, :, None] * r).sum(1); out['knn_decay'][sl] = (sim[:, :, None] * d).sum(1); out['knn_n'][sl] = (c > 0).sum(1)
        out['knn_expo'][sl] = (sim[:, :, None] * np.log1p(e)).sum(1); out['knn_ctr'][sl] = out['knn_click'][sl] / (out['knn_expo'][sl] + 1.)
        out['knn_top_sim'][sl] = np.broadcast_to(sim.max(1)[:, None], (sim.shape[0], 10)); out['knn_sim_sum'][sl] = np.broadcast_to(ssum[:, None], (sim.shape[0], 10))
        cflat = s_c[sl].ravel(); S = np.maximum(S, 0.)
        for al in ALPHAS:  # all users as neighbours, similarity sharpened by a power
            W = S ** al; ws = np.maximum(W.sum(1), 1e-12)[:, None]; ck = gather(XT, W, cflat) / ws; ex = gather(EBT, W, cflat) / ws; cl = gather(XLT, W, cflat) / ws
            out[f'ucf{al}_click'][sl] = ck; out[f'ucf{al}_lift'][sl] = ck / (pop_click[s_c[sl]] + 1e-5); out[f'ucf{al}_expo'][sl] = ex
            out[f'ucf{al}_ctr'][sl] = (cl + 20 * g_ctr * 1e-4) / (ex + 1e-4 * 20 / 1.); out[f'ucf{al}_click_vs_expo'][sl] = ck / (ex + 1e-4)
        if (off // chunk) % 10 == 0: log(json.dumps(dict(stage='knn', done=int(off + S.shape[0]), elapsed_s=round(time.monotonic() - tic, 1))))
        del S, nb, sim, keys, c, d, r, e, W
    return z, {k: v.ravel().astype(np.float64) for k, v in out.items()}, dict(K=K, tau_days=tau_days, cutoff=int(cut), elapsed_s=time.monotonic() - tic)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, default=6); ap.add_argument('--K', type=int, default=100); ap.add_argument('--tau', type=float, default=2.); ap.add_argument('--tag', default='screen'); a = ap.parse_args()
    z, f, info = features_for(a.day, a.K, a.tau)
    np.savez_compressed(R / f'nx_userknn_{a.tag}_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
    if z is not None:
        y, e, v53 = z['y'], z['events'], z['pred']; base = weighted_auc(e, y, v53); rv = rank(v53); rep = dict(info=info, v53_adjusted=base, features={})
        H = np.load(R / 'train_hist.npy', mmap_mode='r'); hl = (H[e] >= 0).sum(1); short = hl <= 4
        for k, v in f.items():
            nz = float((v != 0).mean()); r = rank(v); d = dict(nonzero_share=nz, raw_auc=float(roc_auc_score(y, v)), raw_adjusted=weighted_auc(e, y, v), raw_auc_short=float(roc_auc_score(y[short], v[short])), v53_auc_short=float(roc_auc_score(y[short], v53[short])))
            for w in (.02, .05, .1, .2): p = (1 - w) * rv + w * r; d[f'blend{w}_adjusted_gain'] = weighted_auc(e, y, p) - base
            rep['features'][k] = d; print(k, json.dumps(d), flush=True)
        (R / f'nx_userknn_{a.tag}_day{a.day}_report.json').write_text(json.dumps(rep, indent=1))
