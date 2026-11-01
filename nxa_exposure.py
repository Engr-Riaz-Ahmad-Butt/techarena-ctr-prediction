"""nxa idea C (owner: Idea-A improver session; copy of nx_exposure.py with nxa_ outputs): exposure-policy factorization. The simulator's recommender chose the 10 candidates from the user's latent state,
so P(item shown | user) carries taste even for users with 0-4 history clicks (they still have pre-cutoff impressions).

ONE code path, --day 5 / 6 / 7. For cutoff c: E = user x item exposure counts from impressions with time < c (no labels),
log1p + IDF + row normalisation, truncated SVD (k factors) -> user exposure factors Ue, item factors Ve.
Features per candidate i for user u: expo_dot = Ue[u].Ve[i]; expo_cos; expo_rank = rank of i among all 50k items under expo_dot
(catalogue-level, 0 = top); expo_typ = cosine(Ve[i], mean Ve of the items previously shown to u); expo_n = log1p(#prior impressions of u);
click_rank = rank of i among 50k under the CLICK model (ledger clicks SVD, same construction) for comparison.
Screen: standalone adjusted AUC, V53 rank blends, and AUC restricted to the 0-4 history bins.
"""
import argparse, json, time, gc
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import build_ledger, weighted_auc, rank, cutoff_of, NI, NU, DAY


def factorize(M, k, seed):
    idf = np.log((1. + NU) / (1. + np.asarray((M > 0).sum(0)).ravel())).astype(np.float32); M = M.multiply(idf).tocsr(); M.data = np.log1p(M.data)
    norm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); M = sp.csr_matrix(sp.diags(1. / np.maximum(norm, 1e-8)) @ M)
    svd = TruncatedSVD(n_components=k, n_iter=5, random_state=seed); U = svd.fit_transform(M).astype(np.float32); V = (svd.components_.T * svd.singular_values_[None, :] ** 0).astype(np.float32)
    return U, V


def features_for(day, k=32, seed=4403, log=print):
    cut = cutoff_of(day); tic = time.monotonic(); ledger = build_ledger(cut); rows = ledger.pop('rows'); n = ledger['n_events']
    if day == 7:
        srows = np.asarray(np.load(R / 'evaluation.npy', mmap_mode='r')); z = None
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); t = np.load(R / 'train.npy', mmap_mode='r'); ei = z['events'].reshape(-1, 10)[:, 0]
        srows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()]); assert np.array_equal(srows[:, 5], z['y'])
    s_u = srows[::10, 0].astype(np.int64); s_t = srows[::10, 2].astype(np.int64); s_c = srows[:, 1].reshape(-1, 10).astype(np.int64); assert s_t.min() >= cut
    E = sp.csr_matrix((np.ones(n * 10, np.float32), (rows[:, 0], rows[:, 1])), shape=(NU, NI)); E.sum_duplicates(); del rows
    Ue, Ve = factorize(E, k, seed); C = sp.csr_matrix((np.ones(len(ledger['user']), np.float32), (ledger['user'], ledger['item'])), shape=(NU, NI)); C.sum_duplicates(); Uc, Vc = factorize(C, k, seed + 1)
    log(json.dumps(dict(stage='svd', exposures=int(E.nnz), clicks=int(C.nnz), elapsed_s=round(time.monotonic() - tic, 1))))
    en = np.asarray(E.sum(1)).ravel(); Ven = Ve / np.maximum(np.linalg.norm(Ve, axis=1, keepdims=True), 1e-8)
    out = {name: np.zeros((len(s_u), 10), np.float32) for name in ['expo_dot', 'expo_cos', 'expo_rank', 'expo_typ', 'expo_n', 'click_dot', 'click_rank', 'expo_pop']}
    item_pop = np.log1p(np.asarray(E.sum(0)).ravel())
    for off in range(0, len(s_u), 250):
        sl = slice(off, off + 250); u = s_u[sl]; c = s_c[sl]; ue = Ue[u]; uc = Uc[u]
        full = ue @ Ve.T; dots = np.take_along_axis(full, c, 1); out['expo_dot'][sl] = dots
        out['expo_rank'][sl] = (full[:, None, :] > dots[:, :, None]).sum(2) / NI; del full
        out['expo_cos'][sl] = dots / np.maximum(np.linalg.norm(ue, axis=1)[:, None] * np.linalg.norm(Ve[c], axis=2), 1e-8)
        fullc = uc @ Vc.T; dc = np.take_along_axis(fullc, c, 1); out['click_dot'][sl] = dc; out['click_rank'][sl] = (fullc[:, None, :] > dc[:, :, None]).sum(2) / NI; del fullc
        prof = np.asarray(E[u] @ Ven); prof /= np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-8); out['expo_typ'][sl] = np.einsum('bd,bcd->bc', prof, Ven[c])
        out['expo_n'][sl] = np.broadcast_to(np.log1p(en[u])[:, None], (len(u), 10)); out['expo_pop'][sl] = item_pop[c]
    return z, {kk: v.ravel().astype(np.float64) for kk, v in out.items()}, dict(k=k, cutoff=int(cut), elapsed_s=time.monotonic() - tic)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, default=6); ap.add_argument('--k', type=int, default=32); ap.add_argument('--tag', default='screen'); a = ap.parse_args()
    z, f, info = features_for(a.day, a.k)
    np.savez_compressed(R / f'nxa_exposure_{a.tag}_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
    if z is not None:
        y, e, v53 = z['y'], z['events'], z['pred']; base = weighted_auc(e, y, v53); rv = rank(v53); rep = dict(info=info, v53_adjusted=base, features={})
        H = np.load(R / 'train_hist.npy', mmap_mode='r'); hl = (H[e] >= 0).sum(1); short = hl <= 4; zero = hl == 0
        for kk, v in f.items():
            for sign, vv in (('+', v), ('-', -v)):
                r = rank(vv); d = dict(raw_auc=float(roc_auc_score(y, vv)), raw_adjusted=weighted_auc(e, y, vv), raw_auc_short=float(roc_auc_score(y[short], vv[short])), raw_auc_zero=float(roc_auc_score(y[zero], vv[zero])), v53_auc_short=float(roc_auc_score(y[short], v53[short])), v53_auc_zero=float(roc_auc_score(y[zero], v53[zero])))
                for w in (.02, .05, .1): p = (1 - w) * rv + w * r; d[f'blend{w}_adjusted_gain'] = weighted_auc(e, y, p) - base; d[f'blend{w}_short_auc'] = float(roc_auc_score(y[short], p[short]))
                rep['features'][sign + kk] = d; print(sign + kk, json.dumps(d), flush=True)
        (R / f'nxa_exposure_{a.tag}_day{a.day}_report.json').write_text(json.dumps(rep, indent=1))
