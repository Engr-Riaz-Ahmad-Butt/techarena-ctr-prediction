"""nxa idea C, step 2: exposure-policy features, variants + a fixed-form expert (owner: Idea-A improver session).

The simulator's recommender chose each slate from its belief about the user, so the items a user was SHOWN before the cutoff
(no labels) estimate taste even for users with few or no clicks. ONE code path, --day 5 / 6 / 7 (cutoff start + day*86400):
  impressions used = training rows with time < cutoff (label column never read here), all scored events have time >= cutoff.
  Item exposure factors: TruncatedSVD of the IDF-weighted log1p user x item exposure count matrix (k=64, seed 4403).
  For candidate c of user u (all from impressions < cutoff):
    typ{k}_{w}  cosine(candidate factor, user's weighted mean exposure profile), k in {16,32,64},
                weights w: all = uniform, d1 = exp(-age/1 day), r2 = last 2 days only
    cos{k}      cosine of the SVD user factor and the candidate factor (k in {32,64})
    raw32       unnormalised dot of the profile with the candidate factor; n_expo = log1p(#impressions of u)
Output: artifacts/nxa_exposure2_day{d}_features.npz (+ y, events for days 5/6) and a day-6/day-5 screen report.
"""
import argparse, json, time, gc
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc, rank, cutoff_of, NI, NU, DAY

KS = (16, 32, 64)


def load(day):
    cut = cutoff_of(day); t = np.load(R / 'train.npy', mmap_mode='r'); heads_t = np.asarray(t[::10, 2]); n = int(np.searchsorted(heads_t, cut, side='left'))
    imp = np.asarray(t[:n * 10, :3]); assert imp[:, 2].max() < cut  # user, item, time only (no labels)
    if day == 7: srows = np.asarray(np.load(R / 'evaluation.npy', mmap_mode='r')); z = None
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); ei = z['events'].reshape(-1, 10)[:, 0]; srows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()])
        assert np.array_equal(srows[:, 5], z['y']) and np.array_equal(srows[:, 6], z['events'])
    assert srows[:, 2].min() >= cut
    return cut, imp, srows[::10, 0].astype(np.int64), srows[:, 1].reshape(-1, 10).astype(np.int64), z


def features_for(day, seed=4403, log=print):
    tic = time.monotonic(); cut, imp, s_u, s_c, z = load(day)
    age = (cut - imp[:, 2]).astype(np.float64) / DAY
    def mat(w): M = sp.csr_matrix((w.astype(np.float32), (imp[:, 0], imp[:, 1])), shape=(NU, NI)); M.sum_duplicates(); return M
    E = mat(np.ones(len(imp))); Ed = mat(np.exp(-age)); Er = mat((age <= 2).astype(np.float64)); Er.eliminate_zeros(); del imp, age; gc.collect()
    idf = np.log((1. + NU) / (1. + np.asarray((E > 0).sum(0)).ravel())).astype(np.float32)
    M = E.multiply(idf).tocsr(); M.data = np.log1p(M.data); M = sp.csr_matrix(sp.diags(1. / np.maximum(np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()), 1e-8)) @ M)
    svd = TruncatedSVD(n_components=max(KS), n_iter=5, random_state=seed); Us = svd.fit_transform(M).astype(np.float32); Vs = svd.components_.T.astype(np.float32); del M; gc.collect()
    log(json.dumps(dict(stage='svd', day=day, cutoff=int(cut), exposures=int(E.nnz), singular=[float(x) for x in svd.singular_values_[:5]], elapsed_s=round(time.monotonic() - tic, 1))))
    out = {}; nexpo = np.asarray(E.sum(1)).ravel(); out['n_expo'] = np.repeat(np.log1p(nexpo[s_u]), 10)
    for k in KS:
        V = Vs[:, :k]; Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8); vc = Vn[s_c]  # (events, 10, k)
        for wname, W in (('all', E), ('d1', Ed), ('r2', Er)):
            P = np.asarray(W[s_u] @ Vn); pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)
            out[f'typ{k}_{wname}'] = np.einsum('ed,ecd->ec', pn, vc).ravel()
            if k == 32 and wname == 'all': out['raw32'] = np.einsum('ed,ecd->ec', P / np.maximum(np.asarray(W[s_u].sum(1)).ravel()[:, None], 1.), vc).ravel()
        if k in (32, 64):
            U = Us[s_u, :k]; out[f'cos{k}'] = (np.einsum('ed,ecd->ec', U, V[s_c]) / np.maximum(np.linalg.norm(U, axis=1)[:, None] * np.linalg.norm(V[s_c], axis=2), 1e-8)).ravel()
    out = {k: np.asarray(v, np.float64) for k, v in out.items()}
    for k, v in out.items(): assert v.shape == (len(s_u) * 10,) and np.isfinite(v).all(), k
    return z, out, dict(day=day, cutoff=int(cut), seed=seed, elapsed_s=time.monotonic() - tic)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, required=True); a = ap.parse_args()
    z, f, info = features_for(a.day)
    np.savez_compressed(R / f'nxa_exposure2_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
    if z is not None:
        y, e, v53 = z['y'], z['events'], z['pred']; base = weighted_auc(e, y, v53); rv = rank(v53); rep = dict(info=info, v53_adjusted=base, features={})
        H = np.load(R / 'train_hist.npy', mmap_mode='r'); hl = (H[e] >= 0).sum(1); short = hl <= 4
        for k, v in f.items():
            d = dict(raw_adjusted=weighted_auc(e, y, v), raw_auc_short=float(roc_auc_score(y[short], v[short])))
            for w in (.02, .05, .1): d[f'g{w}'] = weighted_auc(e, y, (1 - w) * rv + w * rank(v)) - base
            rep['features'][k] = d; print(k.ljust(10), json.dumps({kk: round(vv, 6) for kk, vv in d.items()}), flush=True)
        (R / f'nxa_exposure2_day{a.day}_report.json').write_text(json.dumps(rep, indent=1))
    else: print(json.dumps(dict(stage='saved', file=f'nxa_exposure2_day{a.day}_features.npz', rows=len(f['n_expo']))))
