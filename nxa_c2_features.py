"""nxa C2 features (owner: Idea-A improver session): sharper estimates of the recommender's view of each user.

ONE code path, --day 5 / 6 / 7 (cutoff start + day*86400). Impressions (user, item, time) with time < cutoff; no labels except in
(b), which uses labels of training events with time < cutoff only.
 (a) one-step policy estimate: item vectors v_i = unit rows of the k=32 exposure-SVD (same SVD as nxa_exposure2.py, seed 4403),
     offsets b_i = log(1 + #impressions of i), q = softmax(b) (population exposure policy). One Newton step from theta = 0 of
        mean_{i shown to u} theta.v_i - log sum_j exp(theta.v_j + b_j) - (lam / (2 n_u)) |theta|^2      (lam = 10):
        theta_u = (Cov_q[v] + lam/n_u I)^-1 (mean_shown_u - E_q[v]).  (The iterative version diverged at eta 8; see journal.)
     Features: pol_dot = theta.v_c, pol_cos, pol_rank = catalogue rank of c under theta.v + b (0 = top), pol_norm = |theta|.
 (b) exposure-space neighbourhood CTR: users compared by their unit mean exposure profile (k=32); K=200 most similar users (self
     excluded); eknn_ctr = (sum sim*clicks(v,c) + 20 p_c) / (sum sim*shown(v,c) + 20), eknn_shown = log1p(sum sim*shown(v,c)).
 (c) catalogue rank under the warm16 choice model (research F2): s_j = bu + bi_j + (U_u + sum_h H_h / sqrt(n)).V_j over all 50k items
     (the event's supplied history; checkpoint imp2_model_warm1.0_d16_pc1_day{5,6}.npz trained on labels < cutoff(d), evaluation:
     imp2_final_model_warm1.0_d16_pc1.npz, labels < cutoff 7). cr_rank = share of catalogue items scoring above c, cr_gap = s_c - max_j s_j,
     cr_logp = s_c - logsumexp_j s_j.
Output: artifacts/nxa_c2_day{d}_features.npz (row arrays; + y, events for days 5/6).
"""
import argparse, json, time, gc
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from config import ARTIFACT_DIR as R
from nx_seq_core import cutoff_of, NI, NU
from nxa_exposure2 import load


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def features_for(day, seed=4403, lam=10., eta=8., steps=15, K=200, log=print):
    tic = time.monotonic(); cut, imp, s_u, s_c, z = load(day); n_ev = len(imp) // 10
    E = sp.csr_matrix((np.ones(len(imp), np.float32), (imp[:, 0], imp[:, 1])), shape=(NU, NI)); E.sum_duplicates()
    idf = np.log((1. + NU) / (1. + np.asarray((E > 0).sum(0)).ravel())).astype(np.float32)
    M = E.multiply(idf).tocsr(); M.data = np.log1p(M.data); M = sp.csr_matrix(sp.diags(1. / np.maximum(np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()), 1e-8)) @ M)
    Vs = TruncatedSVD(n_components=64, n_iter=5, random_state=seed).fit(M).components_.T.astype(np.float32); del M; gc.collect()
    V = unit(Vs[:, :32]).astype(np.float32); b = np.log1p(np.asarray(E.sum(0)).ravel()).astype(np.float32)
    uu = np.unique(s_u); Eu = E[uu]; nu_ = np.maximum(np.asarray(Eu.sum(1)).ravel(), 1.); mean_shown = (np.asarray(Eu @ V) / nu_[:, None]).astype(np.float64)
    q = np.exp(b - b.max()); q /= q.sum(); mq = q @ V; cov = (V * q[:, None]).T @ V - np.outer(mq, mq)
    theta = np.stack([np.linalg.solve(cov + (lam / n) * np.eye(32), m - mq) for m, n in zip(mean_shown, nu_)]).astype(np.float32)
    log(json.dumps(dict(stage='policy_one_step', users=int(len(uu)), mean_norm=float(np.linalg.norm(theta, axis=1).mean()), cov_top_eig=float(np.linalg.eigvalsh(cov)[-1]), elapsed_s=round(time.monotonic() - tic, 1))))
    ix = np.searchsorted(uu, s_u); th = theta[ix]; vc = V[s_c]; dot = np.einsum('ed,ecd->ec', th, vc); out = dict(pol_dot=dot, pol_cos=dot / np.maximum(np.linalg.norm(th, axis=1)[:, None], 1e-8), pol_norm=np.repeat(np.linalg.norm(th, axis=1)[:, None], 10, 1))
    pr = np.zeros(s_c.shape, np.float32)
    for off in range(0, len(s_u), 250):
        sl = slice(off, off + 250); full = th[sl] @ V.T + b; mine = np.take_along_axis(full, s_c[sl], 1); pr[sl] = (full[:, None, :] > mine[:, :, None]).sum(2) / NI
    out['pol_rank'] = pr
    # (b) exposure-space neighbourhood CTR
    P = unit(np.asarray(E @ V)).astype(np.float32); lab = imp[np.asarray(np.load(R / 'train.npy', mmap_mode='r')[:n_ev * 10, 5]) == 1]
    def keyed(rows_u, rows_i, w):
        k = rows_u.astype(np.int64) * NI + rows_i; k, inv = np.unique(k, return_inverse=True); return k, np.bincount(inv, weights=w).astype(np.float32)
    kS, vS = keyed(imp[:, 0], imp[:, 1], np.ones(len(imp))); kC, vC = keyed(lab[:, 0], lab[:, 1], np.ones(len(lab))); del imp, lab; gc.collect()
    def lookup(keys, k, v): p = np.minimum(np.searchsorted(k, keys), len(k) - 1); return np.where(k[p] == keys, v[p], 0.)
    ic = np.zeros(NI); ish = np.zeros(NI); np.add.at(ish, (kS % NI), vS); np.add.at(ic, (kC % NI), vC); g = ic.sum() / ish.sum(); pc = (ic + 50 * g) / (ish + 50)
    ctr = np.zeros(s_c.shape); shown = np.zeros(s_c.shape)
    for off in range(0, len(s_u), 300):
        sl = slice(off, off + 300); S = P[s_u[sl]] @ P.T; S[np.arange(S.shape[0]), s_u[sl]] = -1.; nb = np.argpartition(-S, K, 1)[:, :K]; sim = np.maximum(np.take_along_axis(S, nb, 1), 0.)
        keys = nb[:, :, None].astype(np.int64) * NI + s_c[sl][:, None, :]; sh = (sim[:, :, None] * lookup(keys, kS, vS)).sum(1); cl = (sim[:, :, None] * lookup(keys, kC, vC)).sum(1)
        ctr[sl] = (cl + 20 * pc[s_c[sl]]) / (sh + 20); shown[sl] = np.log1p(sh)
    out.update(eknn_ctr=ctr, eknn_lift=np.log(ctr / pc[s_c]), eknn_shown=shown)
    with np.load(R / ('imp2_final_model_warm1.0_d16_pc1.npz' if day == 7 else f'imp2_model_warm1.0_d16_pc1_day{day}.npz')) as m: U, Vc, Hc, bu, bi = [m[k].astype(np.float64) for k in ('U', 'V', 'H', 'bu', 'bi')]
    hist = np.asarray(np.load(R / 'evaluation_hist.npy', mmap_mode='r')) if day == 7 else np.asarray(np.load(R / 'train_hist.npy', mmap_mode='r')[z['events'].reshape(-1, 10)[:, 0]])
    n_h = (hist >= 0).sum(1); zu = U[s_u] + (Hc[np.maximum(hist, 0)] * (hist >= 0)[..., None]).sum(1) / np.sqrt(np.maximum(n_h, 1))[:, None]
    crr = np.zeros(s_c.shape); crg = np.zeros(s_c.shape); crl = np.zeros(s_c.shape)
    for off in range(0, len(s_u), 250):
        sl = slice(off, off + 250); full = zu[sl] @ Vc.T + bi + bu[s_u[sl]][:, None]; mine = np.take_along_axis(full, s_c[sl], 1); mx = full.max(1, keepdims=True)
        crr[sl] = (full[:, None, :] > mine[:, :, None]).sum(2) / NI; crg[sl] = mine - mx; crl[sl] = mine - (mx + np.log(np.exp(full - mx).sum(1, keepdims=True)))
    out.update(cr_rank=crr, cr_gap=crg, cr_logp=crl)
    out = {k: np.asarray(v, np.float64).ravel() for k, v in out.items()}
    for k, v in out.items(): assert v.shape == (len(s_u) * 10,) and np.isfinite(v).all(), k
    log(json.dumps(dict(stage='done', day=day, cutoff=int(cut), elapsed_s=round(time.monotonic() - tic, 1)))); return z, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, required=True); a = ap.parse_args()
    z, f = features_for(a.day); np.savez_compressed(R / f'nxa_c2_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
