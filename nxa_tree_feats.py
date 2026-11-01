"""nxa T1 features (owner: Idea-A improver session): label-free exposure-policy columns for EVERY cached feature block, so the main
tree expert can use them. Block conventions are those of the cached 354-column matrices (artifacts/features/):
  dayD (D = 2..6): all training rows with cutoff_of(D) <= time < cutoff_of(D+1), train order, statistics cut at cutoff_of(D);
  valD (D = 5, 6): first event per user on day D in user-id order (core_v2.split), cut at cutoff_of(D);
  eval: artifacts/evaluation.npy order, cut at cutoff_of(7) = 1737936000.
Statistics for cut C use ONLY impressions (user, item, time) with time < C; no label column is read.
Columns (float32, 8): typ16, typ32, cos32 (exactly nxa_exposure2.py's typ16_all, typ32_all, cos32: TruncatedSVD k=64 seed 4403 of the
IDF-log1p exposure matrix), n_expo = log1p(#impressions of the user), and the within-event versions typ16_c, typ32_c, cos32_c
(value minus the event mean) and typ32_erank (rank of typ32 inside the event / 9). Output: artifacts/features/nxa_expo_{block}.npy.
"""
import json, time, gc, sys
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from config import ARTIFACT_DIR as R
from nx_seq_core import cutoff_of, NI, NU

FEAT = R / 'features'; NAMES = ['typ16', 'typ32', 'cos32', 'n_expo', 'typ16_c', 'typ32_c', 'cos32_c', 'typ32_erank']
T = np.load(R / 'train.npy', mmap_mode='r'); HEAD_T = np.asarray(T[::10, 2])


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def stats(cut, seed=4403):
    n = int(np.searchsorted(HEAD_T, cut, side='left')); imp = np.asarray(T[:n * 10, :3]); assert imp[:, 2].max() < cut
    E = sp.csr_matrix((np.ones(len(imp), np.float32), (imp[:, 0], imp[:, 1])), shape=(NU, NI)); E.sum_duplicates(); del imp
    idf = np.log((1. + NU) / (1. + np.asarray((E > 0).sum(0)).ravel())).astype(np.float32)
    M = E.multiply(idf).tocsr(); M.data = np.log1p(M.data); M = sp.csr_matrix(sp.diags(1. / np.maximum(np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()), 1e-8)) @ M)
    svd = TruncatedSVD(n_components=64, n_iter=5, random_state=seed); Us = svd.fit_transform(M).astype(np.float32); Vs = svd.components_.T.astype(np.float32); del M; gc.collect()
    return E, Us, Vs


def block_rows(name):
    if name == 'eval': return np.asarray(np.load(R / 'evaluation.npy', mmap_mode='r')), 7
    d = int(name[-1]); lo, hi = np.searchsorted(HEAD_T, [cutoff_of(d), cutoff_of(d + 1)], side='left'); rows = np.asarray(T[lo * 10:hi * 10])
    if name.startswith('day'): return rows, d
    _, first = np.unique(rows[::10, 0], return_index=True); return rows[(first[:, None] * 10 + np.arange(10)).ravel()], d


def columns(E, Us, Vs, s_u, s_c):
    out = {}; nexpo = np.asarray(E.sum(1)).ravel(); out['n_expo'] = np.repeat(np.log1p(nexpo[s_u]), 10)
    for k in (16, 32):
        V = Vs[:, :k]; Vn = unit(V)
        for off in range(0, len(s_u), 20000):
            sl = slice(off, off + 20000); P = unit(np.asarray(E[s_u[sl]] @ Vn)); out.setdefault(f'typ{k}', []).append(np.einsum('ed,ecd->ec', P, Vn[s_c[sl]]).ravel())
        out[f'typ{k}'] = np.concatenate(out[f'typ{k}'])
        if k == 32:
            U = Us[:, :k]; parts = []
            for off in range(0, len(s_u), 20000):
                sl = slice(off, off + 20000); u = U[s_u[sl]]; v = V[s_c[sl]]
                parts.append((np.einsum('ed,ecd->ec', u, v) / np.maximum(np.linalg.norm(u, axis=1)[:, None] * np.linalg.norm(v, axis=2), 1e-8)).ravel())
            out['cos32'] = np.concatenate(parts)
    for c in ('typ16', 'typ32', 'cos32'):
        m = out[c].reshape(-1, 10); out[c + '_c'] = (m - m.mean(1, keepdims=True)).ravel()
    out['typ32_erank'] = (np.argsort(np.argsort(out['typ32'].reshape(-1, 10), axis=1), axis=1) / 9.).ravel()
    x = np.column_stack([out[n] for n in NAMES]).astype(np.float32); assert np.isfinite(x).all(); return x


if __name__ == '__main__':
    blocks = sys.argv[1:] or ['val6', 'day3', 'day4', 'day5', 'day6', 'day2', 'val5', 'eval']
    cache = {}; tic = time.monotonic()
    for name in blocks:
        dest = FEAT / f'nxa_expo_{name}.npy'
        if dest.exists(): print('exists', dest.name, flush=True); continue
        rows, cut_day = block_rows(name); cut = cutoff_of(cut_day); assert rows[:, 2].min() >= cut
        if cut not in cache: cache.clear(); gc.collect(); cache[cut] = stats(cut)
        x = columns(*cache[cut], rows[::10, 0].astype(np.int64), rows[:, 1].reshape(-1, 10).astype(np.int64))
        np.save(dest, x); print(json.dumps(dict(block=name, rows=int(len(x)), cutoff=int(cut), elapsed_s=round(time.monotonic() - tic, 1))), flush=True)
        if name in ('val5', 'val6'):  # must equal the cohort features used by nxa_expo / nxa_h1
            f = np.load(R / f'nxa_exposure2_day{name[-1]}_features.npz'); v = np.load(R / f'v53_causal_propensity_day{name[-1]}.npz'); assert np.array_equal(rows[:, 6], v['events'])
            for j, c in ((0, 'typ16_all'), (1, 'typ32_all'), (2, 'cos32')): assert np.max(np.abs(x[:, j] - f[c])) < 1e-4, (name, c, float(np.max(np.abs(x[:, j] - f[c]))))
            print('matches nxa_exposure2 cohort features', name, flush=True)
