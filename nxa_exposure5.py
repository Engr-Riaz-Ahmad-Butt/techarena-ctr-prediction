"""nxa idea C, step 5 (owner: Idea-A improver session): the recommender's item geometry from CO-SLATE statistics.

Every slate is one decision of the recommender, so two items shown in the same slate are "similar" in its eyes. Item embedding =
TruncatedSVD of the positive PMI of item x item co-slate counts over slates with time < cutoff (no labels). For candidate c of
user u: sl{k} = cosine(candidate embedding, mean embedding of all items shown to u before the cutoff), k in {16, 32, 64}.
Day-6 screen alone and combined with the queued expert E3 (nxa_expo).
"""
import argparse, json, time, gc
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc, rank, NI, NU
from nxa_exposure2 import load


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def features_for(day, seed=4405, log=print):
    tic = time.monotonic(); cut, imp, s_u, s_c, z = load(day); items = imp[:, 1].reshape(-1, 10).astype(np.int32)
    ia, ib = np.triu_indices(10, 1); rr = items[:, ia].ravel(); cc = items[:, ib].ravel()
    C = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(NI, NI)); del rr, cc; gc.collect()
    C = (C + C.T).tocsr(); C.setdiag(0); C.eliminate_zeros()
    tot = C.sum(); deg = np.asarray(C.sum(1)).ravel(); Cc = C.tocoo()
    pmi = np.log(Cc.data * tot / (deg[Cc.row] * deg[Cc.col])); keep = pmi > 0
    P = sp.csr_matrix((pmi[keep].astype(np.float32), (Cc.row[keep], Cc.col[keep])), shape=(NI, NI)); del C, Cc, pmi, keep; gc.collect()
    svd = TruncatedSVD(n_components=64, n_iter=5, random_state=seed); X = svd.fit_transform(P).astype(np.float32); del P; gc.collect()
    log(json.dumps(dict(stage='svd', day=day, pairs=int(len(items) * 45), elapsed_s=round(time.monotonic() - tic, 1))))
    E = sp.csr_matrix((np.ones(len(imp), np.float32), (imp[:, 0], imp[:, 1])), shape=(NU, NI)); E.sum_duplicates(); Eu = E[s_u]; del imp, items; gc.collect()
    out = {}
    for k in (16, 32, 64):
        V = unit(X[:, :k]); Pu = unit(np.asarray(Eu @ V)); out[f'sl{k}'] = np.einsum('ed,ecd->ec', Pu, V[s_c]).ravel().astype(np.float64)
    log(json.dumps(dict(stage='done', elapsed_s=round(time.monotonic() - tic, 1)))); return z, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, default=6); a = ap.parse_args()
    z, f = features_for(a.day); np.savez_compressed(R / f'nxa_exposure5_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
    if z is not None:
        y, e = z['y'], z['events']; rv = rank(z['pred']); base = weighted_auc(e, y, z['pred'])
        f2 = np.load(R / f'nxa_exposure2_day{a.day}_features.npz'); e3 = np.mean([rank(f2[c]) for c in ('typ16_all', 'typ32_all', 'cos32')], axis=0)
        def cz(x): c = x - np.repeat(x.reshape(-1, 10).mean(1), 10); return c / c.std()
        rep = {}
        for k, v in list(f.items()) + [('E3', e3)] + [(f'E3+{k}', (3 * e3 + rank(v)) / 4) for k, v in f.items()] + [(f'E3+2{k}', (3 * e3 + 2 * rank(v)) / 5) for k, v in f.items()]:
            d = dict(raw=weighted_auc(e, y, v)); d.update({f'cz{bt}': weighted_auc(e, y, rv + bt * cz(v)) - base for bt in (.01, .015, .02)}); rep[k] = d
            print(k.ljust(10), ' '.join(f'{kk}={vv:+.6f}' for kk, vv in d.items()), flush=True)
        (R / f'nxa_exposure5_day{a.day}_report.json').write_text(json.dumps(rep, indent=1))
