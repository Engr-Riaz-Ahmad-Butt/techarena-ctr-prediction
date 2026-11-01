"""nxa idea C, step 4 (owner: Idea-A improver session): the user's exposure profile expressed in OTHER item spaces.

Same impressions as nxa_exposure2.py (time < cutoff, no labels). For candidate c of user u:
  ctr{k}       cosine(candidate, profile - population mean profile) in the exposure-SVD space (popular-direction contrast)
  tas{k}       cosine of taste_c with the mean taste vector of the items shown to u; taste = preference_stats_{cutoff}[3]
               (co-click SVD built by core_v6 from labels < cutoff), k in {16, 32, 126}
  cho          cosine with the mean warm16 choice-model item vector of the shown items (imp2_model_warm1.0_d16_pc1_day{d}.npz,
               trained on labels < cutoff(d); evaluation: imp2_final_model_warm1.0_d16_pc1.npz)
Day-6 screen only unless a variant beats the queued nxa_expo expert there.
"""
import argparse, json, pickle, time, gc
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc, rank, cutoff_of, NI, NU
from nxa_exposure2 import load


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def features_for(day, seed=4403, log=print):
    tic = time.monotonic(); cut, imp, s_u, s_c, z = load(day)
    E = sp.csr_matrix((np.ones(len(imp), np.float32), (imp[:, 0], imp[:, 1])), shape=(NU, NI)); E.sum_duplicates(); del imp; gc.collect()
    idf = np.log((1. + NU) / (1. + np.asarray((E > 0).sum(0)).ravel())).astype(np.float32)
    M = E.multiply(idf).tocsr(); M.data = np.log1p(M.data); M = sp.csr_matrix(sp.diags(1. / np.maximum(np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()), 1e-8)) @ M)
    Vs = TruncatedSVD(n_components=64, n_iter=5, random_state=seed).fit(M).components_.T.astype(np.float32); del M; gc.collect()
    Eu = E[s_u]; out = {}; item_expo = np.asarray(E.sum(0)).ravel()
    def prof_cos(V, contrast=False):
        P = np.asarray(Eu @ V) / np.maximum(np.asarray(Eu.sum(1)), 1.)
        if contrast: P = P - (item_expo @ V) / item_expo.sum()
        return np.einsum('ed,ecd->ec', unit(P), unit(V[s_c])).ravel()
    for k in (16, 32): out[f'ctr{k}'] = prof_cos(unit(Vs[:, :k]), True)
    with (R / f'preference_stats_{cut}.pkl').open('rb') as f: taste = pickle.load(f)[3]
    for k in (16, 32, 126): out[f'tas{k}'] = prof_cos(unit(taste[:, :k]).astype(np.float32)); out[f'tasc{k}'] = prof_cos(unit(taste[:, :k]).astype(np.float32), True)
    del taste; gc.collect()
    mp = R / ('imp2_final_model_warm1.0_d16_pc1.npz' if day == 7 else f'imp2_model_warm1.0_d16_pc1_day{day}.npz')
    with np.load(mp) as m: V = m['V'].astype(np.float32)
    out['cho'] = prof_cos(V); out['choc'] = prof_cos(V, True)
    out = {k: np.asarray(v, np.float64) for k, v in out.items()}
    for k, v in out.items(): assert np.isfinite(v).all(), k
    log(json.dumps(dict(stage='done', day=day, elapsed_s=round(time.monotonic() - tic, 1)))); return z, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, default=6); a = ap.parse_args()
    z, f = features_for(a.day); np.savez_compressed(R / f'nxa_exposure4_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
    if z is not None:
        y, e = z['y'], z['events']; rv = rank(z['pred']); base = weighted_auc(e, y, z['pred'])
        f2 = np.load(R / f'nxa_exposure2_day{a.day}_features.npz'); e3 = np.mean([rank(f2[c]) for c in ('typ16_all', 'typ32_all', 'cos32')], axis=0)
        def cz(x): c = x - np.repeat(x.reshape(-1, 10).mean(1), 10); return c / c.std()
        rep = {}
        for k, v in list(f.items()) + [('E3', e3)] + [(f'E3+{k}', (3 * e3 + rank(v)) / 4) for k, v in f.items()]:
            d = dict(raw=weighted_auc(e, y, v)); d.update({f'cz{bt}': weighted_auc(e, y, rv + bt * cz(v)) - base for bt in (.01, .015, .02)}); rep[k] = d
            print(k.ljust(10), ' '.join(f'{kk}={vv:+.6f}' for kk, vv in d.items()), flush=True)
        (R / f'nxa_exposure4_day{a.day}_report.json').write_text(json.dumps(rep, indent=1))
