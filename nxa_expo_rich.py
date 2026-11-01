"""nxa idea 2 (owner: Idea-A improver session): richer exposure-policy features for the scored cohorts / evaluation rows.

ONE code path, --day 5 / 6 / 7. Same exposure SVD as nxa_exposure2.py (impressions < cutoff, k=64, seed 4403; unit item vectors
v_i over the first 32 components; P_u = unit mean of v over the items shown to u before the cutoff). Columns:
  aud32      cos(P_u, A_c), A_c = unit mean of P_v over the users v shown c before the cutoff (item-side audience)
  xcon32     cos(P_u - H_u, v_c), H_u = unit mean of v over the event's own supplied click history (stamps < event time):
             what the recommender keeps showing that the user has not clicked
  hcl32      cos(H_u, v_c) (click history in the recommender's item space; 0 if no history)
  pct32      share of the user's pre-cutoff impressions whose typicality cos(P_u, v_i) is below the candidate's cos(P_u, v_c)
  rec32      cos(R_u, v_c), R_u = unit mean of v over the user's impressions in the last 2 days before the cutoff (0 if none)
  nrec       log1p(#impressions in the last 2 days)
Output: artifacts/nxa_expo_rich_day{d}_features.npz.
"""
import argparse, json, time, gc
import numpy as np
import scipy.sparse as sp
from config import ARTIFACT_DIR as R
from nx_seq_core import cutoff_of, NI, NU, DAY
from nxa_tree_feats import stats, unit
from nxa_exposure2 import load


def features_for(day, log=print):
    tic = time.monotonic(); cut, imp, s_u, s_c, z = load(day)
    s_h = np.asarray(np.load(R / 'evaluation_hist.npy', mmap_mode='r')) if day == 7 else np.asarray(np.load(R / 'train_hist.npy', mmap_mode='r')[z['events'].reshape(-1, 10)[:, 0]])
    recent = imp[imp[:, 2] >= cut - 2 * DAY]; Er = sp.csr_matrix((np.ones(len(recent), np.float32), (recent[:, 0], recent[:, 1])), shape=(NU, NI)); Er.sum_duplicates(); del imp, recent
    E, Us, Vs = stats(cut); V = unit(Vs[:, :32]); P_all = unit(np.asarray(E @ V)); A = unit(np.asarray(E.T @ P_all)); out = {}
    P = P_all[s_u]; vc = V[s_c]
    out['aud32'] = np.einsum('ed,ecd->ec', P, A[s_c])
    hv = (V[np.maximum(s_h, 0)] * (s_h >= 0)[..., None]).sum(1); H = unit(hv); has = (s_h >= 0).any(1)
    out['xcon32'] = np.einsum('ed,ecd->ec', unit(P - H * has[:, None]), vc); out['hcl32'] = np.einsum('ed,ecd->ec', H, vc) * has[:, None]
    typ = np.einsum('ed,ecd->ec', P, vc); pct = np.zeros_like(typ); Eu = E[s_u]
    for r in range(len(s_u)):
        lo, hi = Eu.indptr[r], Eu.indptr[r + 1]
        if hi == lo: continue
        sims = np.sort(V[Eu.indices[lo:hi]] @ P[r]); w = Eu.data[lo:hi]  # counts ignored in the sort: use unique exposed items
        pct[r] = np.searchsorted(sims, typ[r]) / len(sims)
    out['pct32'] = pct
    Rr = Er[s_u]; Rv = np.asarray(Rr @ V); nrec = np.asarray(Rr.sum(1)).ravel(); out['rec32'] = np.einsum('ed,ecd->ec', unit(Rv), vc) * (nrec > 0)[:, None]
    out['nrec'] = np.repeat(np.log1p(nrec)[:, None], 10, 1)
    out = {k: np.asarray(v, np.float64).ravel() for k, v in out.items()}
    for k, v in out.items(): assert v.shape == (len(s_u) * 10,) and np.isfinite(v).all(), k
    log(json.dumps(dict(stage='done', day=day, cutoff=int(cut), elapsed_s=round(time.monotonic() - tic, 1)))); return z, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, required=True); a = ap.parse_args()
    z, f = features_for(a.day); np.savez_compressed(R / f'nxa_expo_rich_day{a.day}_features.npz', **f, **({} if z is None else dict(y=z['y'], events=z['events'])))
