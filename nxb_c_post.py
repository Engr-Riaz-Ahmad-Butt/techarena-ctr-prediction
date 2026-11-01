"""nxb Track C post-processing: 'u shrunk toward the policy belief p' for short-history events, from a saved C2 model.

kappa_hat = least-squares slope of U on P over users with >= 10 pre-cut events (fitted on the model's own pre-cut
parameters only). For scored events whose history has <= 4 clicks: u' = U + alpha * kappa_hat * P (alpha in a small grid);
other events keep U. Writes artifacts/nxb_c_<name>_shrink_day{d}.npz (raw = prediction for each alpha) and prints the
adjusted AUC of each variant (day-6 selection subset is the only selection set).
Usage: python nxb_c_post.py <name> <target>
"""
import sys, json
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import nxb_c_mf as C

name, d = sys.argv[1], int(sys.argv[2])
st = torch.load(C.R / f'nxb_c_{name}_t{d}.pt')
raw = C.Raw(); share = C.eval_share(); c = C.S + d * C.DAY
nev = np.bincount(raw.ev_user[raw.ev_ts < c], minlength=C.NU)
U, P, V, b, w, c0 = st['U'], st['P'], st['V'], st['b'], st['w'], st['c0']
m = torch.from_numpy(nev >= 10)
kap = float((U[m] * P[m]).sum() / (P[m] * P[m]).sum())
if d in (5, 6):
    ref = np.load(C.R / f'nxa_h1_day{d}.npz'); E = raw.events(ref['events'][::10])
else:
    E = raw.eval_events()
short = torch.from_numpy(E['hlen'] <= 4)
out = {}; preds = {}
for alpha in (0.0, 0.25, 0.5, 1.0):
    Uu = U[E['user']] + alpha * kap * P[E['user']] * short.unsqueeze(1).float()
    s = (Uu.unsqueeze(1) * V[E['items']]).sum(-1) + b[E['items']]
    s0 = c0 + w[E['user']]
    p = torch.softmax(torch.cat([s, s0.unsqueeze(1)], 1), 1)[:, :10].numpy().ravel().astype(np.float64)
    preds[f'a{alpha}'] = p
    if d in (5, 6):
        out[f'a{alpha}'] = dict(adj_full=C.adj(E, p, share))
        if d == 6:
            s6 = np.load(C.R / 'selection_day6.npz'); fi = s6['full_indices']
            out[f'a{alpha}']['adj_sel'] = float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
        hb = np.repeat(E['hlen'] <= 4, 10)
        out[f'a{alpha}']['auc_short'] = float(roc_auc_score(E['y'][hb], p[hb]))
np.savez(C.R / f'nxb_c_{name}_shrink_day{d}.npz', **preds)
print(json.dumps(dict(kappa_hat=kap, n_short_events=int(short.sum()), variants=out), indent=1))
