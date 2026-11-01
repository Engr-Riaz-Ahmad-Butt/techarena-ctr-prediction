"""nxb Track C extension: Laplace-predictive scoring of a saved C1/C2 model.

The per-user Newton Hessian at the MAP (pre-cut events + ledger term + prior) gives a Gaussian posterior
u ~ N(u_hat, H^-1) per user (16 dims + no-click offset). Scored events use the posterior-averaged choice probabilities
E_u[softmax] (Monte Carlo, S samples, fixed seed) instead of the plug-in softmax(u_hat). Users with few events get flatter,
better-calibrated rows, which matters for a global (cross-event) AUC. Everything uses data < the model's cut only.
Usage: python nxb_c_laplace.py <name> <target> <same fit args as the model, e.g. --ledger-w 1 --v-anchor ...>
"""
import sys, json, argparse
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import nxb_c_mf as C

name, d = sys.argv[1], int(sys.argv[2])
ap = argparse.ArgumentParser()
ap.add_argument('--dim', type=int, default=16)
for k, v in dict(lu=1.0, lw=1.0, lv=1.0, lb=1.0, lp=1.0, la=1.0, lh=1.0).items():
    ap.add_argument('--' + k, type=float, default=v)
ap.add_argument('--vscale', type=float, default=1.0); ap.add_argument('--v-anchor', action='store_true')
ap.add_argument('--user-offset', action='store_true'); ap.add_argument('--ledger-w', type=float, default=0.0)
ap.add_argument('--ledger-src', default='first'); ap.add_argument('--expo-w', type=float, default=0.0)
ap.add_argument('--kappa-fit', action='store_true'); ap.add_argument('--samples', type=int, default=64)
a = ap.parse_args(sys.argv[3:]); torch.set_num_threads(12)
st = torch.load(C.R / f'nxb_c_{name}_t{d}.pt')
raw = C.Raw(); share = C.eval_share(); cut = C.S + d * C.DAY
m = C.Model(a, (st['V'], st['b'], st['Hc'], st['A']))
for k in ('U', 'w', 'V', 'b', 'c0', 'P', 'A', 'Hc'):
    setattr(m, k, st[k])
m.kappa = st['kappa']
E = raw.events(np.flatnonzero(raw.ev_ts < cut))
if a.ledger_w > 0:
    m.cats.append(C.Cat(raw.ledger(cut, a.ledger_src), a.ledger_w, 'U', 'Hc'))
# --- Hessian of the user block at the MAP (same terms as Model.user_step) ---
dd = a.dim; k = dd + (1 if a.user_offset else 0)
H = torch.zeros(C.NU, k, k, dtype=torch.float64)
for s in range(0, len(E['user']), 65536):
    u = E['user'][s:s + 65536]; it = E['items'][s:s + 65536]
    p = torch.softmax(m.logits(u, it), 1)
    X = m.V[it]; x0 = torch.zeros(len(u), 1, k)
    if a.user_offset:
        X = torch.cat([X, torch.zeros(len(u), 10, 1)], 2); x0[:, 0, dd] = 1.0
    X = torch.cat([X, x0], 1)
    mm = (p.unsqueeze(-1) * X).sum(1)
    H.index_add_(0, u, (torch.einsum('bj,bjk,bjl->bkl', p, X, X) - mm.unsqueeze(2) * mm.unsqueeze(1)).double())
for c in m.cats:
    if c.utab == 'U':
        _, h = c.user_gh(m); H[:, :dd, :dd] += h
reg = torch.full((k,), 2 * a.lu, dtype=torch.float64)
if a.user_offset:
    reg[dd] = 2 * a.lw
H += torch.diag_embed(reg.expand(C.NU, k))
L = torch.linalg.cholesky(torch.linalg.inv(H)).float()              # posterior covariance factor
# --- scored set ---
if d in (5, 6):
    ref = np.load(C.R / f'nxa_h1_day{d}.npz'); S_ = raw.events(ref['events'][::10])
else:
    S_ = raw.eval_events()
g = torch.Generator().manual_seed(9101)
us, it = S_['user'], S_['items']
acc = torch.zeros(len(us), 10)
th = torch.cat([m.U, m.w.unsqueeze(1)], 1) if a.user_offset else m.U
for _ in range(a.samples):
    z = torch.randn(len(us), k, generator=g)
    smp = th[us] + torch.einsum('bkl,bl->bk', L[us], z)
    Uu = smp[:, :dd]
    s = (Uu.unsqueeze(1) * m.V[it]).sum(-1) + m.b[it]
    s0 = m.c0 + (smp[:, dd] if a.user_offset else 0.0)
    acc += torch.softmax(torch.cat([s, s0.unsqueeze(1) if torch.is_tensor(s0) and s0.dim() else s0.expand(len(us)).unsqueeze(1)], 1), 1)[:, :10]
p = (acc / a.samples).numpy().astype(np.float64).ravel()
plug = m.predict(S_)
out = {}
if d in (5, 6):
    out = dict(plugin_adj=C.adj(S_, plug, share), laplace_adj=C.adj(S_, p, share))
    if d == 6:
        s6 = np.load(C.R / 'selection_day6.npz'); fi = s6['full_indices']
        out['plugin_sel'] = float(roc_auc_score(s6['y'], plug[fi], sample_weight=s6['weights']))
        out['laplace_sel'] = float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
    np.savez(C.R / f'nxb_c_{name}lap_day{d}.npz', y=S_['y'], events=ref['events'], raw=p, pred=p)
else:
    np.save(C.R / f'nxb_c_{name}lap_eval.npy', p)
print(json.dumps(out, indent=1))
