"""nxb Track B (v2): MAP-style 16-dim choice factorization trained to convergence with large-batch Adam.

logit(e, c) = b0 + bu[u] + bi[c] + (U[u] + a * mean-pooled H over the event's history (+ time) [+ DIN attention]) . V[c]
              + repeat / repeat-recency / prior-exposure terms;  no-click logit = 0;  11-way softmax.
Global L2 on the full tables (MAP), Adam on all parameters, batch of events, early stopping on the first-per-user events of
the pre-cutoff day d-1 (phase 1, data < c - DAY), then a refit on all events < c for the selected number of epochs (phase 2).
Warm start: click-SVD (rank = dim) of the pre-cutoff click ledger (labelled clicks + histories of events before the cut).
Data helpers are shared with nxb_b_seq.py (same author, same prefix).
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score
import nxb_b_seq as B

R = B.R; S, DAY, NI, NU = B.S, B.DAY, B.NI, B.NU


class MF(nn.Module):
    def __init__(self, dim, attn=False, hist_w=True):
        super().__init__()
        self.dim, self.attn, self.hist_w = dim, attn, hist_w
        self.V = nn.Parameter(torch.randn(NI + 1, dim) * 0.05); self.bi = nn.Parameter(torch.zeros(NI + 1))
        self.H = nn.Parameter(torch.zeros(NI + 1, dim))
        self.T = nn.Parameter(torch.zeros(25, dim))
        self.U = nn.Parameter(torch.zeros(NU, dim)); self.bu = nn.Parameter(torch.zeros(NU))
        self.rep = nn.Parameter(torch.zeros(6)); self.repgap = nn.Parameter(torch.zeros(25)); self.ex = nn.Parameter(torch.zeros(8))
        self.b0 = nn.Parameter(torch.tensor(-1.8))
        if attn:
            self.Wq = nn.Linear(dim, dim, bias=False); self.Wk = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.Wq.weight)
            self.Wk.weight.data.copy_(torch.eye(dim))
            self.null = nn.Parameter(torch.zeros(1))

    def forward(self, user, items, expo, hist, gb, valid):
        v = self.V[items]
        vm = valid.float()
        e = self.H[hist] + self.T[gb]
        n = vm.sum(1, keepdim=True).clamp(min=1)
        z = (e * vm.unsqueeze(-1)).sum(1) / n.sqrt()
        z = z.unsqueeze(1) + self.U[user].unsqueeze(1)
        if self.attn:
            q = self.Wq(v); k = self.Wk(e)
            a = torch.einsum('bcd,bqd->bcq', q, k) / math.sqrt(self.dim)
            a = a.masked_fill(~valid.unsqueeze(1), -1e4)
            a = torch.cat([self.null.expand(a.shape[0], a.shape[1], 1), a], -1)
            w = torch.softmax(a, -1)[..., 1:]
            z = z + torch.einsum('bcq,bqd->bcd', w, e)
        same = (hist.unsqueeze(1) == items.unsqueeze(2)) & valid.unsqueeze(1)
        cnt = same.sum(-1).clamp(max=5)
        lastgb = torch.where(same, gb.unsqueeze(1), torch.full_like(gb.unsqueeze(1), 24)).min(-1).values
        return (self.b0 + self.bu[user].unsqueeze(1) + self.bi[items] + (z * v).sum(-1)
                + self.rep[cnt] + self.repgap[lastgb] + self.ex[expo])

    def l2(self, lu, lv, lh):
        return lu * (self.U.pow(2).sum() + self.bu.pow(2).sum()) + lv * (self.V.pow(2).sum() + self.bi.pow(2).sum()) \
            + lh * (self.H.pow(2).sum() + self.T.pow(2).sum())


def predict(model, d, bs=8192):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(d['user']), bs):
            tt = B.to_tensors(d, np.arange(s, min(s + bs, len(d['user']))))
            out.append(B.Model.prob(model(*tt[:6])).numpy())
    return np.concatenate(out).ravel()


def fit(a, tr, epochs, val=None, share=None, init=None, log=print, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = MF(a.dim, attn=a.attn)
    if init is not None:
        V0, U0, bi0, bu0 = init
        with torch.no_grad():
            model.V[:NI] = torch.from_numpy(V0); model.H[:NI] = torch.from_numpy(a.h_scale * V0)
            model.U[:] = torch.from_numpy(U0); model.bi[:NI] = torch.from_numpy(bi0); model.bu[:] = torch.from_numpy(bu0)
    model.anchor = {k: getattr(model, k).detach().clone() * (0.0 if a.zero_anchor else 1.0) for k in ('U', 'bu', 'V', 'bi', 'H')}
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n = len(tr['user']); best = (-1.0, 0); hist = []
    nb_per_epoch = math.ceil(n / a.bs)
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); perm = rng.permutation(n); tot = 0.0
        for s in range(0, n, a.bs):
            idx = np.sort(perm[s:s + a.bs]); tt = B.to_tensors(tr, idx)
            logit = model(*tt[:6])
            full = torch.cat([logit, torch.zeros_like(logit[:, :1])], 1)
            ce = Fn.cross_entropy(full, tt[6], reduction='sum')
            # MAP: data term over the batch + full-table L2 scaled by the batch's share of the data
            loss = (ce + model.l2(a.lu, a.lv, a.lh) * (len(idx) / n)) / len(idx)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(ce.detach())
        msg = dict(epoch=ep, ce=round(tot / n, 5), sec=round(time.time() - t0, 1))
        if val is not None:
            p = predict(model, val)
            msg['val_adj'] = round(B.adj_auc(val, p, share), 6); msg['val_plain'] = round(float(roc_auc_score(val['y'], p)), 6)
            if msg['val_adj'] > best[0]:
                best = (msg['val_adj'], ep)
        log(json.dumps(msg)); hist.append(msg)
        if val is not None and ep - best[1] >= a.patience:
            break
    return model, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=6); ap.add_argument('--name', default='m1')
    ap.add_argument('--dim', type=int, default=16); ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.01); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lu', type=float, default=1.0); ap.add_argument('--lv', type=float, default=1.0)
    ap.add_argument('--lh', type=float, default=1.0); ap.add_argument('--h-scale', type=float, default=0.25)
    ap.add_argument('--attn', action='store_true'); ap.add_argument('--no-warm', action='store_true')
    ap.add_argument('--phase', default='both', choices=['select', 'both']); ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--fixed-epochs', type=int, default=0)
    ap.add_argument('--zero-anchor', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    logf = open(R / f'nxb_b_{a.name}_t{a.target}.log', 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(json.dumps(vars(a))); t0 = time.time()
    D = B.Data()
    eh = np.load(R / 'evaluation_hist.npy')
    share = np.bincount(np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True), minlength=5) / len(eh)
    c = S + a.target * DAY
    if a.fixed_epochs:
        best = (None, a.fixed_epochs)
    else:
        tr1 = D.events(np.flatnonzero(D.ev_ts < c - DAY)); val = D.events(D.first_per_user(c - DAY, c))
        init1 = None if a.no_warm else B.warm_init(D, c - DAY, a.dim, log)
        _, best, h1 = fit(a, tr1, a.epochs, val, share, init1, log)
        log(f'phase1 best epoch {best[1]} val adj {best[0]:.6f} ({time.time()-t0:.0f}s)')
        (R / f'nxb_b_{a.name}_select_t{a.target}.json').write_text(json.dumps(dict(args=vars(a), phase1=h1, best=best), indent=1))
        del tr1
        if a.phase == 'select':
            return
    tr2 = D.events(np.flatnonzero(D.ev_ts < c))
    init2 = None if a.no_warm else B.warm_init(D, c, a.dim, log)
    model, _, _ = fit(a, tr2, best[1], None, None, init2, log)
    if a.target in (5, 6):
        ref = np.load(R / f'nxa_h1_day{a.target}.npz'); d = D.events(ref['events'][::10])
        assert np.array_equal(d['y'], ref['y'])
        p = predict(model, d)
        hb = np.digitize((d['hist'] >= 0).sum(1), [0, 4, 9, 19], right=True)
        res = dict(target=a.target, epochs=best[1], adj_full=B.adj_auc(d, p, share), plain=float(roc_auc_score(d['y'], p)))
        if a.target == 6:
            s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
            res['adj_sel'] = float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
        res['bins'] = {int(b): float(roc_auc_score(d['y'][np.repeat(hb == b, 10)], p[np.repeat(hb == b, 10)])) for b in range(5)}
        log(json.dumps(res))
        np.savez(R / f'nxb_b_{a.name}_day{a.target}.npz', y=d['y'], events=ref['events'], raw=p, pred=p)
    else:
        d = D.eval_events(); p = predict(model, d); np.save(R / f'nxb_b_{a.name}_eval.npy', p); log(f'eval rows {len(p)}')
    torch.save(model.state_dict(), R / f'nxb_b_{a.name}_t{a.target}.pt')
    log(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
