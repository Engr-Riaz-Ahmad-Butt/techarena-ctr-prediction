"""nxb Track B: candidate-aware sequence/choice model (DIN-style target attention over the 20 timestamped
history clicks + user/item embeddings), 11-way softmax over the 10 candidates + no-click. Self-contained.

For target day d (cutoff c = S + d*DAY; d = 5, 6, or 7 = evaluation):
  phase 1: train on events with ts < c - DAY, validate each epoch on the first-per-user events of day d-1
           (adjusted AUC, evaluation history mix) -> best epoch E*  (never touches day d)
  phase 2: retrain from scratch on all events with ts < c for E* epochs, score day d's cohort / evaluation rows.
Label-free per-candidate exposure count = number of earlier rows with the same (user, item) strictly before
the event (impressions only). Histories are used as given (strictly before the event).
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent / 'artifacts'
S, DAY, NI, NU = 1737331200, 86400, 50000, 54456


def gap_bucket(sec):
    """log2 buckets of minutes: 0..23."""
    m = np.maximum(sec, 0) / 60.0
    return np.clip(np.floor(np.log2(m + 1.0)), 0, 23).astype(np.int64)


class Data:
    K = 5

    def __init__(self):
        t = np.load(R / 'train.npy')
        self.t = t
        self.hist = np.load(R / 'train_hist.npy'); self.htim = np.load(R / 'train_times.npy')
        # prior exposure count of (user, item) strictly before the row's event (label-free)
        key = t[:, 0].astype(np.int64) * NI + t[:, 1]
        order = np.lexsort((t[:, 2], key))
        ks = key[order]
        start = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
        grp_start = np.repeat(start, np.diff(np.r_[start, len(ks)]))
        rank = np.arange(len(ks)) - grp_start
        # rows of the same event share a timestamp: ties (same key, same ts) get the count of strictly earlier rows
        ts_s = t[order, 2]
        tie_first = np.r_[0, np.flatnonzero((ks[1:] != ks[:-1]) | (ts_s[1:] != ts_s[:-1])) + 1]
        tie_rank = np.repeat(tie_first, np.diff(np.r_[tie_first, len(ks)])) - grp_start
        expo = np.empty(len(t), np.int32); expo[order] = tie_rank
        self.expo = expo
        self.key_sorted, self.key_counts = np.unique(key, return_counts=True)
        self.ev_ts = t[::10, 2].astype(np.int64)
        self.ev_user = t[::10, 0].astype(np.int64)
        # previous events of the same user (strictly earlier in time order; label use only through rows of earlier events)
        KM = 8
        order = np.argsort(self.ev_user, kind='stable')          # events are time-sorted -> per-user time order
        uo = self.ev_user[order]
        prev = np.full((len(order), KM), -1, np.int64)
        for k in range(1, KM + 1):
            ok = np.r_[np.zeros(k, bool), uo[k:] == uo[:-k]]
            src = np.r_[np.full(k, -1), order[:-k]]
            prev[order[ok], k - 1] = src[ok]
        assert np.all(np.where(prev >= 0, self.ev_ts[np.maximum(prev, 0)] <= self.ev_ts[:, None], True))
        self.prev = prev
        last = np.flatnonzero(np.r_[uo[1:] != uo[:-1], True])    # last event of each user (position in order)
        self.last_prev = {}
        lp = np.full((NU, KM), -1, np.int64)
        lp[uo[last], 0] = order[last]
        lp[uo[last], 1:] = prev[order[last], :KM - 1]
        self.user_last = lp

    def neg_items(self, pe, ts, K):
        pe = pe[:, :K].copy()
        pe[(pe >= 0) & (self.ev_ts[np.maximum(pe, 0)] >= ts[:, None])] = -1      # strictly before the event
        rows = (np.maximum(pe, 0)[:, :, None] * 10 + np.arange(10)).reshape(len(pe), -1)
        it = self.t[rows, 1].astype(np.int32); ck = self.t[rows, 5]
        ok = np.repeat(pe >= 0, 10, axis=1) & (ck == 0)
        return np.where(ok, it, NI).astype(np.int32)

    def events(self, ev_idx):
        """Arrays for train events ev_idx (sorted)."""
        t = self.t
        rows = (ev_idx[:, None] * 10 + np.arange(10)).ravel()
        y = t[rows, 5].reshape(-1, 10)
        target = np.where(y.sum(1) > 0, y.argmax(1), 10)
        return dict(user=self.ev_user[ev_idx], items=t[rows, 1].reshape(-1, 10).astype(np.int64),
                    expo=self.expo[rows].reshape(-1, 10).astype(np.int64), target=target.astype(np.int64),
                    hist=self.hist[ev_idx].astype(np.int64), htim=self.htim[ev_idx].astype(np.int64),
                    ts=self.ev_ts[ev_idx], y=y.ravel().astype(np.float64), ev=ev_idx,
                    neg=self.neg_items(self.prev[ev_idx], self.ev_ts[ev_idx], Data.K))

    def eval_events(self):
        e = np.load(R / 'evaluation.npy')
        eh = np.load(R / 'evaluation_hist.npy'); et = np.load(R / 'evaluation_times.npy')
        ev = e[::10, 6]
        key = e[:, 0].astype(np.int64) * NI + e[:, 1]
        i = np.searchsorted(self.key_sorted, key); i = np.minimum(i, len(self.key_sorted) - 1)
        expo = np.where(self.key_sorted[i] == key, self.key_counts[i], 0)
        eu = e[::10, 0].astype(np.int64)
        neg = self.neg_items(self.user_last[eu], e[::10, 2].astype(np.int64), Data.K)
        return dict(neg=neg, user=e[::10, 0].astype(np.int64), items=e[:, 1].reshape(-1, 10).astype(np.int64),
                    expo=expo.reshape(-1, 10).astype(np.int64), target=np.zeros(len(ev), np.int64),
                    hist=eh[ev].astype(np.int64), htim=et[ev].astype(np.int64), ts=e[::10, 2].astype(np.int64),
                    y=np.full(len(e), -1.0), ev=ev)

    def first_per_user(self, lo, hi):
        m = np.flatnonzero((self.ev_ts >= lo) & (self.ev_ts < hi))
        u = self.ev_user[m]
        _, first = np.unique(u, return_index=True)   # events are time-sorted, so the first index is the earliest
        return np.sort(m[first])


def to_tensors(d, idx):
    h = d['hist'][idx]; valid = h >= 0
    gb = gap_bucket(d['ts'][idx][:, None] - d['htim'][idx])
    return (torch.from_numpy(d['user'][idx]), torch.from_numpy(d['items'][idx]),
            torch.from_numpy(np.minimum(d['expo'][idx], 7)), torch.from_numpy(np.where(valid, h, NI)),
            torch.from_numpy(np.where(valid, gb, 24)), torch.from_numpy(valid), torch.from_numpy(d['target'][idx]),
            torch.from_numpy(d['neg'][idx].astype(np.int64)))


class Model(nn.Module):
    def __init__(self, dim=32, attn=True, user_emb=True):
        super().__init__()
        self.dim, self.attn, self.user_emb = dim, attn, user_emb
        self.extras = True
        self.V = nn.Embedding(NI + 1, dim, sparse=True); self.bi = nn.Embedding(NI + 1, 1, sparse=True)
        self.H = nn.Embedding(NI + 1, dim, padding_idx=NI, sparse=True)
        self.T = nn.Embedding(25, dim, padding_idx=24)
        self.U = nn.Embedding(NU, dim, sparse=True); self.bu = nn.Embedding(NU, 1, sparse=True)
        self.N = nn.Embedding(NI + 1, dim, padding_idx=NI, sparse=True); self.nb = nn.Embedding(NI + 1, 1, padding_idx=NI, sparse=True)
        self.use_neg = False
        self.rep = nn.Embedding(6, 1); self.repgap = nn.Embedding(25, 1); self.ex = nn.Embedding(8, 1)
        self.b0 = nn.Parameter(torch.tensor(-1.8))
        self.Wq = nn.Linear(dim, dim, bias=False); self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.null_k = nn.Parameter(torch.zeros(dim))
        for e in (self.V, self.H):
            nn.init.normal_(e.weight, 0, 0.05)
        nn.init.zeros_(self.U.weight)
        for e in (self.bi, self.bu, self.rep, self.repgap, self.ex, self.T, self.N, self.nb):
            nn.init.zeros_(e.weight)
        with torch.no_grad():
            self.H.weight[NI].zero_(); self.T.weight[24].zero_()

    def forward(self, user, items, expo, hist, gb, valid, neg=None):
        v = self.V(items)                                            # B,10,D
        e = self.H(hist) + (self.T(gb) if self.extras else 0.0)       # B,20,D
        vm = valid.float()
        n = vm.sum(1, keepdim=True).clamp(min=1)
        m = (e * vm.unsqueeze(-1)).sum(1) / n.sqrt()                  # B,D
        z = m.unsqueeze(1).expand(-1, 10, -1)
        if self.user_emb:
            z = z + self.U(user).unsqueeze(1)
        nbias = 0.0
        if self.use_neg and neg is not None:
            nm = (neg != NI).float(); cnt_n = nm.sum(1, keepdim=True)
            z = z + (self.N(neg) * nm.unsqueeze(-1)).sum(1).unsqueeze(1) / cnt_n.clamp(min=1).sqrt().unsqueeze(1)
            nbias = (self.nb(neg).squeeze(-1) * nm).sum(1, keepdim=True) / cnt_n.clamp(min=1).sqrt()
        if self.attn:
            q = self.Wq(v)                                           # B,10,D
            k = self.Wk(e)                                           # B,20,D
            a = torch.einsum('bcd,bqd->bcq', q, k) / math.sqrt(self.dim)
            a = a.masked_fill(~valid.unsqueeze(1), -1e4)
            a0 = (q * self.null_k).sum(-1, keepdim=True) / math.sqrt(self.dim)
            w = torch.softmax(torch.cat([a0, a], -1), -1)[..., 1:]   # B,10,20
            r = torch.einsum('bcq,bqd->bcd', w, self.Wv(e))
            z = z + r
        same = (hist.unsqueeze(1) == items.unsqueeze(2)) & valid.unsqueeze(1)      # B,10,20
        cnt = same.sum(-1).clamp(max=5)
        lastgb = torch.where(same, gb.unsqueeze(1), torch.full_like(gb.unsqueeze(1), 24)).min(-1).values
        logit = self.b0 + self.bi(items).squeeze(-1) + (z * v).sum(-1) + nbias
        if self.extras:
            logit = logit + self.rep(cnt).squeeze(-1) + self.repgap(lastgb).squeeze(-1) + self.ex(expo).squeeze(-1)
        if self.user_emb:
            logit = logit + self.bu(user)
        return logit                                                 # B,10 ; no-click logit = 0

    @staticmethod
    def loss(logit, target):
        full = torch.cat([logit, torch.zeros_like(logit[:, :1])], 1)  # index 10 = no click
        return Fn.cross_entropy(full, target)

    @staticmethod
    def prob(logit):
        full = torch.cat([logit, torch.zeros_like(logit[:, :1])], 1)
        return torch.softmax(full, 1)[:, :10]


def predict(model, d, bs=4096):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(d['user']), bs):
            idx = np.arange(s, min(s + bs, len(d['user'])))
            tt = to_tensors(d, idx)
            out.append(Model.prob(model(*tt[:6], neg=tt[7])).numpy())
    return np.concatenate(out).ravel()


def adj_auc(d, p, eval_share):
    hb = np.digitize((d['hist'] >= 0).sum(1), [0, 4, 9, 19], right=True)
    w = (eval_share / (np.bincount(hb, minlength=5) / len(hb)))[hb]
    return float(roc_auc_score(d['y'], p, sample_weight=np.repeat(w, 10)))


def warm_init(D, cut, dim, log=print):
    """Pre-cutoff click-SVD item/user factors and smoothed CTR biases (all from rows / clicks with ts < cut)."""
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    t = D.t; n = int(np.searchsorted(t[:, 2], cut)); P = t[:n]
    pe = P[::10, 6]; lh = D.hist[pe]; lu = np.repeat(P[::10, 0].astype(np.int64), 20); v = lh.ravel() >= 0
    m = P[:, 5] > 0
    U_ = np.r_[lu[v], P[m, 0]]; I_ = np.r_[lh.ravel()[v], P[m, 1]].astype(np.int64)
    ui = np.unique(U_ * NI + I_)
    X = sp.csr_matrix((np.ones(len(ui), np.float32), (ui // NI, ui % NI)), shape=(NU, NI))
    pop = np.asarray(X.sum(0)).ravel().astype(np.float64)
    idf = np.log1p(max((X.getnnz(1) > 0).sum(), 1) / np.maximum(pop, 1))
    Xw = X.multiply(idf[None, :]).tocsr()
    rn = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel()); rn[rn == 0] = 1
    Xw = sp.diags(1 / rn) @ Xw
    if getattr(warm_init, 'mode', 'user') == 'cooc':
        # item-item co-click matrix, normalised by sqrt((n_a+10)(n_b+10)); drop the leading (popularity) component
        C = (X.T @ X).tocsr(); C.setdiag(0); C.eliminate_zeros()
        r_, c_ = C.nonzero(); vals = np.asarray(C[r_, c_]).ravel() / np.sqrt((pop[r_] + 10) * (pop[c_] + 10))
        C = sp.csr_matrix((vals, (r_, c_)), shape=(NI, NI))
        svd = TruncatedSVD(dim + 1, n_iter=6, random_state=7101).fit(C)
        comp = (svd.components_.T * np.sqrt(svd.singular_values_))[:, 1:]
    else:
        comp = TruncatedSVD(dim, n_iter=6, random_state=7101).fit(Xw).components_.T   # items x dim
    def nrm(a):
        z = np.linalg.norm(a, axis=1, keepdims=True); z[z == 0] = 1; return a / z
    V0 = nrm(comp).astype(np.float32); U0 = (0.25 * nrm(np.asarray(Xw @ comp))).astype(np.float32)
    y = P[:, 5].astype(np.float64); g = y.mean()
    ii = np.bincount(P[:, 1], minlength=NI); ic = np.bincount(P[:, 1], y, minlength=NI)
    bi0 = np.clip(np.log(((ic + 50 * g) / (ii + 50)) / g), -2, 2).astype(np.float32)
    ui_ = np.bincount(P[:, 0], minlength=NU); uc = np.bincount(P[:, 0], y, minlength=NU)
    tt = np.clip((uc + 50 * g) / (ui_ + 50) * 10, .02, .98)
    bu0 = np.clip(np.log(tt / (1 - tt) / 10) + 1.8, -2, 2).astype(np.float32)
    log(f'warm init cut {cut}: {len(ui)} user-item click pairs')
    warm_init.pairs = (ui // NI, ui % NI)
    # first-event histories only (pre-window / warm-up clicks of each user's earliest event before the cut), as warm16
    ev_u = P[::10, 0].astype(np.int64); _, fi = np.unique(ev_u, return_index=True)
    fh = D.hist[pe[fi]]; fu = np.repeat(ev_u[fi], 20); fv = fh.ravel() >= 0
    fp = np.unique(fu[fv] * NI + fh.ravel()[fv])
    warm_init.first_pairs = (fp // NI, fp % NI)
    return V0, U0, bi0, bu0


def train_model(args, tr, epochs, val=None, eval_share=None, log=print, seed=0, init=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Model(args.dim, attn=not args.no_attn, user_emb=not args.no_user)
    model.extras = not args.no_extras
    if init is not None:
        V0, U0, bi0, bu0 = init
        with torch.no_grad():
            model.V.weight[:NI] = torch.from_numpy(V0); model.H.weight[:NI] = torch.from_numpy(args.h_scale * V0)
            model.U.weight[:] = torch.from_numpy(U0); model.bi.weight[:NI, 0] = torch.from_numpy(bi0)
            model.bu.weight[:, 0] = torch.from_numpy(bu0)
    model.use_neg = args.neg > 0
    emb = [model.V.weight, model.H.weight, model.U.weight, model.bi.weight, model.bu.weight, model.N.weight, model.nb.weight]
    other = [p for p in model.parameters() if all(p is not q for q in emb)]
    opt_e = torch.optim.Adagrad(emb, lr=args.lr_emb, initial_accumulator_value=0.1)
    opt_d = torch.optim.Adam(other, lr=args.lr)
    n = len(tr['user']); hist_ = []
    best = (-1, 0, None)
    if args.aux > 0:
        aux_u, aux_i = [x.astype(np.int64) for x in (warm_init.first_pairs if args.aux_src == 'first' else warm_init.pairs)]
        cnt = np.bincount(aux_i, minlength=NI + 1) + 1.0
        pop_bias = torch.from_numpy(np.log(cnt / cnt.mean()).astype(np.float32))
    for ep in range(1, epochs + 1):
        model.train(); t = time.time(); perm = rng.permutation(n); tot = 0.0; nb = 0
        for s in range(0, n, args.bs):
            idx = np.sort(perm[s:s + args.bs])
            tt = to_tensors(tr, idx)
            logit = model(*tt[:6], neg=tt[7])
            full = torch.cat([logit, torch.zeros_like(logit[:, :1])], 1)
            loss = Fn.cross_entropy(full, tt[6], reduction='sum')
            # L2 per looked-up occurrence (choice-model style)
            vv = model.V(tt[1]); hh = model.H(tt[3])
            reg = args.l2 * ((vv ** 2).sum() + (hh ** 2).sum() + (model.bi(tt[1]) ** 2).sum())
            if args.neg > 0:
                reg = reg + args.l2 * (model.N(tt[7]) ** 2).sum()
            if args.aux > 0:   # ledger clicks (< cut) vs uniform negatives with popularity log-correction (warm-aux style)
                j = rng.integers(0, len(aux_u), args.bs)
                au = torch.from_numpy(aux_u[j]); ai = torch.from_numpy(np.c_[aux_i[j], rng.integers(0, NI, (args.bs, 10))])
                al = (model.U(au).unsqueeze(1) * model.V(ai)).sum(-1) + pop_bias[ai]
                loss = loss + args.aux * Fn.cross_entropy(al, torch.zeros(args.bs, dtype=torch.long), reduction='sum')
            if not args.no_user:
                reg = reg + args.l2 * ((model.U(tt[0]) ** 2).sum() + (model.bu(tt[0]) ** 2).sum())
            opt_e.zero_grad(); opt_d.zero_grad(); (loss + reg).backward(); opt_e.step(); opt_d.step()
            tot += float(loss.detach()) / len(idx); nb += 1
            if args.max_steps and nb >= args.max_steps:
                break
        msg = dict(epoch=ep, loss=round(tot / nb, 5), sec=round(time.time() - t, 1))
        if val is not None:
            p = predict(model, val)
            msg['val_adj'] = round(adj_auc(val, p, eval_share), 6); msg['val_plain'] = round(float(roc_auc_score(val['y'], p)), 6)
            if msg['val_adj'] > best[0]:
                best = (msg['val_adj'], ep, None)
        log(json.dumps(msg)); hist_.append(msg)
        if val is not None and ep - best[1] >= args.patience:
            break
    return model, best, hist_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=6)
    ap.add_argument('--name', default='b1')
    ap.add_argument('--dim', type=int, default=16)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--lr-emb', type=float, default=0.045)
    ap.add_argument('--h-scale', type=float, default=0.25)
    ap.add_argument('--no-warm', action='store_true')
    ap.add_argument('--aux', type=float, default=0.0)
    ap.add_argument('--no-extras', action='store_true')
    ap.add_argument('--fixed-epochs', type=int, default=0)
    ap.add_argument('--neg', type=int, default=0)
    ap.add_argument('--aux-src', default='ledger', choices=['ledger', 'first'])
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--no-attn', action='store_true')
    ap.add_argument('--no-user', action='store_true')
    ap.add_argument('--subsample', type=float, default=1.0)   # fraction of users (prototype only)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--phase', default='both', choices=['select', 'both'])
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--init', default='user', choices=['user', 'cooc'])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    warm_init.mode = a.init
    Data.K = max(a.neg, 1)
    logf = open(R / f'nxb_b_{a.name}_t{a.target}.log', 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(json.dumps(vars(a)))
    t0 = time.time()
    D = Data()
    eh = np.load(R / 'evaluation_hist.npy')
    eval_share = np.bincount(np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True), minlength=5) / len(eh)
    c = S + a.target * DAY
    keep_user = None
    if a.subsample < 1:
        keep_user = np.random.default_rng(77).random(NU) < a.subsample
    def pick(ev):
        return ev if keep_user is None else ev[keep_user[D.ev_user[ev]]]
    if a.fixed_epochs:
        best = (None, a.fixed_epochs, None)
        log(f'fixed epochs {a.fixed_epochs} (selected earlier on the pre-cutoff day of the day-6 pipeline)')
    else:
        tr1 = D.events(pick(np.flatnonzero(D.ev_ts < c - DAY)))
        val = D.events(pick(D.first_per_user(c - DAY, c)))
        log(f'data {time.time()-t0:.1f}s phase1 train events {len(tr1["user"])} val events {len(val["user"])}')
        init1 = None if a.no_warm else warm_init(D, c - DAY, a.dim, log)
        model, best, hist1 = train_model(a, tr1, a.epochs, val, eval_share, log, init=init1)
        log(f'phase1 best epoch {best[1]} val adj {best[0]:.6f} ({time.time()-t0:.1f}s)')
        sel = dict(args=vars(a), phase1=hist1, best_epoch=best[1], best_val_adj=best[0])
        (R / f'nxb_b_{a.name}_select_t{a.target}.json').write_text(json.dumps(sel, indent=1))
        if a.phase == 'select':
            return
        del tr1
    tr2 = D.events(pick(np.flatnonzero(D.ev_ts < c)))
    init2 = None if a.no_warm else warm_init(D, c, a.dim, log)
    model, _, hist2 = train_model(a, tr2, best[1], None, None, log, init=init2)
    if a.target in (5, 6):
        ref = np.load(R / f'nxa_h1_day{a.target}.npz')
        ev = ref['events'][::10]
        d = D.events(ev)
        assert np.array_equal(d['y'], ref['y'])
        p = predict(model, d)
        hb = np.digitize((d['hist'] >= 0).sum(1), [0, 4, 9, 19], right=True)
        res = dict(target=a.target, epochs=best[1], adj_full=adj_auc(d, p, eval_share), plain=float(roc_auc_score(d['y'], p)),
                   nxa_h1_adj_full=adj_auc(d, ref['pred'], eval_share))
        if a.target == 6:
            s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
            res['adj_sel'] = float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
        res['bins'] = {int(b): dict(n=int((hb == b).sum()), auc=float(roc_auc_score(d['y'][np.repeat(hb == b, 10)], p[np.repeat(hb == b, 10)])))
                       for b in range(5)}
        log(json.dumps(res))
        np.savez(R / f'nxb_b_{a.name}_day{a.target}.npz', y=d['y'], events=ref['events'], raw=p, pred=p)
    else:
        d = D.eval_events(); p = predict(model, d)
        np.save(R / f'nxb_b_{a.name}_eval.npy', p)
        log(f'eval rows {len(p)}')
    torch.save(model.state_dict(), R / f'nxb_b_{a.name}_t{a.target}.pt')
    log(f'total {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
