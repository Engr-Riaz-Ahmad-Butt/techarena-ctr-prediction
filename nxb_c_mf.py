"""nxb Track C: exact MAP choice-factor model (C1) and joint click + exposure model (C2). Self-contained, CPU.

Terms of the (negative log-posterior) objective, all from rows with timestamp < the fitting cut:
  events  : every labelled event, softmax over its 10 candidates + no-click;  s_ej = u.v_j + b_j,  s_e0 = c0 (+ w_u)
  ledger  : (--ledger-w) clicks without a slate: each (user, item) click of the pre-cut ledger ~ softmax over the whole
            catalogue of u.v_i + h_i   (exact, chunked matmuls; --ledger-src first = histories of each user's first
            event before the cut, i.e. mostly warm-up clicks; all = every pre-cut click)
  exposure: (--expo-w, C2) every pre-cut impression (u, i) ~ softmax over the catalogue of p_u.v_i + a_i  (policy belief p)
  priors  : lu|u - kappa p|^2 + lw w^2 + lp|p|^2 + lv|v|^2 + lb b^2 + lh h^2 + la a^2     (kappa refitted with --kappa-fit)
Optimisation: alternating exact per-user Newton for U (and P) with per-user backtracking; item blocks [v, b, h, a] by
Newton with the diag(p) / catalogue curvature bound and global backtracking; c0 by 1-D Newton. Warm start: idf click-SVD.
Protocol: target day d (cut c = S + d*DAY). Phase 1 fits on events < c - DAY and tracks the held-out log-loss and
adjusted AUC of the first-per-user events of day d-1 after each round (best round by held-out log-loss). Phase 2 refits
on everything < c with the same settings and number of rounds and scores day d (cohort) or the evaluation rows (d = 7).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent / 'artifacts'
S, DAY, NI, NU = 1737331200, 86400, 50000, 54456


class Raw:
    def __init__(self):
        self.t = np.load(R / 'train.npy')
        self.hist = np.load(R / 'train_hist.npy')
        self.ev_ts = self.t[::10, 2].astype(np.int64)
        self.ev_user = self.t[::10, 0].astype(np.int64)

    def events(self, ev):
        t = self.t
        rows = (ev[:, None] * 10 + np.arange(10)).ravel()
        y = t[rows, 5].reshape(-1, 10)
        tgt = np.where(y.sum(1) > 0, y.argmax(1), 10)
        return dict(user=torch.from_numpy(self.ev_user[ev]), items=torch.from_numpy(t[rows, 1].reshape(-1, 10).astype(np.int64)),
                    tgt=torch.from_numpy(tgt.astype(np.int64)), y=y.ravel().astype(np.float64),
                    hlen=(self.hist[ev] >= 0).sum(1), ev=ev)

    def first_per_user(self, lo, hi):
        m = np.flatnonzero((self.ev_ts >= lo) & (self.ev_ts < hi))
        _, f = np.unique(self.ev_user[m], return_index=True)
        return np.sort(m[f])

    def eval_events(self):
        e = np.load(R / 'evaluation.npy'); eh = np.load(R / 'evaluation_hist.npy')
        return dict(user=torch.from_numpy(e[::10, 0].astype(np.int64)), items=torch.from_numpy(e[:, 1].reshape(-1, 10).astype(np.int64)),
                    tgt=torch.zeros(len(e) // 10, dtype=torch.long), y=np.full(len(e), -1.0), hlen=(eh[e[::10, 6]] >= 0).sum(1), ev=e[::10, 6])

    def impressions(self, cut):
        n = int(np.searchsorted(self.t[:, 2], cut)); P = self.t[:n]
        k, c = np.unique(P[:, 0].astype(np.int64) * NI + P[:, 1], return_counts=True)
        return torch.from_numpy(k // NI), torch.from_numpy(k % NI), torch.from_numpy(c.astype(np.float32))

    def ledger(self, cut, src):
        n = int(np.searchsorted(self.t[:, 2], cut)); P = self.t[:n]
        pe = P[::10, 6]; pu = P[::10, 0].astype(np.int64)
        if src == 'first':
            _, f = np.unique(pu, return_index=True); pe = pe[f]; pu = pu[f]
        lh = self.hist[pe]; lu = np.repeat(pu, 20); v = lh.ravel() >= 0
        U_ = lu[v]; I_ = lh.ravel()[v].astype(np.int64)
        if src == 'all':
            m = P[:, 5] > 0; U_ = np.r_[U_, P[m, 0]]; I_ = np.r_[I_, P[m, 1].astype(np.int64)]
        k = np.unique(U_ * NI + I_)
        return torch.from_numpy(k // NI), torch.from_numpy(k % NI), torch.ones(len(k))

    def warm(self, cut, dim):
        """idf click-SVD (rank dim) of the full pre-cut click ledger + smoothed item CTR log-ratio + log popularity."""
        n = int(np.searchsorted(self.t[:, 2], cut)); P = self.t[:n]
        pe = P[::10, 6]; lh = self.hist[pe]; lu = np.repeat(P[::10, 0].astype(np.int64), 20); v = lh.ravel() >= 0; m = P[:, 5] > 0
        ui = np.unique(np.r_[lu[v], P[m, 0]] * NI + np.r_[lh.ravel()[v], P[m, 1]].astype(np.int64))
        X = sp.csr_matrix((np.ones(len(ui), np.float32), (ui // NI, ui % NI)), shape=(NU, NI))
        pop = np.asarray(X.sum(0)).ravel().astype(np.float64)
        idf = np.log1p(max((X.getnnz(1) > 0).sum(), 1) / np.maximum(pop, 1))
        Xw = X.multiply(idf[None, :]).tocsr(); rn = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel()); rn[rn == 0] = 1
        Xw = sp.diags(1 / rn) @ Xw
        comp = TruncatedSVD(dim, n_iter=8, random_state=7101).fit(Xw).components_.T
        nrm = np.linalg.norm(comp, axis=1, keepdims=True); nrm[nrm == 0] = 1
        y = P[:, 5].astype(np.float64); g = y.mean()
        ii = np.bincount(P[:, 1], minlength=NI); ic = np.bincount(P[:, 1], y, minlength=NI)
        b0 = np.log(((ic + 50 * g) / (ii + 50)) / g)
        lp = np.log(pop + 1); lp -= lp.mean(); le = np.log(ii + 1.0); le -= le.mean()
        f = lambda x: torch.from_numpy(np.asarray(x, np.float32))
        return f(comp / nrm), f(b0), f(lp), f(le)


class Cat:
    """catalogue-softmax term: each observation (user, item, count) ~ softmax_k(T_user . v_k + bias_k)."""
    def __init__(self, data, weight, utab, btab):
        self.iu, self.ii, self.cnt = data
        self.w, self.utab, self.btab = weight, utab, btab
        self.Nu = torch.zeros(NU).index_add_(0, self.iu, self.cnt)
        self.users = torch.nonzero(self.Nu > 0).squeeze(1)

    def pos_per_user(self, m, Ut):
        out = torch.zeros(NU, dtype=torch.float64); bias = getattr(m, self.btab)
        for s in range(0, len(self.iu), 1 << 20):
            j = slice(s, s + (1 << 20))
            out.index_add_(0, self.iu[j], (self.cnt[j] * ((Ut[self.iu[j]] * m.V[self.ii[j]]).sum(1) + bias[self.ii[j]])).double())
        return out

    def loss_per_user(self, m, Ut=None, cb=512):
        Ut = getattr(m, self.utab) if Ut is None else Ut; bias = getattr(m, self.btab)
        lse = torch.zeros(NU, dtype=torch.float64)
        for s in range(0, len(self.users), cb):
            us = self.users[s:s + cb]
            lse[us] = (self.Nu[us] * torch.logsumexp(Ut[us] @ m.V.T + bias, 1)).double()
        return self.w * (lse - self.pos_per_user(m, Ut))

    def loss(self, m):
        return float(self.loss_per_user(m).sum())

    def user_gh(self, m, cb=512):
        d = m.d; Ut = getattr(m, self.utab); bias = getattr(m, self.btab)
        G = torch.zeros(NU, d, dtype=torch.float64); H = torch.zeros(NU, d, d, dtype=torch.float64)
        for s in range(0, len(self.iu), 1 << 20):
            j = slice(s, s + (1 << 20))
            G.index_add_(0, self.iu[j], -(self.cnt[j].unsqueeze(1) * m.V[self.ii[j]]).double())
        VV = (m.V.unsqueeze(2) * m.V.unsqueeze(1)).reshape(NI, d * d)
        for s in range(0, len(self.users), cb):
            us = self.users[s:s + cb]
            q = torch.softmax(Ut[us] @ m.V.T + bias, 1)
            ev = q @ m.V; evv = (q @ VV).reshape(-1, d, d)
            G[us] += (self.Nu[us].unsqueeze(1) * ev).double()
            H[us] += (self.Nu[us].view(-1, 1, 1) * (evv - ev.unsqueeze(2) * ev.unsqueeze(1))).double()
        return self.w * G, self.w * H

    def item_gh(self, m, cb=512):
        """gradient and curvature bound w.r.t. item blocks [v, bias]."""
        d = m.d; Ut = getattr(m, self.utab); bias = getattr(m, self.btab)
        G = torch.zeros(NI, d + 1, dtype=torch.float64); H = torch.zeros(NI, (d + 1) ** 2, dtype=torch.float64)
        for s in range(0, len(self.iu), 1 << 20):
            j = slice(s, s + (1 << 20))
            Y = torch.cat([Ut[self.iu[j]], torch.ones(len(self.iu[j]), 1)], 1)
            G.index_add_(0, self.ii[j], -(self.cnt[j].unsqueeze(1) * Y).double())
        for s in range(0, len(self.users), cb):
            us = self.users[s:s + cb]
            q = torch.softmax(Ut[us] @ m.V.T + bias, 1) * self.Nu[us].unsqueeze(1)
            Y = torch.cat([Ut[us], torch.ones(len(us), 1)], 1)
            G += (q.T @ Y).double()
            H += (q.T @ (Y.unsqueeze(2) * Y.unsqueeze(1)).reshape(len(us), -1)).double()
        return self.w * G, self.w * H.reshape(NI, d + 1, d + 1)


class Model:
    def __init__(self, a, warm):
        V0, b0, lp, le = warm
        self.a = a; self.d = a.dim
        self.U = torch.zeros(NU, a.dim); self.w = torch.zeros(NU); self.P = torch.zeros(NU, a.dim)
        self.V = V0.clone() * a.vscale; self.b = b0.clone(); self.Hc = lp.clone(); self.A = le.clone()
        self.Vmu = V0.clone() * a.vscale if a.v_anchor else torch.zeros_like(V0)      # prior mean of v
        self.c0 = torch.tensor(1.77); self.kappa = 0.0
        self.cats = []

    def logits(self, user, items, U=None, w=None):
        U = self.U if U is None else U; w = self.w if w is None else w
        s = (U[user].unsqueeze(1) * self.V[items]).sum(-1) + self.b[items]
        s0 = self.c0.expand(len(user))
        if self.a.user_offset:
            s0 = s0 + w[user]
        return torch.cat([s, s0.unsqueeze(1)], 1)

    def event_loss(self, E, U=None, w=None, per_user=False, bs=131072):
        tot = 0.0; pu = torch.zeros(NU, dtype=torch.float64) if per_user else None
        for s in range(0, len(E['user']), bs):
            u = E['user'][s:s + bs]; l = self.logits(u, E['items'][s:s + bs], U, w)
            le = torch.logsumexp(l, 1) - l.gather(1, E['tgt'][s:s + bs].unsqueeze(1)).squeeze(1)
            tot += float(le.double().sum())
            if per_user:
                pu.index_add_(0, u, le.double())
        return tot, pu

    def objective(self, E):
        a = self.a
        f, _ = self.event_loss(E)
        f += a.lu * float(((self.U - self.kappa * self.P) ** 2).sum()) + a.lw * float((self.w ** 2).sum())
        f += a.lv * float(((self.V - self.Vmu) ** 2).sum()) + a.lb * float((self.b ** 2).sum())
        f += a.lh * float((self.Hc ** 2).sum()) + a.la * float((self.A ** 2).sum()) + a.lp * float((self.P ** 2).sum())
        for c in self.cats:
            f += c.loss(self)
        return f

    def user_step(self, E, bs=65536):
        a, d = self.a, self.d
        k = d + (1 if a.user_offset else 0)
        G = torch.zeros(NU, k, dtype=torch.float64); H = torch.zeros(NU, k, k, dtype=torch.float64)
        before = torch.zeros(NU, dtype=torch.float64)
        for s in range(0, len(E['user']), bs):
            u = E['user'][s:s + bs]; it = E['items'][s:s + bs]; tg = E['tgt'][s:s + bs]
            l = self.logits(u, it); p = torch.softmax(l, 1)
            before.index_add_(0, u, (torch.logsumexp(l, 1) - l.gather(1, tg.unsqueeze(1)).squeeze(1)).double())
            X = self.V[it]
            x0 = torch.zeros(len(u), 1, k)
            if a.user_offset:
                X = torch.cat([X, torch.zeros(len(u), 10, 1)], 2); x0[:, 0, d] = 1.0
            X = torch.cat([X, x0], 1)                                         # B,11,k (index 10 = no click)
            m = (p.unsqueeze(-1) * X).sum(1)
            g = m - X[torch.arange(len(u)), tg]
            h = torch.einsum('bj,bjk,bjl->bkl', p, X, X) - m.unsqueeze(2) * m.unsqueeze(1)
            G.index_add_(0, u, g.double()); H.index_add_(0, u, h.double())
        ucats = [c for c in self.cats if c.utab == 'U']
        for c in ucats:
            g, h = c.user_gh(self); G[:, :d] += g; H[:, :d, :d] += h
            before += c.loss_per_user(self)
        th = self.U.double() if not a.user_offset else torch.cat([self.U, self.w.unsqueeze(1)], 1).double()
        mu = (self.kappa * self.P).double()
        if a.user_offset:
            mu = torch.cat([mu, torch.zeros(NU, 1, dtype=torch.float64)], 1)
        reg = torch.full((k,), 2 * a.lu, dtype=torch.float64)
        if a.user_offset:
            reg[d] = 2 * a.lw
        G += reg * (th - mu); H += torch.diag_embed(reg.expand(NU, k))
        before += (reg / 2 * (th - mu) ** 2).sum(1)
        delta = torch.linalg.solve(H, G.unsqueeze(2)).squeeze(2)
        step = torch.ones(NU, 1, dtype=torch.float64); bad = torch.zeros(NU, dtype=torch.bool)
        for _ in range(4):                                                     # per-user backtracking
            new = th - step * delta
            Un = new[:, :d].float(); wn = new[:, d].float() if a.user_offset else self.w
            _, after = self.event_loss(E, Un, wn, per_user=True)
            for c in ucats:
                after += c.loss_per_user(self, Un)
            after += (reg / 2 * (new - mu) ** 2).sum(1)
            bad = after > before + 1e-7
            if not bad.any():
                break
            step[bad] *= 0.5
        new = th - step * delta
        new[bad] = th[bad]
        self.U = new[:, :d].float().contiguous()
        if a.user_offset:
            self.w = new[:, d].float().contiguous()

    def policy_step(self):
        a, d = self.a, self.d
        pc = [c for c in self.cats if c.utab == 'P']
        if not pc:
            return
        G = torch.zeros(NU, d, dtype=torch.float64); H = torch.zeros(NU, d, d, dtype=torch.float64)
        before = torch.zeros(NU, dtype=torch.float64)
        for c in pc:
            g, h = c.user_gh(self); G += g; H += h; before += c.loss_per_user(self)
        th = self.P.double()
        G += 2 * a.lp * th - 2 * a.lu * self.kappa * (self.U.double() - self.kappa * th)
        H += torch.diag_embed(torch.full((NU, d), 2 * a.lp + 2 * a.lu * self.kappa ** 2, dtype=torch.float64))
        before += a.lp * (th ** 2).sum(1) + a.lu * ((self.U.double() - self.kappa * th) ** 2).sum(1)
        delta = torch.linalg.solve(H, G.unsqueeze(2)).squeeze(2)
        step = torch.ones(NU, 1, dtype=torch.float64); bad = torch.zeros(NU, dtype=torch.bool)
        for _ in range(4):
            new = th - step * delta
            after = sum(c.loss_per_user(self, new.float()) for c in pc)
            after = after + a.lp * (new ** 2).sum(1) + a.lu * ((self.U.double() - self.kappa * new) ** 2).sum(1)
            bad = after > before + 1e-7
            if not bad.any():
                break
            step[bad] *= 0.5
        new = th - step * delta; new[bad] = th[bad]
        self.P = new.float().contiguous()

    def item_step(self, E, bs=65536):
        a, d = self.a, self.d
        cols = {'b': d}; k = d + 1
        for c in self.cats:
            cols[c.btab] = k; k += 1
        G = torch.zeros(NI, k, dtype=torch.float64); H = torch.zeros(NI, k * k, dtype=torch.float64)
        for s in range(0, len(E['user']), bs):
            u = E['user'][s:s + bs]; it = E['items'][s:s + bs]; tg = E['tgt'][s:s + bs]; B = len(u)
            p = torch.softmax(self.logits(u, it), 1)[:, :10]
            t = torch.zeros(B, 11); t[torch.arange(B), tg] = 1.0; t = t[:, :10]
            Y = torch.cat([self.U[u], torch.ones(B, 1), torch.zeros(B, k - d - 1)], 1)
            idx = torch.stack([it.reshape(-1), torch.arange(B).repeat_interleave(10)])
            G += torch.sparse.mm(torch.sparse_coo_tensor(idx, (p - t).reshape(-1), (NI, B)), Y).double()
            H += torch.sparse.mm(torch.sparse_coo_tensor(idx, p.reshape(-1), (NI, B)), (Y.unsqueeze(2) * Y.unsqueeze(1)).reshape(B, -1)).double()
        H = H.reshape(NI, k, k)
        for c in self.cats:
            g, h = c.item_gh(self)
            ii = torch.tensor(list(range(d)) + [cols[c.btab]])
            G[:, ii] += g
            H[:, ii.unsqueeze(1), ii.unsqueeze(0)] += h
        tabs = ['b'] + [c.btab for c in self.cats]
        th = torch.cat([self.V] + [getattr(self, t).unsqueeze(1) for t in tabs], 1).double()
        lam = {'b': a.lb, 'Hc': a.lh, 'A': a.la}
        reg = torch.full((k,), 2 * a.lv, dtype=torch.float64)
        for t in tabs:
            reg[cols[t]] = 2 * lam[t]
        mu = torch.zeros_like(th); mu[:, :d] = self.Vmu.double()
        G += reg * (th - mu); H += torch.diag_embed(reg.expand(NI, k))
        delta = torch.linalg.solve(H, G.unsqueeze(2)).squeeze(2)
        f0 = self.objective(E); step = 1.0
        old = {t: getattr(self, t).clone() for t in ['V'] + tabs}
        for _ in range(5):                                                     # global backtracking
            new = th - step * delta
            self.V = new[:, :d].float().contiguous()
            for t in tabs:
                setattr(self, t, new[:, cols[t]].float().contiguous())
            f1 = self.objective(E)
            if f1 <= f0:
                return f1
            step *= 0.5
        for t, v in old.items():
            setattr(self, t, v)
        return f0

    def c0_step(self, E, bs=131072):
        g = 0.0; h = 0.0
        for s in range(0, len(E['user']), bs):
            p0 = torch.softmax(self.logits(E['user'][s:s + bs], E['items'][s:s + bs]), 1)[:, 10]
            t0 = (E['tgt'][s:s + bs] == 10).float()
            g += float((p0 - t0).double().sum()); h += float((p0 * (1 - p0)).double().sum())
        self.c0 = self.c0 - g / max(h, 1e-9)

    def fit_kappa(self):
        num = float((self.U * self.P).sum()); den = float((self.P * self.P).sum())
        self.kappa = num / den if den > 0 else 0.0

    def predict(self, E, bs=131072):
        out = []
        for s in range(0, len(E['user']), bs):
            out.append(torch.softmax(self.logits(E['user'][s:s + bs], E['items'][s:s + bs]), 1)[:, :10])
        return torch.cat(out).numpy().astype(np.float64).ravel()

    def scores(self, E):
        u, it = E['user'], E['items']
        uv = (self.U[u].unsqueeze(1) * self.V[it]).sum(-1)
        pv = (self.P[u].unsqueeze(1) * self.V[it]).sum(-1)
        return uv.numpy().ravel().astype(np.float64), pv.numpy().ravel().astype(np.float64)


def eval_share():
    eh = np.load(R / 'evaluation_hist.npy')
    return np.bincount(np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True), minlength=5) / len(eh)


def adj(E, p, share):
    hb = np.digitize(E['hlen'], [0, 4, 9, 19], right=True)
    w = (share / (np.bincount(hb, minlength=5) / len(hb)))[hb]
    return float(roc_auc_score(E['y'], p, sample_weight=np.repeat(w, 10)))


def run_fit(a, raw, cut, rounds, val=None, share=None, log=print):
    m = Model(a, raw.warm(cut, a.dim))
    E = raw.events(np.flatnonzero(raw.ev_ts < cut))
    if a.ledger_w > 0:
        m.cats.append(Cat(raw.ledger(cut, a.ledger_src), a.ledger_w, 'U', 'Hc'))
    if a.expo_w > 0:
        m.cats.append(Cat(raw.impressions(cut), a.expo_w, 'P', 'A'))
    hist = []; best = None
    for r in range(1, rounds + 1):
        t0 = time.time()
        m.policy_step()
        m.user_step(E)
        if a.kappa_fit and a.expo_w > 0:
            m.fit_kappa()
        m.c0_step(E)
        f = m.item_step(E)
        msg = dict(round=r, obj_per_event=round(f / len(E['user']), 5), c0=round(float(m.c0), 4), kappa=round(m.kappa, 4),
                   unorm=round(float(m.U.norm(dim=1).mean()), 4), pnorm=round(float(m.P.norm(dim=1).mean()), 4),
                   vnorm=round(float(m.V.norm(dim=1).mean()), 4), sec=round(time.time() - t0, 1))
        if val is not None:
            p = m.predict(val)
            msg['val_ll'] = round(m.event_loss(val)[0] / len(val['user']), 6); msg['val_adj'] = round(adj(val, p, share), 6)
            msg['val_plain'] = round(float(roc_auc_score(val['y'], p)), 6)
            if a.expo_w > 0:
                uv, pv = m.scores(val); msg['val_adj_pv'] = round(adj(val, pv, share), 6)
            if best is None or msg['val_ll'] < best[0]:
                best = (msg['val_ll'], r, msg['val_adj'])
        log(json.dumps(msg)); hist.append(msg)
        if val is not None and r - best[1] >= a.patience:
            break
    return m, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=6); ap.add_argument('--name', default='c1')
    ap.add_argument('--dim', type=int, default=16); ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--patience', type=int, default=2)
    for k, v in dict(lu=1.0, lw=1.0, lv=1.0, lb=1.0, lp=1.0, la=1.0, lh=1.0).items():
        ap.add_argument('--' + k, type=float, default=v)
    ap.add_argument('--vscale', type=float, default=1.0)
    ap.add_argument('--v-anchor', action='store_true')
    ap.add_argument('--user-offset', action='store_true')
    ap.add_argument('--ledger-w', type=float, default=0.0); ap.add_argument('--ledger-src', default='first', choices=['first', 'all'])
    ap.add_argument('--expo-w', type=float, default=0.0); ap.add_argument('--kappa-fit', action='store_true')
    ap.add_argument('--phase', default='both', choices=['select', 'both']); ap.add_argument('--fixed-rounds', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    logf = open(R / f'nxb_c_{a.name}_t{a.target}.log', 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(json.dumps(vars(a))); t0 = time.time()
    raw = Raw(); share = eval_share(); c = S + a.target * DAY
    if a.fixed_rounds:
        best = (None, a.fixed_rounds, None)
    else:
        val = raw.events(raw.first_per_user(c - DAY, c))
        _, best, h1 = run_fit(a, raw, c - DAY, a.rounds, val, share, log)
        log(f'phase1 best round {best[1]} held-out logloss {best[0]:.6f} adj {best[2]:.6f} ({time.time()-t0:.0f}s)')
        (R / f'nxb_c_{a.name}_select_t{a.target}.json').write_text(json.dumps(dict(args=vars(a), phase1=h1, best=best), indent=1))
        if a.phase == 'select':
            return
    m, _, _ = run_fit(a, raw, c, best[1], None, None, log)
    if a.target in (5, 6):
        ref = np.load(R / f'nxa_h1_day{a.target}.npz'); E = raw.events(ref['events'][::10])
        assert np.array_equal(E['y'], ref['y'])
        p = m.predict(E); uv, pv = m.scores(E)
        res = dict(target=a.target, rounds=best[1], adj_full=adj(E, p, share), plain=float(roc_auc_score(E['y'], p)),
                   adj_uv=adj(E, uv, share), adj_pv=adj(E, pv, share) if a.expo_w > 0 else None)
        if a.target == 6:
            s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
            res['adj_sel'] = float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
        hb = np.repeat(np.digitize(E['hlen'], [0, 4, 9, 19], right=True), 10)
        res['bins'] = {int(b): float(roc_auc_score(E['y'][hb == b], p[hb == b])) for b in range(5)}
        log(json.dumps(res))
        np.savez(R / f'nxb_c_{a.name}_day{a.target}.npz', y=E['y'], events=ref['events'], raw=p, pred=p, uv=uv, pv=pv)
    else:
        E = raw.eval_events(); p = m.predict(E); uv, pv = m.scores(E)
        np.save(R / f'nxb_c_{a.name}_eval.npy', p); np.savez(R / f'nxb_c_{a.name}_eval_extra.npz', uv=uv, pv=pv)
        log(f'eval rows {len(p)}')
    torch.save(dict(U=m.U, w=m.w, V=m.V, b=m.b, c0=m.c0, P=m.P, A=m.A, Hc=m.Hc, kappa=m.kappa), R / f'nxb_c_{a.name}_t{a.target}.pt')
    log(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
