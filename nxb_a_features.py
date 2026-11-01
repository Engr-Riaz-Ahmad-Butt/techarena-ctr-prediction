"""nxb Track A: clean-room causal features for one day block (self-contained; reads raw arrays only).

Block k (1..6) = all events whose timestamp lies in [S + k*DAY, S + (k+1)*DAY); block 7 = evaluation rows.
Cutoff c = S + k*DAY. Every label statistic, ledger click, impression count and factor model uses rows with
timestamp < c only. The event's own history (clicks strictly before the event) is used as given.
Output: artifacts/nxb_a_feat_k{k}.npy (float32, rows x F), artifacts/nxb_a_meta_k{k}.npz
(y, events, rowidx, first, hbin), artifacts/nxb_a_names.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

R = Path(__file__).resolve().parent / 'artifacts'
S = 1737331200
DAY = 86400
NI = 50000
NU = 54456  # users in train + evaluation (max id 54455)


def smooth(clk, imp, prior, a):
    return (clk + a * prior) / (imp + a)


def lookup(keys, vals, q, default=0.0):
    idx = np.searchsorted(keys, q)
    idx = np.minimum(idx, len(keys) - 1)
    hit = keys[idx] == q
    out = np.full(q.shape, default, dtype=np.float64)
    out[hit] = vals[idx[hit]]
    return out


def count_keys(keys):
    k, c = np.unique(keys, return_counts=True)
    return k, c.astype(np.float64)


def within_event(x):
    """x: rows (multiple of 10) -> rank within event (0 = largest), x - event max, x - event mean."""
    m = x.reshape(-1, 10)
    order = np.argsort(-m, axis=1, kind='stable')
    rk = np.empty_like(order)
    np.put_along_axis(rk, order, np.arange(10)[None, :].repeat(len(m), 0), axis=1)
    return rk.ravel().astype(np.float32), (m - m.max(1, keepdims=True)).ravel(), (m - m.mean(1, keepdims=True)).ravel()


def build(k, log=print):
    t0 = time.time()
    train = np.load(R / 'train.npy')
    hist = np.load(R / 'train_hist.npy')
    htim = np.load(R / 'train_times.npy')
    c = S + k * DAY
    n_past = int(np.searchsorted(train[:, 2], c))  # train is sorted by timestamp, events contiguous
    assert n_past % 10 == 0
    P = train[:n_past]
    assert P[:, 2].max() < c
    if k <= 6:
        hi = int(np.searchsorted(train[:, 2], c + DAY))
        B = train[n_past:hi].copy()
        bh, bt = hist[B[::10, 6]], htim[B[::10, 6]]
        y = B[:, 5].astype(np.float32)
    else:
        B = np.load(R / 'evaluation.npy')
        bh, bt = np.load(R / 'evaluation_hist.npy'), np.load(R / 'evaluation_times.npy')
        bh, bt = bh[B[::10, 6]], bt[B[::10, 6]]
        y = np.full(len(B), -1, np.float32)
    assert B[:, 2].min() >= c
    ne = len(B) // 10
    eu, ets, etb = B[::10, 0].astype(np.int64), B[::10, 2].astype(np.int64), B[::10, 3]
    assert np.all(B[:, 0].reshape(-1, 10) == eu[:, None])
    cand = B[:, 1].astype(np.int64).reshape(-1, 10)
    hv = bh >= 0
    assert np.all(np.where(hv, bt < ets[:, None], True)), 'history must precede the event'
    # first event of each user within the block (block 7: every event is a first event)
    order = np.lexsort((ets, eu))
    first_ev = np.zeros(ne, bool)
    first_ev[order[np.r_[True, eu[order][1:] != eu[order][:-1]]]] = True
    F = {}

    # ---------------- label statistics from P (timestamp < c) ----------------
    pu, pi, pts, ptb, pcl = P[:, 0].astype(np.int64), P[:, 1].astype(np.int64), P[:, 2], P[:, 3], P[:, 5].astype(np.float64)
    g = pcl.mean()
    i_imp = np.bincount(pi, minlength=NI).astype(np.float64)
    i_clk = np.bincount(pi, pcl, minlength=NI)
    i_ctr = smooth(i_clk, i_imp, g, 50.)
    rec = {}
    for w in (1, 3):
        m = pts >= c - w * DAY
        ii = np.bincount(pi[m], minlength=NI).astype(np.float64)
        ic = np.bincount(pi[m], pcl[m], minlength=NI)
        rec[w] = (ii, smooth(ic, ii, i_ctr, 20.))
    ib_imp = np.bincount(pi * 4 + ptb, minlength=NI * 4).astype(np.float64)
    ib_clk = np.bincount(pi * 4 + ptb, pcl, minlength=NI * 4)
    ib_ctr = smooth(ib_clk, ib_imp, np.repeat(i_ctr, 4), 20.)
    i_first = np.full(NI, np.nan); i_last = np.full(NI, np.nan)
    if len(pi):
        _, fidx = np.unique(pi, return_index=True)
        i_first[pi[fidx]] = pts[fidx]
        ridx = len(pi) - 1 - np.unique(pi[::-1], return_index=True)[1]
        i_last[pi[ridx]] = pts[ridx]

    u_imp = np.bincount(pu, minlength=NU).astype(np.float64)
    u_clk = np.bincount(pu, pcl, minlength=NU)
    u_ctr = smooth(u_clk, u_imp, g, 30.)
    pev_u = pu[::10]; pev_c = pcl.reshape(-1, 10).sum(1); pev_t = pts[::10]
    ge = pev_c.mean()
    u_ev = np.bincount(pev_u, minlength=NU).astype(np.float64)
    u_evc = np.bincount(pev_u, pev_c, minlength=NU)
    u_evr = smooth(u_evc, u_ev, ge, 5.)
    m1 = pev_t >= c - DAY
    u_ev1 = np.bincount(pev_u[m1], minlength=NU).astype(np.float64)
    u_evr1 = smooth(np.bincount(pev_u[m1], pev_c[m1], minlength=NU), u_ev1, u_evr, 3.)
    u_lastev = np.full(NU, np.nan)
    ridx = len(pev_u) - 1 - np.unique(pev_u[::-1], return_index=True)[1]
    u_lastev[pev_u[ridx]] = pev_t[ridx]

    ui_k, ui_n = count_keys(pu * NI + pi)                       # exposures (label-free)
    m = pcl > 0
    uic_k, uic_n = count_keys(pu[m] * NI + pi[m])                 # clicks
    log(f'k{k} label stats {time.time()-t0:.1f}s past rows {n_past}')

    # ---------------- click ledger (< c): labelled clicks + histories of events before c ----------------
    pe = P[::10, 6]
    lh, lt = hist[pe], htim[pe]
    lu = np.repeat(P[::10, 0].astype(np.int64), 20)
    lv = lh.ravel() >= 0
    L_u = np.r_[lu[lv], pu[m]]
    L_i = np.r_[lh.ravel()[lv].astype(np.int64), pi[m]]
    L_t = np.r_[lt.ravel()[lv].astype(np.int64), pts[m].astype(np.int64)]
    assert L_t.max() < c
    key = (L_u * NI + L_i) * (1 << 31) + (L_t - 1736000000)
    _, ux = np.unique(key, return_index=True)
    L_u, L_i, L_t = L_u[ux], L_i[ux], L_t[ux]
    del lh, lt, lu, lv, key
    l_ui_k, l_ui_n = count_keys(L_u * NI + L_i)
    l_u_n = np.bincount(L_u, minlength=NU).astype(np.float64)
    pair_u = l_ui_k // NI; pair_i = l_ui_k % NI
    X = sp.csr_matrix((np.ones(len(pair_u), np.float32), (pair_u, pair_i)), shape=(NU, NI))
    i_users = np.asarray(X.sum(0)).ravel().astype(np.float64)    # distinct users who clicked
    log(f'k{k} ledger {len(L_u)} clicks, {len(pair_u)} user-item pairs {time.time()-t0:.1f}s')

    # co-click cosine (diagonal removed)
    C = (X.T @ X).tocsr(); C.setdiag(0); C.eliminate_zeros(); C.sort_indices()
    rows_ = np.repeat(np.arange(NI, dtype=np.int64), np.diff(C.indptr))
    cc_keys = rows_ * NI + C.indices.astype(np.int64)
    sq = np.sqrt(np.maximum(i_users, 1.))
    cc_vals = (C.data / (sq[rows_] * sq[C.indices])).astype(np.float64)
    del rows_, C
    log(f'k{k} co-click nnz {len(cc_keys)} {time.time()-t0:.1f}s')

    # SVD factor model on idf-weighted clicks
    nus = max((l_u_n > 0).sum(), 1)
    idf = np.log1p(nus / np.maximum(i_users, 1.))
    Xw = X.multiply(idf[None, :]).tocsr()
    rn = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel()); rn[rn == 0] = 1
    Xw = sp.diags(1 / rn) @ Xw
    svd = TruncatedSVD(32, algorithm='randomized', n_iter=6, random_state=6101).fit(Xw)
    V = svd.components_.T.astype(np.float32)                      # items x 32
    Ul = Xw @ V                                                   # users x 32 (ledger profile)
    # exposure SVD (label-free)
    E = sp.csr_matrix((np.log1p(ui_n), (ui_k // NI, ui_k % NI)), shape=(NU, NI))
    e_users = np.bincount(ui_k % NI, minlength=NI).astype(np.float64)
    eidf = np.log1p(max((u_imp > 0).sum(), 1) / np.maximum(e_users, 1.))
    E = E.multiply(eidf[None, :]).tocsr()
    rn = np.sqrt(np.asarray(E.multiply(E).sum(1)).ravel()); rn[rn == 0] = 1
    E = sp.diags(1 / rn) @ E
    esvd = TruncatedSVD(32, algorithm='randomized', n_iter=6, random_state=6102).fit(E)
    Ve = esvd.components_.T.astype(np.float32)
    Ue = E @ Ve
    del E, Xw
    log(f'k{k} svd {time.time()-t0:.1f}s')

    # ---------------- per-row features ----------------
    ru = np.repeat(eu, 10); rts = np.repeat(ets, 10).astype(np.float64); rtb = np.repeat(etb, 10)
    ci = cand.ravel()
    hours = 3600.
    F['i_ctr'] = i_ctr[ci]; F['i_imp'] = np.log1p(i_imp[ci]); F['i_clk'] = np.log1p(i_clk[ci])
    F['i_ctr1'] = rec[1][1][ci]; F['i_imp1'] = np.log1p(rec[1][0][ci])
    F['i_ctr3'] = rec[3][1][ci]; F['i_imp3'] = np.log1p(rec[3][0][ci])
    F['i_ctr_tb'] = ib_ctr[ci * 4 + rtb]
    F['i_trend'] = np.log(F['i_ctr1'] / F['i_ctr'])
    F['i_users'] = np.log1p(i_users[ci])
    F['i_age'] = (rts - i_first[ci]) / hours
    F['i_since_last'] = (rts - i_last[ci]) / hours
    F['u_ctr'] = u_ctr[ru]; F['u_imp'] = np.log1p(u_imp[ru]); F['u_evr'] = u_evr[ru]; F['u_ev'] = np.log1p(u_ev[ru])
    F['u_evr1'] = u_evr1[ru]; F['u_ev1'] = np.log1p(u_ev1[ru])
    F['u_since_ev'] = (rts - u_lastev[ru]) / hours
    F['u_ledger'] = np.log1p(l_u_n[ru])
    q = ru * NI + ci
    F['ui_expo'] = lookup(ui_k, ui_n, q); F['ui_clk'] = lookup(uic_k, uic_n, q)
    F['ui_ledger'] = lookup(l_ui_k, l_ui_n, q)
    # history
    hl = hv.sum(1)
    hit = (bh[:, None, :] == cand[:, :, None]) & hv[:, None, :]           # ne x 10 x 20
    F['h_len'] = np.repeat(hl, 10).astype(np.float64)
    last_t = np.where(hl > 0, bt[np.arange(ne), np.maximum(hl - 1, 0)], -1)
    F['h_gap_last'] = np.repeat(np.where(hl > 0, (ets - last_t) / hours, np.nan), 10)
    F['h_span'] = np.repeat(np.where(hl > 1, (last_t - bt[:, 0]) / hours, np.nan), 10)
    F['h_n24'] = np.repeat((((ets[:, None] - bt) < DAY) & hv).sum(1), 10).astype(np.float64)
    srt = np.sort(np.where(hv, bh, -1), 1)
    nun = ((srt[:, 1:] != srt[:, :-1]) & (srt[:, 1:] >= 0)).sum(1) + (srt[:, 0] >= 0)
    F['h_unique'] = np.repeat(nun / np.maximum(hl, 1), 10)
    F['c_hist_cnt'] = hit.sum(2).ravel().astype(np.float64)
    pos = np.arange(20)[None, None, :]
    lastq = np.where(hit, pos, -1).max(2)                                   # ne x 10
    F['c_hist_pos'] = np.where(lastq >= 0, (hl[:, None] - 1 - lastq), 25).ravel().astype(np.float64)
    lastct = np.take_along_axis(bt, np.maximum(lastq, 0), 1)
    F['c_hist_gap'] = np.where(lastq >= 0, (ets[:, None] - lastct) / hours, np.nan).ravel()
    del hit
    # co-click similarity of candidate to the history entries
    w_rec = np.where(hv, 1.0 / (hl[:, None] - np.arange(20)[None, :]).clip(1), 0.)    # 1/(rank from newest+1)
    cc_sum = np.zeros((ne, 10)); cc_max = np.zeros((ne, 10)); cc_rec = np.zeros((ne, 10)); cc_last = np.zeros((ne, 10))
    for s in range(0, ne, 20000):
        e = slice(s, min(s + 20000, ne))
        hh = bh[e].astype(np.int64); cc = cand[e]
        qk = hh[:, None, :] * NI + cc[:, :, None]
        sim = lookup(cc_keys, cc_vals, qk.ravel()).reshape(qk.shape)
        sim *= hv[e][:, None, :]
        cc_sum[e] = sim.sum(2); cc_max[e] = sim.max(2)
        cc_rec[e] = (sim * w_rec[e][:, None, :]).sum(2)
        li = np.maximum(hl[e] - 1, 0)
        cc_last[e] = sim[np.arange(len(hh)), :, li] * (hl[e] > 0)[:, None]
    F['cc_sum'] = cc_sum.ravel(); F['cc_mean'] = (cc_sum / np.maximum(hl, 1)[:, None]).ravel()
    F['cc_max'] = cc_max.ravel(); F['cc_rec'] = cc_rec.ravel(); F['cc_last'] = cc_last.ravel()
    # SVD scores: current history profile and ledger profile
    Hx = sp.csr_matrix((idf[bh[hv]], (np.repeat(np.arange(ne), 20)[hv.ravel()], bh[hv])), shape=(ne, NI))
    Uh = Hx @ V
    nh = np.linalg.norm(Uh, axis=1, keepdims=True); nh[nh == 0] = 1
    Vc = V[cand]                                                            # ne x 10 x 32
    nv = np.linalg.norm(Vc, axis=2); nv[nv == 0] = 1
    F['mf_h'] = (np.einsum('ed,ecd->ec', Uh / nh, Vc)).ravel()
    F['mf_hcos'] = (np.einsum('ed,ecd->ec', Uh / nh, Vc) / nv).ravel()
    ul = Ul[eu]
    F['mf_l'] = np.einsum('ed,ecd->ec', ul, Vc).ravel()
    F['mf_l'][np.repeat(l_u_n[eu] == 0, 10)] = np.nan
    del Vc
    ue = Ue[eu]; ne_ = np.linalg.norm(ue, axis=1, keepdims=True); ne_[ne_ == 0] = 1
    Vec = Ve[cand]; nve = np.linalg.norm(Vec, axis=2); nve[nve == 0] = 1
    ex = np.einsum('ed,ecd->ec', ue / ne_, Vec) / nve
    ex[u_imp[eu] == 0] = np.nan
    F['expo_cos'] = ex.ravel()
    del Vec
    # within-event relative versions and event-level stats
    for f in ['i_ctr', 'i_ctr1', 'cc_sum', 'cc_rec', 'mf_h', 'mf_l', 'expo_cos', 'ui_expo']:
        x = np.nan_to_num(F[f], nan=-1.0) if f in ('mf_l', 'expo_cos') else F[f]
        rk, dmax, dmean = within_event(x)
        F[f + '_rk'] = rk; F[f + '_dmax'] = dmax
        if f in ('i_ctr', 'cc_sum', 'mf_h', 'mf_l', 'expo_cos'):
            F[f + '_dmean'] = dmean
    F['ev_ictr_mean'] = np.repeat(F['i_ctr'].reshape(-1, 10).mean(1), 10)
    F['ev_in_hist'] = np.repeat((F['c_hist_cnt'].reshape(-1, 10) > 0).sum(1), 10).astype(np.float64)
    F['ev_cc_max'] = np.repeat(F['cc_sum'].reshape(-1, 10).max(1), 10)
    F['tb'] = rtb.astype(np.float64)
    names = list(F)
    M = np.empty((len(B), len(names)), np.float32)
    for j, n in enumerate(names):
        M[:, j] = F[n]
    hbin = np.digitize(hl, [0, 4, 9, 19], right=True)
    np.save(R / f'nxb_a_feat_k{k}.npy', M)
    np.savez(R / f'nxb_a_meta_k{k}.npz', y=y, events=B[:, 6], first=np.repeat(first_ev, 10), hbin=np.repeat(hbin, 10),
             users=ru, ts=rts.astype(np.int64))
    (R / 'nxb_a_names.json').write_text(json.dumps(names))
    log(f'k{k} done rows {len(B)} features {len(names)} {time.time()-t0:.1f}s')
    return names


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--blocks', default='6')
    a = ap.parse_args()
    for k in [int(x) for x in a.blocks.split(',')]:
        build(k, log=lambda s: print(s, flush=True))
