"""nx idea A, screen 3: warm-started choice model whose history encoder is a causal transformer over the click ledger.

Matched design: the SAME code with --blocks 0 is the bag-only control (the warm choice model's form: z = U[u] + sum H[h]/sqrt(n),
logit = -1.8 + bu + bi + z.V, 11-way softmax with no-click), with --blocks 2 the treatment adds a zero-initialised transformer path
z += W.LN(transformer(H[h] + gap + position, QUERY token)). Both start from the choice model's own initialisation
(preference_stats at cutoff(day-2): co-click SVD item factors, profile user factors, smoothed CTR biases; no label after that cutoff),
Adagrad on the tables (choice-model style), Adam on the transformer, and a next-click sampled-softmax auxiliary on the prefix states.
ONE code path for --day 5 / 6 / 7 (cutoffs 1737763200 / 1737849600 / 1737936000); same ledger/causality rules as nx_seq_core.py.
"""
import argparse, json, math, pickle, time, gc
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from config import ARTIFACT_DIR as R
from nx_seq_core import (build_ledger, windows, merged_windows, tensors, weighted_auc, rank, cutoff_of, Block,
                         NI, NU, PAD, QUERY, GAP_BUCKETS, SHIFT, DAY)


def choice_init(day, d):
    """Exactly imp2_warm_choice.init32 at cutoff(day-2), any dimension d <= 127."""
    with (R / f'preference_stats_{cutoff_of(day - 2)}.pkl').open('rb') as f: s = pickle.load(f)
    V = normalize(s[3][:, :d]).astype('float32'); U = .25 * normalize(s[4][:, :d]).astype('float32'); H = .25 * V.copy()
    c = s[0][1]; g = c['global']; ur = (c['user_click'] + 50 * g) / (c['user_impr'] + 50); ir = (c['item_click'] + 50 * g) / (c['item_impr'] + 50)
    t = np.clip(ur * 10, .02, .98); bu = np.clip(np.log(t / (1 - t) / 10) + 1.8, -2, 2).astype('float32'); bi = np.clip(np.log(ir / g), -2, 2).astype('float32')
    del s; gc.collect(); return V, H, U, bu, bi


class WarmSeq(nn.Module):
    def __init__(s, init, d, L, heads, blocks, dropout):
        super().__init__(); V, H, U, bu, bi = init
        s.V = nn.Embedding(NI + 2, d); s.H = nn.Embedding(NI + 2, d); s.U = nn.Embedding(NU, d); s.bu = nn.Embedding(NU, 1); s.bi = nn.Embedding(NI + 2, 1)
        with torch.no_grad():
            for e in (s.V, s.H, s.bi): e.weight.zero_()
            s.V.weight[:NI] = torch.from_numpy(V); s.H.weight[:NI] = torch.from_numpy(H); s.U.weight[:] = torch.from_numpy(U); s.bu.weight[:, 0] = torch.from_numpy(bu); s.bi.weight[:NI, 0] = torch.from_numpy(bi)
            s.H.weight[QUERY] = torch.randn(d) * .02
        s.nblocks = blocks
        if blocks > 0:
            s.gap = nn.Embedding(GAP_BUCKETS, d); s.pos = nn.Embedding(L + 1, d); nn.init.normal_(s.gap.weight, std=.02); nn.init.normal_(s.pos.weight, std=.02)
            s.inp = nn.Linear(d, d); s.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(blocks)]); s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, d)
            nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias); s.drop = nn.Dropout(dropout)

    def tables(s): return [s.V.weight, s.H.weight, s.U.weight, s.bu.weight, s.bi.weight]

    def encode(s, items, gaps, ok, users):
        """State at every position t: U[u] + (sum_{j<=t} H[h_j]) / sqrt(n_t) [+ W.LN(transformer_t)]. Last position = QUERY (bag over all)."""
        real = ok.clone(); real[:, -1] = False; he = s.H(items) * real[..., None].float()
        cnt = real.float().cumsum(1); cnt[:, -1] = cnt[:, -2] if items.shape[1] > 1 else 0.
        bag = he.cumsum(1) / cnt.clamp_min(1.).sqrt()[..., None]; z = s.U(users)[:, None, :] + bag
        if s.nblocks > 0:
            B, L1 = items.shape; x = s.drop(s.inp(s.H(items)) + s.gap(gaps) + s.pos.weight[:L1][None])
            mask = (torch.tril(torch.ones(L1, L1, dtype=torch.bool))[None, None] & ok[:, None, None, :]) | torch.eye(L1, dtype=torch.bool)[None, None]
            for b in s.blocks: x = b(x, mask)
            z = z + s.out(s.ln(x))
        return z

    def event_logits(s, zq, cands, users):
        lg = (-1.8 + s.bu(users) + s.bi(cands)[..., 0] + torch.einsum('bd,bcd->bc', zq, s.V(cands))).clamp(-30, 30)
        return torch.cat([lg, torch.zeros(len(lg), 1)], 1)

    def next_loss(s, z, items, ok, neg, logq):
        L = items.shape[1] - 1; valid = ok[:, :L - 1] & ok[:, 1:L]; src = z[:, :L - 1][valid]; tgt = items[:, 1:L][valid]
        if len(tgt) == 0: return z.sum() * 0., 0
        pos = (src * s.V(tgt)).sum(1) + s.bi(tgt)[:, 0] - logq[tgt]; negl = src @ s.V(neg).T + s.bi(neg)[:, 0] - logq[neg]
        negl = negl.masked_fill(tgt[:, None] == neg[None, :], -1e9)
        return F.cross_entropy(torch.cat([pos[:, None], negl], 1), torch.zeros(len(tgt), dtype=torch.long), reduction='sum'), len(tgt)


def run(a):
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); day = a.day; cut = cutoff_of(day); tic = time.monotonic(); log = []
    ledger = build_ledger(cut); rows = ledger.pop('rows'); n = ledger['n_events']
    ev_u = rows[::10, 0].astype(np.int64); ev_t = rows[::10, 2].astype(np.int64); ev_items = rows[:, 1].reshape(-1, 10).astype(np.int64); y = rows[:, 5].reshape(-1, 10)
    target = np.where(y.sum(1) > 0, y.argmax(1), 10).astype(np.int64); del rows, y; gc.collect()
    ev_p = np.searchsorted(ledger['ukey'], ev_u * SHIFT + ev_t, side='left'); ev_start = ledger['start'][ev_u]
    for e in rng.integers(0, n, 2000): seg = slice(ev_start[e], ev_p[e]); assert (ledger['user'][seg] == ev_u[e]).all() and (ledger['time'][seg] < ev_t[e]).all()
    pop = np.bincount(ledger['item'], minlength=NI) + 1.; q = pop ** .75; q /= q.sum(); logq = torch.zeros(NI + 2); logq[:NI] = torch.from_numpy(np.log(q))
    if day == 7:
        srows = np.asarray(np.load(R / 'evaluation.npy', mmap_mode='r')); s_h = np.asarray(np.load(R / 'evaluation_hist.npy', mmap_mode='r')); s_ht = np.asarray(np.load(R / 'evaluation_times.npy', mmap_mode='r')); z = None
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); t = np.load(R / 'train.npy', mmap_mode='r'); h = np.load(R / 'train_hist.npy', mmap_mode='r'); ht = np.load(R / 'train_times.npy', mmap_mode='r')
        ei = z['events'].reshape(-1, 10)[:, 0]; srows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()]); assert np.array_equal(srows[:, 5], z['y']) and np.array_equal(srows[:, 6], z['events'])
        s_h = np.asarray(h[ei]); s_ht = np.asarray(ht[ei]); v53 = z['pred']; base_adj = weighted_auc(z['events'], z['y'], v53)
    s_u = srows[::10, 0].astype(np.int64); s_t = srows[::10, 2].astype(np.int64); s_c = srows[:, 1].reshape(-1, 10).astype(np.int64); assert s_t.min() >= cut and np.all((s_ht < s_t[:, None]) | (s_h < 0))
    model = WarmSeq(choice_init(day, a.dim), a.dim, a.L, a.heads, a.blocks, a.dropout); tab = set(id(p) for p in model.tables())
    opt_t = torch.optim.Adagrad(model.tables(), lr=a.lr_table, initial_accumulator_value=.1); rest = [p for p in model.parameters() if id(p) not in tab]; opt_n = torch.optim.Adam(rest, lr=a.lr) if rest else None
    print(json.dumps(dict(stage='model', day=day, cutoff=int(cut), init_cutoff=int(cutoff_of(day - 2)), transformer_parameters=sum(p.numel() for p in rest), args=vars(a))), flush=True)

    def score(raw_only=False):
        model.eval(); out = np.empty((len(s_u), 10), np.float32)
        with torch.no_grad():
            for off in range(0, len(s_u), 2048):
                sl = slice(off, off + 2048); it, st, ok = merged_windows(ledger, s_u[sl], s_t[sl], s_h[sl], s_ht[sl], a.L); A, G, M = tensors(it, st, ok, s_t[sl])
                us = torch.from_numpy(s_u[sl]); zz = model.encode(A, G, M, us); out[sl] = F.softmax(model.event_logits(zz[:, -1], torch.from_numpy(s_c[sl]), us), 1)[:, :10].numpy()
        return out.ravel().astype(np.float64)

    best = None
    for ep in range(1, a.epochs + 1):
        model.train(); order = rng.permutation(n); te = tn = treg = 0.; cnt_n = 0
        for k, off in enumerate(range(0, n, a.batch)):
            b = order[off:off + a.batch]; it, st, ok = windows(ledger, ev_p[b], ev_start[b], a.L); A, G, M = tensors(it, st, ok, ev_t[b]); us = torch.from_numpy(ev_u[b]); cs = torch.from_numpy(ev_items[b])
            zz = model.encode(A, G, M, us); le = F.cross_entropy(model.event_logits(zz[:, -1], cs, us), torch.from_numpy(target[b]), reduction='sum')
            ln, m = model.next_loss(zz, A, M, torch.from_numpy(rng.choice(NI, a.negatives, p=q)), logq) if a.aux > 0 else (torch.zeros(()), 0)
            real = M.clone(); real[:, -1] = False
            reg = .5 * a.reg * ((model.V(cs) ** 2).sum() + (model.bi(cs) ** 2).sum() + (model.U(us) ** 2).sum() + (model.bu(us) ** 2).sum() + ((model.H(A) ** 2).sum(-1) * real.float()).sum())
            loss = le + a.aux * ln + reg; opt_t.zero_grad()
            if opt_n: opt_n.zero_grad()
            loss.backward(); opt_t.step()
            if opt_n: opt_n.step()
            te += float(le.detach()); tn += float(ln.detach()); cnt_n += m; treg += float(reg.detach())
            if (k + 1) % 300 == 0: print(json.dumps(dict(stage='epoch', epoch=ep, batches=k + 1, event=te / min(off + a.batch, n), next=tn / max(cnt_n, 1), elapsed_s=round(time.monotonic() - tic, 1))), flush=True)
        raw = score(); rec = dict(epoch=ep, event_loss=te / n, next_loss=tn / max(cnt_n, 1), elapsed_s=round(time.monotonic() - tic, 1))
        if z is not None:
            rec.update(raw_auc=float(roc_auc_score(z['y'], raw)), raw_adjusted=weighted_auc(z['events'], z['y'], raw), v53_adjusted=base_adj, blends={})
            for w in (.05, .1, .15, .2, .3): p = (1 - w) * rank(v53) + w * rank(raw); rec['blends'][str(w)] = weighted_auc(z['events'], z['y'], p) - base_adj
            rec['best_gain'] = max(rec['blends'].values())
        print(json.dumps(rec), flush=True); log.append(rec)
        if z is None or not a.select or best is None or rec['best_gain'] > best[0]['best_gain']: best = (rec, raw)
    rec, raw = best; tag = a.tag
    if z is None: np.save(R / f'nx_seqw_{tag}_eval.npy', raw)
    else: np.savez_compressed(R / f'nx_seqw_{tag}_day{day}.npz', y=z['y'], events=z['events'], raw=raw, pred=raw, baseline=v53)
    torch.save(model.state_dict(), R / f'nx_seqw_{tag}_day{day}.pt')
    (R / f'nx_seqw_{tag}_day{day}_log.json').write_text(json.dumps(dict(args=vars(a), cutoff=int(cut), selected_epoch=rec['epoch'], log=log, elapsed_s=time.monotonic() - tic), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, required=True); ap.add_argument('--tag', required=True); ap.add_argument('--dim', type=int, default=32); ap.add_argument('--L', type=int, default=30)
    ap.add_argument('--heads', type=int, default=2); ap.add_argument('--blocks', type=int, default=2); ap.add_argument('--dropout', type=float, default=.1); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-table', type=float, default=.045); ap.add_argument('--reg', type=float, default=.002); ap.add_argument('--batch', type=int, default=256); ap.add_argument('--negatives', type=int, default=256)
    ap.add_argument('--aux', type=float, default=.2); ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--select', action='store_true'); ap.add_argument('--threads', type=int, default=12); ap.add_argument('--seed', type=int, default=7302)
    run(ap.parse_args())
