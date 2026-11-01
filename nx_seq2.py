"""nx idea A, corrected (v2): sequence choice model with candidate-aware scoring.

Defects of nx_seq_core.SeqModel fixed here:
  1. no repeat signal: the score h.v_c cannot express "candidate c is one of the history items" (30% vs 5.7% CTR);
     -> explicit per-candidate repeat count, recency of last occurrence and position of last occurrence, as learned embeddings.
  2. no candidate-to-history interaction: history pooled before the candidate is known;
     -> DIN-style target attention of each candidate over the encoded history states.
  3. undertrained: one epoch with half the compute on the auxiliary loss -> several epochs on the 11-way event loss only.
Data path (ledger, windows, merged scoring windows, time-safety asserts) is reused unchanged from nx_seq_core.
--day 5 / 6 / 7 as in nx_seq_core; day 6 reports sanity floors (repeat flag, click-SVD 0.64) and V53 blends per epoch.
"""
import argparse, json, time, gc
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import (build_ledger, windows, merged_windows, tensors, gap_bucket, weighted_auc, rank, cutoff_of,
                         Block, NI, NU, PAD, QUERY, SHIFT, DAY)

REP_MAX = 5; NOREP_GAP = 26


def cand_features(items, stamps, ok, T, cands):
    """Per candidate: repeat count (0..5), gap bucket of its last occurrence (26 = never), slot of last occurrence from the end (L = never)."""
    B, L = items.shape; match = (items[:, None, :] == cands[:, :, None]) & ok[:, None, :]
    cnt = np.minimum(match.sum(2), REP_MAX)
    last_t = np.where(match, stamps[:, None, :], -1).max(2)
    gb = np.where(cnt > 0, gap_bucket(T[:, None] - last_t), NOREP_GAP)
    pos = np.where(match, np.arange(L)[None, None, :], -1).max(2); back = np.where(cnt > 0, L - 1 - pos, L)
    return torch.from_numpy(cnt), torch.from_numpy(gb.astype(np.int64)), torch.from_numpy(back.astype(np.int64)), torch.from_numpy(match)


class Seq2(nn.Module):
    def __init__(s, d, L, heads, blocks, dropout, use_user=False):
        super().__init__(); s.d = d; s.use_user = use_user
        s.item = nn.Embedding(NI + 2, d, padding_idx=PAD); s.bias = nn.Parameter(torch.zeros(NI + 2))
        s.gap = nn.Embedding(28, d); s.pos = nn.Embedding(L + 1, d); s.user = nn.Embedding(NU, d)
        s.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(blocks)]); s.ln = nn.LayerNorm(d); s.drop = nn.Dropout(dropout)
        s.rep = nn.Embedding(REP_MAX + 1, 8); s.rgap = nn.Embedding(27, 8); s.rback = nn.Embedding(L + 1, 8)
        s.att_q = nn.Linear(d, d, bias=False)
        s.mlp = nn.Sequential(nn.Linear(4 * d + 24 + 2, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        s.noclick = nn.Sequential(nn.Linear(d + 3, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.normal_(s.item.weight, std=.05); nn.init.normal_(s.gap.weight, std=.02); nn.init.normal_(s.pos.weight, std=.02); nn.init.zeros_(s.user.weight)
        with torch.no_grad(): s.item.weight[PAD].zero_()

    def table_parameters(s): return [s.item.weight, s.bias, s.user.weight]

    def forward(s, items, gaps, ok, users, cands, cnt, gb, back, match):
        B, L1 = items.shape
        x = s.item(items) + s.gap(gaps) + s.pos.weight[:L1][None]; x[:, -1] = x[:, -1] + (s.user(users) if s.use_user else 0.)  # free per-user vectors see the user's LATER training labels -> off by default
        x = s.drop(x)
        causal = torch.tril(torch.ones(L1, L1, dtype=torch.bool)); mask = (causal[None, None] & ok[:, None, None, :]) | torch.eye(L1, dtype=torch.bool)[None, None]
        for b in s.blocks: x = b(x, mask)
        x = s.ln(x); h = x[:, -1]; hist = x[:, :-1]; hok = ok[:, :-1]                          # (B,d), (B,L,d)
        vc = s.item(cands)                                                                     # (B,10,d)
        a = torch.einsum('bcd,bld->bcl', s.att_q(vc), hist) / s.d ** .5
        a = a.masked_fill(~hok[:, None, :], -1e9); w = torch.softmax(a, -1) * hok[:, None, :].any(-1, keepdim=True)
        pooled = torch.einsum('bcl,bld->bcd', w, hist)                                         # candidate-specific history summary
        hc = h[:, None, :].expand_as(vc)
        dots = torch.stack([(hc * vc).sum(-1), (pooled * vc).sum(-1)], -1)
        z = torch.cat([hc, vc, pooled, hc * vc, s.rep(cnt), s.rgap(gb), s.rback(back), dots], -1)
        logit = s.mlp(z).squeeze(-1) + dots[..., 0] + s.bias[cands]
        nhist = hok.sum(1, keepdim=True).float()
        ev = torch.cat([h, torch.log1p(nhist), (cnt > 0).sum(1, keepdim=True).float(), cnt.max(1, keepdim=True).values.float()], 1)
        return torch.cat([logit, s.noclick(ev)], 1)


def batch_inputs(it, st, ok, T, users, cands):
    cnt, gb, back, match = cand_features(it, st, ok, T, cands)
    a, g, m = tensors(it, st, ok, T)
    return (a, g, m, torch.from_numpy(users), torch.from_numpy(cands.astype(np.int64)), cnt, gb, back, match)


def run(args):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    day = args.day; cut = cutoff_of(day); tag = args.tag; tic = time.monotonic(); log = []
    ledger = build_ledger(cut); rows = ledger.pop('rows'); n = ledger['n_events']
    ev_u = rows[::10, 0].astype(np.int64); ev_t = rows[::10, 2].astype(np.int64); ev_items = rows[:, 1].reshape(-1, 10).astype(np.int64); y = rows[:, 5].reshape(-1, 10)
    assert np.all(y.sum(1) <= 1); target = np.where(y.sum(1) > 0, y.argmax(1), 10).astype(np.int64); del rows, y; gc.collect()
    ev_p = np.searchsorted(ledger['ukey'], ev_u * SHIFT + ev_t, side='left'); ev_start = ledger['start'][ev_u]
    for e in rng.integers(0, n, 2000):
        seg = ledger['time'][ev_start[e]:ev_p[e]]; assert (ledger['user'][ev_start[e]:ev_p[e]] == ev_u[e]).all() and (seg < ev_t[e]).all()
    tr = np.flatnonzero(ev_t >= cut - args.train_days * DAY)  # recent training events (the ledger still gives them full histories)
    if day == 7:
        ev = np.load(R / 'evaluation.npy'); s_u = ev[::10, 0].astype(np.int64); s_t = ev[::10, 2].astype(np.int64); s_c = ev[:, 1].reshape(-1, 10)
        s_h = np.asarray(np.load(R / 'evaluation_hist.npy')); s_ht = np.asarray(np.load(R / 'evaluation_times.npy')); z = None; assert s_t.min() >= cut
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); t = np.load(R / 'train.npy', mmap_mode='r'); h = np.load(R / 'train_hist.npy', mmap_mode='r'); ht = np.load(R / 'train_times.npy', mmap_mode='r')
        ei = z['events'].reshape(-1, 10)[:, 0]; srows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()]); assert np.array_equal(srows[:, 5], z['y'])
        s_u = srows[::10, 0].astype(np.int64); s_t = srows[::10, 2].astype(np.int64); s_c = srows[:, 1].reshape(-1, 10); s_h = np.asarray(h[ei]); s_ht = np.asarray(ht[ei])
        assert s_t.min() >= cut and s_t.max() < cut + DAY
        v53 = z['pred']; base = weighted_auc(z['events'], z['y'], v53)
    assert np.all((s_ht < s_t[:, None]) | (s_h < 0))
    # scoring windows are fixed: build once
    S = merged_windows(ledger, s_u, s_t, s_h, s_ht, args.L)
    if z is not None:  # sanity floor: candidate-in-history flag
        rep_flag = cand_features(*S, s_t, s_c.astype(np.int64))[0].numpy().ravel().astype(float)
        floor = dict(repeat_count_adjusted=weighted_auc(z['events'], z['y'], rep_flag), v53_adjusted=base); print(json.dumps(floor), flush=True); log.append(floor)
    model = Seq2(args.dim, args.L, args.heads, args.blocks, args.dropout, args.user_vec or args.init_svd)
    if args.init_svd:  # click-SVD factors from the ledger (clicks < cut only); frozen so the network learns to USE taste, not memorise ids
        import scipy.sparse as sp
        from nx_exposure import factorize
        Cm = sp.csr_matrix((np.ones(len(ledger['user']), np.float32), (ledger['user'], ledger['item'])), shape=(NU, NI)); Cm.sum_duplicates()
        Uc, Vc = factorize(Cm, args.dim, args.seed)
        with torch.no_grad():
            model.item.weight[:NI] = torch.from_numpy(Vc / Vc.std() * .1); model.item.weight[NI:] = 0.
            model.user.weight[:] = torch.from_numpy(Uc / max(Uc.std(), 1e-8) * .1)
        model.item.weight.requires_grad_(False); model.user.weight.requires_grad_(False)
        print(json.dumps(dict(stage='init_svd', clicks=int(Cm.nnz))), flush=True)
    tables = set(id(p) for p in model.table_parameters())
    opt = torch.optim.AdamW([dict(params=[p for p in model.table_parameters() if p.requires_grad], lr=args.lr_table, weight_decay=0.), dict(params=[p for p in model.parameters() if id(p) not in tables], lr=args.lr, weight_decay=1e-4)])
    print(json.dumps(dict(stage='model', parameters=sum(p.numel() for p in model.parameters()), train_events=int(len(tr)), args=vars(args))), flush=True)
    if args.smoke: tr = tr[rng.choice(len(tr), 4000, replace=False)]

    def score():
        model.eval(); out = np.empty((len(s_u), 10), np.float32)
        with torch.no_grad():
            for off in range(0, len(s_u), 4096):
                sl = slice(off, off + 4096); it, st, ok = S[0][sl], S[1][sl], S[2][sl]
                out[sl] = F.softmax(model(*batch_inputs(it, st, ok, s_t[sl], s_u[sl], s_c[sl])), 1)[:, :10].numpy()
        return out.ravel().astype(np.float64)

    best = None
    for ep in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(tr); tot = 0.; nb = 0
        for g in opt.param_groups: g['lr'] *= (0.5 if ep > 1 else 1.)
        for off in range(0, len(order), args.batch):
            b = order[off:off + args.batch]; it, st, ok = windows(ledger, ev_p[b], ev_start[b], args.L)
            loss = F.cross_entropy(model(*batch_inputs(it, st, ok, ev_t[b], ev_u[b], ev_items[b])), torch.from_numpy(target[b]))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
            if nb % 300 == 0: print(json.dumps(dict(epoch=ep, batches=nb, loss=tot / nb, elapsed_s=round(time.monotonic() - tic, 1))), flush=True)
        raw = score(); rec = dict(epoch=ep, loss=tot / nb, elapsed_s=round(time.monotonic() - tic, 1))
        if z is not None:
            rec.update(raw_plain=float(roc_auc_score(z['y'], raw)), raw_adjusted=weighted_auc(z['events'], z['y'], raw), blends={})
            for w in (.05, .1, .2, .3): rec['blends'][w] = weighted_auc(z['events'], z['y'], (1 - w) * rank(v53) + w * rank(raw)) - base
            rec['best_gain'] = max(rec['blends'].values())
        print(json.dumps(rec), flush=True); log.append(rec)
        if z is None or not args.select or best is None or rec['best_gain'] > best[0]['best_gain']: best = (rec, raw)
    if args.smoke: return
    rec, raw = best
    if z is None: np.save(R / f'nx_seq2_{tag}_eval.npy', raw)
    else: np.savez_compressed(R / f'nx_seq2_{tag}_day{day}.npz', y=z['y'], events=z['events'], pred=raw, baseline=z['pred'])
    torch.save(model.state_dict(), R / f'nx_seq2_{tag}_day{day}.pt')
    (R / f'nx_seq2_{tag}_day{day}_log.json').write_text(json.dumps(dict(args=vars(args), selected_epoch=rec['epoch'], log=log), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, required=True); ap.add_argument('--tag', default='screen')
    ap.add_argument('--dim', type=int, default=48); ap.add_argument('--L', type=int, default=30); ap.add_argument('--heads', type=int, default=2); ap.add_argument('--blocks', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=.1); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--lr-table', type=float, default=5e-3)
    ap.add_argument('--batch', type=int, default=1024); ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--train-days', type=int, default=6)
    ap.add_argument('--select', action='store_true'); ap.add_argument('--threads', type=int, default=12); ap.add_argument('--seed', type=int, default=7302); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--init-svd', action='store_true', help='initialise item/user tables from pre-cutoff click SVD and freeze them')
    ap.add_argument('--user-vec', action='store_true', help='free per-user vector (leaks later training labels into earlier events; off by default)')
    run(ap.parse_args())
