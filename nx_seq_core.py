"""nx idea A: SASRec-style causal sequence choice model over the full per-user click ledger.

ONE code path for the three prediction sets, selected by --day:
  day 5  cutoff 1737763200  scores the day-5 cohort (artifacts/v53_causal_propensity_day5.npz events)
  day 6  cutoff 1737849600  scores the full day-6 cohort (v53_causal_propensity_day6.npz events)
  day 7  cutoff 1737936000  scores the 529,390 evaluation rows (artifacts/evaluation.npy order)
Ledger(cut)      = unique (user, item, time) clicks taken from the supplied histories AND the labels of events with time < cut.
Training events  = events with time < cut; model input = the user's ledger clicks with time < event time (last L), labels = slot clicked / none.
Scored events    = cohort or evaluation events; input = ledger clicks (all < cut <= event time) UNION the event's own supplied history
                   (all stamps < event time), last L. Nothing with a timestamp >= the event time is ever used; no label >= cut is used.
Model: item table (tied in/out) + gap-to-next-element embedding (log2 minutes) + position embedding, causal pre-LN transformer blocks,
       a QUERY token at the end whose state h scores candidates h.v_c + b_c and a no-click option c0 + w.h (11-way softmax).
       Regulariser / pretraining: sampled-softmax next-click loss at every ledger position (shared popularity negatives, logQ corrected).
Outputs: artifacts/nx_seq_<tag>_day{5,6}.npz (y, events, pred, raw, baseline) or nx_seq_<tag>_eval.npy; artifacts/nx_seq_<tag>_day{d}_log.json.
"""
import argparse, json, math, time, gc
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from config import ARTIFACT_DIR as R

START = 1737331200; DAY = 86400; NI = 50000; NU = 54456; PAD = NI; QUERY = NI + 1; GAP_BUCKETS = 28; QUERY_GAP = 27
SHIFT = np.int64(1 << 31)


def cutoff_of(day): return START + day * DAY
def rank(x): x = np.asarray(x, np.float64); return rankdata(x, method='average') / (len(x) + 1)
def gap_bucket(seconds): return np.minimum(np.floor(np.log2(np.maximum(seconds, 0) / 60. + 1.)), 25).astype(np.int64)


def weighted_auc(events, y, p):
    """Project adjusted AUC: 5 history-length bins reweighted to the evaluation mix (same code as v39_regularized_als32.weighted_auc)."""
    h = np.load(R / 'train_hist.npy', mmap_mode='r'); eh = np.load(R / 'evaluation_hist.npy', mmap_mode='r')
    vg = np.digitize((h[events] >= 0).sum(1), [0, 4, 9, 19], right=True); eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    w = ((np.bincount(eg, minlength=5) / len(eg)) / (np.bincount(vg, minlength=5) / len(vg)))[vg]
    return float(roc_auc_score(y, p, sample_weight=w))


def build_ledger(cut):
    t = np.load(R / 'train.npy', mmap_mode='r'); h = np.load(R / 'train_hist.npy', mmap_mode='r'); ht = np.load(R / 'train_times.npy', mmap_mode='r')
    heads = np.asarray(t[::10]); assert np.all(np.diff(heads[:, 2]) >= 0), 'events must be time ordered'
    n = int(np.searchsorted(heads[:, 2], cut, side='left')); assert n > 0 and heads[n - 1, 2] < cut and (n == len(heads) or heads[n, 2] >= cut)
    hh = np.asarray(h[:n]); tt = np.asarray(ht[:n]); valid = hh >= 0; assert np.all((tt < heads[:n, 2, None]) | ~valid), 'history must precede its event'
    hu = np.repeat(heads[:n, 0], 20)[valid.ravel()]; hi = hh.ravel()[valid.ravel()]; hs = tt.ravel()[valid.ravel()]
    rows = np.asarray(t[:n * 10]); ck = rows[:, 5] == 1; assert int(rows[:, 2].max()) < cut
    u = np.concatenate([hu, rows[ck, 0]]).astype(np.int64); i = np.concatenate([hi, rows[ck, 1]]).astype(np.int64); s = np.concatenate([hs, rows[ck, 2]]).astype(np.int64)
    del hh, tt, valid, hu, hi, hs
    key = (u * SHIFT + s) * NI + i; _, ix = np.unique(key, return_index=True); u, i, s = u[ix], i[ix], s[ix]
    assert s.max() < cut and s.min() > 0 and i.min() >= 0 and i.max() < NI
    start = np.searchsorted(u, np.arange(NU)); end = np.searchsorted(u, np.arange(NU), side='right')
    print(json.dumps(dict(stage='ledger', cutoff=int(cut), events=n, entries=int(len(u)), users=int((end > start).sum()))), flush=True)
    return dict(user=u, item=i.astype(np.int32), time=s, start=start, end=end, ukey=u * SHIFT + s, n_events=n, rows=rows)


def windows(ledger, p, start_u, L):
    """Right-aligned windows [max(start,p-L), p) of the ledger; left padded."""
    idx = p[:, None] - L + np.arange(L)[None, :]; ok = idx >= start_u[:, None]; safe = np.where(ok, idx, 0)
    return np.where(ok, ledger['item'][safe], PAD).astype(np.int64), np.where(ok, ledger['time'][safe], 0).astype(np.int64), ok


def merged_windows(ledger, users, times, hist_items, hist_times, L):
    """Ledger clicks (< cut) union the event's own supplied history, unique by (time, item), sorted, last L. Left padded."""
    n = len(users); items = np.full((n, L), PAD, np.int64); stamps = np.zeros((n, L), np.int64); ok = np.zeros((n, L), bool)
    li, lt, st, en = ledger['item'], ledger['time'], ledger['start'], ledger['end']
    for r in range(n):
        u = users[r]; v = hist_items[r] >= 0
        it = np.concatenate([li[st[u]:en[u]], hist_items[r][v]]).astype(np.int64); ts = np.concatenate([lt[st[u]:en[u]], hist_times[r][v]]).astype(np.int64)
        if len(it) == 0: continue
        assert ts.max() < times[r], 'scoring input must precede the event'
        key = np.unique(ts * NI + it); key = key[-L:]; m = len(key)
        items[r, L - m:] = key % NI; stamps[r, L - m:] = key // NI; ok[r, L - m:] = True
    return items, stamps, ok


def tensors(items, stamps, ok, T):
    """Append the QUERY token; gap of element j = time to element j+1 (to T for the last one)."""
    B, L = items.shape; nxt = np.concatenate([stamps[:, 1:], T[:, None]], 1); gaps = np.where(ok, gap_bucket(nxt - stamps), 0)
    it = np.concatenate([items, np.full((B, 1), QUERY)], 1); gp = np.concatenate([gaps, np.full((B, 1), QUERY_GAP)], 1); ok2 = np.concatenate([ok, np.ones((B, 1), bool)], 1)
    return torch.from_numpy(it), torch.from_numpy(gp), torch.from_numpy(ok2)


class Block(nn.Module):
    def __init__(s, d, heads, dropout):
        super().__init__(); s.h = heads; s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d); s.ln2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)); s.drop = nn.Dropout(dropout); s.p = dropout

    def forward(s, x, mask):
        B, L, d = x.shape; q, k, v = s.qkv(s.ln1(x)).chunk(3, -1)
        q, k, v = [z.view(B, L, s.h, d // s.h).transpose(1, 2) for z in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=s.p if s.training else 0.)
        x = x + s.drop(s.proj(a.transpose(1, 2).reshape(B, L, d))); return x + s.drop(s.ff(s.ln2(x)))


class SeqModel(nn.Module):
    def __init__(s, d, L, heads, blocks, dropout):
        super().__init__(); s.d = d; s.item = nn.Embedding(NI + 2, d, padding_idx=PAD); s.bias = nn.Parameter(torch.zeros(NI + 2)); s.gap = nn.Embedding(GAP_BUCKETS, d); s.pos = nn.Embedding(L + 1, d)
        s.user = nn.Embedding(NU, d)  # per-user vector added to the QUERY token (learns from exposed-but-unclicked items like the choice model's U)
        s.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(blocks)]); s.ln = nn.LayerNorm(d); s.drop = nn.Dropout(dropout); s.noclick = nn.Linear(d, 1)
        nn.init.normal_(s.item.weight, std=.1); nn.init.normal_(s.gap.weight, std=.02); nn.init.normal_(s.pos.weight, std=.02); nn.init.zeros_(s.user.weight)
        with torch.no_grad(): s.item.weight[PAD].zero_(); s.noclick.bias.fill_(1.8)

    def table_parameters(s): return [s.item.weight, s.bias, s.gap.weight, s.pos.weight, s.user.weight]

    def encode(s, items, gaps, ok, users):
        B, L1 = items.shape; x = s.item(items) + s.gap(gaps) + s.pos.weight[:L1][None]; x[:, -1] = x[:, -1] + s.user(users); x = s.drop(x)
        causal = torch.tril(torch.ones(L1, L1, dtype=torch.bool)); mask = (causal[None, None] & ok[:, None, None, :]) | torch.eye(L1, dtype=torch.bool)[None, None]
        for b in s.blocks: x = b(x, mask)
        return s.ln(x)

    def event_logits(s, hq, cands):
        return torch.cat([torch.einsum('bd,bcd->bc', hq, s.item(cands)) + s.bias[cands], s.noclick(hq)], 1)

    def next_loss(s, h, items, ok, neg, logq):
        L = items.shape[1] - 1; valid = ok[:, :L - 1] & ok[:, 1:L]; src = h[:, :L - 1][valid]; tgt = items[:, 1:L][valid]
        if len(tgt) == 0: return h.sum() * 0.
        pos = (src * s.item(tgt)).sum(1) + s.bias[tgt] - logq[tgt]; negl = src @ s.item(neg).T + s.bias[neg] - logq[neg]
        negl = negl.masked_fill(tgt[:, None] == neg[None, :], -1e9)
        return F.cross_entropy(torch.cat([pos[:, None], negl], 1), torch.zeros(len(tgt), dtype=torch.long))


def score_events(model, ledger, users, times, cands, hist_items, hist_times, L, batch=2048):
    model.eval(); out = np.empty((len(users), 10), np.float32)
    with torch.no_grad():
        for off in range(0, len(users), batch):
            sl = slice(off, off + batch); it, st, ok = merged_windows(ledger, users[sl], times[sl], hist_items[sl], hist_times[sl], L)
            a, g, m = tensors(it, st, ok, times[sl]); h = model.encode(a, g, m, torch.from_numpy(users[sl])); lg = model.event_logits(h[:, -1], torch.from_numpy(cands[sl].astype(np.int64)))
            out[sl] = F.softmax(lg, 1)[:, :10].numpy()
    return out


def run(args):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    day = args.day; cut = cutoff_of(day); tag = args.tag; tic = time.monotonic(); log = []
    ledger = build_ledger(cut); rows = ledger.pop('rows'); n = ledger['n_events']
    ev_u = rows[::10, 0].astype(np.int64); ev_t = rows[::10, 2].astype(np.int64); ev_items = rows[:, 1].reshape(-1, 10).astype(np.int64); y = rows[:, 5].reshape(-1, 10)
    assert np.all(y.sum(1) <= 1); target = np.where(y.sum(1) > 0, y.argmax(1), 10).astype(np.int64); del rows, y; gc.collect()
    ev_p = np.searchsorted(ledger['ukey'], ev_u * SHIFT + ev_t, side='left'); ev_start = ledger['start'][ev_u]
    chk = rng.integers(0, n, 2000)
    for e in chk:  # every ledger entry used for a training event precedes that event
        seg = ledger['time'][ev_start[e]:ev_p[e]]; assert (ledger['user'][ev_start[e]:ev_p[e]] == ev_u[e]).all() and (seg < ev_t[e]).all()
    pop = np.bincount(ledger['item'], minlength=NI) + 1.; q = pop ** .75; q /= q.sum(); logq = torch.zeros(NI + 2); logq[:NI] = torch.from_numpy(np.log(q))
    # scoring targets
    if day == 7:
        ev = np.load(R / 'evaluation.npy', mmap_mode='r'); eh = np.load(R / 'evaluation_hist.npy', mmap_mode='r'); et = np.load(R / 'evaluation_times.npy', mmap_mode='r')
        srows = np.asarray(ev); s_u = srows[::10, 0].astype(np.int64); s_t = srows[::10, 2].astype(np.int64); s_c = srows[:, 1].reshape(-1, 10); s_h = np.asarray(eh); s_ht = np.asarray(et); z = None
        assert s_t.min() >= cut
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); t = np.load(R / 'train.npy', mmap_mode='r'); h = np.load(R / 'train_hist.npy', mmap_mode='r'); ht = np.load(R / 'train_times.npy', mmap_mode='r')
        ei = z['events'].reshape(-1, 10); assert np.all(ei == ei[:, :1]); ei = ei[:, 0]; srows = np.asarray(t[(ei[:, None] * 10 + np.arange(10)).ravel()])
        assert np.array_equal(srows[:, 5], z['y']) and np.array_equal(srows[:, 6], z['events']); s_u = srows[::10, 0].astype(np.int64); s_t = srows[::10, 2].astype(np.int64); s_c = srows[:, 1].reshape(-1, 10)
        s_h = np.asarray(h[ei]); s_ht = np.asarray(ht[ei]); assert s_t.min() >= cut and s_t.max() < cut + DAY
        v53 = z['pred']; v39 = np.load(R / f'v39_regularized_als32_day{day}.npz')['pred']; base_adj = weighted_auc(z['events'], z['y'], v53)
    assert np.all((s_ht < s_t[:, None]) | (s_h < 0))
    model = SeqModel(args.dim, args.L, args.heads, args.blocks, args.dropout); tables = set(id(p) for p in model.table_parameters())
    opt = torch.optim.Adam([dict(params=model.table_parameters(), lr=args.lr_table), dict(params=[p for p in model.parameters() if id(p) not in tables], lr=args.lr)])
    print(json.dumps(dict(stage='model', parameters=sum(p.numel() for p in model.parameters()), day=day, cutoff=int(cut), args=vars(args))), flush=True)
    # pretraining windows: non-overlapping chunks of L ledger entries per user, taken from the end
    pw = []; st, en = ledger['start'], ledger['end']
    for u in np.flatnonzero(en - st >= 2):
        p = en[u]
        while p - st[u] >= 2: pw.append((p, st[u])); p -= args.L
    pw = np.array(pw, np.int64); print(json.dumps(dict(stage='pretrain_windows', windows=int(len(pw)))), flush=True)

    def pre_epoch(ep):
        model.train(); order = rng.permutation(len(pw)); tot = 0.; nb = 0
        for off in range(0, len(order), args.batch):
            b = pw[order[off:off + args.batch]]; it, stt, ok = windows(ledger, b[:, 0], b[:, 1], args.L); a, g, m = tensors(it, stt, ok, stt[:, -1])
            neg = torch.from_numpy(rng.choice(NI, args.negatives, p=q)); h = model.encode(a, g, m, torch.from_numpy(ledger['user'][b[:, 0] - 1])); loss = model.next_loss(h, a, m, neg, logq)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
        return tot / max(nb, 1)

    def event_epoch(ep):
        model.train(); order = rng.permutation(n); tot = te = tn = 0.; nb = 0
        for off in range(0, n, args.batch):
            b = order[off:off + args.batch]; it, stt, ok = windows(ledger, ev_p[b], ev_start[b], args.L); a, g, m = tensors(it, stt, ok, ev_t[b])
            h = model.encode(a, g, m, torch.from_numpy(ev_u[b])); le = F.cross_entropy(model.event_logits(h[:, -1], torch.from_numpy(ev_items[b])), torch.from_numpy(target[b]))
            ln = model.next_loss(h, a, m, torch.from_numpy(rng.choice(NI, args.negatives, p=q)), logq) if args.aux > 0 else torch.zeros(())
            loss = le + args.aux * ln; opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); te += float(le.detach()); tn += float(ln.detach()); nb += 1
            if nb % 200 == 0: print(json.dumps(dict(stage='event_epoch', epoch=ep, batches=nb, loss=tot / nb, event=te / nb, next=tn / nb, elapsed_s=round(time.monotonic() - tic, 1))), flush=True)
        return tot / nb, te / nb, tn / nb

    def evaluate(ep):
        raw = score_events(model, ledger, s_u, s_t, s_c, s_h, s_ht, args.L).ravel().astype(np.float64)
        if z is None: return dict(epoch=ep, raw=raw)
        rec = dict(epoch=ep, raw_auc=float(roc_auc_score(z['y'], raw)), raw_adjusted=weighted_auc(z['events'], z['y'], raw), v53_adjusted=base_adj, blends={})
        rb, rv53, rv39 = rank(raw), rank(v53), rank(v39)
        for w in (.05, .1, .15, .2, .3, .4):
            p = (1 - w) * rv53 + w * rb; rec['blends'][f'v53_{w}'] = dict(adjusted=weighted_auc(z['events'], z['y'], p), ordinary=float(roc_auc_score(z['y'], p)))
            p = (1 - w) * rv39 + w * rb; rec['blends'][f'v39_{w}'] = dict(adjusted=weighted_auc(z['events'], z['y'], p), ordinary=float(roc_auc_score(z['y'], p)))
        rec['best_v53_gain'] = max(v['adjusted'] for k, v in rec['blends'].items() if k.startswith('v53')) - base_adj; rec['raw'] = raw; return rec

    if args.smoke:  # tiny end-to-end check: a few training events / windows / scored events, nothing saved
        keep = rng.choice(n, 3000, replace=False); ev_u, ev_t, ev_items, target, ev_p, ev_start = ev_u[keep], ev_t[keep], ev_items[keep], target[keep], ev_p[keep], ev_start[keep]; n = len(keep)
        pw = pw[rng.choice(len(pw), 500, replace=False)]; s_u, s_t, s_c, s_h, s_ht = s_u[:300], s_t[:300], s_c[:300], s_h[:300], s_ht[:300]
        if z is not None: v53, v39 = v53[:3000], v39[:3000]; z = dict(y=z['y'][:3000], events=z['events'][:3000]); base_adj = weighted_auc(z['events'], z['y'], v53)
    history = []; best = None
    for ep in range(1, args.pre_epochs + 1):
        l = pre_epoch(ep); print(json.dumps(dict(stage='pretrain', epoch=ep, loss=l, elapsed_s=round(time.monotonic() - tic, 1))), flush=True); log.append(dict(stage='pretrain', epoch=ep, loss=l))
    for ep in range(1, args.epochs + 1):
        l, le, ln = event_epoch(ep); rec = evaluate(ep); raw = rec.pop('raw'); rec.update(loss=l, event_loss=le, next_loss=ln, elapsed_s=round(time.monotonic() - tic, 1)); history.append(rec); log.append(rec)
        print(json.dumps(rec), flush=True)
        # selection rule: evaluation rows and non-select runs keep the LAST epoch; --select (day 6 only) keeps the epoch with the best V53 blend gain
        if z is None or not args.select or best is None or rec['best_v53_gain'] > best[0]['best_v53_gain']: best = (rec, raw)
    rec, raw = best
    if args.smoke: print(json.dumps(dict(stage='smoke_done', elapsed_s=time.monotonic() - tic))); return
    if z is None:
        np.save(R / f'nx_seq_{tag}_eval.npy', raw); print(json.dumps(dict(stage='saved', file=f'nx_seq_{tag}_eval.npy', rows=int(len(raw)), epoch=rec['epoch'])), flush=True)
    else:
        np.savez_compressed(R / f'nx_seq_{tag}_day{day}.npz', y=z['y'], events=z['events'], raw=raw, pred=raw, baseline=v53)
        print(json.dumps(dict(stage='saved', file=f'nx_seq_{tag}_day{day}.npz', epoch=rec['epoch'], raw_adjusted=rec['raw_adjusted'], best_v53_gain=rec['best_v53_gain'])), flush=True)
    torch.save(model.state_dict(), R / f'nx_seq_{tag}_day{day}.pt')
    (R / f'nx_seq_{tag}_day{day}_log.json').write_text(json.dumps(dict(args=vars(args), cutoff=int(cut), selected_epoch=rec['epoch'], log=log, elapsed_s=time.monotonic() - tic), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, required=True); ap.add_argument('--tag', default='screen'); ap.add_argument('--dim', type=int, default=64); ap.add_argument('--L', type=int, default=50)
    ap.add_argument('--heads', type=int, default=2); ap.add_argument('--blocks', type=int, default=2); ap.add_argument('--dropout', type=float, default=.2); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--lr-table', type=float, default=1e-2)
    ap.add_argument('--batch', type=int, default=512); ap.add_argument('--negatives', type=int, default=256); ap.add_argument('--aux', type=float, default=.5); ap.add_argument('--pre-epochs', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--select', action='store_true', help='keep the best day-6 epoch (day 6 only); otherwise keep the last epoch'); ap.add_argument('--threads', type=int, default=10); ap.add_argument('--seed', type=int, default=7301); ap.add_argument('--smoke', action='store_true')
    run(ap.parse_args())
