"""nxb: choice-level stacking head (design frozen in nxb_journal.md before any score).

Conditional logit over 10 candidates + no-click. Candidate utility = beta . x_j (x = standardised logit-rank of each expert);
no-click utility = history-bin intercepts + event max / mean of each expert. Eval-mix 5-bin event weights, L2 lambda grid
(6 values) chosen on the day-6 SELECTION subset; fit on day 5 -> day 6; day 5 by 2-fold user cross-fit (seed 9901);
evaluation: refit on day 6 with the V39-parent nxa_h1 and apply to the evaluation rows (only with --eval).
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent / 'artifacts'
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
EXPERTS = [('bag1', 'nxb_b_bag1', 'raw'), ('c1', 'nxb_c_c1best', 'raw'), ('c2uv', 'nxb_c_c2best', 'uv')]
h = np.load(R / 'train_hist.npy', mmap_mode='r'); eh = np.load(R / 'evaluation_hist.npy')
eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True); ES = np.bincount(eg, minlength=5) / len(eg)


def lrank(x):
    r = rankdata(x, method='average') / (len(x) + 1)
    return np.log(r / (1 - r))


def load(d, parent='pred', experts=EXPERTS):
    ref = np.load(R / f'nxa_h1_day{d}.npz')
    cols = [lrank(ref[parent])]
    for _, pre, key in experts:
        z = np.load(R / f'{pre}_day{d}.npz'); assert np.array_equal(z['events'], ref['events']); cols.append(lrank(z[key]))
    ev = ref['events'][::10]
    hb = np.digitize((np.asarray(h[ev]) >= 0).sum(1), [0, 4, 9, 19], right=True)
    w = (ES / (np.bincount(hb, minlength=5) / len(hb)))[hb]
    y = ref['y'].reshape(-1, 10); tgt = np.where(y.sum(1) > 0, y.argmax(1), 10)
    users = np.load(R / 'train.npy', mmap_mode='r')[ev * 10, 0]
    return dict(X=np.stack(cols, -1).reshape(-1, 10, len(cols)), hb=hb, w=w, tgt=tgt, y=ref['y'], users=np.asarray(users), ref=ref)


def load_eval(experts=EXPERTS):
    e = np.load(R / 'evaluation.npy')
    cols = [lrank(np.load(R / 'nxa_h1_eval_pred_v39.npy'))]
    for name, pre, key in experts:
        cols.append(lrank(np.load(R / f'{pre}_eval.npy') if key == 'raw' else np.load(R / f'{pre}_eval_extra.npz')[key]))
    hb = np.digitize((eh[e[::10, 6]] >= 0).sum(1), [0, 4, 9, 19], right=True)
    return dict(X=np.stack(cols, -1).reshape(-1, 10, len(cols)), hb=hb)


def design(D, mu, sd):
    X = (D['X'] - mu) / sd
    ev = np.concatenate([X.max(1), X.mean(1)], 1)
    return torch.from_numpy(X), torch.from_numpy(np.eye(5)[D['hb']]), torch.from_numpy(ev)


def fit(D, lam, idx=None):
    sub = D if idx is None else {k: (v[idx] if k in ('X', 'hb', 'w', 'tgt') else v) for k, v in D.items()}
    mu = sub['X'].reshape(-1, sub['X'].shape[-1]).mean(0); sd = sub['X'].reshape(-1, sub['X'].shape[-1]).std(0)
    X, B, Ev = design(sub, mu, sd)
    K = X.shape[-1]
    th = torch.zeros(K + 5 + Ev.shape[1], dtype=torch.float64, requires_grad=True)
    w = torch.from_numpy(sub['w']); t = torch.from_numpy(sub['tgt'])
    opt = torch.optim.LBFGS([th], max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        s = X @ th[:K]; s0 = B @ th[K:K + 5] + Ev @ th[K + 5:]
        l = torch.cat([s, s0.unsqueeze(1)], 1)
        nll = (w * (torch.logsumexp(l, 1) - l.gather(1, t.unsqueeze(1)).squeeze(1))).sum()
        loss = nll + lam * ((th[:K] ** 2).sum() + (th[K + 5:] ** 2).sum())
        loss.backward(); return loss
    opt.step(closure)
    return th.detach(), mu, sd


def predict(D, model, idx=None):
    th, mu, sd = model
    sub = D if idx is None else {k: (v[idx] if k in ('X', 'hb') else v) for k, v in D.items()}
    X, B, Ev = design(sub, mu, sd); K = X.shape[-1]
    l = torch.cat([X @ th[:K], (B @ th[K:K + 5] + Ev @ th[K + 5:]).unsqueeze(1)], 1)
    return torch.softmax(l, 1)[:, :10].numpy().ravel()


def adjw(D):
    return np.repeat(D['w'], 10)


def run(experts, tag, out):
    D5, D6 = load(5, experts=experts), load(6, experts=experts)
    s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
    sel = lambda p: float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
    grid = {}; models = {}
    for lam in LAMBDAS:
        models[lam] = fit(D5, lam); grid[lam] = sel(predict(D6, models[lam]))
    best = max(grid, key=grid.get)
    p6 = predict(D6, models[best])
    fold = (np.random.default_rng(9901).random(D5['users'].max() + 1) < 0.5)[D5['users']]
    p5 = np.zeros(len(D5['y']))
    for f in (False, True):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f)
        m = fit(D5, best, tr)
        p5.reshape(-1, 10)[te] = predict(D5, m, te).reshape(-1, 10)
    full = lambda D, p: float(roc_auc_score(D['y'], p, sample_weight=adjw(D)))
    r = dict(tag=tag, experts=['nxa_h1'] + [e[0] for e in experts], grid_sel6={str(k): v for k, v in grid.items()}, best_lambda=best,
             coef=models[best][0].tolist(),
             head=dict(sel=grid[best], full6=full(D6, p6), d5=full(D5, p5)),
             nxa_h1=dict(sel=sel(D6['ref']['pred']), full6=full(D6, D6['ref']['pred']), d5=full(D5, D5['ref']['pred'])))
    r['gain'] = {k: r['head'][k] - r['nxa_h1'][k] for k in ('sel', 'full6', 'd5')}
    out[tag] = r
    return r, p5, p6, D5, D6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--eval', action='store_true'); a = ap.parse_args()
    out = {}
    r, p5, p6, D5, D6 = run(EXPERTS, 'stack', out)
    rc, *_ = run([], 'control_nxa_h1_only', out)
    # V39-parent variant of the frozen stack (same lambda), for the export recipe
    D5v, D6v = load(5, 'pred_v39'), load(6, 'pred_v39')
    m = fit(D5v, r['best_lambda']); p6v = predict(D6v, m)
    s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
    out['stack_v39_parent'] = dict(sel=float(roc_auc_score(s6['y'], p6v[fi], sample_weight=s6['weights'])),
                                   nxa_h1_v39_sel=float(roc_auc_score(s6['y'], D6v['ref']['pred_v39'][fi], sample_weight=s6['weights'])))
    (R / 'nxb_c_head_grid.json').write_text(json.dumps(out, indent=1))
    for d, p, D in ((5, p5, D5), (6, p6, D6)):
        np.savez(R / f'nxb_c_head_day{d}.npz', y=D['y'], events=D['ref']['events'], pred=p, raw=p)
    print(json.dumps(out, indent=1))
    if a.eval:
        me = fit(D6v, r['best_lambda'])
        pe = predict(load_eval(), me)
        np.save(R / 'nxb_c_head_eval.npy', pe)
        print('eval rows', len(pe), 'range', float(pe.min()), float(pe.max()))


if __name__ == '__main__':
    main()
