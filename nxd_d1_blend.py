"""nxd D1: conditional rank blend of nxa_h1 with bag1, C1 and C2 u.v, weights per cell
cell = candidate-item pre-cut click count bin (<3, 3-9, >=10; labelled train clicks with ts < cut) x user history bin (<10, >=10).
pred = (1 - sum_k w_k,cell) * rank(nxa_h1) + sum_k w_k,cell * rank(expert_k)   (ranks over the whole scored set)
Weight levels {0, 0.05, 0.15} per expert and cell; coordinate ascent over cells (2 passes, start at 0) on the day-6 SELECTION
subset only; grid / path saved. Reference: best global (unconditional) weights from the same 27-point grid.
"""
import json, itertools
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = Path('artifacts'); S, DAY = 1737331200, 86400
EXPERTS = [('bag1', 'nxb_b_bag1', 'raw'), ('c1', 'nxb_c_c1best', 'raw'), ('c2uv', 'nxb_c_c2best', 'uv')]
LEVELS = [0.0, 0.05, 0.15]
rank = lambda x: rankdata(x, method='average') / (len(x) + 1)
t = np.load(R / 'train.npy'); h = np.load(R / 'train_hist.npy', mmap_mode='r'); eh = np.load(R / 'evaluation_hist.npy')
es = np.bincount(np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True), minlength=5) / len(eh)


def item_clicks(cut):
    n = int(np.searchsorted(t[:, 2], cut)); P = t[:n]
    return np.bincount(P[P[:, 5] == 1, 1], minlength=50000)


def load(d):
    ref = np.load(R / f'nxa_h1_day{d}.npz'); ev = ref['events']
    items = t[ev * 10 + np.tile(np.arange(10), len(ev) // 10), 1]
    ic = item_clicks(S + d * DAY)[items]
    hl = (np.asarray(h[ev]) >= 0).sum(1)
    cell = np.digitize(ic, [3, 10]) * 2 + (hl >= 10)
    vg = np.digitize(hl, [0, 4, 9, 19], right=True)
    w = (es / (np.bincount(vg, minlength=5) / len(vg)))[vg]
    ex = []
    for _, pre, key in EXPERTS:
        z = np.load(R / f'{pre}_day{d}.npz'); assert np.array_equal(z['events'], ev); ex.append(rank(z[key]))
    return dict(y=ref['y'], par=rank(ref['pred']), ex=np.stack(ex, 1), cell=cell, w=w, ic=ic, hl=hl, raw_par=ref['pred'], v53=ref['baseline'])


def blend(D, W):            # W: 6 cells x 3 experts
    wc = W[D['cell']]
    return (1 - wc.sum(1)) * D['par'] + (wc * D['ex']).sum(1)


D6, D5 = load(6), load(5)
s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
sel = lambda p: float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
full = lambda D, p: float(roc_auc_score(D['y'], p, sample_weight=D['w']))
combos = list(itertools.product(LEVELS, repeat=3))
# unconditional reference
glob = {str(c): sel(blend(D6, np.tile(np.array(c), (6, 1)))) for c in combos}
gbest = max(glob, key=glob.get)
# conditional: coordinate ascent over cells
W = np.zeros((6, 3)); path = []
cur = sel(blend(D6, W))
for p_ in range(2):
    for c in range(6):
        best_c, best_v = W[c].copy(), cur
        for cb in combos:
            W2 = W.copy(); W2[c] = cb; v = sel(blend(D6, W2))
            if v > best_v + 1e-9:
                best_c, best_v = np.array(cb), v
        W[c] = best_c; cur = best_v; path.append(dict(pass_=p_, cell=c, weights=best_c.tolist(), sel=cur))
names = [f'item{a}_hist{b}' for a in ('<3', '3-9', '>=10') for b in ('<10', '>=10')]
Wg = np.tile(np.array(eval(gbest)), (6, 1))
res = dict(cells=names, cell_share_day6=np.bincount(D6['cell'], minlength=6).tolist(),
           weights={n: dict(zip([e[0] for e in EXPERTS], W[i].tolist())) for i, n in enumerate(names)},
           conditional=dict(sel=cur, full6=full(D6, blend(D6, W)), d5=full(D5, blend(D5, W))),
           unconditional=dict(weights=gbest, sel=glob[gbest], full6=full(D6, blend(D6, Wg)), d5=full(D5, blend(D5, Wg))),
           nxa_h1=dict(sel=sel(D6['raw_par']), full6=full(D6, D6['raw_par']), d5=full(D5, D5['raw_par'])))
for k in ('conditional', 'unconditional'):
    res[k]['gain'] = {m: res[k][m] - res['nxa_h1'][m] for m in ('sel', 'full6', 'd5')}
# per item-frequency bin and history bin plain AUC (day 6), conditional blend vs nxa_h1
pc = blend(D6, W); ib = np.digitize(D6['ic'], [1, 3, 10, 30])
res['day6_item_bins'] = {lab: dict(share=float((ib == b).mean()), nxa_h1=float(roc_auc_score(D6['y'][ib == b], D6['raw_par'][ib == b])),
                                   blend=float(roc_auc_score(D6['y'][ib == b], pc[ib == b])))
                         for b, lab in enumerate(['0', '1-2', '3-9', '10-29', '30+'])}
hb = np.digitize(D6['hl'], [0, 4, 9, 19], right=True)
res['day6_hist_bins'] = {int(b): dict(nxa_h1=float(roc_auc_score(D6['y'][hb == b], D6['raw_par'][hb == b])),
                                      blend=float(roc_auc_score(D6['y'][hb == b], pc[hb == b]))) for b in range(5)}
(R / 'nxd_d1_grid.json').write_text(json.dumps(dict(result=res, path=path, unconditional_grid=glob), indent=1))
for d, D in ((5, D5), (6, D6)):
    np.savez(R / f'nxd_d1_day{d}.npz', y=D['y'], events=np.load(R / f'nxa_h1_day{d}.npz')['events'], pred=blend(D, W), raw=blend(D, W))
print(json.dumps(res, indent=1))
