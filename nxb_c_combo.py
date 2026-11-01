"""nxb Track C: joint rank blend of nxa_h1 with several nxb experts; weights chosen on the day-6 SELECTION subset only.
pred = (1 - sum w) * rank(nxa_h1) + sum_k w_k * rank(expert_k)   (ranks within each scored set)
Usage: python nxb_c_combo.py out_name prefix1 [prefix2 ...]   (artifacts/<prefix>_day{5,6}.npz, key raw)
Saves the grid to artifacts/<out_name>_grid.json and the chosen blends to artifacts/<out_name>_day{5,6}.npz.
"""
import sys, json, itertools
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = Path('artifacts')
out, prefs = sys.argv[1], sys.argv[2:]
rank = lambda x: rankdata(x, method='average') / (len(x) + 1)
h = np.load(R / 'train_hist.npy', mmap_mode='r'); eh = np.load(R / 'evaluation_hist.npy')
es = np.bincount(np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True), minlength=5) / len(eh)
def wts(ev):
    vg = np.digitize((np.asarray(h[ev]) >= 0).sum(1), [0, 4, 9, 19], right=True)
    return (es / (np.bincount(vg, minlength=5) / len(vg)))[vg]
D = {}
for d in (5, 6):
    ref = np.load(R / f'nxa_h1_day{d}.npz')
    ex = []
    for p in prefs:
        z = np.load(R / f'{p}_day{d}.npz'); assert np.array_equal(z['events'], ref['events']); ex.append(rank(z['raw']))
    D[d] = dict(y=ref['y'], par=rank(ref['pred']), ex=ex, w=wts(ref['events']), v53=ref['baseline'])
s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
sel = lambda p: float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
full = lambda d, p: float(roc_auc_score(D[d]['y'], p, sample_weight=D[d]['w']))
blend = lambda d, ws: (1 - sum(ws)) * D[d]['par'] + sum(w * e for w, e in zip(ws, D[d]['ex']))
levels = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15]
grid = {}
for ws in itertools.product(levels, repeat=len(prefs)):
    if sum(ws) <= 0.3:
        grid[str(ws)] = sel(blend(6, ws))
best = max(grid, key=grid.get); ws = eval(best)
res = dict(experts=prefs, best_weights=ws, nxa_h1=dict(sel=sel(D[6]['par']), full6=full(6, D[6]['par']), d5=full(5, D[5]['par'])),
           v53=dict(sel=sel(D[6]['v53']), full6=full(6, D[6]['v53']), d5=full(5, D[5]['v53'])),
           blend=dict(sel=grid[best], full6=full(6, blend(6, ws)), d5=full(5, blend(5, ws))))
res['gain_vs_nxa_h1'] = {k: res['blend'][k] - res['nxa_h1'][k] for k in ('sel', 'full6', 'd5')}
(R / f'{out}_grid.json').write_text(json.dumps(dict(grid=grid, result=res), indent=1))
for d in (5, 6):
    np.savez(R / f'{out}_day{d}.npz', y=D[d]['y'], events=np.load(R / f'nxa_h1_day{d}.npz')['events'], pred=blend(d, ws), raw=blend(d, ws))
print(json.dumps(res, indent=1))
