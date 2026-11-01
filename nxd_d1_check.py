"""nxd D1 checks: frozen per-cell weights (artifacts/nxd_d1_grid.json) applied to the V39 parent (pred_v39) on days 5/6;
paired event bootstrap (400 resamples, seed 3) of the gain vs nxa_h1 on the day-6 selection subset, full day 6 and day 5."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
import importlib.util
spec = importlib.util.spec_from_file_location('d1', 'nxd_d1_blend.py')
R = 'artifacts/'
g = json.load(open(R + 'nxd_d1_grid.json'))['result']
W = np.array([[g['weights'][c][e] for e in ('bag1', 'c1', 'c2uv')] for c in g['cells']])
from scipy.stats import rankdata
rank = lambda x: rankdata(x, method='average') / (len(x) + 1)
t = np.load(R + 'train.npy'); h = np.load(R + 'train_hist.npy', mmap_mode='r'); eh = np.load(R + 'evaluation_hist.npy')
es = np.bincount(np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True), minlength=5) / len(eh)
S, DAY = 1737331200, 86400
def load(d, key):
    ref = np.load(R + f'nxa_h1_day{d}.npz'); ev = ref['events']
    items = t[ev * 10 + np.tile(np.arange(10), len(ev) // 10), 1]
    n = int(np.searchsorted(t[:, 2], S + d * DAY)); P = t[:n]; ic = np.bincount(P[P[:, 5] == 1, 1], minlength=50000)[items]
    hl = (np.asarray(h[ev]) >= 0).sum(1); cell = np.digitize(ic, [3, 10]) * 2 + (hl >= 10)
    vg = np.digitize(hl, [0, 4, 9, 19], right=True); w = (es / (np.bincount(vg, minlength=5) / len(vg)))[vg]
    ex = np.stack([rank(np.load(R + f'{p}_day{d}.npz')[k]) for p, k in (('nxb_b_bag1', 'raw'), ('nxb_c_c1best', 'raw'), ('nxb_c_c2best', 'uv'))], 1)
    par = rank(ref[key]); wc = W[cell]
    return ref['y'], ref[key], (1 - wc.sum(1)) * par + (wc * ex).sum(1), w
out = {}
s6 = np.load(R + 'selection_day6.npz'); fi = s6['full_indices']
for key in ('pred', 'pred_v39'):
    y6, b6, p6, w6 = load(6, key); y5, b5, p5, w5 = load(5, key)
    out[key] = dict(sel=(roc_auc_score(s6['y'], b6[fi], sample_weight=s6['weights']), roc_auc_score(s6['y'], p6[fi], sample_weight=s6['weights'])),
                    full6=(roc_auc_score(y6, b6, sample_weight=w6), roc_auc_score(y6, p6, sample_weight=w6)),
                    d5=(roc_auc_score(y5, b5, sample_weight=w5), roc_auc_score(y5, p5, sample_weight=w5)))
    out[key] = {k: dict(parent=v[0], blend=v[1], gain=v[1] - v[0]) for k, v in out[key].items()}
# paired bootstrap on the V53-parent version
def boot(y, b, p, w, B=400, seed=3):
    rng = np.random.default_rng(seed); ne = len(y) // 10; gains = []
    for _ in range(B):
        e = rng.integers(0, ne, ne); r = (e[:, None] * 10 + np.arange(10)).ravel()
        gains.append(roc_auc_score(y[r], p[r], sample_weight=w[r]) - roc_auc_score(y[r], b[r], sample_weight=w[r]))
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]
y6, b6, p6, w6 = load(6, 'pred'); y5, b5, p5, w5 = load(5, 'pred')
out['bootstrap95'] = dict(sel=boot(s6['y'], b6[fi], p6[fi], s6['weights']), full6=boot(y6, b6, p6, w6), d5=boot(y5, b5, p5, w5))
json.dump(out, open(R + 'nxd_d1_check.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
