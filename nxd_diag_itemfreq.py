"""nxd diagnostic: nxa_h1 plain AUC by the candidate item's pre-cut click count (labelled train clicks with ts < cut)."""
import numpy as np, json
from sklearn.metrics import roc_auc_score
t = np.load('artifacts/train.npy'); S = 1737331200; DAY = 86400
e = np.load('artifacts/evaluation.npy')
bins = [0, 1, 3, 10, 30, 100, 10 ** 9]; out = {}
def counts(cut):
    n = int(np.searchsorted(t[:, 2], cut)); P = t[:n]
    return np.bincount(P[P[:, 5] == 1, 1], minlength=50000)
for d in (5, 6, 7):
    c = counts(S + d * DAY)
    if d < 7:
        ref = np.load(f'artifacts/nxa_h1_day{d}.npz'); it = t[ref['events'] * 10 + np.tile(np.arange(10), len(ref['events']) // 10), 1]
        y, p = ref['y'], ref['pred']
    else:
        it = e[:, 1]
    k = np.digitize(c[it], bins[1:-1], right=False)   # 0:0, 1:1-2, 2:3-9, 3:10-29, 4:30-99, 5:100+
    r = {}
    for b in range(6):
        m = k == b
        r[f'{bins[b]}-{bins[b+1]-1}'] = dict(share=round(float(m.mean()), 4), **({'auc': round(float(roc_auc_score(y[m], p[m])), 4)} if d < 7 and m.sum() > 100 and y[m].sum() > 0 else {}))
    out[f'day{d}'] = r
print(json.dumps(out, indent=1))
