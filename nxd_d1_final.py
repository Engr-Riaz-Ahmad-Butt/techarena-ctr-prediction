"""nxd D1 final: apply the frozen per-cell rank-blend recipe (artifacts/nxd_d1_grid.json) to the evaluation rows.
parent = nxa_h1 exportable evaluation prediction (artifacts/nxa_h1_eval_pred_v39.npy); experts = bag1 / C1 / C2 u.v evaluation
predictions (same code paths, cutoff 1737936000). Cells: candidate-item labelled clicks in train rows (all < cutoff 7) bins
<3 / 3-9 / >=10 x evaluation history length <10 / >=10. Ranks over the whole evaluation set. Output in (0, 1).
"""
import json
import numpy as np
from scipy.stats import rankdata, spearmanr

R = 'artifacts/'
rank = lambda x: rankdata(x, method='average') / (len(x) + 1)
g = json.load(open(R + 'nxd_d1_grid.json'))['result']
W = np.array([[g['weights'][c][e] for e in ('bag1', 'c1', 'c2uv')] for c in g['cells']])
t = np.load(R + 'train.npy'); e = np.load(R + 'evaluation.npy'); eh = np.load(R + 'evaluation_hist.npy')
assert t[:, 2].max() < 1737936000 <= e[:, 2].min()
ic = np.bincount(t[t[:, 5] == 1, 1], minlength=50000)[e[:, 1]]
hl = np.repeat((eh[e[::10, 6]] >= 0).sum(1), 10)
cell = np.digitize(ic, [3, 10]) * 2 + (hl >= 10)
par = np.load(R + 'nxa_h1_eval_pred_v39.npy')
ex = np.stack([rank(np.load(R + 'nxb_b_bag1_eval.npy')), rank(np.load(R + 'nxb_c_c1best_eval.npy')),
               rank(np.load(R + 'nxb_c_c2best_eval_extra.npz')['uv'])], 1)
assert len(par) == len(e) == len(ex)
wc = W[cell]
pred = (1 - wc.sum(1)) * rank(par) + (wc * ex).sum(1)
assert np.all(np.isfinite(pred)) and pred.min() > 0 and pred.max() < 1
np.save(R + 'nxd_d1_eval.npy', pred)
rep = dict(rows=len(pred), min=float(pred.min()), max=float(pred.max()), cell_share=(np.bincount(cell, minlength=6) / len(cell)).round(4).tolist(),
           spearman_vs_nxa_h1=float(spearmanr(pred, par).correlation),
           top_changed_events=int((pred.reshape(-1, 10).argmax(1) != par.reshape(-1, 10).argmax(1)).sum()),
           expert_means=dict(bag1=float(np.load(R + 'nxb_b_bag1_eval.npy').mean()), c1=float(np.load(R + 'nxb_c_c1best_eval.npy').mean())))
json.dump(rep, open(R + 'nxd_d1_eval_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
