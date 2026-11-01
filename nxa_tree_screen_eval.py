"""nxa T1 screen evaluation (owner: Idea-A improver session). Day-6 SELECTION subset only (selection_day6.npz full_indices/weights).
Compares the matched control tree (354 columns) and treatment tree (362 = + exposure columns), both fitted on the same FRAC of
day 3-5 events by nxa_tree.py screen: (1) raw tree adjusted AUC; (2) rank blends with nxa_h1 (w 0.1/0.2/0.3/0.5);
(3) 2-fold user cross-fitted LR [logit rank(nxa_h1), logit rank(tree)] fitted and scored INSIDE the selection subset only.
Gains are reported vs nxa_h1 and treatment minus control. Grid saved to artifacts/nxa_tree_screen_eval.json.
"""
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import rank

sel = np.load(R / 'selection_day6.npz'); FI = sel['full_indices']; W = sel['weights']; Y = sel['y']
s = np.load(R / 'nxa_tree_screen_day6.npz'); h1 = np.load(R / 'nxa_h1_day6.npz'); v = np.load(R / 'v53_causal_propensity_day6.npz')
assert np.array_equal(s['events'], h1['events']) and np.array_equal(h1['events'], v['events'])
ev = v['events'][FI]; users = np.asarray(np.load(R / 'train.npy', mmap_mode='r')[ev.astype(np.int64) * 10, 0])
uniq = np.unique(users); first = set(np.random.default_rng(9901).permutation(uniq)[: len(uniq) // 2].tolist()); fold = np.array([u in first for u in users])
def auc(p): return float(roc_auc_score(Y, p, sample_weight=W))
def lg(x): return logit(np.clip(rank(x), 1e-6, 1 - 1e-6))
base = auc(h1['pred'][FI]); out = dict(cohort='day-6 selection subset', frac=float(s['frac']), nxa_h1=base, v53=auc(v['pred'][FI]))
for arm in ('control', 'treatment'):
    t = s[arm][FI]; hb = h1['pred'][FI]; r = dict(raw_adjusted=auc(t), raw_plain=float(roc_auc_score(Y, t)))
    for w in (.1, .2, .3, .5): r[f'blend{w}_gain_vs_h1'] = auc((1 - w) * rank(hb) + w * rank(t)) - base
    X = np.column_stack([lg(hb), lg(t)]); p = np.zeros(len(Y))
    for a in (fold, ~fold): p[a] = LogisticRegression(C=.01, max_iter=2000).fit(X[~a], Y[~a], sample_weight=W[~a]).decision_function(X[a])
    r['crossfit_head_gain_vs_h1'] = auc(p) - base; out[arm] = r
    print(arm, json.dumps({k: round(x, 6) for k, x in r.items()}), flush=True)
out['treatment_minus_control'] = {k: out['treatment'][k] - out['control'][k] for k in out['control']}
print('treatment - control', json.dumps({k: round(x, 6) for k, x in out['treatment_minus_control'].items()}))
(R / 'nxa_tree_screen_eval.json').write_text(json.dumps(out, indent=1))
