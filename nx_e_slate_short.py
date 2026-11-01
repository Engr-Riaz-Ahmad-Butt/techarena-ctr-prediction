"""Idea E: does the saved V58 slate-composition residual help SHORT histories (0-4 clicks) specifically?

Read-only re-scoring of artifacts/v58_slate_residual_day6.npz (keys: y, events, pred = 0.1 rank blend of
V53 with the slate arm, control = same blend with the matched zero-slate control, raw / control_raw = the
two residual arms alone). Nothing is fitted; no file in artifacts/ is written.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from config import ARTIFACT_DIR as R

z = np.load(R / 'v58_slate_residual_day6.npz'); v = np.load(R / 'v53_causal_propensity_day6.npz')
assert np.array_equal(z['events'], v['events']) and np.array_equal(z['y'], v['y'])
y = z['y']; e = z['events']; base = v['pred']
H = np.load(R / 'train_hist.npy', mmap_mode='r'); EH = np.load(R / 'evaluation_hist.npy', mmap_mode='r')
length = (H[e] >= 0).sum(1); g5 = np.digitize(length, [0, 4, 9, 19], right=True)
eg = np.digitize((EH >= 0).sum(1), [0, 4, 9, 19], right=True); target = np.bincount(eg, minlength=5) / len(eg)
w5 = (target / (np.bincount(g5, minlength=5) / len(g5)))[g5]

def rank(x): return rankdata(x, method='average') / (len(x) + 1)
def auc(mask, p, w=None): return float(roc_auc_score(y[mask], p[mask], sample_weight=None if w is None else w[mask]))

out = {}
arms = {'v53': base, 'slate_blend0.1': z['pred'], 'control_blend0.1': z['control'], 'slate_raw': z['raw'].astype(np.float64), 'control_raw': z['control_raw'].astype(np.float64)}
for wgt in (0.25, 0.5):
    arms[f'slate_blend{wgt}'] = rank((1 - wgt) * rank(base) + wgt * rank(z['raw']))
    arms[f'control_blend{wgt}'] = rank((1 - wgt) * rank(base) + wgt * rank(z['control_raw']))
masks = {'bin0': g5 == 0, 'bin1-4': g5 == 1, 'bin0-4': g5 <= 1, 'bin5-9': g5 == 2, 'bin10-19': g5 == 3, 'bin20': g5 == 4, 'all_adjusted': np.ones(len(y), bool)}
for name, m in masks.items():
    out[name] = {'rows': int(m.sum()), 'events': int(len(np.unique(e[m])))}
    for arm, p in arms.items():
        out[name][arm] = auc(m, p, w5 if name == 'all_adjusted' else None)
    out[name]['slate0.1_minus_v53'] = out[name]['slate_blend0.1'] - out[name]['v53']
    out[name]['slate0.1_minus_control0.1'] = out[name]['slate_blend0.1'] - out[name]['control_blend0.1']
# What an oracle 'short-only' use of the slate arm would give on the adjusted metric: replace V53 by the slate blend only in bins 0-4.
for wgt in (0.1, 0.25, 0.5):
    s = arms[f'slate_blend{wgt}'] if wgt != 0.1 else z['pred']; c = arms[f'control_blend{wgt}'] if wgt != 0.1 else z['control']
    for nm, arm in (('slate', s), ('control', c)):
        mixed = base.copy(); short = g5 <= 1; mixed[short] = arm[short]
        out[f'short_only_{nm}_{wgt}_adjusted_full'] = auc(np.ones(len(y), bool), mixed, w5) - auc(np.ones(len(y), bool), base, w5)
print(json.dumps(out, indent=1))
(R.parent / 'results' / 'nx_e_slate_short.json').write_text(json.dumps(out, indent=1))
