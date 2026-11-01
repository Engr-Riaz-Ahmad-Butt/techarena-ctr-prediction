"""Cheap incremental-signal screen for candidate head features (no V53 refit).

Within the full day-6 cohort, 2-fold user cross-fit (fixed split, seed 9901): logistic regression on
[logit V53, features, features x logit V53] (standardised, C=0.01 like the V53 head) versus the matched control
[logit V53] only. Out-of-fold probabilities are rank-blended 0.75 V53 / 0.25 LR exactly like the V53 head, and the
adjusted AUC of treatment minus control is the screen number (day-6 only; it is not a validation of a refit).
Usage: python nx_headscreen.py FEATURES_NPZ name1,name2,...   (npz must hold y, events and the named row arrays)
"""
import sys, json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc, rank


def screen(day, cols, names, seed=9901, transform=True):
    v = np.load(R / f'v53_causal_propensity_day{day}.npz'); y = v['y']; e = v['events']; b = logit(np.clip(rank(v['pred']), 1e-6, 1 - 1e-6))
    users = np.asarray(np.load(R / 'train.npy', mmap_mode='r')[e.astype(np.int64) * 10, 0])
    uniq = np.unique(users); fold_users = set(np.random.default_rng(seed).permutation(uniq)[: len(uniq) // 2].tolist()); fold = np.array([u in fold_users for u in users])
    X = np.column_stack(cols).astype(np.float64)
    if transform: X = np.sign(X) * np.log1p(np.abs(X))
    def oof(M):
        p = np.zeros(len(y))
        for f in (fold, ~fold):
            sc = StandardScaler().fit(M[~f]); lr = LogisticRegression(C=.01, max_iter=2000).fit(sc.transform(M[~f]), y[~f]); p[f] = lr.predict_proba(sc.transform(M[f]))[:, 1]
        return p
    ctrl = oof(b[:, None]); treat = oof(np.column_stack([b, X, X * b[:, None]]))
    base = weighted_auc(e, y, v['pred']); out = dict(day=day, features=names, v53=base)
    for nm, p in (('control', ctrl), ('treatment', treat)):
        out[nm + '_lr_adjusted'] = weighted_auc(e, y, p); out[nm + '_blend_adjusted'] = weighted_auc(e, y, .75 * rank(v['pred']) + .25 * rank(p))
    out['blend_gain_vs_control'] = out['treatment_blend_adjusted'] - out['control_blend_adjusted']; out['lr_gain_vs_control'] = out['treatment_lr_adjusted'] - out['control_lr_adjusted']
    out['blend_gain_vs_v53'] = out['treatment_blend_adjusted'] - base
    return out


if __name__ == '__main__':
    f = np.load(sys.argv[1]); names = sys.argv[2].split(','); day = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    v = np.load(R / f'v53_causal_propensity_day{day}.npz'); assert np.array_equal(f['events'], v['events']) and np.array_equal(f['y'], v['y'])
    print(json.dumps(screen(day, [f[n] for n in names], names), indent=1))
