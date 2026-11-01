"""nxa H1 (owner: Idea-A improver session): a small exportable logistic head that learns HOW MUCH exposure-policy evidence to add
per history-length segment, instead of the fixed within-event z blend of nxa_expo.

Inputs per row (all available at prediction time): p = logit(rank(parent)) with parent = V53 (validation) or V39 (exportable);
label-free exposure features from nxa_exposure2.py at the scored day's cutoff: z_k = within-event centred typ16_all, typ32_all,
cos32 (each divided by its std over the scored set), E3z = the same for their mean rank (the nxa_expo expert);
segment terms: s = [h == 0, 1 <= h <= 4, 5 <= h <= 9] (h = supplied history length) and their products with E3z and with p.
Model: StandardScaler + LogisticRegression(C=0.01, lbfgs) with the evaluation-mix 5-bin weights (same as the V53 head).
Fitting labels: day 6 predictions <- fit on the full day-5 cohort (labels < day-6 cutoff); day 5 predictions <- 2-fold user
cross-fit inside day 5 (fixed split seed 9901, the V53 convention); evaluation predictions <- fit on the full day-6 cohort.
Feature sets compared on the day-6 SELECTION subset only (grid saved): F0 = [p] (control), F1 = [p, E3z], F2 = F1 + segments,
F3 = F2 + z_k, S0 = [p] + segments + segments x p (matched recalibration control, no exposure), 2b = F1 + segments x E3z. Usage: python nxa_h1.py select | final FEATURESET
"""
import sys, json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import rank, weighted_auc

COLS = ('typ16_all', 'typ32_all', 'cos32')
EH = np.load(R / 'evaluation_hist.npy', mmap_mode='r'); EG = np.digitize((EH >= 0).sum(1), [0, 4, 9, 19], right=True); TARGET = np.bincount(EG, minlength=5) / len(EG)


def zc(x): c = x - np.repeat(x.reshape(-1, 10).mean(1), 10); return c / c.std()


def design(parent, f, hl, fs):
    p = logit(np.clip(rank(parent), 1e-6, 1 - 1e-6)); e3 = zc(np.mean([rank(f[c]) for c in COLS], axis=0)); cols = [p]
    segs = [m.astype(np.float64) for m in (hl == 0, (hl >= 1) & (hl <= 4), (hl >= 5) & (hl <= 9))]
    if fs == 'S0':  # matched control for segment recalibration: no exposure input at all
        for m in segs: cols += [m, m * p]
        return np.column_stack(cols)
    if fs == '2b':  # exposure with segment-specific strength, no parent recalibration terms
        cols.append(e3); cols += [m * e3 for m in segs]; return np.column_stack(cols)
    if fs >= 1: cols.append(e3)
    if fs >= 2:
        for m in segs: cols += [m, m * e3, m * p]
    if fs >= 3: cols += [zc(rank(f[c])) for c in COLS]
    return np.column_stack(cols)


def weights(hl):
    g = np.digitize(hl, [0, 4, 9, 19], right=True); w = (TARGET / (np.bincount(g, minlength=5) / len(g)))[g]; return w / w.mean()


def fit_predict(Xf, yf, wf, Xp):
    sc = StandardScaler().fit(Xf); m = LogisticRegression(C=.01, max_iter=2000, tol=1e-7).fit(sc.transform(Xf), yf, sample_weight=wf)
    return m.decision_function(sc.transform(Xp)), m


def day_inputs(day, parent_name):
    f = np.load(R / f'nxa_exposure2_day{day}_features.npz'); v = np.load(R / f'v53_causal_propensity_day{day}.npz'); assert np.array_equal(f['events'], v['events'])
    parent = v['pred'] if parent_name == 'v53' else np.load(R / f'v39_regularized_als32_day{day}.npz')['pred']
    hl = np.repeat((np.load(R / 'train_hist.npy', mmap_mode='r')[v['events'].reshape(-1, 10)[:, 0]] >= 0).sum(1), 10)
    users = np.asarray(np.load(R / 'train.npy', mmap_mode='r')[v['events'].astype(np.int64) * 10, 0]); return f, v, parent, hl, users


def day6_pred(fs, parent_name):
    f5, v5, p5, h5, _ = day_inputs(5, parent_name); f6, v6, p6, h6, _ = day_inputs(6, parent_name)
    pred, m = fit_predict(design(p5, f5, h5, fs), v5['y'], weights(h5), design(p6, f6, h6, fs)); return pred, v6, p6, m


def day5_pred(fs, parent_name):
    f5, v5, p5, h5, users = day_inputs(5, parent_name); X = design(p5, f5, h5, fs); uniq = np.unique(users)
    first = set(np.random.default_rng(9901).permutation(uniq)[: len(uniq) // 2].tolist()); fold = np.array([u in first for u in users]); pred = np.zeros(len(users))
    for a in (fold, ~fold): pred[a], _ = fit_predict(X[~a], v5['y'][~a], weights(h5[~a]), X[a])
    return pred, v5, p5


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'select':
        sel = np.load(R / 'selection_day6.npz'); FI = sel['full_indices']; out = dict(cohort='day-6 selection subset, selection_day6.npz weights', grid=[])
        for parent_name in ('v53', 'v39'):
            for fs in (0, 1, 'S0', '2b', 2, 3):
                pred, v6, p6, m = day6_pred(fs, parent_name); base = float(roc_auc_score(sel['y'], p6[FI], sample_weight=sel['weights'])); s = float(roc_auc_score(sel['y'], pred[FI], sample_weight=sel['weights']))
                out['grid'].append(dict(parent=parent_name, featureset=fs, parent_selection=base, head_selection=s, gain=s - base, coef=m.coef_.ravel().round(4).tolist()))
                print(parent_name, 'F%s' % fs, 'selection gain %+.6f' % (s - base), m.coef_.ravel().round(3).tolist(), flush=True)
        (R / 'nxa_h1_selection.json').write_text(json.dumps(out, indent=1))
