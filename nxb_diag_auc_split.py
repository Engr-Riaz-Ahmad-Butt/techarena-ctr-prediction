"""nxb diagnostic: where is the global-AUC headroom of nxa_h1 (day 6, adjusted weights)? Labels of the validation cohort only.
Split negatives into (a) other candidates of CLICK events and (b) candidates of NO-CLICK events; within-event AUC; and two
oracles: perfect within-event order (event mass kept) and perfect event propensity (within-event order kept)."""
import numpy as np, json
from sklearn.metrics import roc_auc_score
h = np.load('artifacts/train_hist.npy', mmap_mode='r'); eh = np.load('artifacts/evaluation_hist.npy')
es = np.bincount(np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True), minlength=5) / len(eh)
out = {}
for d in (6, 5):
    ref = np.load(f'artifacts/nxa_h1_day{d}.npz'); y = ref['y']; p = ref['pred']
    vg = np.digitize((np.asarray(h[ref['events']]) >= 0).sum(1), [0, 4, 9, 19], right=True)
    w = (es / (np.bincount(vg, minlength=5) / len(vg)))[vg]
    Y = y.reshape(-1, 10); hasc = np.repeat(Y.sum(1) > 0, 10)
    r = {}
    r['global_adj'] = roc_auc_score(y, p, sample_weight=w)
    m = hasc; r['pos_vs_negs_in_click_events'] = roc_auc_score(y[m], p[m], sample_weight=w[m])
    m = (y == 1) | ~hasc; r['pos_vs_negs_in_noclick_events'] = roc_auc_score(y[m], p[m], sample_weight=w[m])
    P = p.reshape(-1, 10)
    rk = (P[:, :, None] > P[:, None, :]).sum(2) + 0.5 * ((P[:, :, None] == P[:, None, :]).sum(2) - 1)
    ce = Y.sum(1) > 0
    r['within_event_auc'] = float((rk[ce][Y[ce] == 1] / 9).mean())
    # oracle 1: perfect within-event order, event mass kept (clicked item gets the event's max score)
    P1 = P.copy(); i = np.flatnonzero(ce); j = Y[ce].argmax(1); a = P1[i].argmax(1)
    tmp = P1[i, j].copy(); P1[i, j] = P1[i, a]; P1[i, a] = tmp
    r['oracle_within_event'] = roc_auc_score(y, P1.ravel(), sample_weight=w)
    # oracle 2: perfect event propensity, within-event order kept: event mass -> logit shift by +-2 by truth
    sh = np.where(ce, 2.0, -2.0)[:, None]
    from scipy.stats import rankdata
    Q = (rankdata(p) / (len(p) + 1)).reshape(-1, 10)
    lg = np.log(Q / (1 - Q)) + sh
    r['oracle_event_prop_shift2'] = roc_auc_score(y, lg.ravel(), sample_weight=w)
    out[f'day{d}'] = {k: round(float(v), 6) for k, v in r.items()}
print(json.dumps(out, indent=1))
