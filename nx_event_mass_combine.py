"""nx idea G, step 2: probabilistic combination p_row = share_V53(item | click) * P(event has click).
share from V53 'calibrated' (probabilities), event probability = logistic([logit V53 calibrated mass, logit q_event]).
Day 6: coefficients fitted on day-5 cohort (strictly earlier). Day 5: 2-fold user cross-fit within the day. Also a temperature t
on within-event shares (share ∝ cal**t), chosen on day 6 and applied unchanged to day 5."""
import json, numpy as np
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc

D = {}
for day in (5, 6):
    z = np.load(R / f'v53_causal_propensity_day{day}.npz'); q = np.load(R / f'nx_event_mass_day{day}.npz')['q']
    C = z['calibrated'].reshape(-1, 10); m = np.clip(C.sum(1), 1e-4, 1 - 1e-4)
    D[day] = dict(z=z, C=C, X=np.column_stack([logit(m), logit(np.clip(q, 1e-4, 1 - 1e-4))]), hc=(z['y'].reshape(-1, 10).sum(1) > 0))

def score(day, pe, t):
    C = D[day]['C'] ** t; s = C / C.sum(1, keepdims=True); return (s * pe[:, None]).ravel()

rep = {}
lr = LogisticRegression(C=1.0)
for day in (6, 5):
    d = D[day]; z = d['z']; ev, y = z['events'], z['y']; base = weighted_auc(ev, y, z['pred'])
    if day == 6:
        lr.fit(D[5]['X'], D[5]['hc']); pe = lr.predict_proba(d['X'])[:, 1]; coef = lr.coef_.tolist()
        lr0 = LogisticRegression(C=1.0).fit(D[5]['X'][:, :1], D[5]['hc']); pe0 = lr0.predict_proba(d['X'][:, :1])[:, 1]
    else:
        f = np.random.default_rng(0).integers(0, 2, len(d['hc'])); pe = np.zeros(len(f)); pe0 = np.zeros(len(f))
        for k in (0, 1):
            a = LogisticRegression(C=1.0).fit(d['X'][f != k], d['hc'][f != k]); pe[f == k] = a.predict_proba(d['X'][f == k])[:, 1]
            b = LogisticRegression(C=1.0).fit(d['X'][f != k, :1], d['hc'][f != k]); pe0[f == k] = b.predict_proba(d['X'][f == k, :1])[:, 1]
        coef = None
    r = dict(v53_adjusted=base, v53_calibrated_adjusted=weighted_auc(ev, y, z['calibrated']), event_auc_v53=float(roc_auc_score(d['hc'], d['X'][:, 0])),
             event_auc_combined=float(roc_auc_score(d['hc'], pe)), coef=coef, temps={})
    for t in (0.8, 1.0, 1.2, 1.5):
        p = score(day, pe, t); p0 = score(day, pe0, t)
        r['temps'][t] = dict(control_v53mass_only=weighted_auc(ev, y, p0), with_event_model=weighted_auc(ev, y, p),
                             gain_vs_v53=weighted_auc(ev, y, p) - base, gain_vs_control=weighted_auc(ev, y, p) - weighted_auc(ev, y, p0),
                             plain=float(roc_auc_score(y, p)))
    rep[day] = r; print(day, json.dumps(r, indent=1), flush=True)
(R / 'nx_event_mass_combine_report.json').write_text(json.dumps(rep, indent=1))
