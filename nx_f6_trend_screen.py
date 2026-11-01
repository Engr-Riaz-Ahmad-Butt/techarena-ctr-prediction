"""Quick screen of F6: item trend + transductive cohort exposure features, residual over V53 (day 5 and day 6).
Cohort features use ONLY the cohort's own slates (no labels) - the analogue over evaluation.csv is computable.
Label features use rows with t < cutoff only."""
import sys, numpy as np
sys.path.insert(0, r'C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction')
from config import ARTIFACT_DIR as R
from scipy.special import logit
import lightgbm as lgb
from nx_seq_core import weighted_auc

START, DAY, NI = 1737331200, 86400, 50000
t = np.load(R / 'train.npy')
item, ts, clk = t[:, 1], t[:, 2].astype(np.int64), t[:, 5]


def feats(day):
    d = np.load(R / f'v53_causal_propensity_day{day}.npz')
    ev, y, p = d['events'], d['y'], d['pred']
    rows = ev.astype(np.int64) * 10 + np.tile(np.arange(10), len(ev) // 10)
    assert (clk[rows] == y).all()
    cut = START + day * DAY
    it, et = item[rows], ts[rows]
    nev = len(ev) // 10
    F = {}
    # transductive: cohort-slate exposure share, overall and within +-2h of the event
    coh = np.bincount(it, minlength=NI) / nev
    F['coh_share'] = coh[it]
    order = np.argsort(et, kind='stable'); st, si = et[order], it[order]
    win = np.zeros(len(it))
    for i in np.unique(it):  # windowed count per item via searchsorted
        m = si == i; tt = st[m]
        idx = np.where(m)[0]
        win[order[idx]] = np.searchsorted(tt, tt + 7200, 'right') - np.searchsorted(tt, tt - 7200, 'left')
    F['coh_win2h'] = win
    # pre-cutoff trailing stats
    for name, lo in [('all', START), ('d1', cut - DAY), ('h6', cut - 6 * 3600)]:
        m = (ts >= lo) & (ts < cut)
        n = m.sum() / 10
        ex = np.bincount(item[m], minlength=NI); ck = np.bincount(item[m], weights=clk[m], minlength=NI)
        F[f'exp_{name}'] = ex[it] / n
        F[f'ctr_{name}'] = ((ck + 0.063 * 20) / (ex + 20))[it]
    F['trend_coh_all'] = np.log((F['coh_share'] + 1e-4) / (F['exp_all'] + 1e-4))
    F['trend_coh_d1'] = np.log((F['coh_share'] + 1e-4) / (F['exp_d1'] + 1e-4))
    F['trend_d1_all'] = np.log((F['exp_d1'] + 1e-4) / (F['exp_all'] + 1e-4))
    # slate-relative rank of cohort share
    cs = F['coh_share'].reshape(-1, 10)
    F['coh_rank_in_slate'] = (cs[:, :, None] > cs[:, None, :]).sum(2).ravel()
    X = np.column_stack(list(F.values()))
    return ev, y, p, X, list(F)


for day in (6, 5):
    ev, y, p, X, names = feats(day)
    base = weighted_auc(ev, y, p)
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))[:, None]
    print(f'day {day} V53 adjusted {base:.6f}')
    for n, x in zip(names, X.T):
        print(f'   raw AUC {n:18s} {weighted_auc(ev, y, x):.4f}')
    ue = np.arange(len(ev) // 10); fold = np.repeat(np.random.default_rng(0).integers(0, 2, len(ue)), 10)
    for arm, Z in [('control', lp), ('f6', np.hstack([lp, X]))]:
        out = np.zeros(len(y))
        for f in (0, 1):
            tr = fold != f
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1)
            m.fit(Z[tr], y[tr]); out[~tr] = m.predict_proba(Z[~tr])[:, 1]
        print(f'   {arm:8s} crossfit adjusted {weighted_auc(ev, y, out):.6f}  ({weighted_auc(ev, y, out) - base:+.6f} vs V53)')
