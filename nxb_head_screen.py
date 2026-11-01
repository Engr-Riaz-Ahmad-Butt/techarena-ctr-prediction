"""nxb: head screens on top of nxa_h1 (selection on the day-6 SELECTION subset only; grid saved).
(1) bag1 (Track B) through an nxa_h1-style logistic head: LR([logit rank(nxa_h1), within-event z of log bag1 (, logit rank bag1)])
    day 6 <- fitted on full day 5; day 5 <- 2-fold user cross-fit (seed 9901). Eval-mix 5-bin weights.
(2) Track A clean model used only inside 0-history events: rank(nxa_h1) + b * within-event z(clean) on bin-0 rows.
"""
import json, numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
R = Path('artifacts')
def rank(x): return rankdata(x, method='average') / (len(x) + 1)
def logit(p): return np.log(p / (1 - p))
def zc(raw):
    m = np.log(np.clip(raw, 1e-7, 1)).reshape(-1, 10); z = (m - m.mean(1, keepdims=True)).ravel(); return z / z.std()
h = np.load(R / 'train_hist.npy', mmap_mode='r'); eh = np.load(R / 'evaluation_hist.npy')
eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True); es = np.bincount(eg, minlength=5) / len(eg)
def wts(ev):
    vg = np.digitize((np.asarray(h[ev]) >= 0).sum(1), [0, 4, 9, 19], right=True)
    return (es / (np.bincount(vg, minlength=5) / len(vg)))[vg], vg
P = {}
for d in (5, 6):
    ref = np.load(R / f'nxa_h1_day{d}.npz'); b = np.load(R / f'nxb_b_bag1_day{d}.npz'); a = np.load(R / f'nxb_a_a1_day{d}.npz')
    assert np.array_equal(ref['events'], b['events']) and np.array_equal(ref['events'], a['events'])
    w, g = wts(ref['events'])
    P[d] = dict(y=ref['y'], par=ref['pred'], bag=b['raw'], cln=a['raw'], w=w, g=g, ev=ref['events'],
                users=np.load(R / 'train.npy', mmap_mode='r')[ref['events'] * 10, 0])
def X(d, fs):
    p = P[d]; cols = {'par': logit(rank(p['par'])), 'bz': zc(p['bag']), 'br': logit(rank(p['bag']))}
    return np.column_stack([cols[f] for f in fs])
s6 = np.load(R / 'selection_day6.npz'); fi = s6['full_indices']
sel = lambda p: float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
full = lambda d, p: float(roc_auc_score(P[d]['y'], p, sample_weight=P[d]['w']))
def head(fs, C=0.01):
    sc = StandardScaler().fit(X(5, fs)); lr = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(X(5, fs)), P[5]['y'], sample_weight=P[5]['w'])
    p6 = lr.decision_function(sc.transform(X(6, fs)))
    u = P[5]['users']; fold = (np.random.default_rng(9901).random(u.max() + 1) < 0.5)[u]; p5 = np.zeros(len(u))
    for f in (0, 1):
        tr = fold != f; sc2 = StandardScaler().fit(X(5, fs)[tr])
        m = LogisticRegression(C=C, max_iter=1000).fit(sc2.transform(X(5, fs)[tr]), P[5]['y'][tr], sample_weight=P[5]['w'][tr])
        p5[~tr] = m.decision_function(sc2.transform(X(5, fs)[~tr]))
    return p6, p5, lr.coef_.ravel().tolist()
grid = {}; out = {}
base6, base5 = P[6]['par'], P[5]['par']
grid['nxa_h1'] = sel(base6)
for fs in (['par'], ['par', 'bz'], ['par', 'br'], ['par', 'bz', 'br']):
    p6, p5, coef = head(fs); k = 'head_' + '+'.join(fs); grid[k] = sel(p6); out[k] = (p6, p5, coef)
for bta in (0.02, 0.05, 0.1, 0.2, 0.4):
    for d in (6,):
        pass
    p = rank(base6) + bta * zc(P[6]['cln']) * (P[6]['g'] == 0); grid[f'bin0_clean_{bta}'] = sel(p)
best_head = max([k for k in grid if k.startswith('head_')], key=grid.get)
rep = dict(grid_sel6=grid, best_head=best_head)
p6, p5, coef = out[best_head]
rep['best_head_eval'] = dict(coef=coef, d6_sel=sel(p6), d6_full=full(6, p6), d5=full(5, p5),
                             control_head_par_only=dict(d6_sel=sel(out['head_par'][0]), d6_full=full(6, out['head_par'][0]), d5=full(5, out['head_par'][1])),
                             nxa_h1=dict(d6_sel=sel(base6), d6_full=full(6, base6), d5=full(5, base5)))
bb = max([k for k in grid if k.startswith('bin0')], key=grid.get); bta = float(bb.split('_')[-1])
rep['bin0_best'] = dict(key=bb, d6_full=full(6, rank(base6) + bta * zc(P[6]['cln']) * (P[6]['g'] == 0)),
                        d5=full(5, rank(base5) + bta * zc(P[5]['cln']) * (P[5]['g'] == 0)))
(R / 'nxb_head_screen.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
