"""nxb: standalone + blend report for a candidate's day-5/day-6 predictions against nxa_h1 and V53.

Blend recipes (scored on the day-6 SELECTION subset only; grid saved):
  rank: (1-w) * rank(nxa_h1) + w * rank(candidate)
  zev : rank(nxa_h1) + b * zc, zc = within-event centred candidate logit divided by its std (as nxa_expo)
Usage: python nxb_report.py <prefix>   (reads artifacts/<prefix>_day6.npz and, if present, _day5.npz)
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent / 'artifacts'


def rank(x):
    return rankdata(x, method='average') / (len(x) + 1)


def wts(events_rows):
    h = np.load(R / 'train_hist.npy', mmap_mode='r'); eh = np.load(R / 'evaluation_hist.npy')
    vg = np.digitize((np.asarray(h[events_rows]) >= 0).sum(1), [0, 4, 9, 19], right=True)
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    return ((np.bincount(eg, minlength=5) / len(eg)) / (np.bincount(vg, minlength=5) / len(vg)))[vg], vg


def zc(raw):
    x = np.log(np.clip(raw, 1e-7, 1)) if raw.min() >= 0 else raw
    m = x.reshape(-1, 10); z = (m - m.mean(1, keepdims=True)).ravel()
    return z / z.std()


def recipes(par, cand):
    out = {}
    rp = rank(par); rc = rank(cand); z = zc(cand)
    for w in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7):
        out[f'rank{w}'] = (1 - w) * rp + w * rc
    for b in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        out[f'zev{b}'] = rp + b * z
    return out


def main(prefix):
    rep = {}
    d6 = np.load(R / f'{prefix}_day6.npz'); h6 = np.load(R / 'nxa_h1_day6.npz'); s6 = np.load(R / 'selection_day6.npz')
    assert np.array_equal(d6['events'], h6['events']) and np.array_equal(d6['y'], h6['y'])
    fi = s6['full_indices']
    w6, g6 = wts(h6['events']); y6 = h6['y']
    raw6 = d6['raw']
    sel = lambda p: float(roc_auc_score(s6['y'], p[fi], sample_weight=s6['weights']))
    full = lambda p, y=y6, w=w6: float(roc_auc_score(y, p, sample_weight=w))
    rep['day6'] = dict(standalone_full=full(raw6), standalone_sel=sel(raw6), standalone_plain=float(roc_auc_score(y6, raw6)),
                       nxa_h1_full=full(h6['pred']), nxa_h1_sel=sel(h6['pred']), v53_full=full(h6['baseline']), v53_sel=sel(h6['baseline']))
    grid = {k: sel(v) for k, v in recipes(h6['pred'], raw6).items()}
    best = max(grid, key=grid.get)
    rep['grid_sel6'] = grid; rep['best'] = best
    rb = recipes(h6['pred'], raw6)[best]
    rep['day6'].update(blend_sel=grid[best], blend_full=full(rb))
    rep['day6_bins'] = {int(b): dict(n_events=int((g6 == b).sum() // 10), cand=float(roc_auc_score(y6[g6 == b], raw6[g6 == b])),
                                     nxa_h1=float(roc_auc_score(y6[g6 == b], h6['pred'][g6 == b])),
                                     v53=float(roc_auc_score(y6[g6 == b], h6['baseline'][g6 == b])),
                                     blend=float(roc_auc_score(y6[g6 == b], rb[g6 == b]))) for b in range(5)}
    rep['corr_spearman_nxa_h1_d6'] = float(np.corrcoef(rank(raw6), rank(h6['pred']))[0, 1])
    p5 = R / f'{prefix}_day5.npz'
    if p5.exists():
        d5 = np.load(p5); h5 = np.load(R / 'nxa_h1_day5.npz')
        assert np.array_equal(d5['events'], h5['events']) and np.array_equal(d5['y'], h5['y'])
        w5, g5 = wts(h5['events']); y5 = h5['y']
        rb5 = recipes(h5['pred'], d5['raw'])[best]
        rep['day5'] = dict(standalone=full(d5['raw'], y5, w5), nxa_h1=full(h5['pred'], y5, w5), v53=full(h5['baseline'], y5, w5),
                           blend=full(rb5, y5, w5), standalone_plain=float(roc_auc_score(y5, d5['raw'])))
        rep['day5_bins'] = {int(b): dict(cand=float(roc_auc_score(y5[g5 == b], d5['raw'][g5 == b])),
                                         nxa_h1=float(roc_auc_score(y5[g5 == b], h5['pred'][g5 == b]))) for b in range(5)}
    (R / f'{prefix}_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'grid_sel6'}, indent=1))
    print('grid (selection subset):', json.dumps({k: round(v, 6) for k, v in grid.items()}))


if __name__ == '__main__':
    main(sys.argv[1])
