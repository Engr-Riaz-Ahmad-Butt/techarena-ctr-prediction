"""Independent replay of decomposition candidates; no training or promotion."""
import json
import sys
from pathlib import Path

from config import ARTIFACT_DIR as ROOT
import numpy as np
from sklearn.metrics import roc_auc_score
from tst2_candidate import influence, weighted_influence


def audit(day, mix):
    d = np.load(ROOT / f'predictions_imp4_decompose_day{day}.npz')
    b = np.load(ROOT / f'predictions_imp3_wf3_day{day}.npz')
    assert np.array_equal(d['events'], b['events'])
    assert np.array_equal(d['y'], b['y'])
    events = d['events'].reshape(-1, 10)
    assert (events == events[:, :1]).all()
    assert len(np.unique(events[:, 0])) == len(events)
    y = d['y']
    assert (y.reshape(-1, 10).sum(1) <= 1).all()
    h = np.load(ROOT / 'train_hist.npy', mmap_mode='r')
    eh = np.load(ROOT / 'evaluation_hist.npy', mmap_mode='r')
    vg = np.digitize((h[d['events']] >= 0).sum(1), [0, 4, 9, 19], right=True)
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    w = ((np.bincount(eg, minlength=5) / len(eg)) /
         (np.bincount(vg, minlength=5) / len(vg)))[vg]
    old_member = np.load(ROOT / f'predictions_imp3_warmfeat_day{day}.npz')['pred']
    member = (1 - mix) * old_member + mix * d['raw']
    def logit(p):
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return np.log(p) - np.log1p(-p)
    v6 = np.load(ROOT / f'predictions_v6_day{day}.npz')['pred']
    v7 = np.load(ROOT / f'predictions_v7_day{day}.npz')['pred']
    z = .7 * logit(member) + .2 * logit(v6) + .1 * logit(v7)
    p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
    p = np.clip(p, 1e-8, 1 - 1e-8).reshape(-1, 10)
    odds = p / (1 - p)
    p = (.5 * p + .5 * odds / (1 + odds.sum(1, keepdims=True))).ravel()
    warm = np.mean([np.load(ROOT / f'{prefix}_day{day}.npz')['raw'] for prefix in
                    ['imp2_warm1.0_d16_pc1', 'imp3_warm16_s7', 'imp3_warm16_s1009']], axis=0)
    expected = .9 * p + .1 * warm
    err = float(np.max(np.abs(expected - d['pred'])))
    assert err < 1e-12, err
    assert np.isfinite(d['pred']).all()
    assert ((d['pred'] >= 0) & (d['pred'] <= 1)).all()
    result = dict(day=day, rows=len(y), mix=mix, recipe_max_error=err)
    for name, weights, influence_fn in [('auc', None, influence), ('weighted_auc', w, weighted_influence)]:
        old = roc_auc_score(y, b['pred'], sample_weight=weights)
        new = roc_auc_score(y, d['pred'], sample_weight=weights)
        args = () if weights is None else (weights,)
        se = np.sqrt(np.sum((influence_fn(y, d['pred'], *args) - influence_fn(y, b['pred'], *args)) ** 2))
        result[name] = dict(baseline=float(old), candidate=float(new), delta=float(new-old),
                            conditional_paired_event_95pct_interval=[float(new-old-1.96*se), float(new-old+1.96*se)])
    return result


if __name__ == '__main__':
    report = json.loads((ROOT / 'imp4_decompose_report.json').read_text())
    outputs = [audit(r['day'], r['mix']) for r in report]
    result = dict(reports=outputs, note='Intervals exclude repeated selection and distribution uncertainty.')
    (ROOT / 'imp4_test_decomposition_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
