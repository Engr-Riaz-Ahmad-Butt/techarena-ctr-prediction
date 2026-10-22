"""Independent replay of aligned candidates; no training or promotion."""
import argparse
import json
import sys
from pathlib import Path

from config import ARTIFACT_DIR as ROOT
import numpy as np
from sklearn.metrics import roc_auc_score
from tst2_candidate import influence, weighted_influence
TAG = 'aligned'


def check_features(day, events):
    """Replay sampled component products without importing experiment helpers."""
    rotation_file = np.load(ROOT / f'imp5_rotation_cut{day}.npz')
    r = rotation_file['rotation']
    assert r.shape == (16, 16)
    orth = float(np.max(np.abs(r.T @ r - np.eye(16))))
    assert orth < 1e-10
    checkpoint = ('imp2_model_warm1.0_d16_pc1_day5.npz' if day == 5 else
                  'imp2_model_warm1.0_d16_pc1_day6.npz')
    source = np.load(ROOT / checkpoint)
    anchor = np.load(ROOT / 'imp3_model_warm16_cut2.npz')['V'].astype(float)
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    current = unit(source['V'].astype(float))
    cross = current.T @ unit(anchor)
    singular = np.linalg.svd(cross, compute_uv=False)
    objective_error = float(abs(np.trace(r.T @ cross) - singular.sum()))
    assert objective_error < 1e-7
    train = np.load(ROOT / 'train.npy', mmap_mode='r')
    history = np.load(ROOT / 'train_hist.npy', mmap_mode='r')
    indices = np.linspace(0, len(events) - 1, 1000, dtype=int)
    rows = train[events[indices] * 10 + indices % 10]
    assert np.array_equal(rows[:, 6], events[indices])
    u = source['U'].astype(float) @ r
    v = source['V'].astype(float) @ r
    h = (source['H'].astype(float) @ r).astype(np.float32)
    u = unit(u).astype(np.float32)
    v = unit(v).astype(np.float32)
    histories = history[events[indices]]
    valid = histories >= 0
    z = unit((h[np.maximum(histories, 0)] * valid[:, :, None]).sum(1))
    expected = np.column_stack([u[rows[:, 0]] * v[rows[:, 1]], z * v[rows[:, 1]]])
    cached = np.load(ROOT / 'features' / f'imp5_aligned_val{day}.npy', mmap_mode='r')[indices]
    error = float(np.max(np.abs(expected - cached)))
    assert error < 1e-6, error
    decomp = np.load(ROOT / 'features' / f'imp4_decompose_val{day}.npy', mmap_mode='r')[indices]
    cosine_error = float(max(np.max(np.abs(cached[:, :16].sum(1) - decomp[:, 7])),
                             np.max(np.abs(cached[:, 16:].sum(1) - decomp[:, 8]))))
    assert cosine_error < 2e-6, cosine_error
    return dict(orthogonality_error=orth, procrustes_objective_error=objective_error,
                sampled_feature_max_error=error, cosine_sum_max_error=cosine_error)


def audit(day, mix):
    d = np.load(ROOT / f'predictions_imp5_{TAG}_day{day}.npz')
    b = np.load(ROOT / f'predictions_imp4_decompose_day{day}.npz')
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
    old_member = np.load(ROOT / f'predictions_imp4_decompose_day{day}.npz')['raw']
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
    result = dict(day=day, rows=len(y), mix=mix, recipe_max_error=err,
                  aligned_features=check_features(day, d['events']))
    for name, weights, influence_fn in [('auc', None, influence), ('weighted_auc', w, weighted_influence)]:
        old = roc_auc_score(y, b['pred'], sample_weight=weights)
        new = roc_auc_score(y, d['pred'], sample_weight=weights)
        args = () if weights is None else (weights,)
        se = np.sqrt(np.sum((influence_fn(y, d['pred'], *args) - influence_fn(y, b['pred'], *args)) ** 2))
        result[name] = dict(baseline=float(old), candidate=float(new), delta=float(new-old),
                            conditional_paired_event_95pct_interval=[float(new-old-1.96*se), float(new-old+1.96*se)])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', choices=['aligned', 'timecontext'], default='aligned')
    args = parser.parse_args()
    TAG = args.tag
    if TAG == 'timecontext':
        from imp5_test_context_features import check_features
    report = json.loads((ROOT / f'imp5_{TAG}_report.json').read_text())
    outputs = [audit(r['day'], r['mix']) for r in report]
    result = dict(reports=outputs, note='Intervals exclude repeated selection and distribution uncertainty.')
    (ROOT / f'imp5_test_{TAG}_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
