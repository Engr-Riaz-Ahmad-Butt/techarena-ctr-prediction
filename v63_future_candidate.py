"""Attempt a stronger future-day candidate on the V53 feature stack.

This is a valid benchmark experiment under the repo's accepted protocol:
- Train/fit on day5 data only.
- Validate on the genuinely future day6 cohort.
- Compare directly against the saved V53 baseline using the same day6 event set.

It is intentionally limited to a small number of well-motivated, high-capacity
GBDT variants; the goal is to see whether a proper future benchmark can reach
0.85, not to claim a score without independent evidence.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R
import v53_causal_propensity as base

ROOT = Path(__file__).resolve().parent


def load_spline():
    with (R / 'v48_history_calibrator.pkl').open('rb') as f:
        return pickle.load(f)['spline']


def make_model(params):
    defaults = dict(
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=4201,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
        od_type='Iter',
        od_wait=50,
    )
    defaults.update(params)
    return CatBoostClassifier(**defaults)


def compute_day_scores(params, use_sample_weights=True, sample_limit=None):
    t = np.load(R / 'train.npy', mmap_mode='r')
    h = np.load(R / 'train_hist.npy', mmap_mode='r')
    ht = np.load(R / 'train_times.npy', mmap_mode='r')
    spline = load_spline()

    z5, x5, users5, groups5, names5 = base.features(5, t, h, ht, spline)
    z6, x6, users6, groups6, names6 = base.features(6, t, h, ht, spline)

    if sample_limit is not None:
        rng = np.random.default_rng(42)
        ids = np.arange(len(z5['y']))
        keep = rng.choice(ids, size=min(sample_limit, len(ids)), replace=False)
        x5 = x5[keep]
        z5 = {k: v[keep] if isinstance(v, np.ndarray) and v.shape[0] == len(ids) else v for k, v in z5.items()}
        users5 = users5[keep]
        groups5 = groups5[keep]

    if use_sample_weights:
        counts = np.bincount(groups5, minlength=5)
        weights = np.divide(counts.mean(), counts, out=np.zeros_like(counts, dtype='float32'), where=counts > 0)
        sample_weight = weights[groups5].astype('float32')
    else:
        sample_weight = None

    model = make_model(params)
    if sample_weight is None:
        model.fit(x5, z5['y'])
    else:
        model.fit(x5, z5['y'], sample_weight=sample_weight)

    p5 = model.predict_proba(x5)[:, 1]
    p6 = model.predict_proba(x6)[:, 1]

    baseline_day6 = np.load(R / 'v53_causal_propensity_day6.npz')['pred']
    baseline_day5 = np.load(R / 'v53_causal_propensity_day5.npz')['pred']

    day5_auc = float(roc_auc_score(z5['y'], p5))
    day6_auc = float(roc_auc_score(z6['y'], p6))
    v53_day6_auc = float(roc_auc_score(z6['y'], baseline_day6))
    v53_day5_auc = float(roc_auc_score(z5['y'], baseline_day5))

    return {
        'config': params,
        'sample_limit': sample_limit,
        'use_sample_weights': use_sample_weights,
        'day5_auc': day5_auc,
        'day6_auc': day6_auc,
        'v53_day5_auc': v53_day5_auc,
        'v53_day6_auc': v53_day6_auc,
        'day5_gain_vs_v53': day5_auc - v53_day5_auc,
        'day6_gain_vs_v53': day6_auc - v53_day6_auc,
        'target_achieved': day6_auc >= 0.85,
    }


def main():
    configs = [
        {
            'name': 'cb_deep_600',
            'params': dict(iterations=600, depth=8, learning_rate=0.04, l2_leaf_reg=10.0, border_count=254, bagging_temperature=1.0, subsample=0.9, colsample_bylevel=0.8),
            'sample_limit': None,
            'use_sample_weights': True,
        },
        {
            'name': 'cb_deep_1200',
            'params': dict(iterations=1200, depth=8, learning_rate=0.03, l2_leaf_reg=12.0, border_count=254, bagging_temperature=1.0, subsample=0.9, colsample_bylevel=0.85),
            'sample_limit': None,
            'use_sample_weights': True,
        },
        {
            'name': 'cb_fast_800',
            'params': dict(iterations=800, depth=9, learning_rate=0.025, l2_leaf_reg=15.0, border_count=254, bagging_temperature=0.8, subsample=0.85, colsample_bylevel=0.75),
            'sample_limit': None,
            'use_sample_weights': True,
        },
    ]

    results = []
    for cfg in configs:
        start = time.monotonic()
        result = compute_day_scores(cfg['params'], use_sample_weights=cfg['use_sample_weights'], sample_limit=cfg['sample_limit'])
        result['name'] = cfg['name']
        result['elapsed_seconds'] = round(time.monotonic() - start, 3)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    best = max(results, key=lambda r: r['day6_auc'])
    summary = {
        'benchmark': 'valid_future_day6_only',
        'baseline': {
            'version': 'V53',
            'day6_auc': 0.8171690462350286,
            'day5_auc': 0.8180913840683512,
        },
        'results': results,
        'best_candidate': best,
        'target_0_85': best['day6_auc'] >= 0.85,
        'notes': [
            "Each config trains on all day5 labels and scores on the genuine future day6 cohort.",
            "No same-cohort scoring is used; the benchmark is the repo's accepted temporal holdout.",
            "This still does not guarantee a valid 0.85 score unless the model exceeds the threshold under this benchmark."
        ],
    }

    out = ROOT / 'results' / 'v63_future_candidate_summary.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print('\nBEST:', json.dumps(best, indent=2), flush=True)


if __name__ == '__main__':
    main()
