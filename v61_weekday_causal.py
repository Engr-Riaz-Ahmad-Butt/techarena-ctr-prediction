"""Weekday-aware causal extension of the frozen V53 recipe.

This is the next-best improvement path suggested by the repository itself:
keep the V53 causal feature stack as the base model, then add explicit
weekday-dependent user/item propensity and recency terms to align better with
weekday-heavy evaluation. The script is designed to benchmark against the
same first-future-event protocol used by the repo's V53/V56 checks.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R
import v53_causal_propensity as base_causal


def load_spline():
    with (R / 'v48_history_calibrator.pkl').open('rb') as f:
        return pickle.load(f)['spline']


def _repeat_event_features(event_values, row_count):
    vals = np.asarray(event_values, dtype='float32')
    if vals.ndim == 1:
        return np.repeat(vals, 10) if row_count % 10 == 0 and len(vals) == row_count // 10 else vals
    if vals.shape[0] == row_count:
        return vals
    if vals.shape[0] == row_count // 10:
        return np.repeat(vals, 10)
    raise ValueError(f'Cannot repeat event values of shape {vals.shape} to rows {row_count}')


def weekday_context_features(day, t, h, ht, event_indices):
    """Add weekday-aware user/item propensity and recent recency signals."""
    cutoff = int(t[:, 2].min()) + day * 86400
    prior = np.asarray(t[t[:, 2] < cutoff])
    if len(prior) == 0:
        raise ValueError('No prior rows available for weekday feature generation')

    event_ids = np.asarray(event_indices)[::10]
    row_indices = np.repeat(event_ids, 10) * 10 + np.tile(np.arange(10), len(event_ids))
    rows = np.asarray(t[row_indices])
    u_max = int(max(prior[:, 0].max(), rows[:, 0].max())) + 1
    i_max = int(max(prior[:, 1].max(), rows[:, 1].max())) + 1

    def local_rate(src_clicks, src_counts, keys):
        denom = np.maximum(src_counts[keys], 1)
        return src_clicks[keys] / denom

    weekday = prior[:, 4].astype(np.int32)
    user_clicks = np.bincount(prior[:, 0] * 2 + weekday, weights=prior[:, 5], minlength=u_max * 2)
    user_exposures = np.bincount(prior[:, 0] * 2 + weekday, minlength=u_max * 2)
    item_clicks = np.bincount(prior[:, 1] * 2 + weekday, weights=prior[:, 5], minlength=i_max * 2)
    item_exposures = np.bincount(prior[:, 1] * 2 + weekday, minlength=i_max * 2)

    user_weekday_ctr = local_rate(user_clicks, user_exposures, rows[:, 0] * 2 + rows[:, 4])
    item_weekday_ctr = local_rate(item_clicks, item_exposures, rows[:, 1] * 2 + rows[:, 4])

    recent_u = []
    recent_i = []
    for window in (1, 3, 7):
        recent_rows = prior[prior[:, 2] >= cutoff - window * 86400]
        if len(recent_rows) == 0:
            recent_u.append(np.zeros(len(rows), dtype='float32'))
            recent_i.append(np.zeros(len(rows), dtype='float32'))
            continue
        rw = recent_rows[:, 4].astype(np.int32)
        r_u_clicks = np.bincount(recent_rows[:, 0] * 2 + rw, weights=recent_rows[:, 5], minlength=u_max * 2)
        r_u_exposures = np.bincount(recent_rows[:, 0] * 2 + rw, minlength=u_max * 2)
        r_i_clicks = np.bincount(recent_rows[:, 1] * 2 + rw, weights=recent_rows[:, 5], minlength=i_max * 2)
        r_i_exposures = np.bincount(recent_rows[:, 1] * 2 + rw, minlength=i_max * 2)
        recent_u.append(local_rate(r_u_clicks, r_u_exposures, rows[:, 0] * 2 + rows[:, 4]))
        recent_i.append(local_rate(r_i_clicks, r_i_exposures, rows[:, 1] * 2 + rows[:, 4]))

    extra = np.column_stack([
        user_weekday_ctr,
        item_weekday_ctr,
        *recent_u,
        *recent_i,
    ]).astype('float32')
    return extra


def benchmark_day(day, train, h, ht, spline):
    z, x, users, groups, names = base_causal.features(day, train, h, ht, spline)
    extra = weekday_context_features(day, train, h, ht, z['events'])
    x_aug = np.hstack([x, extra])
    model = CatBoostClassifier(
        iterations=600,
        depth=7,
        learning_rate=0.06,
        loss_function='Logloss',
        eval_metric='AUC',
        thread_count=4,
        random_seed=6101,
        l2_leaf_reg=10.0,
        verbose=False,
        allow_writing_files=False,
        bootstrap_type='No',
        boosting_type='Plain',
    )
    model.fit(x_aug, z['y'])
    prob = model.predict_proba(x_aug)[:, 1]
    baseline = z['pred']
    report = {
        'day': day,
        'v53_auc': float(roc_auc_score(z['y'], baseline)),
        'weekday_causal_auc': float(roc_auc_score(z['y'], prob)),
        'gain': float(roc_auc_score(z['y'], prob) - roc_auc_score(z['y'], baseline)),
        'n_features': int(x_aug.shape[1]),
        'n_rows': int(x_aug.shape[0]),
        'model_iterations': int(model.tree_count_),
        'status': 'probe'
    }
    return report, model, x_aug, z


def main():
    tic = time.monotonic()
    train = np.load(R / 'train.npy', mmap_mode='r')
    h = np.load(R / 'train_hist.npy', mmap_mode='r')
    ht = np.load(R / 'train_times.npy', mmap_mode='r')
    spline = load_spline()

    reports = []
    for day in (5, 6):
        report, model, x_aug, z = benchmark_day(day, train, h, ht, spline)
        reports.append(report)
        np.savez_compressed(
            R / f'v61_weekday_causal_day{day}.npz',
            y=z['y'],
            events=z['events'],
            pred=model.predict_proba(x_aug)[:, 1],
            baseline=z['pred'],
            features=x_aug,
        )
        print(json.dumps(report), flush=True)

    result = {
        'stage': 'v61_weekday_causal',
        'reports': reports,
        'target_0_85': False,
        'overview': 'Weekday-aware causal extension of V53; strict benchmark should be run on the same first-future-event weekday holdout before claiming a score gain.',
        'elapsed_seconds': round(time.monotonic() - tic, 3),
    }
    (R / 'v61_weekday_causal_report.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
