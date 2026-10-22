"""Independent saved-LightGBM prediction, blend and score audit; no fitting."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from tst2_candidate import influence, weighted_influence


def settings():
    tree = ast.parse((ROOT / 'imp10_leafwise.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in n.targets)]
    ns = {}
    exec(compile(tree, 'imp10_leafwise.py', 'exec'), ns)
    return ns['SETTINGS']


def check(day, selected):
    rawfile = np.load(ARTIFACT_DIR / f'imp10_leafwise_raw_day{day}.npz')
    baseline = np.load(ARTIFACT_DIR / f'predictions_imp5_warmcontext_day{day}.npz')
    y, events, raw, base = baseline['y'], baseline['events'], rawfile['raw'], baseline['pred']
    assert np.array_equal(rawfile['events'], events) and np.array_equal(rawfile['y'], y)
    groups = events.reshape(-1, 10)
    assert (groups == groups[:, :1]).all() and len(np.unique(groups[:, 0])) == len(groups)
    assert (y.reshape(-1, 10).sum(1) <= 1).all()
    assert np.isfinite(raw).all() and ((raw > 0) & (raw < 1)).all()
    model = lgb.Booster(model_file=str(ARTIFACT_DIR / f'imp10_leafwise_d{day}.txt'))
    assert model.num_feature() == 290
    expected = settings()
    for key in ['num_leaves', 'learning_rate', 'min_data_in_leaf', 'feature_fraction', 'bagging_fraction',
                'bagging_freq', 'lambda_l2', 'max_bin', 'seed', 'num_threads']:
        assert abs(float(model.params[key]) - float(expected[key])) < 1e-12, (key, model.params[key])
    assert model.params['objective'] == 'binary' and 'auc' in model.params['metric']
    assert model.params['deterministic'] and model.params['force_col_wise']
    ix = (np.linspace(0, len(groups) - 1, 100, dtype=int)[:, None] * 10 + np.arange(10)).ravel()
    files = [f'stack_val{day}_c{day}.npy', f'imp3_wf_val{day}.npy', f'imp4_decompose_val{day}.npy',
             f'imp5_aligned_val{day}.npy', f'imp6_context_val{day}.npy']
    matrices = [np.load(ARTIFACT_DIR / 'features' / filename, mmap_mode='r') for filename in files]
    assert [x.shape for x in matrices] == [(len(y), 190), (len(y), 4), (len(y), 32), (len(y), 32), (len(y), 32)]
    features = np.column_stack([x[ix] for x in matrices])
    replay = model.predict(features, num_threads=1)
    model_error = float(np.max(np.abs(replay - raw[ix])))
    assert model_error < 1e-12, model_error
    h = np.load(ARTIFACT_DIR / 'train_hist.npy', mmap_mode='r')
    eh = np.load(ARTIFACT_DIR / 'evaluation_hist.npy', mmap_mode='r')
    vg = np.digitize((h[events] >= 0).sum(1), [0, 4, 9, 19], right=True)
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    sw = ((np.bincount(eg, minlength=5) / len(eg)) / (np.bincount(vg, minlength=5) / len(vg)))[vg]
    result = dict(day=day, rows=len(y), trees=model.num_trees(), features=model.num_feature(),
                  raw_auc=float(roc_auc_score(y, raw)), raw_weighted_auc=float(roc_auc_score(y, raw, sample_weight=sw)),
                  replay_rows=len(ix), saved_model_replay_max_error=model_error)
    if selected is None:
        result['selection'] = 'NO_ACCEPTED_CANDIDATE'
        return result
    d = np.load(ARTIFACT_DIR / f'imp10_leafwise_day{day}.npz')
    assert np.array_equal(d['events'], events) and np.array_equal(d['y'], y)
    assert np.array_equal(d['raw'], raw)
    assert model.num_trees() == selected['trees']
    weight = selected['weight']
    pred = (1 - weight) * base + weight * raw
    error = float(np.max(np.abs(pred - d['pred'])))
    assert error < 1e-15
    assert np.isfinite(pred).all() and ((pred > 0) & (pred < 1)).all()
    result.update(weight=weight, recipe_max_error=error)
    for name, weights, fn in [('auc', None, influence), ('weighted_auc', sw, weighted_influence)]:
        old, new = roc_auc_score(y, base, sample_weight=weights), roc_auc_score(y, pred, sample_weight=weights)
        args = () if weights is None else (weights,)
        se = np.linalg.norm(fn(y, pred, *args) - fn(y, base, *args))
        result[name] = dict(baseline=float(old), candidate=float(new), delta=float(new-old),
                            conditional_paired_event_95pct_interval=[float(new-old-1.96*se), float(new-old+1.96*se)])
    if day == 6:
        choices = []
        for w in [.1, .25, .5, .75, 1.]:
            candidate = (1 - w) * base + w * raw
            auc = roc_auc_score(y, candidate)
            weighted = roc_auc_score(y, candidate, sample_weight=sw)
            if auc > result['auc']['baseline'] and weighted > result['weighted_auc']['baseline']:
                choices.append((auc, w))
        assert choices and max(choices)[1] == weight
    return result


if __name__ == '__main__':
    selected = json.loads((ARTIFACT_DIR / 'imp10_leafwise_report.json').read_text())
    reports = []
    byday = {row['day']: row for row in selected if row is not None}
    for day in [6, 5]:
        if (ARTIFACT_DIR / f'imp10_leafwise_raw_day{day}.npz').exists():
            reports.append(check(day, byday.get(day)))
    assert reports
    if 6 in byday and 5 in byday:
        assert byday[6]['trees'] == byday[5]['trees'] and byday[6]['weight'] == byday[5]['weight']
    both_gain = len(reports) == 2 and all('auc' in r and r['auc']['delta'] > 0 and r['weighted_auc']['delta'] > 0 for r in reports)
    result = dict(status='PASS', reports=reports, both_dates_improved=both_gain,
                  note='Conditional intervals exclude reused-validation model/iteration/blend selection uncertainty.')
    (ARTIFACT_DIR / 'imp10_test_candidate_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
