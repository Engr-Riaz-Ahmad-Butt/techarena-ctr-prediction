"""Fixed causal factor stacking; day5 user-crossfit then day6 future scoring.

Control and treatment share the same small tree recipe. Factor fitting ends
before each feature cohort begins. Reused cohorts remain exploratory.
Day5 meta-crossfit conditions on frozen parent OOF predictions; it does not
retrain all parent ancestors within each meta fold and is not fully nested CV.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from catboost import CatBoostClassifier
from scipy.special import expit, logit, softmax
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

RECIPE = dict(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=50,
              random_strength=0, random_seed=7301, thread_count=2,
              loss_function='Logloss', allow_writing_files=False, verbose=False)
BLEND = 0.1


def rank(values):
    return rankdata(values, method='average')/(len(values)+1)


def invariant_features(weight, bias, global_bias, history, candidates):
    """Only rotation-invariant scalar factor features can transfer across dates."""
    valid = history >= 0
    vectors = weight[np.maximum(history, 0)]*valid[:, :, None]
    count = valid.sum(1)
    profile = vectors.sum(1)/np.maximum(count[:, None], 1)
    candidate_vectors = weight[candidates]
    mean_dot = (profile[:, None, :]*candidate_vectors).sum(-1)
    pn = np.linalg.norm(profile, axis=1)
    cn = np.linalg.norm(candidate_vectors, axis=2)
    cosine = mean_dot/np.maximum(pn[:, None]*cn, 1e-9)
    pairs = np.einsum('bhd,bcd->bch', vectors, candidate_vectors)
    maximum = np.where(valid[:, None, :], pairs, -1e30).max(2)
    maximum[count == 0] = 0
    variance = np.maximum((pairs*pairs).sum(2)/np.maximum(count[:, None], 1)-mean_dot**2, 0)
    logits = mean_dot+bias[candidates]+global_bias
    probability = expit(logits)
    columns = [logits, probability, cosine, np.broadcast_to(pn[:, None], logits.shape),
               cn, bias[candidates], maximum, np.sqrt(variance),
               logits-logits.mean(1, keepdims=True), logits-logits.max(1, keepdims=True),
               softmax(logits, axis=1), np.broadcast_to(probability.sum(1, keepdims=True), logits.shape)]
    return np.stack(columns, axis=-1).reshape(-1, len(columns))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def features(root, run, day):
    frozen = json.loads((run/'frozen.json').read_text())
    assert frozen['day'] == day
    audit = json.loads((run/'independent_audit.json').read_text())
    assert audit['numpy_equation_replay'] == 'PASS'
    for name, expected in frozen['input_hashes'].items():
        assert digest(root/name) == expected
    t = np.load(root/'train.npy', mmap_mode='r')
    h = np.load(root/'train_hist.npy', mmap_mode='r')
    ht = np.load(root/'train_times.npy', mmap_mode='r')
    ref = np.load(root/f'v53_causal_propensity_day{day}.npz')
    z = np.load(run/'predictions.npz')
    assert np.array_equal(ref['events'], z['events']) and np.array_equal(ref['y'], z['y'])
    event_ids = z['events'][::10].astype(np.int64)
    cutoff = int(t[0, 2]) + day*86400
    rows = t[event_ids*10]
    assert (rows[:, 2] >= cutoff).all() and (rows[:, 2] < cutoff+86400).all()
    history = np.asarray(h[event_ids])
    stamps = np.asarray(ht[event_ids])
    assert np.all((stamps < rows[:, 2, None]) | (history < 0))
    count = (history >= 0).sum(1)
    age = (rows[:, 2] - np.where(history >= 0, stamps, 0).max(1))/86400
    age[count == 0] = 30
    control = np.column_stack([ref['pred'], logit(np.clip(ref['pred'], 1e-5, 1-1e-5)),
                               logit(np.clip(ref['calibrated'], 1e-5, 1-1e-5)),
                               np.repeat(count, 10), np.repeat(np.log1p(age), 10),
                               np.repeat(rows[:, 3], 10), np.repeat(rows[:, 4], 10)])
    state = torch.load(run/'model.pt', weights_only=True)
    weight, bias, global_bias = [state[k].numpy() for k in ['items.weight', 'bias', 'global_bias']]
    full_history = np.load(run/'validation_history.npy', mmap_mode='r')
    items = np.load(run/'validation_items.npy', mmap_mode='r')
    extra = np.concatenate([invariant_features(weight, bias, global_bias,
                                              full_history[i:i+512], items[i:i+512])
                            for i in range(0, len(full_history), 512)])
    assert np.max(np.abs(extra[:, 1]-z['pred'])) < 1e-6
    extra = np.column_stack([extra, rank(z['pred'])])
    treatment = np.column_stack([control, extra])
    assert np.isfinite(control).all() and np.isfinite(treatment).all()
    return dict(control=control, treatment=treatment, y=z['y'], events=z['events'],
                baseline=ref['pred'], weights=z['weights'], users=np.repeat(rows[:, 0], 10))


def metrics(data, values):
    return dict(ordinary_auc=float(roc_auc_score(data['y'], values)),
                adjusted_auc=float(roc_auc_score(data['y'], values, sample_weight=data['weights'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--day5-run', type=Path, required=True)
    parser.add_argument('--day6-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = [args.day5_run/p for p in ['model.pt', 'predictions.npz', 'frozen.json']]
    sources += [args.day6_run/p for p in ['model.pt', 'predictions.npz', 'frozen.json']]
    frozen = dict(recipe=RECIPE, blend=BLEND, seed=7301, folds=2,
                  inputs={str(p): digest(p) for p in sources}, source_sha256=digest(__file__),
                  selection='No parameter selection; compare treatment with baseline and matched control on both reused cohorts.')
    (args.output/'frozen.json').write_text(json.dumps(frozen, indent=2))
    data5 = features(args.artifacts, args.day5_run, 5)
    data6 = features(args.artifacts, args.day6_run, 6)
    users = np.unique(data5['users'])
    rng = np.random.default_rng(7301)
    chosen = rng.permutation(users)[:len(users)//2]
    first = np.isin(data5['users'], chosen)
    assert not np.intersect1d(data5['users'][first], data5['users'][~first]).size
    predictions = {5: {}, 6: {}}
    for arm in ['control', 'treatment']:
        oof = np.full(len(data5['y']), np.nan)
        for fold, fit in enumerate([first, ~first]):
            model = CatBoostClassifier(**RECIPE)
            model.fit(data5[arm][fit], data5['y'][fit], sample_weight=data5['weights'][fit])
            oof[~fit] = model.predict_proba(data5[arm][~fit])[:, 1]
            path = args.output/f'{arm}_fold{fold}.cbm'
            model.save_model(path)
            replay = CatBoostClassifier().load_model(path)
            assert np.array_equal(oof[~fit], replay.predict_proba(data5[arm][~fit])[:, 1])
        assert np.isfinite(oof).all()
        model = CatBoostClassifier(**RECIPE)
        model.fit(data5[arm], data5['y'], sample_weight=data5['weights'])
        model.save_model(args.output/f'{arm}_full.cbm')
        future = model.predict_proba(data6[arm])[:, 1]
        replay = CatBoostClassifier().load_model(args.output/f'{arm}_full.cbm')
        assert np.array_equal(future, replay.predict_proba(data6[arm])[:, 1])
        predictions[5][arm] = (1-BLEND)*rank(data5['baseline']) + BLEND*rank(oof)
        predictions[6][arm] = (1-BLEND)*rank(data6['baseline']) + BLEND*rank(future)
        print(json.dumps(dict(stage='tree_arm_complete', arm=arm, elapsed_seconds=time.monotonic()-started)), flush=True)
    reports = []
    for day, data in [(5, data5), (6, data6)]:
        report = dict(day=day, baseline=metrics(data, data['baseline']),
                      control=metrics(data, predictions[day]['control']),
                      treatment=metrics(data, predictions[day]['treatment']))
        report['adjusted_gain'] = report['treatment']['adjusted_auc']-report['baseline']['adjusted_auc']
        report['adjusted_gain_vs_control'] = report['treatment']['adjusted_auc']-report['control']['adjusted_auc']
        reports.append(report)
        np.savez_compressed(args.output/f'day{day}.npz', y=data['y'], events=data['events'],
                            weights=data['weights'], baseline=data['baseline'], **predictions[day])
    result = dict(reports=reports, checkpoint_replay='exact_all_rows_all_six_models',
                  promoted=False, target_achieved=False, fresh_holdout=False, csv_ready=False,
                  ordinary_scope='Day5 conditional meta-user-crossfit on frozen parent OOF arrays, not nested parent CV; day6 future; both reused',
                  development_gain_gate=all(r['adjusted_gain'] >= 0.0005 and
                                           r['adjusted_gain_vs_control'] > 0 for r in reports),
                  elapsed_seconds=time.monotonic()-started)
    (args.output/'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
