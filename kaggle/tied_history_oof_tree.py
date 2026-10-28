"""Full-row temporal OOF factor features with a matched pointwise tree control.

Each day's factors fit strictly earlier events. Tree arms share rows, weights
and settings. Reused validation remains exploratory; no submission is exported.
"""
import gc
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from scipy.special import logit, softmax
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import tied_history_gpu_bag as bag

ROUNDS = 800
PARAMS = dict(objective='binary:logistic', tree_method='hist', device='cuda:0',
              max_bin=63, max_depth=6, eta=.05, min_child_weight=50,
              subsample=.8, colsample_bytree=.9, reg_lambda=5., reg_alpha=.1,
              nthread=2, seed=7501, max_cached_hist_node=32)
BLEND = .1
NAMES = ['factor_probability', 'factor_logit', 'seed_probability_std',
         'mean_dot', 'cosine', 'profile_norm', 'candidate_norm', 'candidate_bias',
         'maximum_history_dot', 'history_dot_std', 'log_full_history_count',
         'slate_centered_logit', 'slate_max_relative_logit', 'slate_logit_std',
         'slate_probability_sum', 'slate_softmax', 'factor_percentile']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def first_ids(data, cutoff):
    ids = np.flatnonzero((data['timestamp'] >= cutoff) & (data['timestamp'] < cutoff+86400))
    _, first = np.unique(data['users'][ids], return_index=True)
    return ids[first]


def score_contexts(data, stamps, cutoff, ids):
    """Only prefix labels and each scored event's supplied prior history."""
    ledger = defaultdict(set)
    for q in np.flatnonzero(data['timestamp'] < cutoff):
        user = int(data['users'][q])
        for item, stamp in zip(data['hist'][q], stamps[q]):
            if item >= 0:
                assert stamp < data['timestamp'][q] < cutoff
                ledger[user].add((int(stamp), int(item)))
        target = int(data['target'][q])
        if target < 10:
            ledger[user].add((int(data['timestamp'][q]), int(data['items'][q, target])))
    ledger = {user: tuple(sorted(values, reverse=True)[:100]) for user, values in ledger.items()}
    contexts = np.full((len(ids), 100), -1, np.int32)
    for n, q in enumerate(ids):
        assert data['timestamp'][q] >= cutoff
        values = set(ledger.get(int(data['users'][q]), ()))
        for item, stamp in zip(data['hist'][q], stamps[q]):
            if item >= 0:
                assert stamp < data['timestamp'][q]
                values.add((int(stamp), int(item)))
        contexts[n] = bag.pad(values)
    return contexts


def geometry(model, history, items):
    valid = history >= 0
    vectors = model.items(history.clamp_min(0))*valid.unsqueeze(-1)
    count = valid.sum(1).clamp_min(1)
    profile = vectors.sum(1)/count[:, None]
    candidates = model.items(items)
    dot = (profile[:, None, :]*candidates).sum(-1)
    pn = torch.linalg.vector_norm(profile, dim=-1)
    cn = torch.linalg.vector_norm(candidates, dim=-1)
    cosine = dot/(pn[:, None]*cn).clamp_min(1e-9)
    pair = torch.einsum('bhd,bcd->bch', vectors, candidates)
    maximum = pair.masked_fill(~valid[:, None, :], -1e30).max(2).values
    maximum[~valid.any(1)] = 0
    variance = ((pair*pair).sum(2)/count[:, None]-dot*dot).clamp_min(0)
    return torch.stack([dot, cosine, pn[:, None].expand_as(dot), cn,
                        model.bias[items], maximum, variance.sqrt()], dim=-1).reshape(-1, 7)


def extra_features(probabilities, mean_geometry, history):
    all_p = np.stack(probabilities).astype('float64')
    probability = all_p.mean(0)
    logits = logit(np.clip(probability, 1e-6, 1-1e-6)).reshape(-1, 10)
    p = probability.reshape(-1, 10)
    values = [probability, logits.ravel(), all_p.std(0)]
    values.extend(mean_geometry[:, i] for i in range(7))
    values.extend([np.repeat(np.log1p((history >= 0).sum(1)), 10),
                   (logits-logits.mean(1, keepdims=True)).ravel(),
                   (logits-logits.max(1, keepdims=True)).ravel(),
                   np.repeat(logits.std(1), 10), np.repeat(p.sum(1), 10),
                   softmax(logits, axis=1).ravel(),
                   rankdata(probability, method='average')/(len(probability)+1)])
    result = np.column_stack(values).astype('float32')
    assert result.shape[1] == len(NAMES) and np.isfinite(result).all()
    return result


def build_day(data, stamps, neural_root, output, day):
    output.mkdir()
    cutoff = int(data['timestamp'][0])+day*86400
    first = first_ids(data, cutoff)
    validation = dict(indices=first, y=(data['target'][first, None] == np.arange(10)).astype('float32').ravel())
    if day in [5, 6]:
        with np.load(neural_root/f'val{day}.npz') as original:
            assert np.array_equal(first, original['indices'])
            assert np.array_equal(validation['y'], original['y'])
    prepared = bag.prepare(data, stamps, validation, cutoff)
    if day in [5, 6]:
        checked = bag.verify_prepared(prepared, day)
        (output/'prepared_hashes.json').write_text(json.dumps(checked, indent=2))
    ids = first if day == 6 else np.flatnonzero((data['timestamp'] >= cutoff) & (data['timestamp'] < cutoff+86400))
    history = score_contexts(data, stamps, cutoff, ids)
    # Confirm scoring contexts on the first-event subset equal the prior audit.
    if day == 6:
        assert np.array_equal(history, prepared['val_h'])
    else:
        assert np.array_equal(history[np.searchsorted(ids, first)], prepared['val_h'])
    prepared['val_h'], prepared['val_i'] = history, data['items'][ids]
    for name in ['fit_h', 'fit_i', 'fit_y', 'aux_h', 'aux_target', 'val_h', 'val_i']:
        prepared[name] = torch.as_tensor(prepared[name], device='cuda:0')
    predictions, mean_geometry = [], None
    for seed in bag.RECIPE['seeds']:
        predictions.append(bag.one_seed(prepared, seed, output, torch.device('cuda:0')))
        model = bag.BagFactor(prepared['n_items'], bag.RECIPE['dimensions']).cuda()
        model.load_state_dict(torch.load(output/f'seed{seed}.pt', weights_only=True))
        pieces = []
        with torch.no_grad():
            for off in range(0, len(ids), 512):
                pieces.append(geometry(model, prepared['val_h'][off:off+512].long(),
                                       prepared['val_i'][off:off+512].long()).cpu().numpy())
        values = np.concatenate(pieces).astype('float64')
        mean_geometry = values if mean_geometry is None else mean_geometry+values
        del model, pieces, values
        gc.collect()
        torch.cuda.empty_cache()
    extra = extra_features(predictions, mean_geometry/len(predictions), history)
    np.save(output/'features.npy', extra)
    np.save(output/'event_ids.npy', ids)
    labels = (data['target'][ids, None] == np.arange(10)).astype('float32').ravel()
    np.save(output/'labels.npy', labels)
    report = dict(day=day, factor_fit_last_timestamp=int(data['timestamp'][data['timestamp'] < cutoff].max()),
                  cutoff=cutoff, score_first_timestamp=int(data['timestamp'][ids].min()),
                  scored_events=len(ids), feature_count=extra.shape[1], feature_sha256=bag.digest(output/'features.npy'),
                  checkpoint_replay='all_five_exact_all_scored_rows', labels_used_for_factor_fit='strict prefix only')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    del prepared, predictions, mean_geometry
    gc.collect()
    torch.cuda.empty_cache()


class Augmented:
    def __init__(self, base, paths):
        self.base = base
        self.parts = [np.load(path, mmap_mode='r') for path in paths]
        assert self.parts and all(part.ndim == 2 and part.shape[1] == len(NAMES) for part in self.parts)
        self.ends = np.cumsum([len(part) for part in self.parts])
        self.starts = np.r_[0, self.ends[:-1]]
        assert len(base) == int(self.ends[-1])

    def __len__(self):
        return len(self.base)

    def block(self, start, end):
        assert 0 <= start < end <= len(self)
        pieces = []
        for left, right, part in zip(self.starts, self.ends, self.parts):
            a, b = max(start, int(left)), min(end, int(right))
            if b > a:
                pieces.append(np.asarray(part[a-int(left):b-int(left)]))
        return np.column_stack([self.base.block(start, end), np.concatenate(pieces)])


class Iterator(xgb.DataIter):
    def __init__(self, features, labels, weights, cp):
        super().__init__(release_data=True)
        self.features, self.labels, self.weights, self.cp = features, labels, weights, cp
        self.position = 0

    def reset(self):
        self.position = 0

    def next(self, input_data):
        if self.position >= len(self.labels):
            return False
        stop = min(self.position+100000, len(self.labels))
        start = self.position
        input_data(data=self.cp.asarray(self.features.block(start, stop)),
                   label=self.cp.asarray(self.labels[start:stop]),
                   weight=self.cp.asarray(self.weights[start:stop]))
        self.position = stop
        return True


class Progress(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        if (epoch+1) % 100 == 0:
            print(json.dumps(dict(stage='tree_training', round=epoch+1, total=ROUNDS)), flush=True)
        return False


def target_share(data, neural_root):
    with np.load(neural_root/'val6.npz') as z:
        groups = np.digitize((data['hist'][z['indices']] >= 0).sum(1), [0, 4, 9, 19], right=True)
        source = np.bincount(groups, minlength=5)/len(groups)
        weights = z['weights'].reshape(-1, 10)[:, 0]
        ratio = np.array([weights[groups == group][0] for group in range(5)])
        assert np.allclose(weights, ratio[groups], rtol=1e-12, atol=1e-12)
        result = source*ratio
        assert np.isclose(result.sum(), 1)
        return result


def fit_trees(root, neural_root, output, data, manifest, helpers, day):
    import cupy as cp
    output.mkdir()
    days = list(range(day-3, day))
    specs = [manifest['train'][str(d)] for d in days]
    labels, ids = [], []
    start = int(data['timestamp'][0])
    for d, spec in zip(days, specs):
        meta = json.loads((root/spec['metadata']).read_text())
        assert meta['last_timestamp'] < start+day*86400 and meta['feature_cutoff'] == start+d*86400
        event_ids = np.load(output.parent/f'factors{d}/event_ids.npy')
        y = np.load(root/spec['labels'])
        assert np.array_equal(y, (data['target'][event_ids, None] == np.arange(10)).ravel())
        labels.append(y)
        ids.append(event_ids)
    labels, ids = np.concatenate(labels).astype('float32'), np.concatenate(ids)
    groups = np.digitize((data['hist'][ids] >= 0).sum(1), [0, 4, 9, 19], right=True)
    shares = target_share(data, neural_root)
    source = np.bincount(groups, minlength=5)/len(groups)
    assert (source > 0).all()
    weights = np.repeat((shares/source)[groups], 10).astype('float32')
    weights /= weights.mean()
    base = helpers.FeatureBlocks(root, [spec['features'] for spec in specs])
    treatment = Augmented(base, [output.parent/f'factors{d}/features.npy' for d in days])
    validation = manifest['validation'][str(day)]
    val = np.load(root/validation['data'])
    val_base = helpers.FeatureBlocks(root, [validation['features']])
    score_ids = np.load(output.parent/f'factors{day}/event_ids.npy')
    desired = val['events'][::10].astype('int64')
    if day == 6:
        assert np.array_equal(desired, score_ids)
        extra = np.load(output.parent/f'factors{day}/features.npy', mmap_mode='r')
    else:
        positions = np.searchsorted(score_ids, desired)
        assert np.array_equal(score_ids[positions], desired)
        extra = np.load(output.parent/f'factors{day}/features.npy', mmap_mode='r').reshape(-1, 10, len(NAMES))[positions].reshape(-1, len(NAMES))
    extra_path = output/'validation_extra.npy'
    np.save(extra_path, extra)
    val_augmented = Augmented(val_base, [extra_path])
    predictions = {}
    for arm, features, future in [('control', base, val_base), ('treatment', treatment, val_augmented)]:
        iterator = Iterator(features, labels, weights, cp)
        matrix = xgb.QuantileDMatrix(iterator, nthread=2, max_bin=63)
        assert matrix.num_row() == len(labels)
        model = xgb.train(PARAMS, matrix, num_boost_round=ROUNDS, callbacks=[Progress()])
        assert model.num_boosted_rounds() == ROUNDS
        model.save_model(output/f'{arm}.ubj')
        del matrix, iterator
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        def predict(member):
            values = []
            for off in range(0, len(future), 25000):
                block = cp.asarray(future.block(off, min(off+25000, len(future))))
                values.append(cp.asnumpy(member.inplace_predict(block)))
            return np.concatenate(values)
        raw = predict(model)
        replay = xgb.Booster()
        replay.load_model(output/f'{arm}.ubj')
        replay.set_param({'device': 'cuda:0', 'nthread': 2})
        assert np.array_equal(raw, predict(replay))
        predictions[arm+'_raw'] = raw
        rank = lambda p: rankdata(np.asarray(p, dtype='float64'), method='average')/(len(p)+1)
        predictions[arm] = (1-BLEND)*rank(val['baseline'])+BLEND*rank(raw)
        print(json.dumps(dict(stage='tree_complete', day=day, arm=arm, model_replay='exact_all_rows')), flush=True)
        del model, replay
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    def metric(p):
        return dict(ordinary_auc=float(roc_auc_score(val['y'], p)),
                    adjusted_auc=float(roc_auc_score(val['y'], p, sample_weight=val['weights'])))
    report = dict(day=day, fitting_days=days, fitting_rows=len(labels), baseline=metric(val['baseline']),
                  control=metric(predictions['control']), treatment=metric(predictions['treatment']))
    report['adjusted_gain'] = report['treatment']['adjusted_auc']-report['baseline']['adjusted_auc']
    report['gain_vs_control'] = report['treatment']['adjusted_auc']-report['control']['adjusted_auc']
    report['ordinary_nonregression'] = report['treatment']['ordinary_auc'] >= report['baseline']['ordinary_auc']
    report['development_gate'] = (report['adjusted_gain'] >= .0005 and report['gain_vs_control'] > 0
                                  and report['ordinary_nonregression'])
    np.savez_compressed(output/'predictions.npz', y=val['y'], events=val['events'],
                        weights=val['weights'], baseline=val['baseline'], **predictions)
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def run(feature_root, neural_root, output):
    feature_root, neural_root, output = map(Path, [feature_root, neural_root, output])
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    frozen = dict(factor_recipe=bag.RECIPE, tree_params=PARAMS, rounds=ROUNDS, blend=BLEND,
                  feature_names=NAMES, source_sha256=bag.digest(__file__),
                  factor_source_sha256=bag.digest(bag.__file__),
                  feature_manifest_sha256=bag.digest(feature_root/'bundle_manifest.json'),
                  gpu=torch.cuda.get_device_name(0), cuda_available=True,
                  fresh_holdout=False, promoted=False, target_achieved=False,
                  scope='Full-row temporal OOF, not the previous first-event meta stack; no validation tuning')
    (output/'frozen.json').write_text(json.dumps(frozen, indent=2))
    print(json.dumps(frozen), flush=True)
    helpers = module('oof_bundle_helpers', feature_root/'runner.py')
    manifest = helpers.load_bundle(feature_root, verify=True)
    data, stamps = bag.load(neural_root)
    for day in [3, 4, 5, 6]:
        build_day(data, stamps, neural_root, output/f'factors{day}', day)
    reports = [fit_trees(feature_root, neural_root, output/'trees6', data, manifest, helpers, 6)]
    if reports[0]['development_gate']:
        build_day(data, stamps, neural_root, output/'factors2', 2)
        reports.append(fit_trees(feature_root, neural_root, output/'trees5', data, manifest, helpers, 5))
    result = dict(reports=reports, day5_run=len(reports)==2, promoted=False, csv_ready=False,
                  fresh_holdout=False, official_auc=None, target_achieved=False,
                  elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result
