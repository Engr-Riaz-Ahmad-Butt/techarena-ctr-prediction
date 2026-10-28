"""Independent saved-score audit, plus optional NumPy replay of all five models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def audit(run, artifacts, require_models=False):
    frozen = json.loads((run/'frozen.json').read_text())
    report = json.loads((run/'day6/report.json').read_text())
    z = np.load(run/'day6/predictions.npz')
    baseline = np.load(artifacts/'v53_causal_propensity_day6.npz')
    assert np.array_equal(z['y'], baseline['y'])
    assert np.array_equal(z['events'], baseline['events'])
    assert np.array_equal(z['baseline'], baseline['pred'])
    seeds = frozen['recipe']['seeds']
    assert seeds == [7201, 7202, 7203, 7204, 7205]
    predictions = [np.load(run/f'day6/seed{seed}_pred.npy') for seed in seeds]
    assert all(np.isfinite(p).all() and p.shape == z['y'].shape for p in predictions)
    mean = np.mean(np.stack(predictions).astype('float64'), axis=0)
    assert np.array_equal(mean, z['bag_raw'])
    assert np.array_equal(predictions[0], z['control_raw'])
    rank = lambda p: rankdata(p, method='average')/(len(p)+1)
    alpha = frozen['recipe']['blend_weight']
    assert alpha == .05
    blend_errors = {}
    for key, values in [('bag', mean), ('control', predictions[0])]:
        replay = (1-alpha)*rank(z['baseline'])+alpha*rank(values)
        blend_errors[key] = float(np.max(np.abs(z[key]-replay)))
        if key == 'bag':
            assert np.array_equal(z[key], replay)
        else:
            # Preserve the observed cross-environment discrepancy explicitly.
            # Do not claim bit-exact control arithmetic or alter saved scores.
            assert blend_errors[key] < 1e-8
    history = np.load(artifacts/'train_hist.npy', mmap_mode='r')
    evaluation = np.load(artifacts/'evaluation_hist.npy', mmap_mode='r')
    source = np.digitize((history[z['events']] >= 0).sum(1), [0, 4, 9, 19], right=True)
    target = np.digitize((evaluation >= 0).sum(1), [0, 4, 9, 19], right=True)
    weights = ((np.bincount(target, minlength=5)/len(target)) /
               (np.bincount(source, minlength=5)/len(source)))[source]
    np.testing.assert_allclose(z['weights'], weights, rtol=1e-12, atol=1e-12)
    checked = {}
    for key in ['baseline', 'control', 'bag', 'raw_control', 'raw_bag']:
        field = {'raw_control': 'control_raw', 'raw_bag': 'bag_raw'}.get(key, key)
        values = z[field]
        metric = dict(ordinary_auc=float(roc_auc_score(z['y'], values)),
                      adjusted_auc=float(roc_auc_score(z['y'], values, sample_weight=z['weights'])))
        for name in metric:
            assert abs(metric[name]-report[key][name]) < 1e-12
        checked[key] = metric
    result = dict(all_five_probability_arrays='PASS', fixed_mean_and_bag_blend='exact',
                  blend_maximum_errors=blend_errors,
                  control_blend_scope='Within 1e-8; not bit-exact across environments; cause not independently established',
                  independent_history_weights='PASS', saved_metrics='PASS',
                  baseline_event_label_identity='PASS', metrics=checked,
                  model_replay={}, fresh_holdout=False, promoted=False, target_achieved=False)
    paths = [run/f'day6/seed{seed}.pt' for seed in seeds]
    if require_models or all(path.exists() for path in paths):
        import torch
        expected = json.loads((run/'model_sha256.json').read_text())
        h = np.load(artifacts/'tied_history_full_001/validation_history.npy', mmap_mode='r')
        items = np.load(artifacts/'tied_history_full_001/validation_items.npy', mmap_mode='r')
        def raw_digest(values):
            view = memoryview(values).cast('B')
            digest = hashlib.sha256()
            for offset in range(0, len(view), 4*1024*1024):
                digest.update(view[offset:offset+4*1024*1024])
            return digest.hexdigest()
        hashes = json.loads((run/'day6/prepared_array_hashes.json').read_text())
        assert raw_digest(h) == hashes['val_h']
        assert raw_digest(items) == hashes['val_i']
        for seed, path, pred in zip(seeds, paths, predictions):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name]
            state = torch.load(path, map_location='cpu', weights_only=True)
            w, b, g = [state[key].numpy() for key in ['items.weight', 'bias', 'global_bias']]
            replay = []
            for offset in range(0, len(h), 512):
                hh = np.asarray(h[offset:offset+512])
                valid = hh >= 0
                profile = (w[np.maximum(hh, 0)]*valid[:, :, None]).sum(1) / np.maximum(valid.sum(1)[:, None], 1)
                ii = items[offset:offset+512]
                logits = (profile[:, None, :]*w[ii]).sum(-1)+b[ii]+g
                replay.append(expit(logits).ravel())
            error = float(np.max(np.abs(pred-np.concatenate(replay))))
            assert error < 1e-6
            result['model_replay'][str(seed)] = dict(status='PASS', maximum_probability_error=error)
    else:
        result['model_replay_scope'] = 'Model files not yet present locally; only saved probabilities and metrics audited'
    (run/'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--require-models', action='store_true')
    args = p.parse_args()
    print(json.dumps(audit(args.run, args.artifacts, args.require_models)), flush=True)
