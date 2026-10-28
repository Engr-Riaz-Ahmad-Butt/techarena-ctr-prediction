"""Independent NumPy equation replay plus sampled raw-history provenance audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    args = p.parse_args()
    r = args.run
    with np.load(r/'predictions.npz') as saved:
        predictions = saved['pred'].copy()
        blend = saved['blend'].copy()
        y, events, weights = [saved[k].copy() for k in ['y', 'events', 'weights']]
    manifest = json.loads((r/'frozen.json').read_text())
    def digest(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    for name, expected in manifest['input_hashes'].items():
        assert digest(args.artifacts/name) == expected
    for name, expected in manifest['source_hashes'].items():
        snapshot = r/'sources'/name
        assert digest(snapshot if snapshot.exists() else Path(__file__).with_name(name)) == expected
    report = json.loads((r/'report.json').read_text())
    state = torch.load(r/'model.pt', weights_only=True)
    w, b, g = [state[k].numpy() for k in ['items.weight', 'bias', 'global_bias']]
    h = np.load(r/'validation_history.npy', mmap_mode='r')
    items = np.load(r/'validation_items.npy', mmap_mode='r')
    replay = []
    for offset in range(0, len(h), 512):
        history = np.asarray(h[offset:offset+512])
        valid = history >= 0
        profile = (w[np.maximum(history, 0)]*valid[:, :, None]).sum(1) / np.maximum(valid.sum(1)[:, None], 1)
        candidates = items[offset:offset+512]
        logits = (profile[:, None, :]*w[candidates]).sum(-1) + b[candidates] + g
        replay.append(expit(logits).ravel())
    replay = np.concatenate(replay)
    error = float(np.max(np.abs(predictions-replay)))
    assert error < 1e-6
    with np.load(r/'comparator.npz') as reference:
        assert np.array_equal(y, reference['y'])
        assert np.array_equal(events, reference['events'])
        assert np.array_equal(weights, reference['weights'])
        rank = lambda x: rankdata(x, method='average')/(len(x)+1)
        alpha = manifest['recipe']['fixed_rank_blend_weight']
        assert np.array_equal(blend, (1-alpha)*rank(reference['pred']) + alpha*rank(predictions))
    computed = dict(ordinary_auc=float(roc_auc_score(y, blend)),
                    adjusted_auc=float(roc_auc_score(y, blend, sample_weight=weights)))
    assert computed == report['blend']
    # Direct raw-array oracle: rebuild known click triples for sampled users.
    t = np.load(args.artifacts/'train.npy', mmap_mode='r')
    hh = np.load(args.artifacts/'train_hist.npy', mmap_mode='r')
    times = np.load(args.artifacts/'train_times.npy', mmap_mode='r')
    heads = t[::10]
    cutoff = int(heads[:, 2].min()) + manifest['day']*86400
    fitting_ids = np.flatnonzero(heads[:, 2] < cutoff)
    rng = np.random.default_rng(7291)
    sampled = rng.choice(len(fitting_ids), 24, replace=False)
    fitting_contexts = np.load(r/'fit_history.npy', mmap_mode='r')
    for local in sampled:
        q = fitting_ids[local]
        user, timestamp = heads[q, :3:2]
        source_ids = fitting_ids[heads[fitting_ids, 0] == user]
        triples = set()
        for s in source_ids:
            triples.update((int(stamp), int(item)) for item, stamp in zip(hh[s], times[s])
                           if item >= 0 and stamp < timestamp)
            if heads[s, 2] < timestamp:
                rows = t[s*10:(s+1)*10]
                triples.update((int(heads[s, 2]), int(item)) for item in rows[rows[:, 5] == 1, 1])
        chosen = sorted(triples, reverse=True)[:manifest['recipe']['context_limit']]
        expected = np.full(manifest['recipe']['context_limit'], -1, np.int32)
        expected[:len(chosen)] = [item for _, item in chosen]
        assert np.array_equal(expected, fitting_contexts[local])
    result = dict(numpy_equation_replay='PASS', maximum_probability_error=error,
                  frozen_inputs_and_source_hashes='PASS',
                  full_blend_and_metric_replay='PASS', raw_fit_context_samples=24,
                  raw_fit_context_oracle='PASS', auxiliary_causality_independently_rebuilt=False,
                  validation_status='reused exploratory', promoted=False, target_achieved=False,
                  blend=computed)
    (r/'independent_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
