"""Frozen full-prefix factor experiment with click-log auxiliary training.

Exploratory only: reused day6 cannot establish an official score or promotion.
No validation early stopping, checkpoint selection, or competition CSV export.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from causal_click_ledger import build_ledger, context_before, other_click_context
from tied_history_model import TiedHistoryModel

RECIPE = dict(steps=2000, batch_size=1024, auxiliary_batch_size=512,
              auxiliary_weight=0.1, dimensions=32, learning_rate=0.01,
              sampled_vector_regularization=0.002, global_weight_decay=0,
              negatives=9, context_limit=100, seed=7201, threads=2,
              fixed_rank_blend_weight=0.05)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pad(values, limit=100):
    result = np.full(limit, -1, np.int32)
    chosen = sorted(values, reverse=True)[:limit]
    result[:len(chosen)] = [item for _, item in chosen]
    return result


def regularization(model, history, candidates):
    """Regularize sampled vectors only, not every item on every minibatch."""
    valid = history >= 0
    squared = model.items.weight[history.clamp_min(0)].square().sum(-1)
    history_term = (squared*valid).sum() / valid.sum().clamp_min(1)
    return (history_term + model.items(candidates).square().sum(-1).mean()
            + model.bias[candidates].square().mean())


def prepare(root, output, day):
    t = np.load(root/'train.npy', mmap_mode='r')
    h = np.load(root/'train_hist.npy', mmap_mode='r')
    ht = np.load(root/'train_times.npy', mmap_mode='r')
    heads = t[::10]
    assert np.array_equal(heads[:, 6], np.arange(len(heads)))
    cutoff = int(heads[:, 2].min()) + day*86400
    fit_ids = np.flatnonzero(heads[:, 2] < cutoff)
    def events():
        for count, q in enumerate(fit_ids):
            row = heads[q]
            history = [(int(i), int(s)) for i, s in zip(h[q], ht[q]) if i >= 0]
            rows = t[q*10:(q+1)*10]
            yield int(row[0]), int(row[2]), history, rows[rows[:, 5] == 1, 1].tolist()
            if count and count % 100000 == 0:
                print(json.dumps(dict(stage='ledger', events=count)), flush=True)
    ledger = build_ledger(events(), cutoff)
    reference = np.load(root/f'v53_causal_propensity_day{day}.npz')
    val_ids = reference['events'][::10].astype(np.int64)
    assert np.array_equal(reference['events'], np.repeat(val_ids, 10))
    assert (heads[val_ids, 2] >= cutoff).all()
    assert len(np.unique(heads[val_ids, 0])) == len(val_ids)
    # V53's stored cohorts use each user's first event within the next UTC day.
    # Keep that same daily horizon on day5 as well as day6; neither is fresh.
    future = np.flatnonzero((heads[:, 2] >= cutoff) & (heads[:, 2] < cutoff+86400))
    _, first = np.unique(heads[future, 0], return_index=True)
    assert np.array_equal(np.sort(future[first]), np.sort(val_ids))
    def contexts(ids, validation=False):
        result = np.full((len(ids), RECIPE['context_limit']), -1, np.int32)
        for n, q in enumerate(ids):
            row = heads[q]
            values = set(context_before(ledger, int(row[0]), int(row[2])))
            if validation:
                assert np.all((ht[q] < row[2]) | (h[q] < 0))
                values.update((int(s), int(i)) for i, s in zip(h[q], ht[q]) if i >= 0)
            result[n] = pad(values)
        return result
    fh = contexts(fit_ids)
    vh = contexts(val_ids, True)
    def candidate_arrays(ids):
        indices = ids[:, None]*10 + np.arange(10)
        items = np.asarray(t[indices, 1], dtype=np.int32)
        labels = np.asarray(t[indices, 5], dtype=np.float32)
        assert np.isin(labels, [0, 1]).all() and (labels.sum(1) <= 1).all()
        return items, labels
    fi, fy = candidate_arrays(fit_ids)
    vi, vy = candidate_arrays(val_ids)
    assert np.array_equal(vy.ravel(), reference['y'])
    # These are prefix-only auxiliary examples, never impression test contexts.
    targets, ah = [], []
    for user, clicks in ledger.items():
        for target in clicks:
            values = other_click_context(ledger, user, target, cutoff)
            if values:
                targets.append(target[1])
                ah.append(pad(values))
    ah = np.asarray(ah, dtype=np.int32)
    targets = np.asarray(targets, dtype=np.int32)
    n_items = max(int(fi.max()), int(vi.max()), int(h.max()))+1
    exposure = np.bincount(fi.ravel(), minlength=n_items).astype(np.float64)
    probability = exposure/exposure.sum()
    assert np.isfinite(probability).all() and np.isclose(probability.sum(), 1)
    eh = np.load(root/'evaluation_hist.npy', mmap_mode='r')
    vg = np.digitize((h[reference['events']] >= 0).sum(1), [0, 4, 9, 19], right=True)
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    vf = np.bincount(vg, minlength=5)/len(vg)
    ef = np.bincount(eg, minlength=5)/len(eg)
    assert (vf > 0).all()
    weights = (ef/vf)[vg]
    for name, array in dict(fit_history=fh, fit_items=fi, fit_labels=fy,
                            auxiliary_history=ah, auxiliary_target=targets,
                            validation_history=vh, validation_items=vi).items():
        np.save(output/f'{name}.npy', array)
    np.savez_compressed(output/'comparator.npz', y=vy.ravel(), events=reference['events'],
                        pred=reference['pred'], weights=weights)
    print(json.dumps(dict(stage='prepared', fitting_events=len(fh), auxiliary_clicks=len(ah),
                          validation_events=len(vh), n_items=n_items)), flush=True)
    return fh, fi, fy, ah, targets, vh, vi, reference, weights, probability, n_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--day', type=int, choices=[5, 6], default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(RECIPE['threads'])
    torch.manual_seed(RECIPE['seed'])
    rng = np.random.default_rng(RECIPE['seed'])
    manifest = dict(recipe=RECIPE, day=args.day, device='cpu', exploratory=True,
                    source_hashes={name: sha(Path(__file__).with_name(name)) for name in
                                   ['tied_history_full.py', 'tied_history_model.py', 'causal_click_ledger.py']},
                    input_hashes={name: sha(args.artifacts/name) for name in
                                  ['train.npy', 'train_hist.npy', 'train_times.npy',
                                   f'v53_causal_propensity_day{args.day}.npz', 'evaluation_hist.npy']})
    (args.output/'sources').mkdir()
    for name in manifest['source_hashes']:
        shutil.copy2(Path(__file__).with_name(name), args.output/'sources'/name)
    (args.output/'frozen.json').write_text(json.dumps(manifest, indent=2))
    fh, fi, fy, ah, targets, vh, vi, ref, weights, probability, n_items = prepare(
        args.artifacts, args.output, args.day)
    assert len(ah) > 0
    model = TiedHistoryModel(n_items, RECIPE['dimensions'])
    with torch.no_grad():
        rate = float(fy.mean())
        model.global_bias.fill_(np.log(rate/(1-rate)))
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['learning_rate'])
    losses = []
    for step in range(RECIPE['steps']):
        indices = rng.integers(len(fh), size=RECIPE['batch_size'])
        ai = rng.integers(len(ah), size=RECIPE['auxiliary_batch_size'])
        ch = torch.tensor(fh[indices], dtype=torch.long)
        ci = torch.tensor(fi[indices], dtype=torch.long)
        cy = torch.tensor(fy[indices])
        ax = torch.tensor(ah[ai], dtype=torch.long)
        negatives = rng.choice(n_items, size=(len(ai), RECIPE['negatives']), p=probability)
        candidates = torch.tensor(np.column_stack([targets[ai], negatives]), dtype=torch.long)
        optimizer.zero_grad()
        event_loss = torch.nn.functional.binary_cross_entropy_with_logits(model(ch, ci), cy)
        aux_loss = model.auxiliary_loss(ax, candidates)
        penalty = regularization(model, ch, ci) + RECIPE['auxiliary_weight']*regularization(model, ax, candidates)
        loss = event_loss + RECIPE['auxiliary_weight']*aux_loss + RECIPE['sampled_vector_regularization']*penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0:
            record = dict(step=step+1, event_loss=float(event_loss.detach()),
                          auxiliary_loss=float(aux_loss.detach()), elapsed_seconds=time.monotonic()-started)
            losses.append(record)
            print(json.dumps(record), flush=True)
    def predict(member):
        with torch.no_grad():
            return np.concatenate([member(torch.tensor(vh[b:b+1024], dtype=torch.long),
                                          torch.tensor(vi[b:b+1024], dtype=torch.long)).sigmoid().numpy().ravel()
                                   for b in range(0, len(vh), 1024)])
    predictions = predict(model)
    torch.save(model.state_dict(), args.output/'model.pt')
    replay = TiedHistoryModel(n_items, RECIPE['dimensions'])
    replay.load_state_dict(torch.load(args.output/'model.pt', weights_only=True))
    assert np.array_equal(predictions, predict(replay))
    rank = lambda x: rankdata(x, method='average')/(len(x)+1)
    blend = (1-RECIPE['fixed_rank_blend_weight'])*rank(ref['pred']) + RECIPE['fixed_rank_blend_weight']*rank(predictions)
    def metrics(values):
        return dict(ordinary_auc=float(roc_auc_score(ref['y'], values)),
                    adjusted_auc=float(roc_auc_score(ref['y'], values, sample_weight=weights)))
    report = dict(status='completed_exploratory', recipe=RECIPE, day=args.day,
                  full_fit_events=len(fh), auxiliary_clicks=len(ah), losses=losses,
                  baseline=metrics(ref['pred']), standalone=metrics(predictions), blend=metrics(blend),
                  full_prediction_replay='exact', promoted=False, target_achieved=False,
                  official_auc=None, fresh_holdout=False, elapsed_seconds=time.monotonic()-started)
    np.savez_compressed(args.output/'predictions.npz', pred=predictions, blend=blend,
                        y=ref['y'], events=ref['events'], weights=weights)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
