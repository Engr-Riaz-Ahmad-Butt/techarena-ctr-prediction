"""Five predeclared tied-history GPU seeds using the existing neural bundle.

Independent full-prefix reconstruction, matched seed7201 control, fixed 5% rank
blend. No validation early stopping, seed selection, or competition CSV export.
The cohorts were already reused and cannot support promotion by themselves.
"""
import gc
import hashlib
import json
import subprocess
import time
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

RECIPE = dict(seeds=[7201, 7202, 7203, 7204, 7205], steps=2000,
              batch_size=1024, auxiliary_batch_size=512, auxiliary_weight=0.1,
              dimensions=32, learning_rate=0.01, sampled_vector_regularization=0.002,
              negatives=9, context_limit=100, blend_weight=0.05,
              minimum_primary_adjusted_gain=0.0005, cpu_threads=2)
EXPECTED = {
    'events.npz': '3e2d628a35a9e69c6c75b0a98da28ee9dab96857209b2d16def26ce7d5c8249e',
    'val5.npz': '6f30b7c3b825a3481006b83f02b81d617b113fb40d4898f34f504b6b9473fd3d',
    'val6.npz': '963b8f1c8ff1edbeb88955cc087ddeccf8a4784472d488b1270b2eb6deff5d8d',
}
PREPARED = {
    6: {
        'fit_h': '91735dca1e65c475254bb60ed09cd8802f45bb9ca152104c6a277cd9e0d9ed59',
        'fit_i': 'a4fcee5ec2b954efb27dc3a6581f85fb0994d3b4494891d1859b6c65a5a4e64d',
        'fit_y': 'c3526f5dc44e90e58c25d18c16415a30ef1b144bfb21c8fd0d388535c993cacc',
        'aux_h': '30ff4574d87fa0a88572401b851b68b84f9b4266cfbd017173542174c2dfa868',
        'aux_target': 'ca8539e97c620ebcadcfb609876788f8615d19fd813ef604266872e0bea6844e',
        'val_h': '652d5c750a633e140f14d20a07e004c433ba145a29a3265f3d75a8bbcd4e25f2',
        'val_i': 'a6fe85b78545ee78477ae90dc02562761a34b0c5100f15ccef2ddf8c98c13019',
    },
    5: {
        'fit_h': 'f1c6da162f62802a6f7ab09358c096e6b54bba2d96e04b04923a0a9841251b39',
        'fit_i': 'c97faba33966617ada0873354c13208bb8e256c78e9933f6e6a77ab59a56a419',
        'fit_y': '4b0f10af758de988b44326a64e6e5bb4672f4ae0581c0a2ecdc97242c8ee0b33',
        'aux_h': '724f9d7a0c5b0a6407249ca53b5613d749149eed011a531161f9f1a622981a36',
        'aux_target': '456a892b8b736f3e4ee83b76d872656c964f90491ea85b1b99285eb3bf94c994',
        'val_h': '3819af854a71b96674f4c936d331864fbe81eda65d4b04fc861643d8444e0d05',
        'val_i': '6915c0db249ea82079daf33e1fac4db1d677ede77156de01ec24fab0e0daaeb4',
    },
}


def verify_prepared(prepared, day):
    checked = {}
    for name, expected in PREPARED[day].items():
        view = memoryview(np.ascontiguousarray(prepared[name])).cast('B')
        value = hashlib.sha256()
        for offset in range(0, len(view), 4*1024*1024):
            value.update(view[offset:offset+4*1024*1024])
        checked[name] = value.hexdigest()
        assert checked[name] == expected, f'CPU context mismatch: day{day} {name}'
    return checked


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class BagFactor(nn.Module):
    def __init__(self, n_items, dimensions=32):
        super().__init__()
        self.items = nn.Embedding(n_items, dimensions)
        self.bias = nn.Parameter(torch.zeros(n_items))
        self.global_bias = nn.Parameter(torch.tensor(-2.2))
        nn.init.normal_(self.items.weight, std=0.005)

    def forward_and_penalty(self, history, candidates):
        valid = history >= 0
        vectors = self.items(history.clamp_min(0))*valid.unsqueeze(-1)
        count = valid.sum(1).clamp_min(1).unsqueeze(-1)
        profile = vectors.sum(1)/count
        candidate_vectors = self.items(candidates)
        bias = self.bias[candidates]
        logits = ((profile.unsqueeze(1)*candidate_vectors).sum(-1)+bias+self.global_bias)
        penalty = (vectors.square().sum()/valid.sum().clamp_min(1)
                   + candidate_vectors.square().sum(-1).mean()+bias.square().mean())
        return logits, penalty

    def forward(self, history, candidates):
        return self.forward_and_penalty(history, candidates)[0]


def restore_timestamps(data):
    valid = data['hist'] >= 0
    gaps = data['gaps'].astype('float64')
    if not np.isfinite(gaps).all() or (gaps < 0).any():
        raise ValueError('Invalid history ages')
    stamps = np.rint(data['timestamp'].astype('int64')[:, None]-gaps*86400).astype('int64')
    if not np.all((stamps < data['timestamp'][:, None]) | ~valid):
        raise ValueError('Noncausal reconstructed history')
    return stamps


def pad(clicks):
    result = np.full(RECIPE['context_limit'], -1, np.int32)
    chosen = sorted(clicks, reverse=True)[:len(result)]
    result[:len(chosen)] = [item for _, item in chosen]
    return result


def prepare(data, stamps, validation, cutoff, progress=True):
    """Reconstruct identical CPU fitting/auxiliary contexts from raw event IDs."""
    fit_ids = np.flatnonzero(data['timestamp'] < cutoff)
    ledger = defaultdict(set)
    for n, q in enumerate(fit_ids):
        user = int(data['users'][q])
        for item, timestamp in zip(data['hist'][q], stamps[q]):
            if item >= 0:
                assert timestamp < data['timestamp'][q] < cutoff
                ledger[user].add((int(timestamp), int(item)))
        target = int(data['target'][q])
        if target < 10:
            ledger[user].add((int(data['timestamp'][q]), int(data['items'][q, target])))
        if progress and n and n % 100000 == 0:
            print(json.dumps(dict(stage='ledger', fitting_events=n)), flush=True)
    ledger = {user: tuple(sorted(clicks)) for user, clicks in ledger.items()}
    def prior(user, timestamp):
        clicks = ledger.get(user, ())
        stop = bisect_left([t for t, _ in clicks], timestamp)
        return clicks[max(0, stop-RECIPE['context_limit']):stop]
    fit_h = np.empty((len(fit_ids), RECIPE['context_limit']), np.int32)
    for n, q in enumerate(fit_ids):
        fit_h[n] = pad(prior(int(data['users'][q]), int(data['timestamp'][q])))
    targets, auxiliary = [], []
    for clicks in ledger.values():
        for target in clicks:
            others = [value for value in clicks if value != target and value[0] < cutoff]
            if others:
                targets.append(target[1])
                auxiliary.append(pad(others))
    auxiliary = np.asarray(auxiliary, dtype=np.int32)
    targets = np.asarray(targets, dtype=np.int32)
    ids = validation['indices'].astype('int64')
    future = np.flatnonzero((data['timestamp'] >= cutoff) & (data['timestamp'] < cutoff+86400))
    _, first = np.unique(data['users'][future], return_index=True)
    assert np.array_equal(np.sort(future[first]), np.sort(ids))
    y = (data['target'][ids, None] == np.arange(10)).astype('float32')
    assert np.array_equal(y.ravel(), validation['y'])
    val_h = np.empty((len(ids), RECIPE['context_limit']), np.int32)
    for n, q in enumerate(ids):
        clicks = set(prior(int(data['users'][q]), int(data['timestamp'][q])))
        clicks.update((int(t), int(i)) for i, t in zip(data['hist'][q], stamps[q]) if i >= 0)
        val_h[n] = pad(clicks)
    n_items = max(int(data['items'].max()), int(data['hist'].max()))+1
    counts = np.bincount(data['items'][fit_ids].ravel(), minlength=n_items).astype('float64')
    probability = counts/counts.sum()
    fit_y = (data['target'][fit_ids, None] == np.arange(10)).astype('float32')
    return dict(fit_h=fit_h, fit_i=data['items'][fit_ids], fit_y=fit_y,
                aux_h=auxiliary, aux_target=targets, val_h=val_h,
                val_i=data['items'][ids], exposure_probability=probability, n_items=n_items)


def load(root):
    for name, expected in EXPECTED.items():
        assert digest(root/name) == expected, f'Unexpected bundle version: {name}'
    with np.load(root/'events.npz') as z:
        data = {key: z[key].copy() for key in z.files}
    assert np.isin(data['target'], np.arange(11)).all()
    assert data['items'].shape == (len(data['timestamp']), 10)
    assert np.all(np.diff(data['timestamp'].astype('int64')) >= 0)
    stamps = restore_timestamps(data)
    return data, stamps


def one_seed(prepared, seed, output, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = BagFactor(prepared['n_items'], RECIPE['dimensions']).to(device)
    with torch.no_grad():
        rate = float(prepared['fit_y'].mean().item())
        model.global_bias.fill_(np.log(rate/(1-rate)))
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['learning_rate'])
    started = time.monotonic()
    for step in range(RECIPE['steps']):
        ii = rng.integers(len(prepared['fit_h']), size=RECIPE['batch_size'])
        ai = rng.integers(len(prepared['aux_h']), size=RECIPE['auxiliary_batch_size'])
        ids = torch.as_tensor(ii, device=device, dtype=torch.long)
        aids = torch.as_tensor(ai, device=device, dtype=torch.long)
        history = prepared['fit_h'][ids].long()
        candidates = prepared['fit_i'][ids].long()
        negatives = rng.choice(prepared['n_items'], size=(len(ai), RECIPE['negatives']),
                               p=prepared['exposure_probability'])
        negative_ids = torch.as_tensor(negatives, device=device, dtype=torch.long)
        aux_candidates = torch.cat([prepared['aux_target'][aids, None].long(), negative_ids], dim=1)
        aux_history = prepared['aux_h'][aids].long()
        optimizer.zero_grad()
        logits, penalty = model.forward_and_penalty(history, candidates)
        aux_logits, aux_penalty = model.forward_and_penalty(aux_history, aux_candidates)
        aux_labels = torch.zeros_like(aux_logits)
        aux_labels[:, 0] = 1
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, prepared['fit_y'][ids])
                + RECIPE['auxiliary_weight']*nn.functional.binary_cross_entropy_with_logits(aux_logits, aux_labels)
                + RECIPE['sampled_vector_regularization']*(penalty+RECIPE['auxiliary_weight']*aux_penalty))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        if (step+1) % 200 == 0:
            print(json.dumps(dict(seed=seed, step=step+1, loss=float(loss.detach()),
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
    def predict(member):
        result = []
        with torch.no_grad():
            for offset in range(0, len(prepared['val_h']), 1024):
                h = prepared['val_h'][offset:offset+1024].long()
                i = prepared['val_i'][offset:offset+1024].long()
                result.append(member(h, i).sigmoid().cpu().numpy().ravel())
        return np.concatenate(result)
    pred = predict(model)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, output/f'seed{seed}.pt')
    replay = BagFactor(prepared['n_items'], RECIPE['dimensions']).to(device)
    replay.load_state_dict(torch.load(output/f'seed{seed}.pt', weights_only=True))
    assert np.array_equal(pred, predict(replay))
    np.save(output/f'seed{seed}_pred.npy', pred)
    print(json.dumps(dict(seed=seed, checkpoint_replay='exact_all_rows',
                          elapsed_seconds=time.monotonic()-started)), flush=True)
    del model, replay, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return pred


def run(root, output):
    root, output = Path(root), Path(output)
    assert torch.cuda.is_available(), 'GPU required; refusing a long CPU fallback'
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(RECIPE['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    started = time.monotonic()
    gpu = dict(cuda_available=True, name=torch.cuda.get_device_name(0),
               torch_version=torch.__version__, memory_bytes=torch.cuda.get_device_properties(0).total_memory,
               nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True))
    frozen = dict(recipe=RECIPE, gpu=gpu, source_sha256=digest(__file__),
                  inputs=EXPECTED, fresh_holdout=False, official_auc=None,
                  note='Fixed five seeds, matched single-seed GPU control; no seed selection or per-seed validation scoring')
    (output/'frozen.json').write_text(json.dumps(frozen, indent=2))
    print(json.dumps(frozen), flush=True)
    data, stamps = load(root)
    reports = []
    for day in [6, 5]:
        day_out = output/f'day{day}'
        day_out.mkdir()
        with np.load(root/f'val{day}.npz') as z:
            validation = {key: z[key].copy() for key in z.files}
        cutoff = int(data['timestamp'][0]) + day*86400
        prepared = prepare(data, stamps, validation, cutoff)
        hashes = verify_prepared(prepared, day)
        (day_out/'prepared_array_hashes.json').write_text(json.dumps(hashes, indent=2))
        print(json.dumps(dict(stage='prepared', day=day, fitting_events=len(prepared['fit_h']),
                              auxiliary_clicks=len(prepared['aux_h']))), flush=True)
        for name in ['val_h', 'val_i']:
            np.save(day_out/f'{name}.npy', prepared[name])
        for name in ['fit_h', 'fit_i', 'fit_y', 'aux_h', 'aux_target', 'val_h', 'val_i']:
            prepared[name] = torch.as_tensor(prepared[name], device=device)
        predictions = [one_seed(prepared, seed, day_out, device) for seed in RECIPE['seeds']]
        control_raw = predictions[0]
        bag_raw = np.mean(np.stack(predictions).astype('float64'), axis=0)
        rank = lambda x: rankdata(x, method='average')/(len(x)+1)
        blend = lambda p: (1-RECIPE['blend_weight'])*rank(validation['baseline']) + RECIPE['blend_weight']*rank(p)
        control, bag = blend(control_raw), blend(bag_raw)
        def metrics(values):
            return dict(ordinary_auc=float(roc_auc_score(validation['y'], values)),
                        adjusted_auc=float(roc_auc_score(validation['y'], values, sample_weight=validation['weights'])))
        record = dict(day=day, baseline=metrics(validation['baseline']), control=metrics(control),
                      bag=metrics(bag), raw_control=metrics(control_raw), raw_bag=metrics(bag_raw),
                      all_five_checkpoint_replays='exact_all_rows', promoted=False)
        record['adjusted_gain'] = record['bag']['adjusted_auc']-record['baseline']['adjusted_auc']
        record['adjusted_gain_vs_control'] = record['bag']['adjusted_auc']-record['control']['adjusted_auc']
        record['development_gate'] = (record['adjusted_gain'] >= RECIPE['minimum_primary_adjusted_gain']
                                      and record['adjusted_gain_vs_control'] > 0
                                      and record['bag']['ordinary_auc'] >= record['baseline']['ordinary_auc'])
        np.savez_compressed(day_out/'predictions.npz', y=validation['y'], events=np.repeat(validation['indices'], 10),
                            weights=validation['weights'], baseline=validation['baseline'],
                            control_raw=control_raw, bag_raw=bag_raw, control=control, bag=bag)
        (day_out/'report.json').write_text(json.dumps(record, indent=2))
        reports.append(record)
        print(json.dumps(record), flush=True)
        del prepared, predictions
        gc.collect()
        torch.cuda.empty_cache()
        if not record['development_gate']:
            break
    summary = dict(reports=reports, day5_run=len(reports)==2, target_achieved=False,
                   local_adjusted_target_reached=reports[0]['bag']['adjusted_auc'] >= 0.83,
                   promoted=False, csv_ready=False, official_auc=None, fresh_holdout=False,
                   elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary
