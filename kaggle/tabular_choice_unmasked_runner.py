"""Permutation-equivariant neural choice expert over causal numeric features.

Fitting and normalization use prior dates only. A fixed blend is scored on
day6 and confirmed unchanged on day5 only after a positive gain gate.
"""
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

RECIPE = dict(epochs=8, batch_size=1024, learning_rate=.0005,
              weight_decay=.005, dropout=.15, seed=6802, blend_weight=.1,
              hidden=128, representation=64)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChoiceNet(nn.Module):
    def __init__(self, mean, scale, recipe):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        hidden, dim = recipe['hidden'], recipe['representation']
        self.encoder = nn.Sequential(nn.Linear(len(mean), hidden), nn.ReLU(),
                                     nn.Linear(hidden, dim), nn.ReLU())
        self.candidate = nn.Sequential(nn.Linear(3*dim, 64), nn.ReLU(),
                                       nn.Dropout(recipe['dropout']), nn.Linear(64, 1))
        self.outside = nn.Sequential(nn.Linear(2*dim, 32), nn.ReLU(), nn.Linear(32, 1))
        # Sensible initial no-click mass; no dependency on candidate position.
        nn.init.constant_(self.candidate[-1].bias, -2.)
        nn.init.zeros_(self.outside[-1].bias)

    def forward(self, x):
        x = ((x-self.mean)/self.scale).clamp(-10, 10)
        z = self.encoder(x)
        avg, peak = z.mean(1), z.max(1).values
        context = torch.cat([avg, peak], -1)
        combined = torch.cat([z, context[:, None, :].expand(-1, 10, -1)], -1)
        return torch.cat([self.candidate(combined).squeeze(-1), self.outside(context)], -1)


@torch.no_grad()
def predict(model, features, device):
    model.eval()
    result = []
    for off in range(0, len(features), 20000):
        x = torch.as_tensor(features.block(off, min(off+20000, len(features))), device=device)
        p = model(x.reshape(-1, 10, 354)).softmax(1)[:, :10]
        result.append(p.cpu().numpy().ravel())
    return np.concatenate(result)


def fit(root, neural_root, output, manifest, day, h, weighting, device):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(RECIPE['seed'])
    specs = [manifest['train'][str(d)] for d in range(day-3, day)]
    validation = manifest['validation'][str(day)]
    labels = []
    for s in specs:
        meta = json.loads((root/s['metadata']).read_text())
        assert meta['last_timestamp'] < validation['cutoff']
        labels.append(np.load(root/s['labels']))
    y = np.concatenate(labels).astype('float32')
    assert len(y) % 10 == 0
    event_y = y.reshape(-1, 10)
    assert ((event_y == 0) | (event_y == 1)).all() and (event_y.sum(1) <= 1).all()
    target = np.where(event_y.sum(1) > 0, event_y.argmax(1), 10)
    weights, weight_recipe = weighting.training_weights(neural_root, day, y, h)
    weights = weights.reshape(-1, 10)
    assert np.array_equal(weights, np.repeat(weights[:, :1], 10, axis=1))
    features = h.FeatureBlocks(root, [s['features'] for s in specs])
    assert len(features) == len(y)
    # Transfer one block at a time. No second full host feature matrix.
    x = torch.empty((len(y), 354), device=device)
    for off in range(0, len(y), 50000):
        end = min(off+50000, len(y))
        x[off:end] = torch.as_tensor(features.block(off, end), device=device)
    variance, mean = torch.var_mean(x, dim=0, unbiased=False)
    scale = variance.sqrt().clamp_min(.01)
    assert torch.isfinite(mean).all() and torch.isfinite(scale).all()
    x = x.reshape(-1, 10, 354)
    target = torch.as_tensor(target, device=device, dtype=torch.long)
    weight = torch.as_tensor(weights[:, 0].copy(), device=device)
    model = ChoiceNet(mean, scale, RECIPE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['learning_rate'],
                                  weight_decay=RECIPE['weight_decay'])
    recipe = dict(recipe=RECIPE, day=day, fit_days=list(range(day-3, day)),
                  fit_rows=len(y), validation_cutoff=validation['cutoff'],
                  weighting=weight_recipe, source_sha256=h.digest(Path(__file__)),
                  manifest_sha256=h.digest(root/'bundle_manifest.json'),
                  torch_version=torch.__version__,
                  selection='Fixed epochs and blend before scoring; no validation optimization')
    (output/'recipe.json').write_text(json.dumps(recipe, indent=2))
    for epoch in range(RECIPE['epochs']):
        model.train()
        order = torch.randperm(len(x), device=device)
        total = mass = 0.
        for off in range(0, len(x), RECIPE['batch_size']):
            ix = order[off:off+RECIPE['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            losses = nn.functional.cross_entropy(model(x[ix]), target[ix], reduction='none')
            loss = (losses*weight[ix]).sum()/weight[ix].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float((losses.detach()*weight[ix]).sum())
            mass += float(weight[ix].sum())
        print(dict(day=day, epoch=epoch+1, epochs=RECIPE['epochs'],
                   loss=total/mass, elapsed_seconds=round(time.monotonic()-started, 1)), flush=True)
    checkpoint = output/'model.pt'
    torch.save(model.state_dict(), checkpoint)
    del x, target, weight, optimizer
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    with np.load(root/validation['data']) as z:
        val = {key:z[key].copy() for key in z.files}
    xv = h.FeatureBlocks(root, [validation['features']])
    assert len(xv) == len(val['y'])
    raw = predict(model, xv, device)
    assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
    assert (raw.reshape(-1, 10).sum(1) <= 1.00001).all()
    pred = h.rank((1-RECIPE['blend_weight'])*h.rank(val['baseline'])
                  + RECIPE['blend_weight']*h.rank(raw))
    m = h.metrics(val['y'], pred, val['baseline'], val['weights'])
    restored = ChoiceNet(mean, scale, RECIPE).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    replay = predict(restored, xv, device)
    error = float(np.max(np.abs(raw-replay)))
    assert error < 1e-7, error
    m.update(day=day, raw_auc=float(h.roc_auc_score(val['y'], raw)),
             saved_model_replay_max_error=error, saved_checkpoint_replay='PASS',
             model_sha256=h.digest(checkpoint), rows=len(raw),
             elapsed_seconds=round(time.monotonic()-started, 1), csv_ready=False, official_auc=None)
    np.savez_compressed(output/'predictions.npz', raw=raw, pred=pred, **val)
    (output/'report.json').write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2), flush=True)
    del model, restored
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    return m


def run(root, neural_root, output, weighting_path, device='cuda'):
    root, neural_root, output = Path(root), Path(neural_root), Path(output)
    assert device == 'cpu' or torch.cuda.is_available()
    torch.set_num_threads(4)
    h = load_module('tabular_choice_helpers', root/'runner.py')
    weighting = load_module('tabular_choice_weights', weighting_path)
    manifest = h.load_bundle(root, verify=True)
    output.mkdir(parents=True, exist_ok=False)
    (output/'tabular_choice_runner.py').write_text(Path(__file__).read_text())
    (output/'weighting_source.py').write_text(Path(weighting_path).read_text())
    reports = [fit(root, neural_root, output/'day6', manifest, 6, h, weighting, device)]
    if reports[0]['positive_gate']:
        reports.append(fit(root, neural_root, output/'day5', manifest, 5, h, weighting, device))
    eligible = len(reports) == 2 and all(r['positive_gate'] for r in reports)
    summary = dict(reports=reports, eligible_for_review=eligible, target=.83,
                   target_achieved=bool(eligible and reports[0]['history_adjusted_auc'] >= .83),
                   csv_ready=False, official_auc=None,
                   note='Reused main validation dates. A local metric, not an official forecast.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
