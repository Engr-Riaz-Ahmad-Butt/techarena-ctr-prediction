"""Synthetic integration and event invariants for the numeric choice expert."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import runner
import tabular_choice_runner as choice


def main():
    torch.set_num_threads(2)
    torch.manual_seed(6802)
    model = choice.ChoiceNet(torch.zeros(354), torch.ones(354), choice.RECIPE).eval()
    x = torch.randn(6, 10, 354)
    perm = torch.tensor([9, 3, 6, 4, 0, 7, 5, 8, 2, 1])
    with torch.no_grad():
        a = model(x).softmax(1)
        b = model(x[:, perm]).softmax(1)
    assert torch.allclose(a[:, perm], b[:, :10], atol=1e-7)
    assert torch.allclose(a[:, 10], b[:, 10], atol=1e-7)
    assert torch.allclose(a.sum(1), torch.ones(6)) and (a[:, :10].sum(1) < 1).all()
    saved = dict(choice.RECIPE)
    try:
        choice.RECIPE['epochs'] = 1
        choice.RECIPE['batch_size'] = 8
        parent = Path(__file__).resolve().parents[1]/'artifacts'
        with tempfile.TemporaryDirectory(dir=parent, prefix='numeric-choice-test-') as temp:
            root = Path(temp)
            rng = np.random.default_rng(6802)
            manifest = dict(train={}, validation={})
            def block(day):
                features = rng.normal(size=(100, 354)).astype('float32')
                labels = np.zeros(100, dtype='float32'); labels[::20] = 1
                np.save(root/f'x{day}.npy', features)
                np.save(root/f'y{day}.npy', labels)
                (root/f'm{day}.json').write_text(json.dumps(dict(last_timestamp=day)))
                return dict(features=[f'x{day}.npy'], labels=f'y{day}.npy', metadata=f'm{day}.json')
            for day in [3, 4, 5]: manifest['train'][str(day)] = block(day)
            val = block(6)
            labels = np.load(root/'y6.npy')
            np.savez(root/'val.npz', y=labels, baseline=.1+.8*labels,
                     weights=np.ones(100), events=np.repeat(np.arange(10), 10))
            manifest['validation']['6'] = dict(features=val['features'], data='val.npz', cutoff=6)
            (root/'bundle_manifest.json').write_text(json.dumps(manifest))
            weighting = SimpleNamespace(training_weights=lambda root, day, y, h:(np.ones(len(y)), {'synthetic':True}))
            with contextlib.redirect_stdout(io.StringIO()):
                result = choice.fit(root, root, root/'result', manifest, 6, runner, weighting, 'cpu')
            assert result['saved_checkpoint_replay'] == 'PASS' and result['saved_model_replay_max_error'] == 0
            checkpoint = torch.load(root/'result/model.pt', weights_only=True)
            expected_mean = np.concatenate([np.load(root/f'x{d}.npy') for d in [3, 4, 5]]).mean(0)
            assert np.allclose(checkpoint['mean'].numpy(), expected_mean, atol=1e-6)
            # No validation-label fitting or normalization; checkpoint buffers use fitting rows only.
            assert not np.allclose(checkpoint['mean'].numpy(), np.load(root/'x6.npy').mean(0))
            (root/'m5.json').write_text(json.dumps(dict(last_timestamp=6)))
            try: choice.fit(root, root, root/'bad_cutoff', manifest, 6, runner, weighting, 'cpu')
            except AssertionError: pass
            else: raise AssertionError('Temporal cutoff accepted held-out rows')
    finally:
        choice.RECIPE.clear(); choice.RECIPE.update(saved)
    report = dict(status='PASS', scope='Synthetic CPU test; not production GPU score',
                  checks=['candidate permutation symmetry', 'outside probability and mass',
                          'one full optimizer epoch', 'fitting-only normalization',
                          'saved checkpoint replay', 'held-out timestamp rejection'])
    destination = Path(__file__).resolve().parents[1]/'results/tabular_choice_driver_test.json'
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
