"""Three fixed seeds of the eight-epoch setwise choice model; no seed selection."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch

SEEDS = (6802, 6803, 6804)


def mean_predictions(arrays):
    assert len(arrays) == len(SEEDS)
    assert all(a.shape == arrays[0].shape for a in arrays)
    assert all(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in arrays)
    return np.mean(np.stack(arrays).astype('float64'), axis=0)


def run(root, neural_root, output, model_source, weighting_source, cache=None, device='cuda'):
    import importlib.util
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m
    root, neural_root, output = Path(root), Path(neural_root), Path(output)
    base = module('bag_numeric', model_source)
    h = module('bag_helpers', root/'runner.py')
    weighting = module('bag_weighting', weighting_source)
    assert base.RECIPE['epochs'] == 8 and base.RECIPE['blend_weight'] == .1
    manifest = h.load_bundle(root, verify=True)
    output.mkdir(parents=True, exist_ok=False)
    for src, name in [(Path(__file__), 'choice_bag_runner.py'),
                      (Path(model_source), 'numeric_source.py'),
                      (Path(weighting_source), 'weighting_source.py')]:
        shutil.copy2(src, output/name)
    recipe = dict(seeds=SEEDS, averaging='Equal mean of three raw probability vectors',
                  blend_weight=.1, model_recipe=base.RECIPE.copy(),
                  selection='Fixed seeds and eight epochs. No seed or blend search.',
                  gates='Positive ordinary and adjusted lower gain interval versus V53 and single6802 on both dates',
                  source_sha256=h.digest(Path(__file__)), model_source_sha256=h.digest(Path(model_source)))
    (output/'recipe.json').write_text(json.dumps(recipe, indent=2))
    original_recipe = base.RECIPE.copy()
    reports = []
    try:
        for day in [6,5]:
            predictions = []
            reference = None
            single_pred = None
            for seed in SEEDS:
                base.RECIPE['seed'] = seed
                destination = output/f'seed{seed}'/f'day{day}'
                if seed == SEEDS[0] and cache is not None and (Path(cache)/f'day{day}'/'model.pt').exists():
                    source = Path(cache)/f'day{day}'
                    cached_recipe = json.loads((source/'recipe.json').read_text())
                    assert cached_recipe['recipe'] == base.RECIPE
                    cached_manifest = cached_recipe.get('manifest_sha256', cached_recipe.get('input_manifest_sha256'))
                    assert cached_manifest == h.digest(root/'bundle_manifest.json')
                    shutil.copytree(source, destination)
                else:
                    base.fit(root, neural_root, destination, manifest, day, h, weighting, device)
                report = json.loads((destination/'report.json').read_text())
                checkpoint = destination/'model.pt'
                assert h.digest(checkpoint) == report['model_sha256']
                with np.load(destination/'predictions.npz') as z:
                    values = {k:z[k].copy() for k in z.files}
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                restored = base.ChoiceNet(state['mean'], state['scale'], base.RECIPE).to(device)
                restored.load_state_dict(state)
                xv = h.FeatureBlocks(root, [manifest['validation'][str(day)]['features']])
                replay = base.predict(restored, xv, device)
                assert np.array_equal(replay, values['raw'])
                del restored, state
                if reference is None:
                    reference = values
                    single_pred = values['pred']
                else:
                    for key in ['y', 'events', 'baseline', 'weights']:
                        assert np.array_equal(values[key], reference[key]), key
                predictions.append(values['raw'])
            raw = mean_predictions(predictions)
            pred = h.rank(.9*h.rank(reference['baseline'])+.1*h.rank(raw))
            m = h.metrics(reference['y'], pred, reference['baseline'], reference['weights'])
            versus_single = h.metrics(reference['y'], pred, single_pred, reference['weights'])
            m.update(day=day, versus_single=versus_single,
                     raw_auc=float(h.roc_auc_score(reference['y'], raw)),
                     saved_checkpoint_replay='PASS, all three seeds exact',
                     rows=len(raw), csv_ready=False, official_auc=None)
            np.savez_compressed(output/f'day{day}.npz', raw=raw, pred=pred,
                                **{k:reference[k] for k in ['y','events','baseline','weights']})
            (output/f'day{day}_report.json').write_text(json.dumps(m, indent=2))
            reports.append(m)
            print('BAGGED_RESULT', json.dumps(m, indent=2), flush=True)
            if not (m['positive_gate'] and versus_single['positive_gate']): break
    finally:
        base.RECIPE.clear(); base.RECIPE.update(original_recipe)
    eligible = len(reports) == 2 and all(r['positive_gate'] and r['versus_single']['positive_gate'] for r in reports)
    summary = dict(reports=reports, eligible_for_review=eligible, target=.83,
                   target_achieved=bool(eligible and reports[0]['history_adjusted_auc'] >= .83),
                   csv_ready=False, official_auc=None,
                   note='Reused development dates. Intervals omit fitting and repeated-selection uncertainty.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
