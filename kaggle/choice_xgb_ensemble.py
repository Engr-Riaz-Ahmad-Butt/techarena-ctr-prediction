"""Fixed 80/10/10 baseline/XGBoost/numeric-choice rank ensemble.

Reproduce the previously frozen XGBoost interaction recipe; validate every
saved neural and tree prediction before combining. No blend search.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

XGB_PARAMETERS = dict(max_depth=8, eta=.035, min_child_weight=100,
                      reg_lambda=15., reg_alpha=.5)
ROUNDS = 1000


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def combine(tree, neural, h):
    for key in ['y', 'events', 'baseline', 'weights']:
        assert np.array_equal(tree[key], neural[key]), 'Misaligned '+key
    assert len(tree['y']) % 10 == 0
    for raw in [tree['raw'], neural['raw']]:
        assert raw.shape == tree['y'].shape
        assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
    p = h.rank(.8*h.rank(tree['baseline'])+.1*h.rank(tree['raw'])+.1*h.rank(neural['raw']))
    return p


def run(root, neural_root, output, neural_source, weighting_source, neural_output):
    root, output, neural_output = Path(root), Path(output), Path(neural_output)
    assert not output.exists(), 'Choose a fresh experiment directory'
    h = module('ensemble_tree', root/'runner.py')
    neural = module('ensemble_numeric', neural_source)
    assert neural.RECIPE['epochs'] == 8 and neural.RECIPE['blend_weight'] == .1
    if not (neural_output/'summary.json').exists():
        neural.run(root, Path(neural_root), neural_output, weighting_source)
    neural_summary = json.loads((neural_output/'summary.json').read_text())
    assert neural_summary['eligible_for_review']
    h.PARAMS.update(XGB_PARAMETERS)
    output.mkdir(parents=True)
    recipe = dict(weights={'V53':.8,'xgboost':.1,'numeric_choice':.1},
                  xgboost_params=h.PARAMS, rounds=ROUNDS, neural_recipe=neural.RECIPE,
                  source_sha256=h.digest(Path(__file__)),
                  neural_source_sha256=h.digest(Path(neural_source)),
                  selection='Fixed before scoring, no blend search or validation-label fitting')
    (output/'recipe.json').write_text(json.dumps(recipe, indent=2))
    (output/'choice_xgb_ensemble.py').write_bytes(Path(__file__).read_bytes())
    (output/'neural_source.py').write_bytes(Path(neural_source).read_bytes())
    tree_output = output/'xgboost'
    tree_output.mkdir()
    tree_summary = h.run(root, tree_output, device='cuda', mode='main', rounds=ROUNDS, verify=True)
    assert tree_summary['eligible_for_review'], 'Reproduced tree did not pass both date gates'
    manifest = h.load_bundle(root, verify=False)
    reports = []
    for day in [6,5]:
        directory = tree_output/f'day{day}'
        with np.load(directory/'predictions.npz') as z:
            tree = {k:z[k].copy() for k in z.files}
        # Rerunning experiment must load the saved model and reproduce all rows.
        h.experiment(root, manifest, directory, day=day, device='cuda', rounds=ROUNDS, mode='main')
        with np.load(directory/'predictions.npz') as z:
            assert np.array_equal(z['raw'], tree['raw'])
            assert np.array_equal(z['pred'], tree['pred'])
        ndir = neural_output/f'day{day}'
        nr = json.loads((ndir/'report.json').read_text())
        assert h.digest(ndir/'model.pt') == nr['model_sha256']
        with np.load(ndir/'predictions.npz') as z:
            nn = {k:z[k].copy() for k in z.files}
        state = torch.load(ndir/'model.pt', map_location='cuda', weights_only=True)
        model = neural.ChoiceNet(state['mean'], state['scale'], neural.RECIPE).to('cuda')
        model.load_state_dict(state)
        xv = h.FeatureBlocks(root, [manifest['validation'][str(day)]['features']])
        replay = neural.predict(model, xv, 'cuda')
        assert np.array_equal(replay, nn['raw'])
        pred = combine(tree, nn, h)
        result = h.metrics(tree['y'], pred, tree['baseline'], tree['weights'])
        versus_tree = h.metrics(tree['y'], pred, tree['pred'], tree['weights'])
        versus_neural = h.metrics(tree['y'], pred, nn['pred'], tree['weights'])
        result.update(day=day, versus_tree=versus_tree, versus_neural=versus_neural,
                      rows=len(pred), saved_checkpoint_replay='PASS, both experts exact',
                      csv_ready=False, official_auc=None)
        np.savez_compressed(output/f'day{day}.npz', pred=pred, tree_pred=tree['pred'],
                            neural_pred=nn['pred'], neural_raw=nn['raw'],
                            **{k:v for k,v in tree.items() if k != 'pred'})
        reports.append(result)
        print(json.dumps(result, indent=2), flush=True)
    eligible = all(r['positive_gate'] and r['versus_tree']['positive_gate']
                   and r['versus_neural']['positive_gate'] for r in reports)
    summary = dict(reports=reports, eligible_for_review=eligible, target=.83,
                   target_achieved=bool(eligible and reports[0]['history_adjusted_auc'] >= .83),
                   csv_ready=False, official_auc=None,
                   note='Reused development dates; intervals omit fitting and repeated selection uncertainty.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
