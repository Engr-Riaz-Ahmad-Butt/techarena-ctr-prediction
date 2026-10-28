"""Small CPU integration for grouped query weights and both-model replay."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import runner
import tabular_choice_setwise_runner as neural
import full_rank_runner as ranker


def main():
    torch.set_num_threads(2)
    rng=np.random.default_rng(6901)
    scores=rng.normal(size=100);mass=rng.random(10)
    a=ranker.probabilities(scores,mass).reshape(-1,10)
    assert np.allclose(a.sum(1),mass)
    perm=np.array([9,3,4,7,6,5,8,1,0,2])
    b=ranker.probabilities(scores.reshape(-1,10)[:,perm],mass).reshape(-1,10)
    assert np.allclose(a[:,perm],b)
    assert np.allclose(a,ranker.probabilities(scores.reshape(-1,10)+25,mass).reshape(-1,10))
    assert not ranker.probabilities(scores,np.zeros(10)).any()
    saved_rounds,saved_params=ranker.ROUNDS,ranker.PARAMS.copy()
    parent=Path(__file__).resolve().parents[1]/'artifacts'
    try:
        ranker.ROUNDS=4;ranker.PARAMS.update(max_depth=2,nthread=2)
        with tempfile.TemporaryDirectory(dir=parent,prefix='full-ranker-test-') as temp:
            root=Path(temp);manifest=dict(train={},validation={})
            for day in [3,4,5]:
                np.save(root/f'x{day}.npy',rng.normal(size=(100,354)).astype('float32'))
                y=np.zeros(100,dtype='float32');y[::20]=1
                np.save(root/f'y{day}.npy',y)
                (root/f'm{day}.json').write_text(json.dumps(dict(last_timestamp=day)))
                manifest['train'][str(day)]=dict(features=[f'x{day}.npy'],labels=f'y{day}.npy',metadata=f'm{day}.json')
            np.save(root/'val.npy',rng.normal(size=(100,354)).astype('float32'))
            np.savez(root/'val.npz',y=y,baseline=.1+.8*y,weights=np.ones(100),events=np.repeat(np.arange(10),10))
            manifest['validation']['6']=dict(features=['val.npy'],data='val.npz',cutoff=6)
            (root/'bundle_manifest.json').write_text(json.dumps(manifest))
            weighting=SimpleNamespace(training_weights=lambda root,day,y,h:(np.ones(len(y),dtype='float32'),{'synthetic':True}))
            with contextlib.redirect_stdout(io.StringIO()):
                neural.fit(root,root,root/'neural/day6',manifest,6,runner,weighting,'cpu')
                report=ranker.fit(root,root,root/'rank',manifest,6,runner,neural,weighting,root/'neural','cpu')
            assert report['saved_checkpoint_replay']=='PASS, ranker and neural exact'
            recipe=json.loads((root/'rank/recipe.json').read_text())
            assert recipe['fit_rows']==300 and recipe['fit_queries']==30 and recipe['clicked_queries']==15
            with np.load(root/'rank/predictions.npz') as z:
                assert np.array_equal(z['raw'],ranker.probabilities(z['scores'],z['event_mass']))
            assert (root/'rank/neural_mass_model.pt').exists()
    finally:
        ranker.ROUNDS=saved_rounds;ranker.PARAMS.clear();ranker.PARAMS.update(saved_params)
    report=dict(status='PASS',scope='Synthetic CPU integration; not a production GPU score',
                checks=['query groups and group weights accepted by XGBoost',
                        'strict pre-cutoff fitting metadata','ranker and neural checkpoint replay',
                        'event click mass conserved','candidate permutation and score-shift invariance',
                        'zero mass','production recipe restored'])
    (parent.parent/'results/full_rank_driver_test.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
