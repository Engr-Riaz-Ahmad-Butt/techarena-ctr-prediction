"""Exercise fresh-seed fitting, cache copying, model replay and averaging."""
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
import runner
import tabular_choice_setwise_runner as model
import choice_bag_runner as bag


def main():
    torch.set_num_threads(2)
    parent = Path(__file__).resolve().parents[1]/'artifacts'
    with tempfile.TemporaryDirectory(dir=parent, prefix='choice-bag-test-') as temp:
        root = Path(temp)
        manifest = dict(schema=1, feature_width=354, train={}, validation={}, files={})
        def record(path):
            manifest['files'][path.name] = dict(bytes=path.stat().st_size, sha256=runner.digest(path))
            return path.name
        shutil.copy2(Path(__file__).with_name('runner.py'), root/'runner.py')
        record(root/'runner.py')
        rng = np.random.default_rng(6803)
        for day in [3,4,5]:
            x = root/f'x{day}.npy'; y = root/f'y{day}.npy'; m = root/f'm{day}.json'
            np.save(x, rng.normal(size=(100,354)).astype('float32'))
            labels = np.zeros(100,dtype='float32'); labels[::20] = 1; np.save(y,labels)
            m.write_text(json.dumps(dict(last_timestamp=day)))
            manifest['train'][str(day)] = dict(features=[record(x)], labels=record(y), metadata=record(m))
        x = root/'val.npy'; v = root/'val.npz'
        np.save(x,rng.normal(size=(100,354)).astype('float32'))
        np.savez(v,y=labels,baseline=.1+.8*labels,weights=np.ones(100),events=np.repeat(np.arange(10),10))
        manifest['validation']['6'] = dict(features=[record(x)], data=record(v), cutoff=6)
        (root/'bundle_manifest.json').write_text(json.dumps(manifest))
        model_source = root/'numeric_source.py'
        shutil.copy2(Path(model.__file__),model_source)
        weighting = root/'weighting.py'
        weighting.write_text('import numpy as np\ndef training_weights(root,day,y,h): return np.ones(len(y)), {"synthetic":True}\n')
        with contextlib.redirect_stdout(io.StringIO()):
            summary = bag.run(root,root,root/'bag',model_source,weighting,device='cpu')
            cached = bag.run(root,root,root/'cached_bag',model_source,weighting,
                             cache=root/'bag/seed6802',device='cpu')
        assert len(summary['reports']) == len(cached['reports']) == 1
        assert not summary['eligible_for_review'] and not summary['csv_ready']
        arrays=[]
        for seed in bag.SEEDS:
            folder=root/'bag'/f'seed{seed}'/'day6'
            arrays.append(np.load(folder/'predictions.npz')['raw'])
            assert json.loads((folder/'recipe.json').read_text())['recipe']['seed'] == seed
        expected=np.mean(np.stack(arrays).astype('float64'),axis=0)
        assert np.array_equal(np.load(root/'bag/day6.npz')['raw'],expected)
        assert np.array_equal(np.load(root/'cached_bag/day6.npz')['raw'],expected)
        assert not (root/'bag/seed6802/day5').exists()
        assert runner.digest(root/'bag/seed6802/day6/model.pt') == runner.digest(root/'cached_bag/seed6802/day6/model.pt')
        bad=[a.copy() for a in arrays];bad[0][0]=np.nan
        try:bag.mean_predictions(bad)
        except AssertionError:pass
        else:raise AssertionError('Non-finite averaging accepted')
    report=dict(status='PASS',scope='Synthetic CPU integration; not production score',
                checks=['three distinct seeds fitted', 'equal probability averaging',
                        'all checkpoints replay', 'verified seed6802 cache reuse',
                        'failed gate skips day5', 'non-finite rejection'])
    (parent.parent/'results/choice_bag_driver_test.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
