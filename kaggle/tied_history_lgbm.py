"""Fixed LightGBM comparison using preserved causal OOF factors, without refitting factors."""
import gc
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PARAMS = dict(objective='binary', learning_rate=.05, num_leaves=31, max_depth=6,
              min_sum_hessian_in_leaf=50, lambda_l2=5., lambda_l1=.1,
              feature_fraction=.9, bagging_fraction=.8, bagging_freq=1,
              max_bin=63, num_threads=4, seed=7502, verbosity=-1)
ROUNDS = 800
BLEND = .1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def run():
    started = time.monotonic()
    roots = list(Path('/kaggle/input').rglob('bundle_manifest.json'))
    neural = list(Path('/kaggle/input').rglob('neural_manifest.json'))
    previous = list(Path('/kaggle/input').rglob('trees6/predictions.npz'))
    assert len(roots) == len(neural) == len(previous) == 1
    root, nr, prior = roots[0].parent, neural[0].parent, previous[0].parent.parent
    out = Path('/kaggle/working/tied-history-lgbm-001')
    out.mkdir(exist_ok=False)
    # Probe only runtime capability; no validation labels or model settings selected.
    device = 'cpu'
    try:
        probe = lgb.Dataset(np.arange(400, dtype='float32').reshape(100,4), label=np.arange(100)%2)
        lgb.train(dict(PARAMS, device_type='gpu'), probe, num_boost_round=1)
        device = 'gpu'
    except lgb.basic.LightGBMError as error:
        print(json.dumps(dict(stage='gpu_probe', fallback='cpu', reason=str(error))), flush=True)
    params = dict(PARAMS, device_type=device)
    frozen = dict(params=params, rounds=ROUNDS, blend=BLEND, factor_retraining=False,
                  source_sha256=digest(__file__), prior_prediction_sha256=digest(previous[0]),
                  expected_prior_prediction_sha256='3ad63970946c7eb4b0e5f51efa9ad4a761a90bee93ab0d79cbb22c24449198ff',
                  fresh_holdout=False, promoted=False, official_auc=None)
    assert frozen['prior_prediction_sha256'] == frozen['expected_prior_prediction_sha256']
    (out/'frozen.json').write_text(json.dumps(frozen, indent=2))
    print(json.dumps(frozen), flush=True)
    spec = importlib.util.spec_from_file_location('causal_features', root/'runner.py')
    helpers = importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    manifest = helpers.load_bundle(root, verify=True)
    with np.load(nr/'events.npz') as z:
        hist, target = z['hist'], z['target']
    with np.load(nr/'val6.npz') as z:
        groups = np.digitize((hist[z['indices']]>=0).sum(1), [0,4,9,19], right=True)
        source = np.bincount(groups,minlength=5)/len(groups)
        w = z['weights'].reshape(-1,10)[:,0]
        shares = source*np.array([w[groups==g][0] for g in range(5)])
        assert np.isclose(shares.sum(),1)
    specs = [manifest['train'][str(d)] for d in [3,4,5]]
    labels, ids, extra = [], [], []
    for d,s in zip([3,4,5],specs):
        factor = prior/f'factors{d}'
        report = json.loads((factor/'report.json').read_text())
        assert report['factor_fit_last_timestamp'] < report['cutoff'] <= report['score_first_timestamp']
        assert digest(factor/'features.npy') == report['feature_sha256']
        dayids = np.load(factor/'event_ids.npy')
        y = np.load(root/s['labels'])
        assert np.array_equal(y,(target[dayids,None]==np.arange(10)).ravel())
        labels.append(y);ids.append(dayids);extra.append(np.load(factor/'features.npy',mmap_mode='r'))
    labels = np.concatenate(labels).astype('float32');ids=np.concatenate(ids)
    groups = np.digitize((hist[ids]>=0).sum(1),[0,4,9,19],right=True)
    source=np.bincount(groups,minlength=5)/len(groups)
    weights=np.repeat((shares/source)[groups],10).astype('float32');weights/=weights.mean()
    base=helpers.FeatureBlocks(root,[s['features'] for s in specs])
    assert len(base)==len(labels)==2937590
    extra=np.concatenate(extra)
    assert extra.shape==(len(labels),17)
    dense=np.lib.format.open_memmap(out/'training_features.npy',mode='w+',dtype='float32',shape=(len(labels),371))
    for off in range(0,len(labels),100000):
        end=min(off+100000,len(labels));dense[off:end,:354]=base.block(off,end);dense[off:end,354:]=extra[off:end]
    dense.flush();del extra,hist,target,ids;gc.collect()
    validation=manifest['validation']['6'];val=np.load(root/validation['data'])
    old=np.load(previous[0])
    for key in ['y','events','weights','baseline']:
        np.testing.assert_allclose(val[key],old[key],rtol=0,atol=1e-12)
    vb=helpers.FeatureBlocks(root,[validation['features']])
    vx=np.empty((len(vb),371),dtype='float32')
    for off in range(0,len(vb),100000):
        end=min(off+100000,len(vb));vx[off:end,:354]=vb.block(off,end)
    vx[:,354:]=np.load(prior/'trees6/validation_extra.npy')
    predictions={}
    for arm,width in [('control',354),('treatment',371)]:
        print(json.dumps(dict(stage='tree_start',arm=arm,rows=len(labels),features=width)),flush=True)
        dataset=lgb.Dataset(dense[:,:width],label=labels,weight=weights,free_raw_data=True,
                            params={'max_bin':63})
        def progress(env):
            if (env.iteration+1)%100==0:
                print(json.dumps(dict(stage='tree_training',arm=arm,round=env.iteration+1,total=ROUNDS)),flush=True)
        model=lgb.train(params,dataset,num_boost_round=ROUNDS,callbacks=[progress])
        assert model.current_iteration()==ROUNDS
        model.save_model(str(out/f'{arm}.txt'))
        raw=model.predict(vx[:,:width],num_threads=4)
        replay=lgb.Booster(model_file=str(out/f'{arm}.txt'))
        assert np.array_equal(raw,replay.predict(vx[:,:width],num_threads=4))
        rank=lambda p:rankdata(np.asarray(p,dtype='float64'),method='average')/(len(p)+1)
        predictions[arm+'_raw']=raw
        predictions[arm]=(1-BLEND)*rank(val['baseline'])+BLEND*rank(raw)
        del model,replay,dataset;gc.collect()
    def metric(p):
        return dict(ordinary_auc=float(roc_auc_score(val['y'],p)),
                    adjusted_auc=float(roc_auc_score(val['y'],p,sample_weight=val['weights'])))
    report=dict(baseline=metric(val['baseline']),**{k:metric(v) for k,v in predictions.items()},
                factor_retraining=False,training_rows=len(labels),elapsed_seconds=time.monotonic()-started,
                fresh_holdout=False,promoted=False,csv_ready=False,official_auc=None,target_achieved=False)
    report['adjusted_gain']=report['treatment']['adjusted_auc']-report['baseline']['adjusted_auc']
    report['gain_vs_control']=report['treatment']['adjusted_auc']-report['control']['adjusted_auc']
    report['development_gate']=(report['adjusted_gain']>=.0005 and report['gain_vs_control']>0
                                and report['treatment']['ordinary_auc']>=report['baseline']['ordinary_auc'])
    np.savez_compressed(out/'predictions.npz',y=val['y'],events=val['events'],weights=val['weights'],
                        baseline=val['baseline'],**predictions)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    # Keep the saved models and predictions; remove only our disposable design matrix.
    del dense;gc.collect()
    (out/'training_features.npy').unlink()
    return report
