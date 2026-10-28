"""Full-population grouped ranking with a causal neural event-click mass.

Different from the earlier 30k-event sampled ranker: all prior-date queries,
history-population query weights and a fixed GPU recipe. No held-out tuning.
"""
import gc
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb

ROUNDS = 600
PARAMS = dict(objective='rank:pairwise', tree_method='hist', max_depth=6,
              max_bin=63, eta=.05, min_child_weight=50, subsample=.8,
              colsample_bytree=.9, reg_lambda=10., reg_alpha=.1,
              nthread=4, seed=6901, max_cached_hist_node=32)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def probabilities(scores, mass):
    scores = np.asarray(scores).reshape(-1,10).astype('float64')
    mass = np.asarray(mass).reshape(-1)
    assert len(scores) == len(mass) and np.isfinite(scores).all()
    assert np.isfinite(mass).all() and ((mass >= 0) & (mass <= 1)).all()
    exp = np.exp(scores-scores.max(1,keepdims=True))
    return (exp/exp.sum(1,keepdims=True)*mass[:,None]).ravel()


def fit(root, neural_root, output, manifest, day, h, neural, weighting, neural_output, device):
    output.mkdir(parents=True, exist_ok=False)
    specs = [manifest['train'][str(d)] for d in range(day-3,day)]
    validation = manifest['validation'][str(day)]
    labels = []
    for spec in specs:
        meta = json.loads((root/spec['metadata']).read_text())
        assert meta['last_timestamp'] < validation['cutoff']
        labels.append(np.load(root/spec['labels']))
    y = np.concatenate(labels).astype('float32')
    assert len(y)%10 == 0 and ((y == 0) | (y == 1)).all()
    assert (y.reshape(-1,10).sum(1) <= 1).all()
    weights, weight_recipe = weighting.training_weights(neural_root,day,y,h)
    assert np.array_equal(weights.reshape(-1,10),np.repeat(weights.reshape(-1,10)[:,:1],10,axis=1))
    features = h.FeatureBlocks(root,[s['features'] for s in specs])
    assert len(features) == len(y)
    dev, cp = h.choose_device(device)
    matrix = xgb.QuantileDMatrix(h.BatchIterator(features,y,cp),nthread=4,max_bin=63)
    matrix.set_group(np.full(len(y)//10,10,dtype='uint32'))
    matrix.set_weight(weights[::10])
    recipe = dict(params=dict(PARAMS,device=dev),rounds=ROUNDS,fit_rows=len(y),
                  fit_queries=len(y)//10,fit_days=list(range(day-3,day)),
                  clicked_queries=int((y.reshape(-1,10).sum(1)>0).sum()),
                  validation_cutoff=validation['cutoff'],weighting=weight_recipe,
                  blend_weight=.1,source_sha256=h.digest(Path(__file__)),
                  manifest_sha256=h.digest(root/'bundle_manifest.json'),
                  xgboost=xgb.__version__,
                  selection='Fixed full-population ranking; no early stopping or validation-label training')
    (output/'recipe.json').write_text(json.dumps(recipe,indent=2))
    print(dict(stage='fit_full_ranker',day=day,rows=len(y),queries=len(y)//10),flush=True)
    model = xgb.train(recipe['params'],matrix,num_boost_round=ROUNDS,
                      callbacks=[h.Progress(output,ROUNDS)])
    checkpoint = output/'model.ubj'; model.save_model(checkpoint)
    del matrix,y,features
    gc.collect()
    if cp is not None: cp.get_default_memory_pool().free_all_blocks()
    val = dict(np.load(root/validation['data']))
    xv = h.FeatureBlocks(root,[validation['features']])
    def predict(m):
        result=[]
        for off in range(0,len(xv),25000):
            x=xv.block(off,min(off+25000,len(xv)))
            p=m.inplace_predict(cp.asarray(x) if cp is not None else x)
            result.append(cp.asnumpy(p) if cp is not None else p)
        return np.concatenate(result)
    scores = predict(model)
    restored = xgb.Booster(); restored.load_model(checkpoint)
    restored.set_param({'device':dev,'nthread':4})
    assert np.array_equal(predict(restored),scores)
    directory = neural_output/f'day{day}'
    nr = json.loads((directory/'report.json').read_text())
    assert h.digest(directory/'model.pt') == nr['model_sha256']
    nn = dict(np.load(directory/'predictions.npz'))
    for key in ['y','events','baseline','weights']:
        assert np.array_equal(nn[key],val[key]),key
    state = torch.load(directory/'model.pt',map_location=dev,weights_only=True)
    nm = neural.ChoiceNet(state['mean'],state['scale'],neural.RECIPE).to(dev)
    nm.load_state_dict(state)
    assert np.array_equal(neural.predict(nm,xv,dev),nn['raw'])
    mass = np.clip(nn['raw'].reshape(-1,10).sum(1),0,1)
    raw = probabilities(scores,mass)
    pred = h.rank(.9*h.rank(val['baseline'])+.1*h.rank(raw))
    m = h.metrics(val['y'],pred,val['baseline'],val['weights'])
    versus_single = h.metrics(val['y'],pred,nn['pred'],val['weights'])
    m.update(day=day,versus_single=versus_single,raw_auc=float(h.roc_auc_score(val['y'],raw)),
             saved_checkpoint_replay='PASS, ranker and neural exact',rows=len(raw),
             model_sha256=h.digest(checkpoint),csv_ready=False,official_auc=None)
    np.savez_compressed(output/'predictions.npz',raw=raw,pred=pred,scores=scores,event_mass=mass,**val)
    shutil.copy2(directory/'model.pt',output/'neural_mass_model.pt')
    shutil.copy2(directory/'recipe.json',output/'neural_mass_recipe.json')
    (output/'report.json').write_text(json.dumps(m,indent=2))
    print('FULL_RANK_RESULT',json.dumps(m,indent=2),flush=True)
    return m


def run(root,neural_root,output,model_source,weighting_source,neural_output,device='cuda'):
    root,neural_root,output,neural_output=map(Path,[root,neural_root,output,neural_output])
    h=module('full_rank_helpers',root/'runner.py')
    neural=module('full_rank_neural',model_source)
    weighting=module('full_rank_weights',weighting_source)
    assert neural.RECIPE['epochs']==8 and neural.RECIPE['seed']==6802
    manifest=h.load_bundle(root,verify=True)
    if not (neural_output/'summary.json').exists():
        neural.run(root,neural_root,neural_output,weighting_source,device)
    assert json.loads((neural_output/'summary.json').read_text())['eligible_for_review']
    output.mkdir(parents=True,exist_ok=False)
    for src,name in [(Path(__file__),'full_rank_runner.py'),(Path(model_source),'neural_source.py'),
                     (Path(weighting_source),'weighting_source.py')]:
        shutil.copy2(src,output/name)
    reports=[]
    for day in [6,5]:
        r=fit(root,neural_root,output/f'day{day}',manifest,day,h,neural,weighting,neural_output,device)
        reports.append(r)
        if not (r['positive_gate'] and r['versus_single']['positive_gate']):break
    eligible=len(reports)==2 and all(r['positive_gate'] and r['versus_single']['positive_gate'] for r in reports)
    summary=dict(reports=reports,eligible_for_review=eligible,target=.83,
                 target_achieved=bool(eligible and reports[0]['history_adjusted_auc']>=.83),
                 csv_ready=False,official_auc=None,note='Reused development dates; no official forecast.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary
