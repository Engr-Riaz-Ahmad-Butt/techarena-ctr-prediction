"""Independent saved neural/scaler, causal features and blend audit; no fit."""
import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tst2_candidate import influence, weighted_influence


def caches(name, day):
    if name == 'val':
        files = [f'stack_val{day}_c{day}.npy', f'imp3_wf_val{day}.npy', f'imp4_decompose_val{day}.npy',
                 f'imp5_aligned_val{day}.npy', f'imp6_context_val{day}.npy']
    else:
        files = [f'stack_day{day}_c{day}.npy', f'imp3_wf_day{day}.npy', f'imp4_decompose_day{day}.npy',
                 f'imp5_aligned_day{day}.npy', f'imp6_context_day{day}.npy']
    matrices = [np.load(ARTIFACT_DIR / 'features' / filename, mmap_mode='r') for filename in files]
    assert [x.shape[1] for x in matrices] == [190, 4, 32, 32, 32]
    assert len({len(x) for x in matrices}) == 1
    return matrices


def planned_settings():
    tree = ast.parse((ROOT / 'imp13_neural.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in n.targets)]
    ns = {}
    exec(compile(tree, 'imp13_neural.py', 'exec'), ns)
    return ns['SETTINGS']


def audit(day, selection, verify_scaler):
    d = np.load(ARTIFACT_DIR / f'imp13_neural_day{day}.npz')
    b = np.load(ARTIFACT_DIR / f'imp11_bilinear_day{day}.npz')
    artifact = joblib.load(ARTIFACT_DIR / f'imp13_neural_d{day}.joblib')
    model, scaler = artifact['model'], artifact['scaler']
    y, events, raw, base = b['y'], b['events'], d['raw'], b['pred']
    assert np.array_equal(d['events'], events) and np.array_equal(d['y'], y)
    groups = events.reshape(-1, 10)
    assert (groups == groups[:, :1]).all() and len(np.unique(groups[:, 0])) == len(groups)
    assert (y.reshape(-1, 10).sum(1) <= 1).all()
    assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
    for name, expected in planned_settings().items():
        assert model.get_params()[name] == expected, name
    assert model.n_features_in_ == scaler.n_features_in_ == 290
    assert np.array_equal(model.classes_, [0, 1])
    training_rows = sum(len(caches('day', t)[0]) for t in [day-3, day-2, day-1])
    assert int(scaler.n_samples_seen_) == artifact['training_rows'] == training_rows
    assert artifact['epoch'] == selection['epoch']
    assert artifact['partial_fit_calls'] == artifact['epoch'] * ((training_rows+65535)//65536)
    assert int(model.t_) == artifact['epoch'] * training_rows
    assert model._optimizer.t == artifact['epoch'] * ((training_rows+4095)//4096)
    assert all(np.isfinite(value).all() for value in [*model.coefs_, *model.intercepts_, scaler.mean_, scaler.var_, scaler.scale_])
    data = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    assert artifact['cutoff'] == int(data[:, 2].min()) + day * 86400
    matrices = caches('val', day)
    assert len(matrices[0]) == len(y)
    ix = (np.linspace(0, len(groups)-1, 100, dtype=int)[:, None]*10 + np.arange(10)).ravel()
    features = np.column_stack([matrix[ix] for matrix in matrices]).astype(np.float32)
    normalized = np.clip(scaler.transform(features.copy()), -8, 8)
    replay = model.predict_proba(normalized)[:, 1]
    model_error = float(np.max(np.abs(replay-raw[ix])))
    # Different GEMM batch size can change float32 final rounding.
    assert model_error < 1e-6, model_error
    weight = selection['weight']
    pred = (1-weight)*base + weight*raw
    recipe_error = float(np.max(np.abs(pred-d['pred'])))
    assert recipe_error < 1e-15
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    hist = np.load(ARTIFACT_DIR / 'train_hist.npy', mmap_mode='r')
    ehist = np.load(ARTIFACT_DIR / 'evaluation_hist.npy', mmap_mode='r')
    vg = np.digitize((hist[events]>=0).sum(1), [0,4,9,19], right=True)
    eg = np.digitize((ehist>=0).sum(1), [0,4,9,19], right=True)
    sw = ((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
    result = dict(day=day, rows=len(y), epoch=artifact['epoch'], weight=weight,
                  training_rows=training_rows, partial_fit_calls=artifact['partial_fit_calls'],
                  adam_steps=int(model._optimizer.t), replay_rows=len(ix),
                  saved_model_replay_max_error=model_error, recipe_max_error=recipe_error)
    for name, weights, fn in [('auc', None, influence), ('weighted_auc', sw, weighted_influence)]:
        old, new = roc_auc_score(y,base,sample_weight=weights), roc_auc_score(y,pred,sample_weight=weights)
        args=() if weights is None else (weights,)
        se=np.linalg.norm(fn(y,pred,*args)-fn(y,base,*args))
        result[name]=dict(baseline=float(old), candidate=float(new), delta=float(new-old),
                          conditional_paired_event_95pct_interval=[float(new-old-1.96*se),float(new-old+1.96*se)])
    if day==6:
        trials=json.loads((ARTIFACT_DIR/'imp13_neural_trials_day6.json').read_text())
        eligible=[r for r in trials if r['auc']>r['baseline_auc'] and r['reweighted']>r['baseline_reweighted']]
        best=max(eligible,key=lambda r:r['auc'])
        assert best['epoch']==selection['epoch'] and best['weight']==weight
        assert all(r['epoch'] in [2,4,6,8] and r['weight'] in [.1,.25,.5,.75,1.] for r in trials)
    if verify_scaler:
        independent=StandardScaler()
        for t in [day-3,day-2,day-1]:
            source=caches('day',t)
            for off in range(0,len(source[0]),65536):
                independent.partial_fit(np.column_stack([matrix[off:off+65536] for matrix in source]))
        mean_error=float(np.max(np.abs(independent.mean_-scaler.mean_)))
        variance_error=float(np.max(np.abs(independent.var_-scaler.var_)))
        assert independent.n_samples_seen_==scaler.n_samples_seen_
        assert np.allclose(independent.mean_,scaler.mean_,rtol=1e-10,atol=1e-10)
        assert np.allclose(independent.var_,scaler.var_,rtol=1e-10,atol=1e-10)
        result['independent_training_only_scaler']=dict(mean_max_error=mean_error,variance_max_error=variance_error)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-scaler',action='store_true')
    args=parser.parse_args()
    selections=json.loads((ARTIFACT_DIR/'imp13_neural_report.json').read_text())
    assert selections and all(r is not None for r in selections),'No accepted model to audit'
    assert len({r['epoch'] for r in selections})==len({r['weight'] for r in selections})==1
    with threadpool_limits(limits=1):
        reports=[audit(r['day'],r,args.verify_scaler) for r in selections]
    result=dict(status='PASS',reports=reports,
                both_dates_improved=len(reports)==2 and all(r[k]['delta']>0 for r in reports for k in ['auc','weighted_auc']),
                note='Conditional intervals exclude reused-validation epoch/learner/blend selection uncertainty.')
    (ARTIFACT_DIR/'imp13_test_candidate_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
