"""Independent 63-leaf LightGBM prediction, rank blend and frozen-baseline audit; no fitting."""
import ast
import hashlib
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
ROOT = ARTIFACT_DIR
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from tst2_candidate import influence, weighted_influence


def settings():
    tree = ast.parse((SOURCE_ROOT / 'imp10_leafwise.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in n.targets)]
    ns = {}
    exec(compile(tree, 'imp10_leafwise.py', 'exec'), ns)
    return dict(ns['SETTINGS'], num_leaves=63)


def load_frozen_baseline():
    manifest=ROOT/'imp20_baseline.json'
    frozen=json.loads(manifest.read_text())
    frozen['_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    for day in [6,5]:
        member=ROOT/f"{frozen['baseline']}_day{day}.npz"
        assert hashlib.sha256(member.read_bytes()).hexdigest()==frozen['sha256'][str(day)]
    audit=ROOT/frozen['audit_filename']
    assert hashlib.sha256(audit.read_bytes()).hexdigest()==frozen['independent_audit_sha256']
    accepted=json.loads(audit.read_text())
    assert accepted['status']=='PASS' and accepted['both_dates_improved']
    return frozen


def check_record_manifest(record,frozen):
    assert record['baseline']==frozen['baseline']
    assert record['baseline_manifest_sha256']==frozen['_manifest_sha256']


def check_checkpoints(cuts):
    data = np.load(ROOT / 'train.npy', mmap_mode='r')
    start = int(data[:, 2].min())
    results = []
    for cut in sorted(cuts):
        filename = f'imp15_model_day{cut}.npz' if cut in [5, 6] else f'imp17_als_cut{cut}.npz'
        with np.load(ROOT / filename) as source:
            model = {key: source[key] for key in source.files}
        assert int(model['cutoff']) == start + cut * 86400
        assert int(model['iteration']) == 6
        nusers, nitems = len(model['U']), len(model['V'])
        assert model['U'].shape == (nusers, 16) and model['V'].shape == (nitems, 16)
        assert model['bu'].shape == model['exposures'].shape == (nusers,)
        assert model['bi'].shape == (nitems,)
        assert all(np.isfinite(model[key]).all() for key in ['U', 'V', 'bu', 'bi', 'g', 'exposures'])
        prefix = data[data[:, 2] < int(model['cutoff'])]
        assert np.array_equal(np.bincount(prefix[:, 0], minlength=nusers), model['exposures'])
        assert abs(float(prefix[:, 5].mean()) - float(model['g'])) < 1e-15
        with np.load(ROOT / 'imp17_als_cut2.npz') as anchor:
            anchor_v=anchor['V'].astype(float)
        rotation_path=ROOT/f'imp19_rotation_cut{cut}.npz'
        with np.load(rotation_path) as archive:
            r=archive['rotation'];s=archive['singular_values']
            assert int(archive['cutoff'])==int(model['cutoff'])
            assert int(archive['anchor_cutoff'])==start+2*86400<=int(archive['cutoff'])
        assert r.shape==(16,16) and s.shape==(16,) and np.isfinite(r).all() and np.isfinite(s).all()
        orth=float(np.max(np.abs(r.T@r-np.eye(16))))
        assert orth<1e-10
        def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        cross=unit(model['V']).T@unit(anchor_v)
        singular=np.linalg.svd(cross,compute_uv=False)
        objective_error=float(abs(np.trace(r.T@cross)-singular.sum()))
        assert objective_error<1e-8 and np.allclose(s,singular,rtol=0,atol=1e-9)
        results.append(dict(cut=cut, checkpoint=filename, cutoff=int(model['cutoff']),
                            iteration=6, prefix_exposures_and_global_ctr_exact=True,
                            rotation_anchor_cut=2,rotation_orthogonality_error=orth,
                            procrustes_objective_error=objective_error))
    return results


def check_manual_products(day,events,indices,actual):
    data=np.load(ROOT/'train.npy',mmap_mode='r')
    histories=np.load(ROOT/'train_hist.npy',mmap_mode='r')
    times=np.load(ROOT/'train_times.npy',mmap_mode='r')
    rows=data[events[indices]*10+indices%10]
    assert np.array_equal(rows[:,6],events[indices])
    with np.load(ROOT/f'imp15_model_day{day}.npz') as source:
        U,V,bu,bi,exposures=[source[key] for key in ['U','V','bu','bi','exposures']]
        g=float(source['g'])
    with np.load(ROOT/f'imp19_rotation_cut{day}.npz') as archive:r=archive['rotation']
    def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    design=np.column_stack([np.ones(len(V)),V]);gram=design.T@design
    rhs=-design.T@(g+bi)
    expected=[];cosine_errors=[];cold=0;empty=0
    for event,row in enumerate(rows[::10]):
        h=histories[row[6]];valid=h>=0;h=h[valid];t=times[row[6]][valid]
        assert (t<row[2]).all()
        user=int(row[0])
        if exposures[user]:u=U[user]
        else:
            cold+=1
            unique=sorted(set(zip(h.tolist(),t.tolist())))
            ids=np.array([item for item,stamp in unique],int)
            x=design[ids];confidence=9*len(ids)/len(V)
            answer=np.linalg.solve(confidence*gram+x.T@x+np.diag(np.r_[20.,np.ones(16)]),
                                   confidence*rhs+x.T@(1-g-bi[ids]))
            u=answer[1:]
        profile=V[h].mean(0) if len(h) else np.zeros(16)
        if not len(h):empty+=1
        for item in rows[event*10:event*10+10,1]:
            p=np.r_[unit(u@r)*unit(V[item]@r),unit(profile@r)*unit(V[item]@r)]
            expected.append(p)
            cosine_errors.extend([abs(p[:16].sum()-unit(u)@unit(V[item])),
                                  abs(p[16:].sum()-unit(profile)@unit(V[item]))])
    error=float(np.max(np.abs(np.array(expected,np.float32)-actual)))
    assert error<1e-6 and max(cosine_errors)<1e-10
    return dict(events=len(rows)//10,cold_dense_profiles=cold,empty_histories=empty,
                product_max_error=error,cosine_sum_error=max(cosine_errors))


def check(day, selected):
    rawfile = np.load(ROOT / f'imp20_tree_raw_day{day}.npz')
    frozen=load_frozen_baseline()
    baseline_path=ROOT/f"{frozen['baseline']}_day{day}.npz"
    baseline=np.load(baseline_path)
    y, events, raw, base = baseline['y'], baseline['events'], rawfile['raw'], baseline['pred']
    assert np.array_equal(rawfile['events'], events) and np.array_equal(rawfile['y'], y)
    groups = events.reshape(-1, 10)
    assert (groups == groups[:, :1]).all() and len(np.unique(groups[:, 0])) == len(groups)
    assert (y.reshape(-1, 10).sum(1) <= 1).all()
    data=np.load(ROOT/'train.npy',mmap_mode='r')
    cutoff=int(data[:,2].min())+day*86400
    event_stream=data[::10]
    day_events=event_stream[(event_stream[:,2]>=cutoff)&(event_stream[:,2]<cutoff+86400)]
    _,first=np.unique(day_events[:,0],return_index=True)
    assert np.array_equal(day_events[first,6],groups[:,0])
    rows=data[(groups[:,0,None]*10+np.arange(10)).ravel()]
    assert np.array_equal(rows[:,5],y) and np.array_equal(rows[:,6],events)
    assert np.isfinite(raw).all() and ((raw > 0) & (raw < 1)).all()
    model = lgb.Booster(model_file=str(ROOT / f'imp20_tree_d{day}.txt'))
    assert model.num_feature() == 354
    assert 0<model.num_trees()<=1600
    expected = settings()
    for key in ['num_leaves', 'learning_rate', 'min_data_in_leaf', 'feature_fraction', 'bagging_fraction',
                'bagging_freq', 'lambda_l2', 'max_bin', 'seed', 'num_threads']:
        assert abs(float(model.params[key]) - float(expected[key])) < 1e-12, (key, model.params[key])
    assert model.params['objective'] == 'binary' and 'auc' in model.params['metric']
    assert model.params['deterministic'] and model.params['force_col_wise']
    ix = (np.linspace(0, len(groups) - 1, 100, dtype=int)[:, None] * 10 + np.arange(10)).ravel()
    files = [f'stack_val{day}_c{day}.npy', f'imp3_wf_val{day}.npy', f'imp4_decompose_val{day}.npy',
             f'imp5_aligned_val{day}.npy', f'imp6_context_val{day}.npy', f'imp17_als_val{day}.npy', f'imp19_als_aligned_val{day}.npy']
    matrices = [np.load(ROOT / 'features' / filename, mmap_mode='r') for filename in files]
    assert [x.shape for x in matrices] == [(len(y), 190), (len(y), 4), (len(y), 32), (len(y), 32), (len(y), 32), (len(y), 32), (len(y), 32)]
    features = np.column_stack([x[ix] for x in matrices])
    replay = model.predict(features, num_threads=1)
    model_error = float(np.max(np.abs(replay - raw[ix])))
    assert model_error < 1e-12, model_error
    h = np.load(ROOT / 'train_hist.npy', mmap_mode='r')
    eh = np.load(ROOT / 'evaluation_hist.npy', mmap_mode='r')
    vg = np.digitize((h[events] >= 0).sum(1), [0, 4, 9, 19], right=True)
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    sw = ((np.bincount(eg, minlength=5) / len(eg)) / (np.bincount(vg, minlength=5) / len(vg)))[vg]
    result = dict(day=day, rows=len(y), trees=model.num_trees(), features=model.num_feature(),
                  raw_auc=float(roc_auc_score(y, raw)), raw_weighted_auc=float(roc_auc_score(y, raw, sample_weight=sw)),
                  replay_rows=len(ix), saved_model_replay_max_error=model_error)
    result['manual_new32']=check_manual_products(day,events,ix,matrices[-1][ix])
    old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    accepted=json.loads((ROOT/frozen['audit_filename']).read_text())
    assert accepted['status']=='PASS' and accepted['both_dates_improved']
    incumbent=next(r for r in accepted['reports'] if r['day']==day)
    assert abs(old-incumbent['auc']['candidate'])<1e-12
    assert abs(oldw-incumbent['weighted_auc']['candidate'])<1e-12
    result['accepted_incumbent_metrics_verified']=True
    base_rank=rankdata(base,method='average')/(len(base)+1)
    raw_rank=rankdata(raw,method='average')/(len(raw)+1)
    trial_path=ROOT/f'imp20_tree_trials_day{day}.json'
    trials=json.loads(trial_path.read_text())
    if day==6:
        assert len(trials)==5 and {r['weight'] for r in trials}=={.1,.25,.5,.75,1.}
    else:assert len(trials)==1
    eligible=[]
    for trial in trials:
        check_record_manifest(trial,frozen)
        w=trial['weight'];p=(1-w)*base_rank+w*raw_rank
        a=float(roc_auc_score(y,p));wa=float(roc_auc_score(y,p,sample_weight=sw))
        assert trial['day']==day and trial['trees']==model.num_trees()
        for key,value in [('auc',a),('reweighted',wa),('baseline_auc',old),('baseline_reweighted',oldw),('raw_auc',result['raw_auc'])]:
            assert abs(trial[key]-value)<1e-12
        if a>old and wa>oldw:eligible.append((a,w))
    result['recorded_trials_independently_recomputed']=len(trials)
    if selected is None:
        assert not eligible
        result['selection'] = 'NO_ACCEPTED_CANDIDATE'
        return result
    check_record_manifest(selected,frozen)
    d = np.load(ROOT / f'imp20_tree_day{day}.npz')
    assert np.array_equal(d['events'], events) and np.array_equal(d['y'], y)
    assert np.array_equal(d['raw'], raw)
    assert model.num_trees() == selected['trees']
    weight = selected['weight']
    base_rank = rankdata(base, method='average') / (len(base) + 1)
    raw_rank = rankdata(raw, method='average') / (len(raw) + 1)
    pred = (1 - weight) * base_rank + weight * raw_rank
    error = float(np.max(np.abs(pred - d['pred'])))
    assert error < 1e-15
    assert np.isfinite(pred).all() and ((pred > 0) & (pred < 1)).all()
    result.update(weight=weight, recipe_max_error=error)
    for name, weights, fn in [('auc', None, influence), ('weighted_auc', sw, weighted_influence)]:
        old, new = roc_auc_score(y, base, sample_weight=weights), roc_auc_score(y, pred, sample_weight=weights)
        args = () if weights is None else (weights,)
        se = np.linalg.norm(fn(y, pred, *args) - fn(y, base, *args))
        result[name] = dict(baseline=float(old), candidate=float(new), delta=float(new-old),
                            conditional_paired_event_95pct_interval=[float(new-old-1.96*se), float(new-old+1.96*se)])
    assert abs(result['raw_auc']-selected['raw_auc'])<1e-12
    for name,a,b,d,interval in [('auc','auc','baseline_auc','delta','paired_interval'),
                                ('weighted_auc','reweighted','baseline_reweighted','delta_reweighted','weighted_paired_interval')]:
        value=result[name]
        assert abs(value['candidate']-selected[a])<1e-12
        assert abs(value['baseline']-selected[b])<1e-12 and abs(value['delta']-selected[d])<1e-12
        assert np.allclose(value['conditional_paired_event_95pct_interval'],selected[interval],rtol=0,atol=1e-12)
    if day == 6:
        choices = []
        for w in [.1, .25, .5, .75, 1.]:
            candidate = (1 - w) * base_rank + w * raw_rank
            auc = roc_auc_score(y, candidate)
            weighted = roc_auc_score(y, candidate, sample_weight=sw)
            if auc > result['auc']['baseline'] and weighted > result['weighted_auc']['baseline']:
                choices.append((auc, w))
        assert choices and max(choices)[1] == weight
    return result


if __name__ == '__main__':
    capacity_prefit=json.loads((ROOT/'imp20_test_baseline_report.json').read_text())
    assert capacity_prefit['status']=='PASS'
    for name,digest in capacity_prefit['source_sha256'].items():
        assert hashlib.sha256((SOURCE_ROOT/name).read_bytes()).hexdigest()==digest
    frozen=load_frozen_baseline()
    prefit=json.loads((ROOT/'imp19_test_features_report.json').read_text())
    assert prefit['status']=='PASS'
    for name,digest in prefit['source_sha256'].items():
        assert hashlib.sha256((SOURCE_ROOT/name).read_bytes()).hexdigest()==digest
    selected = json.loads((ROOT / 'imp20_tree_report.json').read_text())
    reports = []
    byday = {row['day']: row for row in selected if row is not None}
    with threadpool_limits(limits=1):
        for day in [6, 5]:
            if (ROOT / f'imp20_tree_raw_day{day}.npz').exists():
                reports.append(check(day, byday.get(day)))
    assert reports
    cuts = {cut for row in reports for cut in range(row['day'] - 3, row['day'] + 1)}
    with threadpool_limits(limits=1):checkpoints = check_checkpoints(cuts)
    if 6 in byday and 5 in byday:
        assert byday[6]['trees'] == byday[5]['trees'] and byday[6]['weight'] == byday[5]['weight']
    both_gain = len(reports) == 2 and all('auc' in r and r['auc']['delta'] > 0 and r['weighted_auc']['delta'] > 0 for r in reports)
    selection=next((r for r in reports if r['day']==6),None)
    conditional_gate=bool(selection and 'auc' in selection and
                          selection['auc']['conditional_paired_event_95pct_interval'][0]>0 and
                          selection['weighted_auc']['conditional_paired_event_95pct_interval'][0]>0)
    result = dict(status='PASS', reports=reports, causal_als_checkpoints=checkpoints, both_dates_improved=both_gain,
                  day6_conditional_gate_passed=conditional_gate,acceptance_gates_passed=both_gain and conditional_gate,
                  baseline=frozen['baseline'],baseline_manifest_sha256=frozen['_manifest_sha256'],
                  baseline_audit_sha256=frozen['independent_audit_sha256'],
                  ranking_scores_only=True, note='Conditional intervals exclude reused-validation model/iteration/blend selection uncertainty.')
    (ROOT / 'imp20_test_candidate_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
