"""Independent frozen NCE candidate/provenance audit; no parameter fitting."""
import hashlib
import json
import sys
from pathlib import Path

from config import ARTIFACT_DIR
ROOT=ARTIFACT_DIR
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from tst2_candidate import influence,weighted_influence
import imp18_nce_core as core


def audit(day,selected,frozen):
    baseline=frozen['baseline']
    baseline_path=ROOT/f'{baseline}_day{day}.npz'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==frozen['sha256'][str(day)]
    reference=np.load(baseline_path)
    candidate=np.load(ROOT/f'imp18_nce_day{day}.npz')
    with np.load(ROOT/f'imp18_nce_model_day{day}.npz') as source:
        model={name:source[name] for name in source.files}
    y,events,raw,base=reference['y'],reference['events'],candidate['raw'],reference['pred']
    assert np.array_equal(candidate['y'],y) and np.array_equal(candidate['events'],events)
    groups=events.reshape(-1,10)
    assert (groups==groups[:,:1]).all() and len(np.unique(groups[:,0]))==len(groups)
    data=np.load(ROOT/'train.npy',mmap_mode='r')
    hist=np.load(ROOT/'train_hist.npy',mmap_mode='r')
    times=np.load(ROOT/'train_times.npy',mmap_mode='r')
    start=int(data[:,2].min());cutoff=start+day*86400
    assert int(model['cutoff'])==selected['cutoff']==cutoff
    assert int(model['epoch'])==selected['epoch'] and selected['epoch'] in [2,4,6]
    assert str(model['baseline'])==selected['baseline']==baseline and int(model['seed'])==42
    initial_name=f'imp2_model_warm1.0_d16_pc1_day{day}.npz'
    assert str(model['initial_checkpoint'])==initial_name
    assert hashlib.sha256((ROOT/initial_name).read_bytes()).hexdigest()==str(model['initial_sha256'])
    current_core_hash=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    native_core_test=json.loads((ROOT/'imp18_test_core_report.json').read_text())
    assert native_core_test['status']=='PASS'
    assert str(model['core_sha256'])==current_core_hash==native_core_test['core_sha256']
    U,V,H,bu,bi,g=[model[k] for k in ['U','V','H','bu','bi','g']]
    nu,ni=len(U),len(V)
    assert U.shape==(nu,16) and V.shape==H.shape==(ni,16)
    assert bu.shape==model['aw'].shape==(nu,) and bi.shape==model['offset'].shape==(ni,) and g.shape==(1,)
    for name,acc in [('U','aU'),('V','aV'),('H','aH'),('bu','abu'),('bi','abi'),('g','ag'),('aw','aaw')]:
        assert model[name].shape==model[acc].shape
        assert model[name].dtype==model[acc].dtype==np.float32
        assert np.isfinite(model[name]).all() and np.isfinite(model[acc]).all()
        assert (model[acc]>=np.float32(.1)).all()
    rng_state=json.loads(str(model['rng_state']))
    assert rng_state['bit_generator']=='PCG64'
    restored_rng=np.random.default_rng()
    restored_rng.bit_generator.state=rng_state
    assert np.isfinite(restored_rng.random(2)).all()
    # Independently reconstruct the exact first-event cohort from the training stream.
    event_stream=data[::10]
    event_day=event_stream[(event_stream[:,2]>=cutoff)&(event_stream[:,2]<cutoff+86400)]
    _,first=np.unique(event_day[:,0],return_index=True)
    assert np.array_equal(event_day[first,6],groups[:,0])
    rows=data[(groups[:,0,None]*10+np.arange(10)).ravel()]
    assert np.array_equal(rows[:,5],y) and np.array_equal(rows[:,6],events)
    h=np.asarray(hist[groups[:,0]])
    assert ((times[groups[:,0]]<rows[::10,2,None])|(h<0)).all()
    assert (y.reshape(-1,10).sum(1)<=1).all()
    replay=core.predict(rows[::10,0].copy(),rows[:,1].reshape(-1,10).copy(),h,U,V,H,bu,bi,g)
    raw_error=float(np.max(np.abs(replay-raw)))
    assert raw_error<1e-12 and np.isfinite(raw).all()
    # Prefix data, warm records, and fixed contrastive offset use no held-out rows.
    prefix=data[data[:,2]<cutoff]
    assert len(prefix)//10==selected['real_events']
    e=prefix[::10]
    _,first=np.unique(e[:,0],return_index=True)
    earliest=e[first]
    wh=hist[earliest[:,6]];wt=times[earliest[:,6]];valid=wh>=0
    assert ((wt<earliest[:,2,None])|~valid).all()
    users=np.broadcast_to(earliest[:,0,None],wh.shape)
    records=np.column_stack([users[valid],wh[valid],wt[valid]])
    unique=np.unique(records,axis=0)
    assert len(unique)==selected['warm_records']
    counts=np.bincount(unique[:,1],minlength=ni)+1
    offset=np.log(counts/counts.mean()).astype(np.float32)
    assert np.array_equal(offset,model['offset'])
    weight=selected['weight']
    base_rank=rankdata(base,method='average')/(len(base)+1)
    raw_rank=rankdata(raw,method='average')/(len(raw)+1)
    pred=(1-weight)*base_rank+weight*raw_rank
    recipe_error=float(np.max(np.abs(pred-candidate['pred'])))
    assert recipe_error<1e-15 and np.isfinite(pred).all() and ((pred>0)&(pred<1)).all()
    vg=np.digitize((hist[events]>=0).sum(1),[0,4,9,19],right=True)
    eh=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
    eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
    sw=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
    result=dict(day=day,rows=len(y),epoch=selected['epoch'],weight=weight,baseline=baseline,
                raw_model_replay_max_error=raw_error,rank_recipe_max_error=recipe_error,
                exact_first_event_cohort=True,real_events=len(prefix)//10,warm_records=len(unique),
                warm_offset_exact=True,initial_checkpoint_sha256_verified=True,
                core_sha256=current_core_hash,adagrad_and_rng_metadata_valid=True)
    result['raw_auc']=float(roc_auc_score(y,raw))
    assert abs(result['raw_auc']-selected['raw_auc'])<1e-12
    for name,weights,fn in [('auc',None,influence),('weighted_auc',sw,weighted_influence)]:
        old,new=roc_auc_score(y,base,sample_weight=weights),roc_auc_score(y,pred,sample_weight=weights)
        args=() if weights is None else (weights,)
        se=np.linalg.norm(fn(y,pred,*args)-fn(y,base,*args))
        result[name]=dict(baseline=float(old),candidate=float(new),delta=float(new-old),
                          conditional_paired_event_95pct_interval=[float(new-old-1.96*se),float(new-old+1.96*se)])
    for name,reported_auc,reported_base,reported_delta,reported_interval in [
        ('auc','auc','baseline_auc','delta','paired_interval'),
        ('weighted_auc','reweighted','baseline_reweighted','delta_reweighted','weighted_paired_interval')]:
        computed=result[name]
        assert abs(computed['candidate']-selected[reported_auc])<1e-12
        assert abs(computed['baseline']-selected[reported_base])<1e-12
        assert abs(computed['delta']-selected[reported_delta])<1e-12
        assert np.allclose(computed['conditional_paired_event_95pct_interval'],
                           selected[reported_interval],rtol=0,atol=1e-12)
    if day==6:
        possibilities=[]
        for w in [.05,.1,.2]:
            p=(1-w)*base_rank+w*raw_rank
            a=roc_auc_score(y,p);wa=roc_auc_score(y,p,sample_weight=sw)
            if a>result['auc']['baseline'] and wa>result['weighted_auc']['baseline']:possibilities.append((a,w))
        assert possibilities and max(possibilities)[1]==weight
    return result


if __name__=='__main__':
    selected=json.loads((ROOT/'imp18_nce_report.json').read_text())
    assert selected and all(r is not None for r in selected),'No accepted NCE candidate exists'
    assert len({r['epoch'] for r in selected})==len({r['weight'] for r in selected})==1
    frozen=json.loads((ROOT/'imp18_baseline.json').read_text())
    for day in [6,5]:
        path=ROOT/f"{frozen['baseline']}_day{day}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['sha256'][str(day)]
    trials=json.loads((ROOT/'imp18_nce_trials_day6.json').read_text())
    expected_grid={(epoch,weight) for epoch in [2,4,6] for weight in [.05,.1,.2]}
    observed_grid=[(r['epoch'],r['weight']) for r in trials]
    assert len(observed_grid)==9 and len(set(observed_grid))==9 and set(observed_grid)==expected_grid
    eligible=[r for r in trials if r['auc']>r['baseline_auc'] and r['reweighted']>r['baseline_reweighted']]
    best=max(eligible,key=lambda r:r['auc']);selection=next(r for r in selected if r['day']==6)
    assert best['epoch']==selection['epoch'] and best['weight']==selection['weight']
    with threadpool_limits(limits=1):
        reports=[audit(r['day'],r,frozen) for r in selected]
    result=dict(status='PASS',reports=reports,
                both_dates_improved=len(reports)==2 and all(r[k]['delta']>0 for r in reports for k in ['auc','weighted_auc']),
                ranking_scores_only=True,official_score=None,
                note='Conditional intervals exclude reused-validation model/epoch/blend selection uncertainty.')
    (ROOT/'imp18_test_candidate_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
