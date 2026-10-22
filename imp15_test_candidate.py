"""Independent exact ALS score/rank recipe and frozen validation audit."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from tst2_candidate import influence, weighted_influence


def audit(day, selected):
    with np.load(ARTIFACT_DIR / f'imp15_model_day{day}.npz') as f:
        model = {k: f[k] for k in f.files}
    d = np.load(ARTIFACT_DIR / f'imp15_als_day{day}.npz')
    b = np.load(ARTIFACT_DIR / f'imp11_bilinear_day{day}.npz')
    U,V,bu,bi = [model[k] for k in ['U','V','bu','bi']]
    g = float(model['g'])
    nu,ni = len(U),len(V)
    assert U.shape == (nu,16) and V.shape == (ni,16)
    assert bu.shape == (nu,) and bi.shape == (ni,) and model['exposures'].shape == (nu,)
    assert all(np.isfinite(model[k]).all() for k in ['U','V','bu','bi','g','exposures'])
    data = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    hist = np.load(ARTIFACT_DIR / 'train_hist.npy', mmap_mode='r')
    times = np.load(ARTIFACT_DIR / 'train_times.npy', mmap_mode='r')
    cutoff = int(data[:,2].min()) + day*86400
    assert int(model['cutoff']) == cutoff and int(model['iteration']) == selected['iteration']
    prefix = data[data[:,2] < cutoff]
    assert np.array_equal(np.bincount(prefix[:,0],minlength=nu), model['exposures'])
    assert abs(float(prefix[:,5].mean())-g) < 1e-15
    y,events,raw,base = b['y'], b['events'], d['raw'], b['pred']
    assert np.array_equal(y,d['y']) and np.array_equal(events,d['events'])
    groups = events.reshape(-1,10)
    assert (groups==groups[:,:1]).all() and len(np.unique(groups[:,0]))==len(groups)
    assert np.isfinite(raw).all()
    eids = groups[:,0]
    rows = data[(eids[:,None]*10+np.arange(10)).ravel()]
    assert np.array_equal(rows[:,5],y) and np.array_equal(rows[:,6],events)
    e = rows[::10]
    assert ((e[:,2]>=cutoff)&(e[:,2]<cutoff+86400)).all()
    users,items = e[:,0], rows[:,1].reshape(-1,10)
    known = model['exposures'][users]>0
    direct = g+bu[users[known],None]+bi[items[known]]+np.einsum('ed,ecd->ec',U[users[known]],V[items[known]])
    known_error = float(np.max(np.abs(direct-raw.reshape(-1,10)[known])))
    assert known_error < 1e-12
    cold = np.flatnonzero(~known)
    sample = cold[np.linspace(0,len(cold)-1,min(100,len(cold)),dtype=int)] if len(cold) else []
    design = np.column_stack([np.ones(ni),V])
    gram = design.T@design
    background_rhs = -design.T@(g+bi)
    errors=[]
    for event in sample:
        h,t = hist[eids[event]],times[eids[event]]
        valid = h>=0
        assert (t[valid]<e[event,2]).all()
        unique = sorted(set(zip(h[valid].tolist(),t[valid].tolist())))
        history_items = np.array([item for item,stamp in unique],dtype=int)
        x = design[history_items]
        confidence = 9*len(unique)/ni
        a = confidence*gram + x.T@x + np.diag(np.r_[20.,np.ones(16)])
        rhs = confidence*background_rhs+x.T@(1-g-bi[history_items])
        values = np.linalg.solve(a,rhs)
        expected = g+values[0]+bi[items[event]]+V[items[event]]@values[1:]
        errors.append(float(np.max(np.abs(expected-raw.reshape(-1,10)[event]))))
    cold_error = max(errors,default=0.)
    assert cold_error < 2e-10
    weight = selected['weight']
    base_rank = rankdata(base,method='average')/(len(base)+1)
    raw_rank = rankdata(raw,method='average')/(len(raw)+1)
    pred = (1-weight)*base_rank+weight*raw_rank
    recipe_error = float(np.max(np.abs(pred-d['pred'])))
    assert recipe_error < 1e-15
    assert ((pred>0)&(pred<1)).all()
    vg=np.digitize((hist[events]>=0).sum(1),[0,4,9,19],right=True)
    eh=np.load(ARTIFACT_DIR/'evaluation_hist.npy',mmap_mode='r')
    eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
    sw=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
    result=dict(day=day,rows=len(y),iteration=int(model['iteration']),weight=weight,
                prefix_global_ctr=g,prefix_exposures_exact=True,known_events=int(known.sum()),
                full_known_raw_replay_max_error=known_error,cold_events=len(cold),cold_dense_checked=len(errors),
                cold_dense_raw_replay_max_error=cold_error,rank_recipe_max_error=recipe_error)
    for name,weights,fn in [('auc',None,influence),('weighted_auc',sw,weighted_influence)]:
        old,new=roc_auc_score(y,base,sample_weight=weights),roc_auc_score(y,pred,sample_weight=weights)
        args=() if weights is None else (weights,)
        se=np.linalg.norm(fn(y,pred,*args)-fn(y,base,*args))
        result[name]=dict(baseline=float(old),candidate=float(new),delta=float(new-old),
                          conditional_paired_event_95pct_interval=[float(new-old-1.96*se),float(new-old+1.96*se)])
    if day==6:
        candidates=[]
        for w in [.05,.1,.2]:
            p=(1-w)*base_rank+w*raw_rank
            auc=roc_auc_score(y,p);weighted=roc_auc_score(y,p,sample_weight=sw)
            if auc>result['auc']['baseline'] and weighted>result['weighted_auc']['baseline']:
                candidates.append((auc,w))
        assert candidates and max(candidates)[1]==weight
    return result


if __name__=='__main__':
    selections=json.loads((ARTIFACT_DIR/'imp15_als_report.json').read_text())
    assert {r['day'] for r in selections}=={5,6}
    assert len({r['iteration'] for r in selections})==len({r['weight'] for r in selections})==1
    trials=json.loads((ARTIFACT_DIR/'imp15_als_trials_day6.json').read_text())
    eligible=[r for r in trials if r['auc']>r['baseline_auc'] and r['reweighted']>r['baseline_reweighted']]
    best=max(eligible,key=lambda r:r['auc'])
    frozen=next(r for r in selections if r['day']==6)
    assert best['iteration']==frozen['iteration'] and best['weight']==frozen['weight']
    with threadpool_limits(limits=1):
        reports=[audit(r['day'],r) for r in selections]
    result=dict(status='PASS',reports=reports,
                both_dates_improved=all(r[k]['delta']>0 for r in reports for k in ['auc','weighted_auc']),
                ranking_scores_only=True,official_score=None,
                note='Uniform unknown alternatives are a modeling assumption. Intervals exclude reused-validation selection uncertainty.')
    (ARTIFACT_DIR/'imp15_test_candidate_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
