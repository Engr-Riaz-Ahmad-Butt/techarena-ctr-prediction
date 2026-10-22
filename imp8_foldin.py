"""Fixed-item, conditional clicked-history recency adaptation."""
from imp4_decompose import *
from imp2_warm_choice import warm_data
from numba import njit, prange, set_num_threads
import math
from threadpoolctl import threadpool_limits
set_num_threads(8)


@njit
def partition(z, background):
    logits=background@z
    maximum=logits.max()
    exp=np.exp(logits-maximum)
    return maximum+math.log(exp.mean()),exp/exp.sum()


@njit(parallel=True)
def adapt(anchors, histories, stamps, event_times, V, background, penalty):
    out=np.zeros_like(anchors)
    objectives=np.zeros((len(anchors),2))
    for e in prange(len(anchors)):
        anchor=anchors[e];z=anchor.copy();n=0.;target=np.zeros(16)
        for q in range(20):
            if histories[e,q]<0:continue
            weight=math.exp(-math.log(2.)*(event_times[e]-stamps[e,q])/(48*3600.))
            target+=weight*V[histories[e,q]];n+=weight
        if n==0:continue
        lse,p=partition(z,background)
        objective=n*lse-np.dot(target,z)
        objectives[e,0]=objective
        for iteration in range(4):
            mean=p@background
            second=p@(background*background)
            gradient=n*mean-target+penalty*(z-anchor)
            curvature=n*np.maximum(second-mean*mean,0)+penalty
            step=gradient/curvature
            step/=max(1.,np.linalg.norm(step))
            accepted=False
            for trial in range(8):
                candidate=z-step*(.5**trial)
                nlse,np_=partition(candidate,background)
                difference=candidate-anchor
                obj=n*nlse-np.dot(target,candidate)+.5*penalty*np.dot(difference,difference)
                if obj<=objective+1e-12:
                    z=candidate;p=np_;objective=obj;accepted=True;break
            if not accepted:break
        out[e]=z-anchor;objectives[e,1]=objective
    return out,objectives


def run(day,chosen=None):
    reference=np.load(ROOT/f'predictions_imp5_warmcontext_day{day}.npz')
    y=reference['y'];events=reference['events'];base=reference['pred']
    _,rows=split(day)
    assert np.array_equal(rows[:,6],events) and np.array_equal(rows[:,5],y)
    a=rows[::10];h=np.asarray(hist[a[:,6]])
    times=np.asarray(np.load(ROOT/'train_times.npy',mmap_mode='r')[a[:,6]])
    assert np.all((times<a[:,2,None])|(h<0))
    with np.load(model_path(day)) as model:
        U,V,H=[model[k].astype(np.float64) for k in ['U','V','H']]
    anchor=U[a[:,0]]+(H[np.maximum(h,0)]*(h>=0)[:,:,None]).sum(1)/np.sqrt(np.maximum((h>=0).sum(1)[:,None],1))
    # Deduplicated warm counts are reconstructed independently of held-out rows.
    prefix=np.asarray(train[train[:,2]<cutoff_of(day)])
    wu,wi=warm_data(prefix);del prefix,wu
    probability=(np.bincount(wi,minlength=len(V))+1).astype(float);probability/=probability.sum()
    pool=np.random.default_rng(814).choice(len(V),1024,replace=True,p=probability)
    background=np.ascontiguousarray(V[pool])
    sw,_=history_weights(events)
    old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    assert ((base>0)&(base<1)).all()
    base_logits=logit(base).reshape(-1,10)
    trials=[];best=None
    for penalty in ([chosen['penalty']] if chosen else [4.,16.,64.]):
        with threadpool_limits(limits=1):
            residual,objectives=adapt(np.ascontiguousarray(anchor),h,times,a[:,2].copy(),V,background,penalty)
        assert np.isfinite(residual).all() and np.all(residual[(h>=0).sum(1)==0]==0)
        assert np.all(objectives[:,1]<=objectives[:,0]+1e-9)
        correction=np.einsum('ed,ecd->ec',residual,V[rows[:,1].reshape(-1,10)])
        for strength in ([chosen['strength']] if chosen else [.1,.25,.5]):
            logits=base_logits+strength*correction
            pred=sigmoid(logits).ravel()
            auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
            result=dict(day=day,penalty=penalty,strength=strength,auc=auc,reweighted=weighted,base_auc=old,base_reweighted=oldw)
            trials.append(result);print(json.dumps(result),flush=True)
            if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
                best=result.copy()
                se=np.linalg.norm(influence(y,pred)-influence(y,base))
                sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
                best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
                np.savez_compressed(ROOT/f'imp8_foldin_day{day}.npz',y=y,events=events,pred=pred,residual=residual,pool=pool,objectives=objectives)
    (ROOT/f'imp8_foldin_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    return best


if __name__=='__main__':
    path=ROOT/'imp8_foldin_report.json'
    assert not path.exists(),'Do not overwrite an existing experiment.'
    selected=run(6);reports=[selected];path.write_text(json.dumps(reports,indent=2))
    if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
        reports.append(run(5,selected));path.write_text(json.dumps(reports,indent=2))
    print('Completed. Independent audit required for any gain.',flush=True)
