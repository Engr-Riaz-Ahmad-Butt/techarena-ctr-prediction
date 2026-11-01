"""Matched frozen-ALS vs positive-only anchored warm-user refresh.

No fitting, V53 replay, fake negative generation, or external data. The
simplified matched blend contains a frozen warm16 expert already used in V53.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from horizon_foldin import refresh,self_test

BASE=Path(__file__).resolve().parent;A=BASE/'artifacts';R=A/'v54_horizon';RUN=R/'matched_refresh_001'
def sha(p):
    hh=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):hh.update(b)
    return hh.hexdigest()
def rank(x):return rankdata(x,method='average')/(len(x)+1)
def auc(y,p,w=None):return float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None

class AUCPlan:
    """Sort once and replay weighted tie-correct AUC for user bootstraps."""
    def __init__(self,y,p,w):
        self.order=np.argsort(p,kind='stable');self.y=y[self.order];self.w=w[self.order];s=p[self.order];self.starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    def value(self,mult):
        w=self.w*mult[self.order];pos=np.add.reduceat(w*self.y,self.starts);neg=np.add.reduceat(w*(1-self.y),self.starts);totalpos=pos.sum();totalneg=neg.sum()
        return float(np.dot(pos,np.cumsum(neg)-.5*neg)/(totalpos*totalneg)) if totalpos>0 and totalneg>0 else np.nan

def compare(y,control,candidate,eventweights,events,mask,label,boots,seed):
    selected=np.flatnonzero(mask);iy=(selected[:,None]*10+np.arange(10)).ravel();yy=y[iy];p=control[iy];q=candidate[iy];w=np.repeat(eventweights[selected],10)
    result=dict(group=label,events=len(selected),rows=len(yy),clicked_events=int(yy.sum()),effective_events=float(eventweights[selected].sum()**2/np.square(eventweights[selected]).sum()),weighted_control_auc=auc(yy,p,w),weighted_candidate_auc=auc(yy,q,w),unweighted_control_auc=auc(yy,p),unweighted_candidate_auc=auc(yy,q))
    if result['weighted_control_auc'] is None:
        result.update(weighted_delta=None,bootstrap_delta_95=None,bootstrap_replicates=0);return result
    result['weighted_delta']=result['weighted_candidate_auc']-result['weighted_control_auc'];result['unweighted_delta']=result['unweighted_candidate_auc']-result['unweighted_control_auc']
    pc=AUCPlan(yy,p,w);qc=AUCPlan(yy,q,w);assert abs(pc.value(np.ones(len(yy)))-result['weighted_control_auc'])<1e-10;assert abs(qc.value(np.ones(len(yy)))-result['weighted_candidate_auc'])<1e-10
    rng=np.random.default_rng(seed);deltas=[]
    for _ in range(boots):
        count=np.bincount(rng.integers(0,len(selected),len(selected)),minlength=len(selected));m=np.repeat(count,10);d=qc.value(m)-pc.value(m)
        if np.isfinite(d):deltas.append(d)
    result['bootstrap_delta_95']=np.quantile(deltas,[.025,.975]).tolist();result['bootstrap_replicates']=len(deltas);result['bootstrap_se']=float(np.std(deltas,ddof=1))
    return result

def warm_scores(rows,hist,model):
    users=rows[::10,0];items=rows[:,1].reshape(-1,10);hh=np.asarray(hist[rows[::10,6]]);valid=hh>=0
    profile=model['U'][users]+(model['H'][np.maximum(hh,0)]*valid[:,:,None]).sum(1)/np.sqrt(np.maximum(valid.sum(1),1))[:,None]
    logits=-1.8+model['bu'][users,None]+model['bi'][items]+np.einsum('ed,ecd->ec',profile,model['V'][items]);logits=np.clip(logits,-30,30);m=np.maximum(logits.max(1),0);p=np.exp(logits-m[:,None]);denom=p.sum(1)+np.exp(-m);return (p/denom[:,None]).ravel()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=400);args=ap.parse_args();RUN.mkdir(exist_ok=True);tic=time.monotonic()
    outpath=RUN/'report.json'
    if outpath.exists():raise FileExistsError('Completed matched experiment exists: inspect it instead of rerunning.')
    meta=json.loads((R/'cohort_manifest.json').read_text());cut=int(meta['cutoff']);rows=np.load(R/'score_rows.npy',mmap_mode='r');co=dict(np.load(R/'cohort.npz'));hist=np.load(A/'train_hist.npy',mmap_mode='r');times=np.load(A/'train_times.npy',mmap_mode='r')
    alsfile=A/'imp17_als_cut3.npz';warmfile=A/'imp3_model_warm16_cut3.npz'
    protected={str(p):sha(p) for p in [alsfile,warmfile,A/'train.npy',A/'v53_causal_propensity_models.pkl'] if p.name!='train.npy'}
    model=dict(np.load(alsfile));assert int(model['cutoff'])==cut and int(model['iteration'])==6
    # Verify archived warm16 recipe: its cut3 initialization is pre-cutoff2.
    assert cut==1737590400
    warm=dict(np.load(warmfile));assert all(v.shape[0] in (54456,50000) for v in warm.values())
    frozen=dict(model='pre-cutoff3 ALS16 checkpoint',model_sha256=sha(alsfile),warm_expert_sha256=sha(warmfile),cutoff=cut,anchor_precision=16.,half_life_days=2.,refresh='warm users only; deduplicated supplied history at cutoff<=click_time<event_time; positive-only ridge anchored at cutoff vector',fixed_biases=True,cold_users='Cutoff checkpoint user vectors/biases held identical in both arms; existing cold fold-in not applied because it assumes unseen background negatives.',simplified_blend='0.95 rank(frozen warm16 expert) + 0.05 rank(ALS), jointly ranked over full 4-day horizon; same other expert in both arms',bootstrap=args.bootstrap,seed=541701,cohort_sha256=sha(R/'cohort.npz'),scoring_rows_sha256=sha(R/'score_rows.npy'),source_sha256=sha(Path(__file__)),kernel_sha256=sha(BASE/'horizon_foldin.py'),gate='Advance to full V53 rebuild if 20% screen weighted overall gain>=.0002 OR later-arrival(3+) gain has paired95% lower bound>0. Full cohort diagnostic is not V53 promotion.',no_holdout=True,no_submission=True)
    (RUN/'frozen.json').open('x').write(json.dumps(frozen,indent=2))
    assert np.all(rows[:,5]==-1);users=rows[::10,0];items=rows[:,1].reshape(-1,10);ev=co['events'];assert np.array_equal(rows[::10,6],ev)
    valid=hist[ev]>=0;assert np.all((times[ev]<rows[::10,2,None])|~valid)
    old=model['U'][users].astype(float);new=old.copy();warmmask=model['exposures'][users]>0;newcounts=np.zeros(len(users),int)
    with threadpool_limits(limits=1):
        for k in np.flatnonzero(warmmask):
            new[k],newcounts[k]=refresh(old[k],model['V'],model['bi'],float(model['g']),model['bu'][users[k]],hist[ev[k]],times[ev[k]],cut,int(rows[k*10,2]),16.,2.)
    untouched=(~warmmask)|(newcounts==0);assert np.array_equal(new[untouched],old[untouched])
    bias=float(model['g'])+model['bu'][users,None]+model['bi'][items]
    control=(bias+np.einsum('ed,ecd->ec',old,model['V'][items])).ravel();candidate=(bias+np.einsum('ed,ecd->ec',new,model['V'][items])).ravel()
    assert np.array_equal(control.reshape(-1,10)[untouched],candidate.reshape(-1,10)[untouched])
    other=warm_scores(rows,hist,warm);bc=.95*rank(other)+.05*rank(control);bq=.95*rank(other)+.05*rank(candidate)
    assert all(np.isfinite(p).all() for p in (control,candidate,other,bc,bq))
    np.savez_compressed(RUN/'predictions.npz',control_als=control,candidate_als=candidate,control_blend=bc,candidate_blend=bq,other_expert=other,events=ev,new_click_counts=newcounts,warm=warmmask)
    scoring_seconds=time.monotonic()-tic
    print(json.dumps(dict(stage='scored_without_labels',seconds=scoring_seconds,events=len(ev),warm_users=int(warmmask.sum()),refreshed_users=int((newcounts>0).sum()))),flush=True)
    # Outcomes enter only after both arms have been produced and persisted.
    y=np.load(R/'validation_labels.npy').ravel();assert len(y)==len(control)
    metrics=[]
    for scope,take in [('screen20',co['sample']),('full',np.ones(len(ev),bool))]:
        ew=co['event_weight'].copy()
        if scope=='screen20':ew[take]=co['sample_event_weight']
        groups=[('overall',take),('arrival1',take&(co['arrival']==0)),('arrival2',take&(co['arrival']==1)),('arrival3plus',take&(co['arrival']>=2)),('history0',take&(co['history_length']==0)),('history1to5',take&(co['history_length']>=1)&(co['history_length']<=5)),('history6to20',take&(co['history_length']>=6))]
        for arm,p,q in [('standalone_ALS',control,candidate),('simplified_blend',bc,bq)]:
            for j,(name,m) in enumerate(groups):
                r=compare(y,p,q,ew,ev,m,name,args.bootstrap,541701+j);r.update(scope=scope,arm=arm);metrics.append(r)
            print(json.dumps(dict(stage='bootstrap_complete',scope=scope,arm=arm,overall=next(r for r in metrics if r['scope']==scope and r['arm']==arm and r['group']=='overall'),elapsed_seconds=time.monotonic()-tic)),flush=True)
    # Replay independent normal equations for a selection of real changed users.
    checks=[]
    for k in np.flatnonzero(newcounts>0)[::max(1,int((newcounts>0).sum())//25)][:25]:
        h=np.asarray(hist[ev[k]]);tt=np.asarray(times[ev[k]]);m=(h>=0)&(tt>=cut)&(tt<int(rows[k*10,2]));records=set(zip(h[m].tolist(),tt[m].tolist()));vv=np.array([model['V'][i] for i,_ in records]);ww=np.array([2**(-(int(rows[k*10,2])-ts)/(2*86400)) for _,ts in records]);target=np.array([1-float(model['g'])-model['bu'][users[k]]-model['bi'][i] for i,_ in records]);grad=16*(new[k]-old[k])+vv.T@(ww*(vv@new[k]-target));assert np.linalg.norm(grad)<1e-8;checks.append(int(ev[k]))
    assert all(sha(Path(p))==s for p,s in protected.items())
    advance=[]
    for arm in ('standalone_ALS','simplified_blend'):
        overall=next(r for r in metrics if r['scope']=='screen20' and r['arm']==arm and r['group']=='overall');later=next(r for r in metrics if r['scope']=='screen20' and r['arm']==arm and r['group']=='arrival3plus')
        if overall['weighted_delta']>=.0002 or later['bootstrap_delta_95'][0]>0:advance.append(arm)
    result=dict(status='matched_control_complete',metrics=metrics,advance_to_full_v53_rebuild=bool(advance),qualifying_arms=advance,refreshed_events=int((newcounts>0).sum()),warm_events=int(warmmask.sum()),total_events=len(ev),new_click_count_total=int(newcounts.sum()),refresh_rate_by_arrival={str(d+1):float((newcounts[co['arrival']==d]>0).mean()) for d in range(4)},scoring_seconds=scoring_seconds,elapsed_seconds=time.monotonic()-tic,real_gradient_checks=checks,unit_tests=self_test(),protected_hashes_unchanged=True,limitations=['Simplified blend is not V53 and cannot establish gain inside full V53.','Cold users are fixed at cutoff in both arms; no fabricated negative impressions were added.','Refresh adds positive-history evidence under the explicit squared-loss assumption; no new item fitting.','Day5-7 evaluation tail unsupported; arrival4 screen has only309 users.','Bootstrap intervals condition on fixed item factors, refresh settings, and weighting; no multiple-testing adjustment.','Archived cutoff3 factor checkpoint reused; historical settings were chosen on reused dates, not newly independent.'])
    outpath.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2),flush=True)

if __name__=='__main__':main()
