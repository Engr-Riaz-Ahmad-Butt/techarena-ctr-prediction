"""Fixed latent taste-group interaction screen; isolated from Claude's nx/nxa work."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib,pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from horizon_matched_control import AUCPlan
B=Path(__file__).resolve().parent;A=B/'artifacts';H=A/'v54_horizon';R=H/'cdx_taste_groups_001'
K=16;NI=50000;PRIOR=20.;ALPHA=.02;SEED=8127
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ecdf(s,x):return (np.searchsorted(s,x,'left')+np.searchsorted(s,x,'right'))/(2*len(s))
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def fit_state(train,model,cut):
    assert int(model['cutoff'])==cut;mask=train[:,2]<cut;rows=np.asarray(train[mask]);assert rows[:,2].max()<cut;nu=len(model['U']);warm=model['exposures']>0;vec=unit(model['U']);km=MiniBatchKMeans(n_clusters=K,n_init=1,max_iter=50,batch_size=2048,random_state=SEED,reassignment_ratio=0.)
    km.fit(vec[warm]);groups=km.predict(vec);key=groups[rows[:,0]]*NI+rows[:,1];n=np.bincount(key,minlength=K*NI).reshape(K,NI);c=np.bincount(key,weights=rows[:,5],minlength=K*NI).reshape(K,NI)
    userkey=rows[:,0].astype(np.int64)*NI+rows[:,1];unique,ix=np.unique(userkey,return_inverse=True);un=np.bincount(ix);uc=np.bincount(ix,weights=rows[:,5]);user_n=np.bincount(rows[:,0],minlength=nu);user_c=np.bincount(rows[:,0],weights=rows[:,5],minlength=nu)
    state=dict(cutoff=cut,km=km,groups=groups,warm=warm,n=n,c=c,userkey=unique,userpair_n=un,userpair_c=uc,user_n=user_n,user_c=user_c,item_n=n.sum(0),item_c=c.sum(0),group_n=n.sum(1),group_c=c.sum(1),n_all=len(rows),c_all=float(rows[:,5].sum()))
    return state
def score(state,rows):
    assert rows[::10,2].min()>=state['cutoff'];u=rows[:,0];i=rows[:,1];g=state['groups'][u];query=u.astype(np.int64)*NI+i;keys=state['userkey'];p=np.minimum(np.searchsorted(keys,query),len(keys)-1);ok=keys[p]==query;own_n=np.where(ok,state['userpair_n'][p],0);own_c=np.where(ok,state['userpair_c'][p],0)
    # Every group/item/global count below excludes all of this user's outcomes.
    global_g=(state['c_all']-state['user_c'][u])/(state['n_all']-state['user_n'][u]);item_p=(state['item_c'][i]-own_c+50*global_g)/(state['item_n'][i]-own_n+50);group_p=(state['group_c'][g]-state['user_c'][u]+100*global_g)/(state['group_n'][g]-state['user_n'][u]+100)
    expected=expit(logit(np.clip(item_p,1e-6,1-1e-6))+logit(np.clip(group_p,1e-6,1-1e-6))-logit(np.clip(global_g,1e-6,1-1e-6)))
    nn=state['n'][g,i]-own_n;cc=state['c'][g,i]-own_c;assert (nn>=0).all() and (cc>=0).all();posterior=(cc+PRIOR*expected)/(nn+PRIOR);residual=logit(np.clip(posterior,1e-6,1-1e-6))-logit(np.clip(expected,1e-6,1-1e-6));warm=state['warm'][u];residual[~warm]=0;assert np.isfinite(residual).all();return residual,warm,nn
def compare(y,base,p,weights,mask,boots):
    yy=y[mask].ravel();bb=base[mask].ravel();pp=p[mask].ravel();w=np.repeat(weights[mask],10);cp=AUCPlan(yy,bb,w);qp=AUCPlan(yy,pp,w);one=np.ones(len(yy));ca=cp.value(one);pa=qp.value(one);assert abs(ca-roc_auc_score(yy,bb,sample_weight=w))<1e-10;rng=np.random.default_rng(SEED);delta=[];ne=int(mask.sum())
    for _ in range(boots):
        m=np.repeat(np.bincount(rng.integers(0,ne,ne),minlength=ne),10);delta.append(qp.value(m)-cp.value(m))
    return dict(events=ne,control_auc=ca,candidate_auc=pa,gain=pa-ca,paired95=np.quantile(delta,[.025,.975]).tolist(),bootstraps=boots)
def main():
    R.mkdir(exist_ok=True);assert not (R/'report.json').exists(),'Do not repeat a completed screen';tic=time.monotonic();t=np.load(A/'train.npy',mmap_mode='r');co=np.load(H/'cohort.npz');rows=np.load(H/'score_rows.npy',mmap_mode='r');cut=int(rows[::10,2].min());start=int(t[:,2].min());assert cut==start+3*86400;assert np.all(rows[:,5]==-1)
    recipe=dict(hypothesis='Latent taste groups share item-specific response beyond low-rank dot products; global user/item propensities removed',method='Own simplified latent-class-inspired hard grouping, not an exact implementation of source EM/co-clustering',source='https://bigdata.oden.utexas.edu/publication/information-theoretic-co-clustering/',source_limit='Source establishes co-clustering family, not gains on this competition. IJCAI1999 paper search found but full page could not be opened.',cluster_count=K,cluster_algorithm='MiniBatchKMeans on normalized existing pre-cutoff ALS user vectors',seed=SEED,n_init=1,max_iter=50,prior_strength=PRIOR,score_scale=ALPHA,fit_labels='strictly before each cutoff; all counts leave the scored user out',reference_cdf='unlabelled earlier day2 first-user events scored from prefix2 state',fixed_candidate='cached simplified horizon blend + .02*(historical residual CDF - historical CDF at0), warm users only',screen_gate='Weighted gain>=.0002 overall and each of four arrival days',confirmation_gate='Weighted gain>=.0005 overall and each arrival day; no holdout',baseline='Frozen simplified V53-component blend; full V53 horizon not rebuilt',source_sha256=sha(__file__),cohort_sha256=sha(H/'cohort.npz'),fit_checkpoints={str(p.relative_to(B)):sha(p) for p in [A/'imp17_als_cut2.npz',A/'imp17_als_cut3.npz']},closed_directions='No global popularity/recency, ignored-impression timing, KNN, sequence, stacking or training-weight search. Only group-item interaction residual.',no_evaluation_statistics=True,no_holdout=True,no_submission=True)
    (R/'frozen.json').open('x').write(json.dumps(recipe,indent=2))
    with threadpool_limits(limits=2):
        # Timing pilot: a single small clustering pass, no labels or scores.
        m2=dict(np.load(A/'imp17_als_cut2.npz'));vp=unit(m2['U'][m2['exposures']>0])[:4096];pt=time.monotonic();MiniBatchKMeans(n_clusters=K,n_init=1,max_iter=10,batch_size=1024,random_state=SEED).fit(vp);pilot=time.monotonic()-pt;print(json.dumps(dict(stage='timing_pilot',users=len(vp),seconds=pilot,full_job_estimate_seconds=180)),flush=True)
        s2=fit_state(t,m2,start+2*86400);fit=np.load(H/'event_measurement_001/measurement_arrays.npz');ee=fit['fit_events'];heads=t[ee*10];ee=ee[heads[:,2]>=start+2*86400];reference=np.asarray(t[(ee[:,None]*10+np.arange(10)).ravel()]);reference[:,5]=-1;rr,_,_=score(s2,reference);cdf=np.sort(rr);del s2
        print(json.dumps(dict(stage='historical_cdf_frozen',events=len(ee),seconds=time.monotonic()-tic)),flush=True)
        m3=dict(np.load(A/'imp17_als_cut3.npz'));s3=fit_state(t,m3,cut);r,warm,support=score(s3,rows);zero=float(ecdf(cdf,np.array([0.]))[0]);shift=np.where(warm,ALPHA*(ecdf(cdf,r)-zero),0).reshape(-1,10);base=np.load(H/'matched_refresh_001/predictions.npz')['control_blend'].reshape(-1,10);candidate=base+shift
        # Counterfactual label invariance of the entire new signal.
        probe=np.asarray(rows[:2560]).copy();probe[:,5]=1;r2,w2,_=score(s3,probe);assert np.array_equal(r2,r[:2560]) and np.array_equal(w2,warm[:2560]);assert np.isfinite(candidate).all()
        np.savez_compressed(R/'frozen_predictions.npz',events=co['events'],control=base,candidate=candidate,residual=r.reshape(-1,10),support=support.reshape(-1,10),warm=warm.reshape(-1,10));(R/'model.pkl').write_bytes(pickle.dumps(dict(state=s3,cdf=cdf,recipe=recipe)))
        # Outcomes are loaded only after both score arrays are fixed on disk.
        y=np.load(H/'validation_labels.npy');sample=co['sample'];sy=y[sample];sb=base[sample];sp=candidate[sample];sw=co['sample_event_weight'];arrival=co['arrival'][sample];history=co['history_group'][sample];reports=[]
        for name,mask in [('overall',np.ones(len(sy),bool))]+[(f'arrival{k+1}',arrival==k) for k in range(4)]+[(f'history{k}',history==k) for k in range(3)]:
            result=compare(sy,sb,sp,sw,mask,200);result['group']=name;reports.append(result)
        survives=all(x['gain']>=.0002 for x in reports[:5]);full=[]
        if survives:
            for name,mask in [('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',co['arrival']==k) for k in range(4)]:
                result=compare(y,base,candidate,co['event_weight'],mask,200);result['group']=name;full.append(result)
        protected=json.loads((H/'protected_before.json').read_text());assert all(sha(B/p)==meta['sha256'] for p,meta in protected.items())
        result=dict(status='screen_complete',screen=reports,screen_passed=bool(survives),full=full,confirmed=bool(full and all(x['gain']>=.0005 for x in full)),seconds=time.monotonic()-tic,pilot_seconds=pilot,new_signal_nonzero_share=float((r!=0).mean()),other_user_pair_support_share=float((support>0).mean()),group_sizes=np.bincount(s3['groups'][s3['warm']],minlength=K).tolist(),label_invariance=True,protected_files_unchanged=len(protected),holdout_opened=False,evaluation_scored=False,submission_created=False,official_gain=None,decision='Retain for independent full-V53 confirmation only' if survives else 'Reject fixed latent-taste-group screen; no expansion or tuning',limitations=['Pre-existing fixed ALS recipe reused; development cohorts have been repeatedly reused.','Simplified horizon baseline is not full V53; no improvement on V53 can be claimed.','The grouping is a simple hard-assignment diagnostic, not the full latent-class EM model.'])
        (R/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
