"""Idea2 step1 only: temporal event-click measurement, no product candidate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
BASE=Path(__file__).resolve().parent;A=BASE/'artifacts';H=A/'v54_horizon';R=H/'event_measurement_001'
import json,time,hashlib
import numpy as np
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

FEATURES=['log_days_since_last_activity','no_known_activity','log_hours_since_cutoff','history_length','smoothed_user_event_click_rate','time_bucket','day_type','ALS_max','ALS_mean','ALS_top2_gap']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def auc(y,p,w=None):return float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None
def ensure_als1():
    p=H/'imp17_als_cut1.npz'
    if not p.exists():
        os.environ['CTR_ARTIFACT_DIR']=str(H)
        from numba import set_num_threads
        set_num_threads(4)
        from imp17_als_features import ensure_model
        with threadpool_limits(limits=1):ensure_model(1)
    return p

def features(rows,cutoff,model,t,h,ht,nu):
    # rows may contain arbitrary label sentinels; no current row labels used.
    head=rows[::10];users=head[:,0];history=np.asarray(h[head[:,6]]);times=np.asarray(ht[head[:,6]]);valid=history>=0;assert np.all((times<head[:,2,None])|~valid)
    prior=np.asarray(t[t[:,2]<cutoff]);ev=prior[::10];eventclick=prior[:,5].reshape(-1,10).sum(1);assert ev[:,2].max()<cutoff<=head[:,2].min()
    n=np.bincount(ev[:,0],minlength=nu);c=np.bincount(ev[:,0],weights=eventclick,minlength=nu);g=float(eventclick.mean());rate=(c[users]+20*g)/(n[users]+20)
    last=np.zeros(nu,dtype=np.int32);np.maximum.at(last,ev[:,0],ev[:,2]);known=np.maximum(last[users],np.where(valid,times,0).max(1));missing=known==0;gap=np.where(missing,7.,(head[:,2]-known)/86400.)
    items=rows[:,1].reshape(-1,10);score=float(model['g'])+model['bu'][users,None]+model['bi'][items]+np.einsum('ed,ecd->ec',model['U'][users],model['V'][items]);top=np.partition(score,-2,axis=1)[:,-2:]
    x=np.column_stack([np.log1p(gap),missing,np.log1p((head[:,2]-cutoff)/3600.),valid.sum(1),rate,head[:,3],head[:,4],score.max(1),score.mean(1),top.max(1)-top.min(1)]).astype(np.float64)
    assert np.isfinite(x).all();return x

def bootstrap(y,p,w,boots=400,seed=542001,other=None):
    from horizon_matched_control import AUCPlan
    plan=AUCPlan(y,p,w);control=AUCPlan(y,other,w) if other is not None else None;rng=np.random.default_rng(seed);values=[];delta=[]
    for _ in range(boots):
        count=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y));v=plan.value(count)
        if np.isfinite(v):values.append(v)
        if control is not None:
            d=v-control.value(count)
            if np.isfinite(d):delta.append(d)
    result=dict(auc95=np.quantile(values,[.025,.975]).tolist(),replicates=len(values))
    if delta:result['delta_vs_existing_event_max95']=np.quantile(delta,[.025,.975]).tolist()
    return result

def summary(y,p,w,mask,name,other=None):
    yy=y[mask];pp=p[mask];ww=w[mask];r=dict(group=name,events=int(mask.sum()),click_events=int(yy.sum()),weighted_click_share=float(np.average(yy,weights=ww)),unweighted_auc=auc(yy,pp),weighted_auc=auc(yy,pp,ww),weighted_brier=float(brier_score_loss(yy,pp,sample_weight=ww)),weighted_logloss=float(log_loss(yy,pp,sample_weight=ww,labels=[0,1])))
    r.update(bootstrap(yy,pp,ww,other=other[mask] if other is not None else None))
    if other is not None:r.update(existing_blend_event_max_auc=auc(yy,other[mask],ww),delta_vs_existing_event_max=auc(yy,pp,ww)-auc(yy,other[mask],ww))
    return r

def main():
    R.mkdir(exist_ok=True);tic=time.monotonic()
    if (R/'report.json').exists():raise FileExistsError('Completed measurement exists; do not repeat.')
    params=dict(n_estimators=120,num_leaves=7,max_depth=3,learning_rate=.04,min_child_samples=100,reg_lambda=20,reg_alpha=2,random_state=5420,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
    frozen=dict(stage='measurement_only',train_cutoff_days=1,fit_label_end_days=3,validation_cutoff_days=3,train_event_selection='first event per user after day1 cutoff over days1/2',features=FEATURES,model_params=params,source_sha256=sha(__file__),missing_activity='7day sentinel + explicit missing flag',last_activity='max(last pre-cutoff observed impression, latest supplied prior click)',user_rate='pre-cutoff event clicks / event count, shrunk with20 global-prior events',calibration='No calibration fitted; native classifier probability assessed only.',candidate_product_built=False)
    (R/'frozen.json').open('x').write(json.dumps(frozen,indent=2))
    als1=ensure_als1();als3=A/'imp17_als_cut3.npz';m1=dict(np.load(als1));m3=dict(np.load(als3));t=np.load(A/'train.npy',mmap_mode='r');h=np.load(A/'train_hist.npy',mmap_mode='r');ht=np.load(A/'train_times.npy',mmap_mode='r');start=int(t[:,2].min());cut=start+3*86400;assert int(m1['cutoff'])==start+86400 and int(m3['cutoff'])==cut
    # Historical first-event labels lie entirely before the held-out horizon.
    head=t[::10];available=np.flatnonzero((head[:,2]>=start+86400)&(head[:,2]<cut));_,ix=np.unique(head[available,0],return_index=True);fit_events=available[np.sort(ix)];fit_rows=np.asarray(t[(fit_events[:,None]*10+np.arange(10)).ravel()]);fit_y=fit_rows[:,5].reshape(-1,10).sum(1).astype(np.int8);nu=len(m3['U'])
    x=features(fit_rows,start+86400,m1,t,h,ht,nu);fit_rows[:,5]=-1
    rows=np.load(H/'score_rows.npy',mmap_mode='r');co=dict(np.load(H/'cohort.npz'));xv=features(rows,cut,m3,t,h,ht,nu)
    # Counterfactual label invariance: fit rows/statistics stay fixed.
    probe=np.asarray(rows[:2560]).copy();probe[:,5]=1;assert np.array_equal(features(probe,cut,m3,t,h,ht,nu),xv[:256])
    from lightgbm import LGBMClassifier
    model=LGBMClassifier(**params);model.fit(x,fit_y);pred=model.predict_proba(xv)[:,1];assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();model.booster_.save_model(str(R/'event_model.txt'));np.savez_compressed(R/'measurement_arrays.npz',event_probability=pred,features=xv,train_features=x,train_y=fit_y,fit_events=fit_events,validation_events=co['events'])
    print(json.dumps(dict(stage='prediction_frozen_before_validation_outcomes',fit_events=len(fit_events),fit_click_share=float(fit_y.mean()),seconds=time.monotonic()-tic)),flush=True)
    y=np.load(H/'validation_labels.npy').sum(1).astype(np.int8);w=co['event_weight'];previous=np.load(H/'matched_refresh_001/predictions.npz')['control_blend'].reshape(-1,10);eventmax=previous.max(1);reports=[]
    groups=[('overall',np.ones(len(y),bool))]+[(f'arrival{d+1}',co['arrival']==d) for d in range(4)]+[('arrival3plus',co['arrival']>=2)]
    for name,mask in groups:reports.append(summary(y,pred,w,mask,name,eventmax))
    sample=co['sample'];reports.append(summary(y[sample],pred[sample],co['sample_event_weight'],np.ones(int(sample.sum()),bool),'screen20_overall',eventmax[sample]))
    comparisons=dict(frozen_blend_max=auc(y,eventmax,w),frozen_blend_mean=auc(y,previous.mean(1),w),ALS_max=auc(y,xv[:,7],w),ALS_mean=auc(y,xv[:,8],w),ALS_top2_gap=auc(y,xv[:,9],w),shrunk_user_rate=auc(y,xv[:,4],w),history_length=auc(y,xv[:,3],w),hours_since_cutoff=auc(y,xv[:,2],w))
    v53=[]
    for day in (5,6):
        z=np.load(A/f'selection_day{day}.npz');yy=z['y'].reshape(-1,10).sum(1);p=z['baseline'].reshape(-1,10);bins=np.minimum((rankdata(p.max(1))/(len(p)+1)*5).astype(int),4);rates=[]
        for b in range(5):
            take=bins==b;rates.append(dict(score_quintile=b+1,events=int(take.sum()),event_click_rate=float(yy[take].mean())))
        v53.append(dict(day=day,events=len(yy),event_click_share=float(yy.mean()),max_rank_score_event_auc=auc(yy,p.max(1)),mean_rank_score_event_auc=auc(yy,p.mean(1)),best_score_quintile_click_rates=rates,note='Saved V53 selection scores only. Cohorts differ from horizon validation; day5 user-OOF and day6 later-date scores. No reserved holdout loaded.'))
    protected=json.loads((H/'protected_before.json').read_text())
    for name,item in protected.items():assert sha(BASE/name)==item['sha256']
    overall=reports[0];gate=overall['weighted_auc']>=.53 and overall['auc95'][0]>.5
    result=dict(status='measurement_complete',event_model_results=reports,existing_score_event_aucs=comparisons,saved_V53_event_measurements=v53,measurement_gate_passed=bool(gate),decision='Supports considering matched probability-product test; pause for user review.' if gate else 'Below meaningful event-discrimination threshold; stop Idea2.',fit_events=len(fit_events),last_fit_label_timestamp=int(t[fit_events*10,2].max()),validation_first_timestamp=int(rows[::10,2].min()),fit_horizon_max_hours=float(np.expm1(x[:,2]).max()),validation_horizon_max_hours=float(np.expm1(xv[:,2]).max()),fit_activity_gap_days_quantiles=np.quantile(np.expm1(x[:,0]),[.1,.5,.9]).tolist(),validation_activity_gap_days_quantiles=np.quantile(np.expm1(xv[:,0]),[.1,.5,.9]).tolist(),feature_importance=dict(zip(FEATURES,map(int,model.feature_importances_))),label_invariance_check=True,protected_files_unchanged=len(protected),holdout_opened=False,candidate_product_built=False,elapsed_seconds=time.monotonic()-tic,limitations=['Historical training horizon has2arrival days; validation has4. Long-gap extrapolation may be limited.','Event AUC above.53 alone does not establish incremental global row-AUC gain.','Native event probabilities were not externally calibrated; Brier/logloss are reported.','Archived factor settings were historically selected on reused dates; dates are not independent.','V53 event measurements use different cohorts and cannot be subtracted from the new model AUC as a matched effect.'])
    (R/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
