"""Experiment 2: causal repeat/exposure features and matched user-OOF screen."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R

def auc(y,p,w=None):
    return float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None

def lookup(keys,values,q,default=0):
    pos=np.searchsorted(keys,q); valid=pos<len(keys); safe=np.minimum(pos,len(keys)-1)
    valid &= keys[safe]==q
    return np.where(valid,values[safe],default)

def grouped(y,p,rep,length):
    out=[]
    for label,mask in [('all',np.ones(len(y),bool)),('repeat',rep),('non_repeat',~rep)]:
        for hlabel,hm in [('all',np.ones(len(y),bool)),('0',length==0),('1-5',(length>=1)&(length<=5)),('6-20',length>=6)]:
            m=mask&hm
            out.append(dict(repeat_status=label,history_length=hlabel,rows=int(m.sum()),clicks=int(y[m].sum()),click_rate=float(y[m].mean()) if m.any() else None,auc=auc(y[m],p[m])))
    return out

def prepare():
    tic=time.monotonic(); t=np.load(R/'train.npy',mmap_mode='r'); h=np.load(R/'train_hist.npy',mmap_mode='r'); ht=np.load(R/'train_times.npy',mmap_mode='r')
    ni=int(t[:,1].max())+1; nu=int(t[:,0].max())+1; start=int(t[:,2].min()); reports=[]
    for day in (5,6):
        z=np.load(R/f'selection_day{day}.npz'); ev=z['events'][::10]; users=t[ev*10,0]
        sample=(users.astype(np.uint64)*np.uint64(2654435761)%100)<20; ev=ev[sample]
        rows=np.asarray(t[(ev[:,None]*10+np.arange(10)).ravel()]); u=rows[:,0]; i=rows[:,1]; ts=rows[:,2]; b=rows[:,3]
        hh=np.repeat(np.asarray(h[ev]),10,axis=0); tt=np.repeat(np.asarray(ht[ev]),10,axis=0); valid=hh>=0
        assert np.all((tt<ts[:,None])|~valid)
        length=valid.sum(1); match=(hh==i[:,None])&valid; count=match.sum(1); rep=count>0
        lastpos=np.where(match,np.arange(20),-1).max(1); lasttime=np.where(match,tt,0).max(1)
        clicksago=np.where(rep,length-1-lastpos,21); gap=np.where(rep,(ts-lasttime)/86400,0)
        same=(match&((tt%86400//21600)==b[:,None])).sum(1)
        base=z['baseline'].reshape(-1,10)[sample].ravel(); y=z['y'].reshape(-1,10)[sample].ravel(); weights=z['weights'].reshape(-1,10)[sample].ravel()
        baseline_groups=grouped(y,base,rep,length)
        print(json.dumps(dict(stage='baseline_segments',day=day,segments=baseline_groups)),flush=True)
        cutoff=start+day*86400; prior=np.asarray(t[t[:,2]<cutoff]); assert prior[:,2].max()<cutoff<=ts.min()
        # Global per-item repeat rates and user/bucket repeat propensities.
        priorrep=np.zeros(len(prior),bool); priorsame=np.zeros(len(prior),bool)
        for off in range(0,len(prior),100000):
            a=prior[off:off+100000]; ph=np.asarray(h[a[:,6]]); pt=np.asarray(ht[a[:,6]])
            m=(ph==a[:,1,None])&(ph>=0); priorrep[off:off+len(a)]=m.any(1)
            priorsame[off:off+len(a)]=(m&((pt%86400//21600)==a[:,3,None])).any(1)
        py=prior[:,5]; clicks=py==1; pu=prior[:,0]; pi=prior[:,1]; ub=pu*4+prior[:,3]; qi=u*4+b
        totalc=np.bincount(pu,weights=py,minlength=nu); rc=np.bincount(pu,weights=py*priorrep,minlength=nu); globalrepeat=float(rc.sum()/max(totalc.sum(),1))
        ur=(rc+10*globalrepeat)/(totalc+10)
        ic=np.bincount(pi,weights=py,minlength=ni); irc=np.bincount(pi,weights=py*priorrep,minlength=ni); itemrate=(irc+20*globalrepeat)/(ic+20)
        bn=np.bincount(ub,weights=priorrep,minlength=nu*4); bc=np.bincount(ub,weights=py*priorrep,minlength=nu*4)
        globalrctr=float((py*priorrep).sum()/max(priorrep.sum(),1)); bucketrate=(bc+20*globalrctr)/(bn+20)
        sn=np.bincount(ub,weights=priorsame,minlength=nu*4); sc=np.bincount(ub,weights=py*priorsame,minlength=nu*4); samerate=(sc+20*globalrctr)/(sn+20)
        # Pair exposure features use only observed training presentations, never inferred warm-up exposures.
        keys=pu.astype(np.int64)*ni+pi; unique,inverse=np.unique(keys,return_inverse=True)
        first=np.full(len(unique),np.iinfo(np.int32).max,dtype=np.int32); np.minimum.at(first,inverse,prior[:,2])
        seen=prior[:,2]>first[inverse]  # strict timestamps, including tied events
        seen_n=np.bincount(pu,weights=seen,minlength=nu); new_n=np.bincount(pu,weights=~seen,minlength=nu)
        seen_c=np.bincount(pu,weights=py*seen,minlength=nu); new_c=np.bincount(pu,weights=py*~seen,minlength=nu)
        seenctr=(seen_c+20*float(py.mean()))/(seen_n+20); newctr=(new_c+20*float(py.mean()))/(new_n+20)
        seenfraction=(seen_c+10*.5)/(totalc+10)
        # Counts are from pre-cutoff rows; hence strictly earlier than every scored event.
        neg=np.bincount(inverse,weights=1-py,minlength=len(unique)); pc=np.bincount(inverse,weights=py,minlength=len(unique))
        lastneg=np.zeros(len(unique),dtype=np.int32); np.maximum.at(lastneg,inverse[~clicks],prior[~clicks,2])
        q=u.astype(np.int64)*ni+i; pn=lookup(unique,neg,q); pp=lookup(unique,pc,q); ln=lookup(unique,lastneg,q)
        repeat_features=np.column_stack([clicksago,np.log1p(gap),same/np.maximum(count,1),ur[u],itemrate[i],bucketrate[qi],samerate[qi],rep*ur[u],rep*itemrate[i]])
        exposure_features=np.column_stack([np.log1p(pn),pn>0,np.where(ln>0,np.log1p((ts-ln)/86400),0),np.log1p(pn/(pp+1)),np.log1p(pp),seenctr[u],newctr[u],seenfraction[u],(pn>0)*seenctr[u],(pn>0)*newctr[u]])
        control=np.column_stack([rankdata(base)/(len(base)+1),rep,count,length,b])
        x=np.column_stack([control,repeat_features,exposure_features]).astype(np.float32); assert np.isfinite(x).all()
        np.savez_compressed(R/f'v54_repeat_screen_day{day}.npz',x=x,y=y,baseline=base,weights=weights,users=u,events=rows[:,6],repeat=rep,length=length)
        report=dict(day=day,baseline_auc=auc(y,base),segments=baseline_groups,prior_rows=len(prior),pair_negative_exposure_share=float((pn>0).mean()),pair_prior_click_share=float((pp>0).mean()),repeat_share=float(rep.mean()),sample_events=len(ev),cutoff=cutoff,prior_max_timestamp=int(prior[:,2].max()))
        reports.append(report); print(json.dumps(dict(stage='features_ready',**{k:v for k,v in report.items() if k!='segments'})),flush=True)
    result=dict(reports=reports,elapsed_seconds=time.monotonic()-tic,feature_names=['baseline_rank','repeat_existing','candidate_prior_click_count_existing','history_length','time_bucket','clicks_ago','log_last_click_gap_days','fraction_repeat_clicks_same_bucket','user_repeat_click_fraction','item_repeat_click_fraction','user_bucket_repeat_ctr','user_same_bucket_repeat_ctr','repeat_x_user_repeat','repeat_x_item_repeat','log_user_item_negative_exposures','has_negative_exposure','log_last_negative_exposure_gap_days','log_negative_to_clicked_ratio_smoothed','log_user_item_training_clicks','user_seen_item_ctr','user_new_item_ctr','user_seen_item_click_fraction','seen_x_user_seen_ctr','seen_x_user_new_ctr'],protocol='Frozen pre-cutoff aggregates. Histories strictly earlier than event. No holdout/evaluation accessed. Seen/new means observed in earlier training events; warm-up exposures are unknown.')
    (R/'v54_repeat_diagnostics.json').write_text(json.dumps(result,indent=2))

def fit():
    from catboost import CatBoostClassifier
    tic=time.monotonic(); data={d:dict(np.load(R/f'v54_repeat_screen_day{d}.npz')) for d in (5,6)}
    results=[]; diagnostics=json.loads((R/'v54_repeat_diagnostics.json').read_text()); names=diagnostics['feature_names']
    for arm,cols in [('control',list(range(5))),('repeat',list(range(14))),('exposure',list(range(5))+list(range(14,24))),('combined',list(range(24)))]:
        a=data[5]; x=a['x'][:,cols]; unique=np.unique(a['users']); foldusers=(unique.astype(np.uint64)*np.uint64(2246822519)%2)==0; fold=np.isin(a['users'],unique[foldusers]); pred=np.empty(len(x))
        params=dict(iterations=180,depth=4,learning_rate=.04,l2_leaf_reg=20,loss_function='Logloss',random_seed=5402,thread_count=4,allow_writing_files=False,verbose=False)
        for take in (fold,~fold):
            assert not np.intersect1d(a['users'][take],a['users'][~take]).size
            model=CatBoostClassifier(**params); model.fit(x[take],a['y'][take]); pred[~take]=model.predict_proba(x[~take])[:,1]
        full=CatBoostClassifier(**params); full.fit(x,a['y']); p6=full.predict_proba(data[6]['x'][:,cols])[:,1]
        for d,p in ((5,pred),(6,p6)):
            a=data[d]; br=rankdata(a['baseline'])/(len(p)+1); pr=rankdata(p)/(len(p)+1)
            for weight in (.05,.1,.25):
                score=(1-weight)*br+weight*pr; gain=auc(a['y'],score)-auc(a['y'],a['baseline'])
                if gain>.01: raise RuntimeError('Gain > .01: stop for leakage audit')
                results.append(dict(arm=arm,day=d,weight=weight,baseline_auc=auc(a['y'],a['baseline']),auc=auc(a['y'],score),gain=gain,adjusted_auc=auc(a['y'],score,a['weights']),segments=grouped(a['y'],score,a['repeat'],a['length'])))
                np.save(R/f'v54_repeat_{arm}_day{d}_w{weight}.npy',score)
        print(json.dumps(dict(arm=arm,results=[{k:v for k,v in r.items() if k!='segments'} for r in results if r['arm']==arm])),flush=True)
    options=[]
    for arm in ('repeat','exposure','combined'):
        for weight in (.05,.1,.25):
            rr=[r for r in results if r['arm']==arm and r['weight']==weight]
            controls=[r for r in results if r['arm']=='control' and r['weight']==weight]
            options.append(dict(arm=arm,weight=weight,minimum_gain=min(r['gain'] for r in rr),gains=[r['gain'] for r in rr],increment_over_control=[r['auc']-c['auc'] for r,c in zip(rr,controls)]))
    best=max(options,key=lambda r:r['minimum_gain'])
    out=dict(results=results,best=best,sample_gate_passed=best['minimum_gain']>=.0005,elapsed_seconds=time.monotonic()-tic,protocol='Day5 two-fold user-disjoint OOF; day6 model fits day5 only. Fixed CatBoost settings, no early stopping. Matched control and feature ablations. Screening only; baseline predictions come from prior V53 OOF pipeline. No holdout/evaluation opened.',feature_names=names)
    (R/'v54_repeat_screen_report.json').write_text(json.dumps(out,indent=2)); print(json.dumps(dict(best=best,sample_gate_passed=out['sample_gate_passed'],elapsed_seconds=out['elapsed_seconds'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','fit']); args=parser.parse_args()
    prepare() if args.stage=='prepare' else fit()
