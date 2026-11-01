"""History-matched validation and causal distribution/segment screens."""
import argparse,ast,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v54_abcd import load_samples,aligned_source,lgb_fit

TARGET=np.array([.0126,.1637,.8237])
def rank(x):return rankdata(x)/(len(x)+1)
def group(n):return np.select([n==0,n<=5],[0,1],default=2)
def weights(n):
    g=group(n);share=np.bincount(g,minlength=3)/len(g);return (TARGET/share)[g]
def auc(y,p,w=None):return float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None
def save(tag,obj):
    (R/f'v54_distribution_{tag}.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj),flush=True)
def evaluate(a,p,day,label):
    m=a['test'];y=a['y'][m];base=a['baseline'][m];p=p[m];w=weights(a['length'][m]);gain=auc(y,p,w)-auc(y,base,w)
    if gain>.01:raise RuntimeError('Audit required: single-step weighted gain > .01')
    out=dict(day=day,label=label,rows=int(m.sum()),baseline_weighted_auc=auc(y,base,w),weighted_auc=auc(y,p,w),weighted_gain=gain,baseline_unweighted_auc=auc(y,base),unweighted_auc=auc(y,p),unweighted_gain=auc(y,p)-auc(y,base))
    for name,mask in [('zero',a['length'][m]==0),('short',a['length'][m]<=5),('bucket3',a['rows'][m,3]==3)]:out[name]=dict(rows=int(mask.sum()),before=auc(y[mask],base[mask]),after=auc(y[mask],p[mask]))
    return out
def step1():
    t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');reports=[]
    for d in (5,6):
        z=np.load(R/f'selection_day{d}.npz');n=(h[z['events']]>=0).sum(1);users=t[z['events']*10,0];sample=(users.astype(np.uint64)*np.uint64(2654435761)%100)<20
        for label,m in [('full_selection',np.ones(len(n),bool)),('sample20',sample)]:
            w=weights(n[m]);reports.append(dict(day=d,cohort=label,rows=int(m.sum()),unweighted_auc=auc(z['y'][m],z['baseline'][m]),weighted_auc=auc(z['y'][m],z['baseline'][m],w),effective_mix=[float(w[group(n[m])==k].sum()/w.sum()) for k in range(3)]))
    data,cut=load_samples()
    for d,a in data.items():reports.append(dict(cohort='model_evaluation_early40_later60',**evaluate(a,a['baseline'],d,'V53')))
    save('step1',dict(target_mix=TARGET.tolist(),reports=reports,note='Weights are target share / cohort share, constant within history group. Day6 selection excludes reserved holdout. No evaluation access needed: user supplied mix.'))

def original_features():
    # Compile only original pure feature definitions, avoiding imports with side effects.
    from scipy.special import logit
    env=dict(np=np,R=R,rank=rank,logit=logit)
    def load(file,functions,bindings=None):
        scope=env.copy();scope.update(bindings or {});tree=ast.parse(Path(file).read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions];exec(compile(tree,file,'exec'),scope);return scope
    f48=load('v48_history_calibration.py',['features'])['features']
    sources=[('imp19_tree','pred'),('v23_outside','raw'),('v36_recent_context','raw'),('v36_recent_context','old'),('v39_regularized_als32','raw'),('imp15_als','raw')]
    f49=load('v49_expert_calibration.py',['features'],dict(base_features=f48,SOURCES=sources))['features']
    return load('v53_causal_propensity.py',['features','rate'],dict(prior_features=f49))['features']

def cache_original():
    from threadpoolctl import threadpool_limits
    tic=time.monotonic();features=original_features();t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');data,_=load_samples()
    with (R/'v53_causal_propensity_models.pkl').open('rb') as f:bundle=pickle.load(f)
    for d,a in data.items():
        path=R/f'v54_distribution_original_features_day{d}.npz'
        if path.exists():continue
        with threadpool_limits(limits=1):z,x,users,groups,names=features(d,t,h,ht,bundle['spline'])
        src=z['events'].astype(np.int64)*10+np.tile(np.arange(10),len(z['events'])//10);q=a['events'].astype(np.int64)*10+np.tile(np.arange(10),len(a['events'])//10);order=np.argsort(src);ix=order[np.searchsorted(src[order],q)];assert np.array_equal(src[ix],q)
        assert x.shape[1]==bundle['full']['model'].coef_.shape[1]
        np.savez_compressed(path,x=x[ix].astype(np.float64),parent=z['pred'][ix])
        print(json.dumps(dict(stage='exact_original_feature_recipe',day=d,shape=x[ix].shape,elapsed=time.monotonic()-tic)),flush=True)

def step2():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    tic=time.monotonic();cache_original();data,cut=load_samples();fit=data[5]['fit'];y=data[5]['y'][fit];cache={d:dict(np.load(R/f'v54_distribution_original_features_day{d}.npz')) for d in (5,6)}
    scaler=StandardScaler().fit(cache[5]['x'][fit]);xx={d:scaler.transform(c['x']) for d,c in cache.items()};basew=weights(data[5]['length'][fit]);short=data[5]['length'][fit]<=5;reports=[]
    for strength in (0,1,2,4):
        w=np.ones(len(y)) if strength==0 else basew*np.where(short,strength,1);w/=w.mean()
        model=LogisticRegression(C=.01,solver='lbfgs',max_iter=1000,tol=1e-6)
        with threadpool_limits(limits=1):model.fit(xx[5][fit],y,sample_weight=w)
        assert model.n_iter_[0]<1000
        for d,a in data.items():
            with threadpool_limits(limits=1):p=model.predict_proba(xx[d])[:,1]
            m=a['test'];pred=a['baseline'].copy();pred[m]=.75*rank(cache[d]['parent'][m])+.25*rank(p[m]);label=f'original_V53_final_head_weight{strength}';reports.append(evaluate(a,pred,d,label));np.save(R/f'v54_distribution_step2_{label}_day{d}.npy',pred)
    save('step2',dict(reports=reports,fit_cutoff=cut,fit_rows=int(fit.sum()),recipe='Exact original 190-feature V53 logistic final head: StandardScaler, C=.01, same .75 parent/.25 head rank blend; parent experts frozen. Weight0 unweighted matched control; weight1 matches target; 2/4 additionally multiply <=5 histories. No new stacker architecture.',limitation='Only final V53 head refitted, not every upstream expert. Earlier40% day5 fits; later60% and day6 test. Original parent V53 day5 meta features inherit historical OOF dependencies.',elapsed_seconds=time.monotonic()-tic))

FEATURE_NAMES=['history_length','history_repeat_count','bucket','daytype','log_all_user_clicks','log_all_user_events','log_user_item_training_clicks','pair_bayes5','pair_bayes20','user_ctr_bayes20','item_ctr','log_item_exposures','log_joint_item_popularity','joint_item_ctr','log_item_first_appearance_age','ALS_item_bias','ALS_direct_score','allclick_profile5_dot','allclick_profile20_dot','user_bucket_click_affinity','user_bucket_ctr','user_bucket_support']
def segment_features(day,rows):
    from scipy.sparse import coo_matrix
    t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');start=int(t[:,2].min());cut=start+day*86400;prior=np.asarray(t[t[:,2]<cut]);assert prior[:,2].max()<cut<=rows[:,2].min();nu=int(t[:,0].max())+1;ni=int(t[:,1].max())+1
    u=rows[:,0];i=rows[:,1];pu=prior[:,0];pi=prior[:,1];y=prior[:,5];click=prior[y==1];uc=np.bincount(click[:,0],minlength=nu);un=np.bincount(prior[::10,0],minlength=nu);ic=np.bincount(pi,weights=y,minlength=ni);inn=np.bincount(pi,minlength=ni);globalctr=float(y.mean());itemctr=(ic+100*globalctr)/(inn+100)
    pairs=coo_matrix((np.ones(len(click),dtype=np.float32),(click[:,0],click[:,1])),shape=(nu,ni)).tocsr();pc=np.asarray(pairs[u,i]).ravel()
    joint=prior[:,3]*2+prior[:,4];ix=rows[:,3]*2+rows[:,4];jn=np.bincount(pi*8+joint,minlength=ni*8).reshape(ni,8);jc=np.bincount(pi*8+joint,weights=y,minlength=ni*8).reshape(ni,8)
    jointctr=(jc[i,ix]+100*itemctr[i])/(jn[i,ix]+100)
    first=np.full(ni,cut,dtype=np.int32);np.minimum.at(first,pi,prior[:,2]);firstage=(rows[:,2]-first[i])/86400
    path=R/(f'imp17_als_cut{day}.npz' if day in (3,4) else f'imp15_model_day{day}.npz');model=np.load(path);assert int(model['cutoff'])<=cut
    V=model['V'];U=model['U'];bi=model['bi'];bu=model['bu'];g=float(model['g']);profile=pairs@V;priorv=V.mean(0);dots=[]
    for strength in (5,20):
        shrunk=(profile+strength*priorv)/(uc[:,None]+strength);dots.append(np.einsum('ij,ij->i',shrunk[u],V[i]))
    direct=g+bu[u]+bi[i]+np.einsum('ij,ij->i',U[u],V[i])
    ub=pu*4+prior[:,3];qb=u*4+rows[:,3];ubn=np.bincount(ub,minlength=nu*4);ubc=np.bincount(ub,weights=y,minlength=nu*4);globalb=np.bincount(click[:,3],minlength=4)/max(len(click),1)
    affinity=(ubc[qb]+10*globalb[rows[:,3]])/(uc[u]+10);userctr=(uc+20*10*globalctr)/(un+20);bctr=(ubc[qb]+20*userctr[u]/10)/(ubn[qb]+20)
    hh=np.asarray(h[rows[:,6]]);n=(hh>=0).sum(1);rep=((hh==i[:,None])&(hh>=0)).sum(1)
    x=np.column_stack([n,rep,rows[:,3],rows[:,4],np.log1p(uc[u]),np.log1p(un[u]),np.log1p(pc),(pc+5*itemctr[i])/(uc[u]+5),(pc+20*itemctr[i])/(uc[u]+20),userctr[u],itemctr[i],np.log1p(inn[i]),np.log1p(jn[i,ix]),jointctr,np.log1p(firstage),bi[i],direct,*dots,affinity,bctr,np.log1p(ubn[qb])]).astype(np.float32)
    assert np.isfinite(x).all();return x

def prepare_segments():
    tic=time.monotonic();t=np.load(R/'train.npy',mmap_mode='r');start=int(t[:,2].min());data,_=load_samples()
    for day in (3,4,5,6):
        path=R/f'v54_distribution_segment_features_day{day}.npz'
        if path.exists():continue
        if day in (3,4):
            ev=np.flatnonzero((t[::10,2]>=start+day*86400)&(t[::10,2]<start+(day+1)*86400));_,first=np.unique(t[ev*10,0],return_index=True);ev=ev[np.sort(first)];u=t[ev*10,0];ev=ev[(u.astype(np.uint64)*np.uint64(2654435761)%100)<20];rows=np.asarray(t[(ev[:,None]*10+np.arange(10)).ravel()])
        else:rows=data[day]['rows']
        x=segment_features(day,rows);np.savez_compressed(path,x=x,y=rows[:,5],events=rows[:,6],time=rows[:,2]);print(json.dumps(dict(stage='causal_segment_features',day=day,rows=len(rows),elapsed=time.monotonic()-tic)),flush=True)

def segment_model(step):
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    tic=time.monotonic();prepare_segments();data,cut=load_samples();training=[dict(np.load(R/f'v54_distribution_segment_features_day{d}.npz')) for d in (3,4)];x=np.concatenate([z['x'] for z in training]);y=np.concatenate([z['y'] for z in training]);times=np.concatenate([z['time'] for z in training]);assert times.max()<data[5]['time'].min()
    # Step3 excludes dedicated bucket propensity features; step4 tests their residual effect.
    cols=list(range(19)) if step==3 else list(range(22));fit=x[:,0]<=5 if step==3 else np.ones(len(y),bool)
    model=lgb_fit(x[fit][:,cols],y[fit],seed=5430+step,repeat=step==3)
    a=data[5];early=a['fit'];mapmodel=LogisticRegression(C=1000,max_iter=1000).fit(a['baseline'][early,None],a['y'][early]);slope=float(mapmodel.coef_[0,0]);intercept=float(mapmodel.intercept_[0]);assert slope>0
    reports=[];cold=[]
    for d,a in data.items():
        z=dict(np.load(R/f'v54_distribution_segment_features_day{d}.npz'));assert np.array_equal(z['events'],a['events']);pred=model.predict_proba(z['x'][:,cols])[:,1];bp=expit(slope*a['baseline']+intercept)
        gates=[('short_only',a['length']<=5)] if step==3 else [('bucket3_only',a['rows'][:,3]==3),('all_buckets',np.ones(len(pred),bool))]
        for glabel,gate in gates:
            for w in (.05,.1,.25,.5):
                out=a['baseline'].copy();pp=(1-w)*bp[gate]+w*pred[gate];out[gate]=(logit(np.clip(pp,1e-8,1-1e-8))-intercept)/slope;assert np.array_equal(out[~gate],a['baseline'][~gate]);label=f'{glabel}_w{w}';reports.append(evaluate(a,out,d,label));np.save(R/f'v54_distribution_step{step}_{label}_day{d}.npy',out)
        zero=a['test']&(a['length']==0)
        for name,col in [('joint_item_popularity',12),('joint_item_ctr',13),('item_first_appearance_age',14),('ALS_item_bias',15)]:cold.append(dict(day=d,feature=name,zero_rows=int(zero.sum()),zero_positives=int(a['y'][zero].sum()),auc=auc(a['y'][zero],z['x'][zero,col])))
    save(f'step{step}',dict(reports=reports,zero_history_probes=cold,train_days=[3,4],fit_rows=int(fit.sum()),last_fit_label_timestamp=int(times[fit].max()),first_validation_timestamp=int(data[5]['time'].min()),features=[FEATURE_NAMES[c] for c in cols],protocol='Feature counts frozen strictly before each training/validation day. Whole earlier training-window click matrix (not truncated history), Bayesian shrunk full-click item-factor profile. Dedicated short model fits only <=5 histories; context model includes smoothed joint item CTR and user bucket propensity. Monotone probability mapping fits earlyday5 only; output blended only inside named gate.',elapsed_seconds=time.monotonic()-tic))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['1','2','3','4']);args=p.parse_args();segment_model(int(args.step)) if args.step in ('3','4') else globals()['step'+args.step]()
