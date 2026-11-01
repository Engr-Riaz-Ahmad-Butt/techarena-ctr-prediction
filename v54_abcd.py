"""Sequential A-D research screens. No baseline mutations or submissions."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R,EVALUATION_CSV

def auc(y,p):return float(roc_auc_score(y,p))
def rank(p):return rankdata(p,method='average')/(len(p)+1)
def shares(length,repeat):
    return dict(rows=len(length),history_0=int((length==0).sum()),history_1_5=int(((length>=1)&(length<=5)).sum()),history_6_20=int((length>=6).sum()),repeat=int(repeat.sum()),non_repeat=int((~repeat).sum()))
def save(name,obj):
    (R/f'v54_abcd_{name}.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj),flush=True)
def stage_a():
    tic=time.monotonic();t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');reports=[]
    for day in (5,6):
        z=np.load(R/f'selection_day{day}.npz');ev=z['events'];items=t[ev*10+np.tile(np.arange(10),len(ev)//10),1];hh=h[ev];length=(hh>=0).sum(1);rep=((hh==items[:,None])&(hh>=0)).any(1)
        for scope,take in [('full_selection',np.ones(len(ev),bool)),('sample20',(t[ev*10,0].astype(np.uint64)*np.uint64(2654435761)%100)<20)]:
            y=z['y'][take];p=z['baseline'][take];zero=length[take]==0;oracle=p.copy();oracle[zero&(y==1)]=p.max()+1;oracle[zero&(y==0)]=p.min()-1
            reports.append(dict(day=day,scope=scope,counts=shares(length[take],rep[take]),baseline_auc=auc(y,p),perfect_zero_history_auc=auc(y,oracle),maximum_zero_history_gain=auc(y,oracle)-auc(y,p),oracle_note='Label-informed upper bound: cold positives above all rows, cold negatives below all rows; includes cross-segment pairs. Not achievable calibration.'))
    # DictReader accesses only the three named fields; no evaluation label read.
    counts=dict(rows=0,history_0=0,history_1_5=0,history_6_20=0,repeat=0,non_repeat=0)
    with EVALUATION_CSV.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            history=json.loads(row['history_item_ids']); n=len(history); rep=str(row['item_id']) in {str(v) for v in history}
            counts['rows']+=1;counts['history_0' if n==0 else 'history_1_5' if n<=5 else 'history_6_20']+=1;counts['repeat' if rep else 'non_repeat']+=1
    assert counts['rows']==529390
    save('A',dict(reports=reports,evaluation_counts=counts,evaluation_access='Only item_id and history_item_ids values used, solely descriptive counts; no evaluation labels/features fitted.',elapsed_seconds=time.monotonic()-tic))

def load_samples():
    t=np.load(R/'train.npy',mmap_mode='r'); out={}
    for d in (5,6):
        a=dict(np.load(R/f'v54_repeat_screen_day{d}.npz')); ev=a['events']; rows=np.asarray(t[ev*10+np.tile(np.arange(10),len(ev)//10)])
        assert np.array_equal(rows[:,6],ev)
        a['rows']=rows;a['time']=rows[:,2];out[d]=a
    cut=int(np.quantile(out[5]['time'][::10],.4,method='lower'))
    out[5]['fit']=out[5]['time']<cut;out[5]['test']=out[5]['time']>=cut;out[6]['test']=np.ones(len(out[6]['y']),bool)
    assert out[5]['time'][out[5]['fit']].max()<out[5]['time'][out[5]['test']].min()
    assert out[5]['time'][out[5]['fit']].max()<out[6]['time'].min()
    return out,cut

def metrics(a,p,day,label):
    take=a['test']; y=a['y'][take];base=a['baseline'][take]; pred=p[take]; rep=a['repeat'][take]
    gain=auc(y,pred)-auc(y,base)
    if gain>.01:raise RuntimeError('Gain > .01: audit required')
    return dict(day=day,label=label,rows=len(y),baseline_auc=auc(y,base),auc=auc(y,pred),gain=gain,repeat_baseline_auc=auc(y[rep],base[rep]),repeat_auc=auc(y[rep],pred[rep]))

def lgb_fit(x,y,seed=5410,repeat=False):
    import lightgbm as lgb
    model=lgb.LGBMClassifier(n_estimators=120,num_leaves=7,max_depth=3,learning_rate=.025,min_child_samples=50 if repeat else 250,reg_lambda=50,reg_alpha=2,colsample_bytree=.85,subsample=.8,subsample_freq=1,random_state=seed,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(x,y);return model

def bfeatures(a):
    p=a['baseline'].reshape(-1,10);er=np.apply_along_axis(rankdata,1,p).ravel()/10
    return np.column_stack([a['baseline'],a['length'],a['repeat'],a['x'][:,5],a['rows'][:,3],a['rows'][:,4],er])

def stage_b():
    from sklearn.isotonic import IsotonicRegression
    tic=time.monotonic();data,cut=load_samples();a=data[5];fit=a['fit'];x={d:bfeatures(v) for d,v in data.items()}; model=lgb_fit(x[5][fit],a['y'][fit]);pred={d:model.predict_proba(x[d])[:,1] for d in (5,6)}
    globaliso=IsotonicRegression(out_of_bounds='clip').fit(a['baseline'][fit],a['y'][fit]);isos={}
    def segment(v):return np.select([v['length']==0,v['length']<=5],[0,1],default=2)*2+v['repeat']
    sg=segment(a)
    for group in np.unique(sg):
        take=fit&(sg==group)
        if take.sum()>=100 and a['y'][take].sum()>=10:isos[int(group)]=IsotonicRegression(out_of_bounds='clip').fit(a['baseline'][take],a['y'][take])
    results=[]
    for d,a in data.items():
        ip=globaliso.predict(a['baseline']);groups=segment(a)
        for g,m in isos.items():ip[groups==g]=m.predict(a['baseline'][groups==g])
        for label,p in [('shallow_lightgbm',pred[d]),('segment_isotonic',ip)]:
            # Ranks only over the evaluated cohort, never mixed with training scores.
            for w in (.1,.25,.5,1.):
                take=a['test'];score=a['baseline'].copy();score[take]=(1-w)*rank(a['baseline'][take])+w*rank(p[take]);tag=f'{label}_w{w}'
                results.append(metrics(a,score,d,tag));np.save(R/f'v54_abcd_B_{tag}_day{d}.npy',score)
    save('B',dict(results=results,fit_before_timestamp=cut,fit_rows=int(fit.sum()),day5_test_rows=int(data[5]['test'].sum()),features=['V53 score','history length','repeat flag','repeat clicks ago','time bucket','day type','within-event V53 rank'],protocol='User-approved early 40% day5 time block only fits both models; later day5 and sampled day6 evaluated. Isotonic groups history 0/1-5/6-20 x repeat, sparse groups fall back to global calibration.',elapsed_seconds=time.monotonic()-tic))

def aligned_source(a,day,name,key):
    z=np.load(R/f'{name}_day{day}.npz'); source=z['events'].astype(np.int64)*10+np.tile(np.arange(10),len(z['events'])//10);query=a['events'].astype(np.int64)*10+np.tile(np.arange(10),len(a['events'])//10); order=np.argsort(source);pos=np.searchsorted(source[order],query);assert np.all(source[order][pos]==query)
    assert np.array_equal(z['y'][order[pos]],a['y']);return z[key][order[pos]]

def stage_c():
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit,logit
    tic=time.monotonic(); data,cut=load_samples();t=np.load(R/'train.npy',mmap_mode='r');start=int(t[:,2].min());xx={}
    for d,a in data.items():
        prior=np.asarray(t[t[:,2]<start+d*86400]); pop=np.bincount(prior[:,1],minlength=int(t[:,1].max())+1)
        als=aligned_source(a,d,'imp15_als','raw')
        xx[d]=np.column_stack([a['x'][:,5],a['x'][:,2],a['x'][:,6],a['x'][:,7],a['x'][:,9],a['x'][:,8],np.log1p(pop[a['rows'][:,1]]),als])
    a=data[5];fit=a['fit']&a['repeat'];model=lgb_fit(xx[5][fit],a['y'][fit],repeat=True)
    calibration=LogisticRegression(C=1000,max_iter=1000).fit(a['baseline'][a['fit'],None],a['y'][a['fit']]);slope=float(calibration.coef_[0,0]);intercept=float(calibration.intercept_[0]);assert slope>0
    results=[]
    for d,a in data.items():
        p=model.predict_proba(xx[d])[:,1];bp=expit(slope*a['baseline']+intercept)
        for w in (.1,.25,.5,1.):
            score=a['baseline'].copy();rep=a['repeat'];new=(1-w)*bp[rep]+w*p[rep];score[rep]=(logit(np.clip(new,1e-8,1-1e-8))-intercept)/slope
            assert np.array_equal(score[~rep],a['baseline'][~rep]);tag=f'repeat_only_w{w}';results.append(metrics(a,score,d,tag));np.save(R/f'v54_abcd_C_{tag}_day{d}.npy',score)
    save('C',dict(results=results,fit_before_timestamp=cut,repeat_fit_rows=int(fit.sum()),features=['clicks ago','history count','last click gap','same time bucket fraction','item global repeat rate','user repeat rate','item exposure popularity','ALS raw score'],protocol='Dedicated repeat-only LightGBM fitted on early-day5 repeats. Logistic monotone probability map fitted on early-day5 only; inverse map preserves every non-repeat V53 score exactly. Prefix statistics cached by experiment2.',elapsed_seconds=time.monotonic()-tic))

def stage_d():
    tic=time.monotonic();data,cut=load_samples();a=data[5];x={d:bfeatures(v) for d,v in data.items()};models=[lgb_fit(x[5][a['fit']],a['y'][a['fit']],seed=s) for s in (5411,5412,5413)];results=[]
    for d,a in data.items():
        seeds=[]
        for s in (0,1,2):
            z=np.load(R/f'v21_fastseeds_day{d}_s{s}_ep1_raw.npz'); src=z['events'].astype(np.int64)*10+np.tile(np.arange(10),len(z['events'])//10);q=a['events'].astype(np.int64)*10+np.tile(np.arange(10),len(a['events'])//10);order=np.argsort(src);ix=np.searchsorted(src[order],q);assert np.array_equal(src[order][ix],q);assert np.array_equal(z['y'][order[ix]],a['y']);seeds.append(z['raw'][order[ix]])
        take=a['test'];seedmean=np.mean([rank(p[take]) for p in seeds],axis=0);tree=aligned_source(a,d,'imp19_tree','raw');newmean=np.mean([rank(m.predict_proba(x[d][take])[:,1]) for m in models],axis=0)
        different=aligned_source(a,d,'imp15_als','raw')
        arms={'saved_three_neural_seeds':seedmean,'different_tree':rank(tree[take]),'different_ALS':rank(different[take]),'new_three_stacker_seeds':newmean,'seeds_plus_tree':.5*seedmean+.5*rank(tree[take])}
        for label,p in arms.items():
            for w in (.05,.1,.25,.5):
                score=a['baseline'].copy();score[take]=(1-w)*rank(a['baseline'][take])+w*rank(p);tag=f'{label}_w{w}';results.append(metrics(a,score,d,tag));np.save(R/f'v54_abcd_D_{tag}_day{d}.npy',score)
    save('D',dict(results=results,fit_before_timestamp=cut,protocol='Reuse existing three causal neural-seed predictions and distinct tree/ALS predictions (already ingredients of V53); test extra rank averaging. Additionally train three new bagged shallow stacker seeds on approved early-day5 block. Full V53 pipeline not retrained for seeds: original logistic meta layer is deterministic.',elapsed_seconds=time.monotonic()-tic))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['A','B','C','D']);args=p.parse_args();globals()['stage_'+args.stage.lower()]()
