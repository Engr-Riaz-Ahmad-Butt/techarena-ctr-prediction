"""V7: logistic matrix factorization of earlier impressions, added to V6 features.

Two arms share identical CatBoost settings so the comparison is fair:
  A: V6 features only (separates the effect of training more trees)
  B: V6 features + factorization features
Feature matrices are cached under features/ so later experiments skip rebuilding them.
"""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v6 import *

FEAT=ROOT/'features';FEAT.mkdir(exist_ok=True)
THREADS=12
PARAMS=dict(depth=7,learning_rate=.1,loss_function='Logloss',thread_count=THREADS,random_seed=42,allow_writing_files=False)

def cached(name,build):
    path=FEAT/(name+'.npy')
    if path.exists():return np.load(path)
    x=build();np.save(path,x);return x

def v6_features(tag,parts,cutoff):
    """Returns V6 feature matrices for the given (name, rows, histories) parts, building stats only if needed."""
    names=[tag+'_'+name for name,_,_ in parts]
    if all((FEAT/(n+'.npy')).exists() for n in names):
        return [np.load(FEAT/(n+'.npy')) for n in names]
    stats=preference_stats(cutoff);out=[]
    for n,(_,a,h) in zip(names,parts):out.append(cached(n,lambda:preference_features(a,h,stats)))
    del stats;gc.collect()
    return out

def fit_mf(rows,dim=32,lam=.01,epochs=5,lr=.1,seed=7,batch=32768):
    """Logistic factorization: logit = g + bu + bi + p_u . q_i, trained with Adagrad."""
    u=rows[:,0].astype(np.int64);i=rows[:,1].astype(np.int64);y=rows[:,5].astype('float32')
    rng=np.random.default_rng(seed)
    P=(rng.standard_normal((nu,dim))*.05).astype('float32');Q=(rng.standard_normal((ni,dim))*.05).astype('float32')
    bu=np.zeros(nu,'float32');bi=np.zeros(ni,'float32')
    hP=np.full((nu,dim),1e-3,'float32');hQ=np.full((ni,dim),1e-3,'float32')
    hbu=np.full(nu,1e-3,'float32');hbi=np.full(ni,1e-3,'float32')
    m=float(y.mean());g=np.float32(np.log(m/(1-m)))
    for _ in range(epochs):
        order=rng.permutation(len(y))
        for s in range(0,len(y),batch):
            k=order[s:s+batch];uu=u[k];ii=i[k];pu=P[uu];qi=Q[ii]
            z=g+bu[uu]+bi[ii]+(pu*qi).sum(axis=1)
            err=(1/(1+np.exp(-z))-y[k]).astype('float32')
            for M,H,b,hb,idx,grad in [(P,hP,bu,hbu,uu,err[:,None]*qi+lam*pu),(Q,hQ,bi,hbi,ii,err[:,None]*pu+lam*qi)]:
                uniq,inv=np.unique(idx,return_inverse=True)
                gs=np.zeros((len(uniq),dim),'float32');np.add.at(gs,inv,grad)
                gb=np.bincount(inv,weights=err,minlength=len(uniq)).astype('float32')
                H[uniq]+=gs*gs;M[uniq]-=lr*gs/np.sqrt(H[uniq])
                hb[uniq]+=gb*gb;b[uniq]-=lr*gb/np.sqrt(hb[uniq])
    return g,P,Q,bu,bi

def mf_score(model,rows):
    g,P,Q,bu,bi=model
    return g+bu[rows[:,0]]+bi[rows[:,1]]+(P[rows[:,0]]*Q[rows[:,1]]).sum(axis=1)

LAMBDA=[]

def mf_stats(cutoff):
    path=FEAT/('mf_'+str(cutoff)+'.pkl')
    if path.exists():
        with path.open('rb') as f:out=pickle.load(f)
        if not LAMBDA:LAMBDA.append(out[2])
        return out
    a=np.asarray(train[train[:,2]<cutoff])
    if not LAMBDA:
        # Choose regularization once, on the last day before the first cutoff, fitting on earlier rows only.
        early=a[a[:,2]<cutoff-DAY];late=a[a[:,2]>=cutoff-DAY];best=None
        for lam in [.003,.01,.03]:
            auc=roc_auc_score(late[:,5],mf_score(fit_mf(early,lam=lam),late))
            print('MF lambda',lam,'next-day AUC',round(auc,6),flush=True)
            if best is None or auc>best[0]:best=(auc,lam)
        LAMBDA.append(best[1]);del early,late
    model=fit_mf(a,lam=LAMBDA[0])
    seen=np.bincount(a[:,0],minlength=nu)
    out=(model,seen,LAMBDA[0])
    with path.open('wb') as f:pickle.dump(out,f,protocol=5)
    del a;gc.collect()
    return out

def mf_features(a,histories,stats):
    (g,P,Q,bu,bi),seen,_=stats;cols=[]
    u=a[:,0];i=a[:,1]
    dot=(P[u]*Q[i]).sum(axis=1)
    cols.extend([g+bu[u]+bi[i]+dot,dot,bi[i],bu[u],np.log1p(seen[u])])
    hv=[]
    for off in range(0,len(a),20000):
        b=a[off:off+20000];h=np.asarray(histories[b[:,6]]);valid=h>=0;n=valid.sum(axis=1)
        qh=Q[np.maximum(h,0)]*valid[:,:,None]
        mean=qh.sum(axis=1)/np.maximum(n[:,None],1);qc=Q[b[:,1]]
        last=Q[np.maximum(h[np.arange(len(b)),np.maximum(n-1,0)],0)]*(n>0)[:,None]
        sims=(qh*qc[:,None,:]).sum(axis=2)
        hv.append(np.column_stack([(mean*qc).sum(axis=1),(last*qc).sum(axis=1),
                                   np.where(valid,sims,-1e9).max(axis=1).clip(-50,50)*(n>0)]))
    hv=np.concatenate(hv)
    # A user unseen before the cutoff has no learned vector: fall back to history items.
    fold=np.where(seen[u]>0,dot,hv[:,0])
    cols.extend([hv[:,0],hv[:,1],hv[:,2],fold,g+bi[i]+fold])
    x=np.column_stack(cols).astype('float32');relative=[]
    for col in [0,1,5,8,9]:
        v=x[:,col].reshape(-1,10);e=np.exp(v-v.max(axis=1,keepdims=True))
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),(v-v.max(axis=1,keepdims=True)).ravel(),
                         (e/e.sum(axis=1,keepdims=True)).ravel(),v.argsort(axis=1).argsort(axis=1).ravel()])
    return np.column_stack([x,*relative]).astype('float32')

def train_arm(xf,yf,xv,yv,iterations=None):
    if iterations is None:
        model=CatBoostClassifier(iterations=4000,eval_metric='AUC',verbose=200,**PARAMS)
        model.fit(xf,yf,eval_set=(xv,yv),early_stopping_rounds=100)
    else:
        model=CatBoostClassifier(iterations=iterations,verbose=500,**PARAMS);model.fit(xf,yf)
    return model,float(roc_auc_score(yv,model.predict_proba(xv)[:,1]))

if __name__=='__main__':
    t0=time.time()
    v6={}
    path=ROOT/'validation_v6.json'
    if path.exists():v6={r['validation_day']:r['v6_auc'] for r in json.loads(path.read_text())}
    reports=[];chosen={}
    for day in [6,5]:
        print('FACTORIZATION VALIDATION DAY',day,flush=True)
        cutoff=start+(day-2)*DAY;fit,val=split(day)
        xf,xv=v6_features('v6_day'+str(day),[('fit',fit,hist),('val',val,hist)],cutoff)
        ms=mf_stats(cutoff)
        mf_f=cached('mf_day'+str(day)+'_fit',lambda:mf_features(fit,hist,ms))
        mf_v=cached('mf_day'+str(day)+'_val',lambda:mf_features(val,hist,ms))
        print('Factorization AUC alone',round(roc_auc_score(val[:,5],mf_v[:,0]),6),'lambda',ms[2],flush=True)
        report={'validation_day':day,'rows':len(val),'v6_auc':v6.get(day)}
        for arm,(a_f,a_v) in [('A_v6_more_trees',(xf,xv)),('B_v7',(np.hstack([xf,mf_f]),np.hstack([xv,mf_v])))]:
            print('ARM',arm,'features',a_f.shape[1],flush=True)
            model,auc=train_arm(a_f,fit[:,5],a_v,val[:,5],chosen.get(arm))
            if day==6:chosen[arm]=int(model.tree_count_)
            report[arm+'_auc']=auc;report[arm+'_trees']=chosen[arm]
            if arm=='B_v7':np.savez_compressed(ROOT/('predictions_mf_day'+str(day)+'.npz'),y=val[:,5],pred=model.predict_proba(a_v)[:,1],events=val[:,6])
            print(arm,'AUC',round(auc,6),'elapsed min',round((time.time()-t0)/60,1),flush=True)
            del model,a_f,a_v;gc.collect()
        report['v7_minus_A']=report['B_v7_auc']-report['A_v6_more_trees_auc']
        reports.append(report);(ROOT/'validation_mf.json').write_text(json.dumps(reports,indent=2));print(report,flush=True)
        del xf,xv,mf_f,mf_v,fit,val,ms;gc.collect()

    use_mf=all(r['v7_minus_A']>0 for r in reports)
    arm='B_v7' if use_mf else 'A_v6_more_trees'
    print('Selected arm',arm,flush=True)
    cutoff=start+6*DAY;fit=np.asarray(train[train[:,2]>=cutoff])
    xf,xe=v6_features('v6_final',[('fit',fit,hist),('eval',evaluation,ehist)],cutoff)
    if use_mf:
        ms=mf_stats(cutoff)
        xf=np.hstack([xf,cached('mf_final_fit',lambda:mf_features(fit,hist,ms))])
        xe=np.hstack([xe,cached('mf_final_eval',lambda:mf_features(evaluation,ehist,ms))])
    model=CatBoostClassifier(iterations=chosen[arm],verbose=500,**PARAMS)
    model.fit(xf,fit[:,5]);model.save_model(str(ROOT/'model_mf.cbm'))
    prob=model.predict_proba(xe)[:,1]
    assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    source=EVALUATION_CSV;seen=set()
    with source.open(newline='',encoding='utf-8') as srcfile,(ROOT/'submission_mf.csv').open('w',newline='',encoding='utf-8') as dest:
        writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
        for row,p in zip(csv.DictReader(srcfile),prob,strict=True):
            assert row['rowid'] not in seen
            seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')])
    assert len(seen)==529390
    print('Verified MF submission',len(seen),'rows, arm',arm,'minutes',round((time.time()-t0)/60,1),flush=True)
