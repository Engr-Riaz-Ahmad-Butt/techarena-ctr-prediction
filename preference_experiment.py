"""V6: richer item factors and exposure-adjusted user preference profiles."""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v5 import *
from threadpoolctl import threadpool_limits

threadpool_limits(limits=6)

def preference_stats(cutoff):
    cache=ROOT/('warm_stats_'+str(cutoff)+'.pkl')
    if cache.exists():
        with cache.open('rb') as f:base=pickle.load(f)
    else:
        base=build_warm_stats(cutoff)
        with cache.open('wb') as f:pickle.dump(base,f,protocol=5)
    relations,context=base
    _,keys,counts,hc,cc,_,_=relations
    row=keys//ni;col=keys%ni
    matrix=coo_matrix((counts/np.sqrt((hc[row]+10)*(cc[col]+10)),(row,col)),shape=(ni,ni)).tocsr()
    svd=TruncatedSVD(n_components=128,n_iter=5,random_state=46)
    us=svd.fit_transform(matrix);s=np.maximum(svd.singular_values_,1e-8)
    # Remove the strongest common direction when constructing taste profiles.
    left=(us/np.sqrt(s)).astype('float32');right=(svd.components_.T*np.sqrt(s)).astype('float32')
    taste=normalize(right[:,1:]).astype('float32')
    del matrix,us,svd,row,col
    a=np.asarray(train[train[:,2]<cutoff]);events=a[::10]
    _,first=np.unique(events[:,0],return_index=True)
    e=events[first];h=np.asarray(hist[e[:,6]])
    hu=np.repeat(e[:,0],20);hi=h.ravel();valid=hi>=0
    clicked=a[a[:,5]==1]
    users=np.concatenate([hu[valid],clicked[:,0]]);items=np.concatenate([hi[valid],clicked[:,1]])
    positive=coo_matrix((np.ones(len(users),dtype='float32'),(users,items)),shape=(nu,ni)).tocsr()
    exposure=coo_matrix((np.ones(len(a),dtype='float32'),(a[:,0],a[:,1])),shape=(nu,ni)).tocsr()
    pc=np.asarray(positive.sum(axis=1)).ravel();ec=np.asarray(exposure.sum(axis=1)).ravel()
    pm=(positive@taste)/np.maximum(pc[:,None],1)
    em=(exposure@taste)/np.maximum(ec[:,None],1)
    profile=normalize(pm).astype('float32')
    contrast=normalize(pm-em).astype('float32')
    # Approximate variance adjustment accounts for globally common directions.
    variance=np.maximum(np.mean(taste*taste,axis=0),1e-6)
    adjusted=normalize((pm-em)/np.sqrt(variance)).astype('float32')
    del a,events,e,h,hu,hi,clicked,positive,exposure,pm,em
    gc.collect()
    print('128-factor item model and user preference profiles ready',flush=True)
    return base,left,right,taste,profile,contrast,adjusted,pc

def preference_features(a,histories,stats):
    base,left,right,taste,profile,contrast,adjusted,pc=stats
    x=context_features(a,histories,base);parts=[]
    for off in range(0,len(a),10000):
        b=a[off:off+10000];h=np.asarray(histories[b[:,6]])
        valid=h>=0;safe=np.maximum(h,0);n=valid.sum(axis=1)
        age=n[:,None]-1-np.arange(20)[None,:]
        l=left[safe];r=right[b[:,1]];vals=[]
        for rank in [8,16,32,64,128]:
            # Shared popularity direction is excluded, retaining taste factors.
            ls=l[:,:,1:rank];rs=r[:,1:rank]
            raw=(ls*rs[:,None,:]).sum(axis=2)*valid
            sim=raw/np.maximum(np.linalg.norm(ls,axis=2)*np.linalg.norm(rs,axis=1)[:,None],1e-8)
            vals.extend([raw.sum(axis=1)/np.maximum(n,1),
                         sim.sum(axis=1)/np.maximum(n,1),sim.max(axis=1)])
            for recent in [3,5]:
                mask=valid&(age<recent)
                vals.append((sim*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1))
        t=taste[b[:,1]];u=b[:,0]
        p=profile[u];c=contrast[u];ad=adjusted[u]
        vals.extend([(p*t).sum(axis=1),(c*t).sum(axis=1),(ad*t).sum(axis=1),np.log1p(pc[u])])
        ht=(taste[safe]*valid[:,:,None]).sum(axis=1)/np.maximum(n[:,None],1)
        ht=normalize(ht)
        vals.extend([(ht*t).sum(axis=1),(ht*p).sum(axis=1)])
        # Component-wise interactions let trees combine preference dimensions.
        parts.append(np.column_stack([*vals,(p*t)[:,:16],(ht*t)[:,:16]]).astype('float32'))
    extra=np.concatenate(parts);relative=[]
    for col in [1,6,11,16,21,25,26,27,29]:
        v=extra[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),(v-v.max(axis=1,keepdims=True)).ravel()])
    return np.column_stack([x,extra,*relative]).astype('float32')

old={r['validation_day']:r for r in json.loads((ROOT/'validation_v5.json').read_text())}
reports=[];chosen=0
for day in [6,5]:
    print('PREFERENCE VALIDATION DAY',day,flush=True)
    stats=preference_stats(start+(day-2)*DAY);fit,val=split(day)
    xf=preference_features(fit,hist,stats);xv=preference_features(val,hist,stats)
    print('Feature count',xf.shape[1],flush=True)
    model=CatBoostClassifier(iterations=750 if day==6 else chosen,depth=7,learning_rate=.08,loss_function='Logloss',eval_metric='AUC',thread_count=6,random_seed=42,verbose=100,allow_writing_files=False)
    if day==6:
        model.fit(xf,fit[:,5],eval_set=(xv,val[:,5]),early_stopping_rounds=65)
        chosen=int(model.tree_count_);model.save_model(str(ROOT/'validation_v6.cbm'))
    else:model.fit(xf,fit[:,5])
    pred=model.predict_proba(xv)[:,1];auc=float(roc_auc_score(val[:,5],pred))
    previous=np.load(ROOT/('predictions_v5_day'+str(day)+'.npz'))
    assert np.array_equal(previous['events'],val[:,6]) and np.array_equal(previous['y'],val[:,5])
    assert abs(roc_auc_score(val[:,5],previous['pred'])-old[day]['v5_auc'])<1e-12
    report={'validation_day':day,'rows':len(val),'v5_auc':old[day]['v5_auc'],'v6_auc':auc,'improvement':auc-old[day]['v5_auc'],'iterations':chosen}
    reports.append(report);(ROOT/'validation_v6.json').write_text(json.dumps(reports,indent=2));print(report,flush=True)
    np.savez_compressed(ROOT/('predictions_v6_day'+str(day)+'.npz'),y=val[:,5],pred=pred,events=val[:,6])
    del stats,xf,xv,fit,val,model,previous
    gc.collect()

if not all(r['improvement']>0 for r in reports):
    print('Improvement not confirmed on both splits. Keep V5.',flush=True)
    sys.exit(0)
print('Both comparisons passed. Fitting final V6.',flush=True)
stats=preference_stats(start+6*DAY);fit=np.asarray(train[train[:,2]>=start+6*DAY])
xf=preference_features(fit,hist,stats)
model=CatBoostClassifier(iterations=chosen,depth=7,learning_rate=.08,loss_function='Logloss',thread_count=6,random_seed=42,verbose=100,allow_writing_files=False)
model.fit(xf,fit[:,5]);model.save_model(str(ROOT/'model_v6.cbm'));del xf,fit
xe=preference_features(evaluation,ehist,stats);prob=model.predict_proba(xe)[:,1]
assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
source=EVALUATION_CSV;seen=set()
with source.open(newline='',encoding='utf-8') as srcfile,(ROOT/'submission_v6.csv').open('w',newline='',encoding='utf-8') as dest:
    writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
    for row,p in zip(csv.DictReader(srcfile),prob,strict=True):
        assert row['rowid'] not in seen
        seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')])
assert len(seen)==529390
print('Verified V6 submission',len(seen),'rows',flush=True)
