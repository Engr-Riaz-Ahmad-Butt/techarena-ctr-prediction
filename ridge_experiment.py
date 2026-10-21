"""V7: regularized user models in latent item space, blended with V6."""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v6 import *

def ridge_stats(cutoff):
    cache=ROOT/('preference_stats_'+str(cutoff)+'.pkl')
    if cache.exists():
        with cache.open('rb') as f:base=pickle.load(f)
    else:
        base=preference_stats(cutoff)
        with cache.open('wb') as f:pickle.dump(base,f,protocol=5)
    a=np.asarray(train[train[:,2]<cutoff]);events=a[::10]
    _,first=np.unique(events[:,0],return_index=True)
    e=events[first];h=np.asarray(hist[e[:,6]])
    hu=np.repeat(e[:,0],20);hi=h.ravel();valid=hi>=0
    warm=coo_matrix((np.ones(valid.sum(),dtype='float32'),(hu[valid],hi[valid])),shape=(nu,ni)).tocsr()
    exposure=coo_matrix((np.ones(len(a),dtype='float32'),(a[:,0],a[:,1])),shape=(nu,ni)).tocsr()
    clicks=coo_matrix((a[:,5].astype('float32'),(a[:,0],a[:,1])),shape=(nu,ni)).tocsr();clicks.eliminate_zeros()
    wc=np.asarray(warm.sum(axis=1)).ravel()
    ec=np.asarray(exposure.sum(axis=1)).ravel();cc=np.asarray(clicks.sum(axis=1)).ravel()
    rate=(cc+50*a[:,5].mean())/(ec+50)
    models=[]
    for dim in [16,32]:
        t=normalize(base[3][:,:dim]).astype('float32')
        mean=t.mean(axis=0);t=t-mean
        pop_cov=t.T@t/len(t)
        outer=(t[:,:,None]*t[:,None,:]).reshape(ni,-1)
        covariance=(exposure@outer).reshape(nu,dim,dim)
        # Warm-up clicks have no observed candidate lists. Approximate their
        # exposure covariance with ten draws from the global item distribution.
        covariance+=10*wc[:,None,None]*pop_cov[None,:,:]
        rhs=clicks@t+warm@t-rate[:,None]*(exposure@t)
        del outer
        for penalty in [1.,10.]:
            coef=np.empty((nu,dim),dtype='float32')
            for off in range(0,nu,1024):
                mat=covariance[off:off+1024].copy()
                mat[:,np.arange(dim),np.arange(dim)]+=penalty
                coef[off:off+1024]=np.linalg.solve(mat,rhs[off:off+1024,:,None])[:,:,0]
            models.append((t,coef))
        del covariance,rhs
        print('Regularized user models ready:',dim,'dimensions',flush=True)
    del a,exposure,clicks,warm
    gc.collect()
    tables={dim:tuple(normalize(p[:,:dim]).astype('float32') for p in [base[3],base[4],base[5],base[6]])
            for dim in [8,16,32,64,127]}
    return base,models,tables

def ridge_features(a,histories,stats):
    base,models,tables=stats
    context=base[0][1];taste=base[3];profile=base[4];contrast=base[5];adjusted=base[6]
    blocks=[]
    for off in range(0,len(a),20000):
        b=a[off:off+20000];u=b[:,0];i=b[:,1];bucket=b[:,3]
        h=np.asarray(histories[b[:,6]]);mask=h>=0;safe=np.maximum(h,0);n=mask.sum(axis=1)
        g=context['global'];values=[bucket,b[:,4],n,np.log1p(base[7][u]),(h==i[:,None]).sum(axis=1)]
        for name,ident in [('user',u),('item',i)]:
            impressions=context[name+'_impr'][ident];clicks=context[name+'_click'][ident]
            prior=(clicks+50*g)/(impressions+50);key=ident*4+bucket
            bi=context[name+'_bucket_impr'][key];bc=context[name+'_bucket_click'][key]
            recent_i=context[name+'_recent_impr'][ident];recent_c=context[name+'_recent_click'][ident]
            values.extend([prior,np.log1p(impressions),(bc+30*prior)/(bi+30),(recent_c+30*prior)/(recent_i+30)])
        for dim in [8,16,32,64,127]:
            item_table,*user_tables=tables[dim]
            t=item_table[i]
            for p in user_tables:
                values.append((p[u]*t).sum(axis=1))
            assert len(b)%10==0 and np.all(b[:,6].reshape(-1,10)==b[::10,6,None])
            history_vec=item_table[safe[::10]]
            sim=np.einsum('ekd,ejd->ekj',t.reshape(-1,10,dim),history_vec,optimize=True).reshape(-1,20)*mask
            age=n[:,None]-1-np.arange(20)[None,:];recent=mask&(age<5)
            values.extend([sim.sum(axis=1)/np.maximum(n,1),sim.max(axis=1),
                           (sim*recent).sum(axis=1)/np.maximum(recent.sum(axis=1),1)])
        for t,coef in models:
            v=t[i];p=coef[u]
            score=(p*v).sum(axis=1)
            values.extend([score,score/np.maximum(np.linalg.norm(p,axis=1)*np.linalg.norm(v,axis=1),1e-8),
                           np.log1p(np.linalg.norm(p,axis=1))])
        blocks.append(np.column_stack(values).astype('float32'))
    x=np.concatenate(blocks);relative=[]
    for col in [9,11,13,19,25,31,37,43,46,49,52]:
        v=x[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),(v-v.max(axis=1,keepdims=True)).ravel()])
    return np.column_stack([x,*relative]).astype('float32')

old={r['validation_day']:r for r in json.loads((ROOT/'validation_v6.json').read_text())}
reports=[];chosen=0;weight=None
for day in [6,5]:
    print('RIDGE VALIDATION DAY',day,flush=True)
    stats=ridge_stats(start+(day-2)*DAY);fit,val=split(day)
    xf=ridge_features(fit,hist,stats);xv=ridge_features(val,hist,stats)
    print('Feature count',xf.shape[1],flush=True)
    model=CatBoostClassifier(iterations=650 if day==6 else chosen,depth=7,learning_rate=.08,loss_function='Logloss',eval_metric='AUC',thread_count=6,random_seed=47,verbose=100,allow_writing_files=False)
    if day==6:
        model.fit(xf,fit[:,5],eval_set=(xv,val[:,5]),early_stopping_rounds=60)
        chosen=int(model.tree_count_);model.save_model(str(ROOT/'validation_v7.cbm'))
    else:model.fit(xf,fit[:,5])
    pred=model.predict_proba(xv)[:,1]
    previous=np.load(ROOT/('predictions_v6_day'+str(day)+'.npz'))
    assert np.array_equal(previous['events'],val[:,6]) and np.array_equal(previous['y'],val[:,5])
    if day==6:
        grid={str(w):float(roc_auc_score(val[:,5],w*pred+(1-w)*previous['pred'])) for w in [0,.25,.5,.75,1.]}
        weight=float(max(grid,key=grid.get))
        print('Blend selection:',grid,'chosen weight:',weight,flush=True)
    blend=weight*pred+(1-weight)*previous['pred']
    auc=float(roc_auc_score(val[:,5],blend))
    report={'validation_day':day,'rows':len(val),'v6_auc':old[day]['v6_auc'],'new_model_auc':float(roc_auc_score(val[:,5],pred)),
            'v7_blend_auc':auc,'improvement':auc-old[day]['v6_auc'],'new_model_weight':weight,'iterations':chosen}
    reports.append(report);(ROOT/'validation_v7.json').write_text(json.dumps(reports,indent=2));print(report,flush=True)
    np.savez_compressed(ROOT/('predictions_v7_day'+str(day)+'.npz'),y=val[:,5],pred=blend,new_model_pred=pred,events=val[:,6])
    del stats,xf,xv,fit,val,model,previous
    gc.collect()

if not all(r['improvement']>0 for r in reports):
    print('Improvement not confirmed on both splits. Keep V6.',flush=True)
    sys.exit(0)
print('Both comparisons passed. Fitting final V7.',flush=True)
stats=ridge_stats(start+6*DAY);fit=np.asarray(train[train[:,2]>=start+6*DAY])
xf=ridge_features(fit,hist,stats)
model=CatBoostClassifier(iterations=chosen,depth=7,learning_rate=.08,loss_function='Logloss',thread_count=6,random_seed=47,verbose=100,allow_writing_files=False)
model.fit(xf,fit[:,5]);model.save_model(str(ROOT/'model_v7_component.cbm'));del xf,fit
xe=ridge_features(evaluation,ehist,stats);prob=model.predict_proba(xe)[:,1]
source=EVALUATION_CSV;seen=set()
with source.open(newline='',encoding='utf-8') as srcfile,(ROOT/'submission_v6.csv').open(newline='',encoding='utf-8') as oldfile,(ROOT/'submission_v7.csv').open('w',newline='',encoding='utf-8') as dest:
    writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
    for row,oldrow,p in zip(csv.DictReader(srcfile),csv.DictReader(oldfile),prob,strict=True):
        assert row['rowid']==oldrow['rowid'] and row['rowid'] not in seen
        blend=weight*p+(1-weight)*float(oldrow['predicted_probability'])
        assert np.isfinite(blend) and 0<=blend<=1
        seen.add(row['rowid']);writer.writerow([row['rowid'],format(blend,'.10g')])
assert len(seen)==529390
(ROOT/'v7_blend.json').write_text(json.dumps({'v7_weight':weight,'v6_weight':1-weight,'iterations':chosen},indent=2))
print('Verified V7 submission',len(seen),'rows',flush=True)
