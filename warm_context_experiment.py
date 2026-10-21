"""V5: warm-up click relationships and smoothed contextual click rates."""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v4 import *
from scipy.sparse import csr_matrix
import gc

def build_warm_stats(cutoff):
    path=ROOT/('stats_'+str(cutoff)+'.pkl')
    with path.open('rb') as f:base=pickle.load(f)
    a=np.asarray(train[train[:,2]<cutoff])
    clicked=a[a[:,5]==1]
    h=np.asarray(hist[clicked[:,6]])
    left=h.ravel();right=np.repeat(clicked[:,1],20);mask=left>=0
    matrix=coo_matrix((np.ones(mask.sum(),dtype='float32'),(left[mask],right[mask])),shape=(ni,ni)).tocsr()
    del h,left,right,mask,clicked
    events=a[::10]
    _,first=np.unique(events[:,0],return_index=True)
    histories=np.asarray(hist[events[first,6]])
    # Only the first available pre-cutoff history per user is used, avoiding
    # repeated counting of the same warm-up clicks across overlapping histories.
    # Every pair is directed from an older click to a newer click.
    warm_pairs=0
    for j in range(1,20):
        previous=histories[:,:j];following=histories[:,j]
        source=previous.ravel();target=np.repeat(following,j)
        mask=(source>=0)&(target>=0)
        matrix += coo_matrix((np.ones(mask.sum(),dtype='float32'),(source[mask],target[mask])),shape=(ni,ni)).tocsr()
        warm_pairs+=int(mask.sum())
    del histories,events,previous,following,source,target,mask
    hc=np.asarray(matrix.sum(axis=1)).ravel();cc=np.asarray(matrix.sum(axis=0)).ravel()
    co=matrix.tocoo();del matrix
    keys=co.row.astype('int64')*ni+co.col
    order=np.argsort(keys);keys=keys[order];counts=co.data[order];del order
    co.data /= np.sqrt((hc[co.row]+10)*(cc[co.col]+10))
    svd=TruncatedSVD(n_components=64,n_iter=5,random_state=44)
    us=svd.fit_transform(co.tocsr());s=np.maximum(svd.singular_values_,1e-8)
    src=(us/np.sqrt(s)).astype('float32');dst=(svd.components_.T*np.sqrt(s)).astype('float32')
    print('Warm-up pairs added:',warm_pairs,'Distinct combined pairs:',len(keys),flush=True)
    del co,us,svd
    context={}
    y=a[:,5];g=float(y.mean())
    bucket_impr=np.bincount(a[:,3],minlength=4)
    bucket_click=np.bincount(a[:,3],weights=y,minlength=4)
    context['global']=g
    context['bucket_ctr']=(bucket_click+100*g)/(bucket_impr+100)
    for name,col,size in [('user',0,nu),('item',1,ni)]:
        ids=a[:,col];key=ids*4+a[:,3]
        context[name+'_impr']=np.bincount(ids,minlength=size)
        context[name+'_click']=np.bincount(ids,weights=y,minlength=size)
        context[name+'_bucket_impr']=np.bincount(key,minlength=size*4)
        context[name+'_bucket_click']=np.bincount(key,weights=y,minlength=size*4)
        recent=a[:,2]>=cutoff-DAY
        context[name+'_recent_impr']=np.bincount(ids[recent],minlength=size)
        context[name+'_recent_click']=np.bincount(ids[recent],weights=y[recent],minlength=size)
    del a
    gc.collect()
    return (base,keys,counts,hc,cc,src,dst),context

def context_features(a,histories,stats):
    relations,context=stats
    x=relation_features(a,histories,relations)
    bucket=a[:,3];g=context['global'];columns=[]
    for name,col in [('user',0),('item',1)]:
        ids=a[:,col];key=ids*4+bucket
        exposure=context[name+'_impr'][ids];clicks=context[name+'_click'][ids]
        prior=(clicks+50*g)/(exposure+50)
        # Conditional prior accounts for each bucket's overall click propensity.
        conditional=np.clip(prior*context['bucket_ctr'][bucket]/g,1e-5,.999)
        ni_=context[name+'_bucket_impr'][key];nc=context[name+'_bucket_click'][key]
        ri=context[name+'_recent_impr'][ids];rc=context[name+'_recent_click'][ids]
        columns.extend([(clicks+10*g)/(exposure+10),(clicks+150*g)/(exposure+150),
                        (nc+20*conditional)/(ni_+20),(nc+80*conditional)/(ni_+80),
                        np.log1p(ni_),(rc+30*prior)/(ri+30),
                        (rc+30*prior)/(ri+30)-prior,ni_/np.maximum(exposure,1)])
    columns.append(context['bucket_ctr'][bucket])
    extra=np.column_stack(columns).astype('float32');relative=[]
    for col in [8,10,11,13,14]:
        v=extra[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),(v-v.max(axis=1,keepdims=True)).ravel()])
    return np.column_stack([x,extra,*relative]).astype('float32')

old={r['validation_day']:r for r in json.loads((ROOT/'validation_v4.json').read_text())}
reports=[];chosen=0
for day in [6,5]:
    print('WARM CONTEXT VALIDATION DAY',day,flush=True)
    stats=build_warm_stats(start+(day-2)*DAY);fit,val=split(day)
    xf=context_features(fit,hist,stats);xv=context_features(val,hist,stats)
    print('Feature count',xf.shape[1],flush=True)
    model=CatBoostClassifier(iterations=750 if day==6 else chosen,depth=7,learning_rate=.08,loss_function='Logloss',eval_metric='AUC',thread_count=6,random_seed=42,verbose=100,allow_writing_files=False)
    if day==6:
        model.fit(xf,fit[:,5],eval_set=(xv,val[:,5]),early_stopping_rounds=65)
        chosen=int(model.tree_count_);model.save_model(str(ROOT/'validation_v5.cbm'))
    else:model.fit(xf,fit[:,5])
    pred=model.predict_proba(xv)[:,1];auc=float(roc_auc_score(val[:,5],pred))
    previous=np.load(ROOT/('predictions_v4_day'+str(day)+'.npz'))
    assert np.array_equal(previous['events'],val[:,6]) and np.array_equal(previous['y'],val[:,5])
    assert abs(roc_auc_score(val[:,5],previous['pred'])-old[day]['v4_auc'])<1e-12
    report={'validation_day':day,'rows':len(val),'v4_auc':old[day]['v4_auc'],'v5_auc':auc,'improvement':auc-old[day]['v4_auc'],'iterations':chosen}
    reports.append(report);(ROOT/'validation_v5.json').write_text(json.dumps(reports,indent=2));print(report,flush=True)
    np.savez_compressed(ROOT/('predictions_v5_day'+str(day)+'.npz'),y=val[:,5],pred=pred,events=val[:,6])
    del stats,xf,xv,fit,val,model,previous
    gc.collect()

if not all(r['improvement']>0 for r in reports):
    print('Improvement not confirmed on both splits. Keep V4.',flush=True)
    sys.exit(0)
print('Both comparisons passed. Fitting final V5.',flush=True)
stats=build_warm_stats(start+6*DAY);fit=np.asarray(train[train[:,2]>=start+6*DAY])
xf=context_features(fit,hist,stats)
model=CatBoostClassifier(iterations=chosen,depth=7,learning_rate=.08,loss_function='Logloss',thread_count=6,random_seed=42,verbose=100,allow_writing_files=False)
model.fit(xf,fit[:,5]);model.save_model(str(ROOT/'model_v5.cbm'));del xf,fit
xe=context_features(evaluation,ehist,stats);prob=model.predict_proba(xe)[:,1]
assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
source=EVALUATION_CSV;seen=set()
with source.open(newline='',encoding='utf-8') as srcfile,(ROOT/'submission_v5.csv').open('w',newline='',encoding='utf-8') as dest:
    writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
    for row,p in zip(csv.DictReader(srcfile),prob,strict=True):
        assert row['rowid'] not in seen
        seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')])
assert len(seen)==529390
print('Verified V5 submission',len(seen),'rows',flush=True)
