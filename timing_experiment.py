"""Actual elapsed-time features, compared against V2 on rolling time splits."""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v2 import *
import pickle

times=np.load(ROOT/'train_times.npy',mmap_mode='r')
etimes=np.load(ROOT/'evaluation_times.npy',mmap_mode='r')

def cached_stats(cutoff):
    path=ROOT/('stats_'+str(cutoff)+'.pkl')
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    stats=extra_statistics(cutoff)
    with path.open('wb') as f:pickle.dump(stats,f,protocol=5)
    return stats

def timing_features(a,histories,timestamps,stats):
    x=extra_features(a,histories,stats)
    emb=stats[1]; result=[]
    for off in range(0,len(a),20000):
        b=a[off:off+20000];h=np.asarray(histories[b[:,6]])
        ts=np.asarray(timestamps[b[:,6]]);valid=h>=0;length=valid.sum(axis=1)
        hours=np.maximum(b[:,2,None].astype('float64')-ts,0)/3600
        sim=(emb[np.maximum(h,0)]*emb[b[:,1],None,:]).sum(axis=2)*valid
        repeat=(h==b[:,1,None])&valid
        vals=[]
        for half in [1,6,24,72]:
            w=np.exp2(-hours/half)*valid
            vals.extend([(sim*w).sum(axis=1)/np.maximum(w.sum(axis=1),1e-10),
                         (sim*w).sum(axis=1),w.sum(axis=1),(repeat*w).sum(axis=1)])
        for window in [1,6,24,72]:
            mask=(hours<window)&valid
            vals.extend([mask.sum(axis=1),(sim*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1)])
        last=np.where(valid,hours,1e6).min(axis=1)
        oldest=np.where(valid,hours,0).max(axis=1)
        avg=(hours*valid).sum(axis=1)/np.maximum(length,1)
        repeat_age=np.where(repeat,hours,1e6).min(axis=1)
        for v in [last,oldest,avg,repeat_age]:vals.append(np.log1p(v))
        vals.append(length/np.maximum(oldest,1))
        samebucket=valid&((ts%86400//21600)==b[:,3,None])
        vals.extend([samebucket.sum(axis=1),
                     (sim*samebucket).sum(axis=1)/np.maximum(samebucket.sum(axis=1),1)])
        result.append(np.column_stack(vals).astype('float32'))
    t=np.concatenate(result)
    relative=[]
    for col in [0,1,4,5,8,9,12,13,30]:
        v=t[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),
                         (v-v.max(axis=1,keepdims=True)).ravel()])
    return np.column_stack([x,t,*relative]).astype('float32')

old_reports={r['validation_day']:r for r in json.loads((ROOT/'validation_v2.json').read_text())}
reports=[];chosen=0
for day in [6,5]:
    print('TIMING VALIDATION DAY',day,flush=True)
    stats=cached_stats(start+(day-2)*DAY);fit,val=split(day)
    xf=timing_features(fit,hist,times,stats);xv=timing_features(val,hist,times,stats)
    print('Feature count',xf.shape[1],flush=True)
    model=CatBoostClassifier(iterations=600 if day==6 else chosen,depth=7,learning_rate=.08,loss_function='Logloss',eval_metric='AUC',thread_count=6,random_seed=42,verbose=100,allow_writing_files=False)
    if day==6:
        model.fit(xf,fit[:,5],eval_set=(xv,val[:,5]),early_stopping_rounds=65)
        chosen=int(model.tree_count_);model.save_model(str(ROOT/'validation_v3.cbm'))
    else:model.fit(xf,fit[:,5])
    pred=model.predict_proba(xv)[:,1]
    auc=float(roc_auc_score(val[:,5],pred))
    result={'validation_day':day,'rows':len(val),'v2_auc':old_reports[day]['new_auc'],'v3_auc':auc,'improvement':auc-old_reports[day]['new_auc'],'iterations':chosen}
    reports.append(result);(ROOT/'validation_v3.json').write_text(json.dumps(reports,indent=2));print(result,flush=True)
    np.savez_compressed(ROOT/('predictions_v3_day'+str(day)+'.npz'),y=val[:,5],pred=pred,events=val[:,6])
    del xf,xv,stats,fit,val,model

if not all(r['improvement']>0 for r in reports):
    print('Timing features did not improve both splits. Keep submission_v2.csv.',flush=True)
    sys.exit(0)
print('Timing improvement confirmed. Fitting final V3.',flush=True)
stats=cached_stats(start+6*DAY);fit=np.asarray(train[train[:,2]>=start+6*DAY])
xf=timing_features(fit,hist,times,stats)
model=CatBoostClassifier(iterations=chosen,depth=7,learning_rate=.08,loss_function='Logloss',thread_count=6,random_seed=42,verbose=100,allow_writing_files=False)
model.fit(xf,fit[:,5]);model.save_model(str(ROOT/'model_v3.cbm'))
del xf,fit
xe=timing_features(evaluation,ehist,etimes,stats);prob=model.predict_proba(xe)[:,1]
assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
source=EVALUATION_CSV;seen=set()
with source.open(newline='',encoding='utf-8') as src,(ROOT/'submission_v3.csv').open('w',newline='',encoding='utf-8') as dest:
    writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
    for row,p in zip(csv.DictReader(src),prob,strict=True):
        assert row['rowid'] not in seen
        seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')])
assert len(seen)==529390
print('Verified V3 submission',len(seen),'rows',flush=True)
