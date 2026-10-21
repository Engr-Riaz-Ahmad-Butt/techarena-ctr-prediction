"""Stacked-day training: V6 features, but the classifier learns from three days instead of one.

Every block of rows (day d) gets its features from statistics built strictly before day d,
so no block ever sees its own labels. Validation day 6 fits on days 3-5, validation day 5
on days 2-4, and the final model on days 4-6 with evaluation features cut off at day 7.
The final V6 model fit on 979,510 rows; this fits on about three times as many.
"""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v6 import *
from threadpoolctl import threadpool_limits
threadpool_limits(limits=12)

FEAT=ROOT/'features';FEAT.mkdir(exist_ok=True)
THREADS=12
PARAMS=dict(depth=7,learning_rate=.08,loss_function='Logloss',thread_count=THREADS,random_seed=42,allow_writing_files=False)

def cutoff_of(day):return start+day*DAY

def ensure_base(cutoff):
    path=ROOT/('stats_'+str(cutoff)+'.pkl')
    if not path.exists():
        base=extra_statistics(cutoff)
        with path.open('wb') as f:pickle.dump(base,f,protocol=5)
        del base;gc.collect()

def stats_for(day):
    cutoff=cutoff_of(day);cache=ROOT/('preference_stats_'+str(cutoff)+'.pkl')
    if cache.exists():
        with cache.open('rb') as f:return pickle.load(f)
    ensure_base(cutoff);s=preference_stats(cutoff)
    with cache.open('wb') as f:pickle.dump(s,f,protocol=5)
    return s

def day_rows(day):
    return np.asarray(train[(train[:,2]>=cutoff_of(day))&(train[:,2]<cutoff_of(day+1))])

def first_events(rows):
    _,first=np.unique(rows[::10,0],return_index=True)
    return rows[(first[:,None]*10+np.arange(10)).ravel()]

def block(name,rows,histories,day):
    """Features for rows, built from statistics cut off at the start of `day`; cached on disk."""
    path=FEAT/('stack_'+name+'_c'+str(day)+'.npy')
    if path.exists():return np.load(path)
    t=time.time();s=stats_for(day);x=preference_features(rows,histories,s)
    np.save(path,x);del s;gc.collect()
    print('Built',path.name,x.shape,'in',round((time.time()-t)/60,1),'min',flush=True)
    return x

def fit_blocks(days):
    xs=[];ys=[]
    for d in days:
        rows=day_rows(d);xs.append(block('day'+str(d),rows,hist,d));ys.append(rows[:,5].copy());del rows
    return np.concatenate(xs),np.concatenate(ys)

if __name__=='__main__':
    t0=time.time()
    v6={r['validation_day']:r['v6_auc'] for r in json.loads((ROOT/'validation_v6.json').read_text())}
    reports=[];chosen=0
    for day in [6,5]:
        print('STACK VALIDATION DAY',day,flush=True)
        xf,yf=fit_blocks([day-3,day-2,day-1])
        val=first_events(day_rows(day));xv=block('val'+str(day),val,hist,day);yv=val[:,5]
        previous=np.load(ROOT/('predictions_v6_day'+str(day)+'.npz'))
        assert np.array_equal(previous['events'],val[:,6]) and np.array_equal(previous['y'],yv)
        print('Fit rows',len(yf),'validation rows',len(yv),'features',xf.shape[1],'elapsed min',round((time.time()-t0)/60,1),flush=True)
        if day==6:
            model=CatBoostClassifier(iterations=2500,eval_metric='AUC',verbose=100,**PARAMS)
            model.fit(xf,yf,eval_set=(xv,yv),early_stopping_rounds=100);chosen=int(model.tree_count_)
        else:
            model=CatBoostClassifier(iterations=chosen,verbose=500,**PARAMS);model.fit(xf,yf)
        pred=model.predict_proba(xv)[:,1];auc=float(roc_auc_score(yv,pred))
        blend=float(roc_auc_score(yv,.5*pred+.5*previous['pred']))
        report={'validation_day':day,'fit_rows':int(len(yf)),'rows':int(len(yv)),'v6_auc':v6[day],'stack_auc':auc,
                'improvement':auc-v6[day],'half_blend_with_v6_auc':blend,'iterations':chosen}
        reports.append(report);(ROOT/'validation_stack.json').write_text(json.dumps(reports,indent=2));print(report,flush=True)
        np.savez_compressed(ROOT/('predictions_stack_day'+str(day)+'.npz'),y=yv,pred=pred,events=val[:,6])
        del xf,yf,xv,yv,val,model,previous,pred;gc.collect()

    if not all(r['improvement']>0 for r in reports):
        print('Stacked training did not improve both splits. Keep V6/V8.',flush=True);sys.exit(0)
    print('Both comparisons passed. Fitting final stacked model on days 4-6.',flush=True)
    xf,yf=fit_blocks([4,5,6])
    xe=block('eval',np.asarray(evaluation),ehist,7)
    model=CatBoostClassifier(iterations=chosen,verbose=500,**PARAMS);model.fit(xf,yf)
    model.save_model(str(ROOT/'model_stack.cbm'));del xf,yf
    prob=model.predict_proba(xe)[:,1]
    assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    source=EVALUATION_CSV;seen=set()
    with source.open(newline='',encoding='utf-8') as srcfile,(ROOT/'submission_stack.csv').open('w',newline='',encoding='utf-8') as dest:
        writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
        for row,p in zip(csv.DictReader(srcfile),prob,strict=True):
            assert row['rowid'] not in seen
            seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')])
    assert len(seen)==529390
    print('Verified stacked submission',len(seen),'rows, minutes',round((time.time()-t0)/60,1),flush=True)
