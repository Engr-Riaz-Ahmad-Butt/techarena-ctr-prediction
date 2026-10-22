"""Fit the existing leaf-wise learner with 32 causal exposure-ALS features."""
from imp6_warmcontext import *
from imp17_als_features import als_features
from imp10_leafwise import lgb,SETTINGS
from scipy.stats import rankdata

_base_combined=combined_features


def combined(name,rows,histories,cut):
    return np.column_stack([_base_combined(name,rows,histories,cut),als_features(name,rows,histories,cut)])


def setup():
    configure();engine.TAG='alsfeatures';engine.WIDTH=322;engine.aligned_features=combined


def fit(day,chosen=None):
    setup();x,y,path=engine.assemble_aligned([day-3,day-2,day-1],f'd{day}')
    xv,prior,sw=engine.aligned_validation(day)
    ref=np.load(ROOT/f'imp15_als_day{day}.npz')
    assert np.array_equal(prior['events'],ref['events']) and np.array_equal(prior['y'],ref['y'])
    data=lgb.Dataset(x,label=y,free_raw_data=True)
    validation=lgb.Dataset(xv,label=ref['y'],reference=data,free_raw_data=True)
    callbacks=[lgb.log_evaluation(100)]
    if chosen is None:callbacks.append(lgb.early_stopping(120,first_metric_only=True))
    model=lgb.train(SETTINGS,data,num_boost_round=1600 if chosen is None else chosen['trees'],
                    valid_sets=[validation],valid_names=['fixed_validation'],callbacks=callbacks)
    trees=int(model.best_iteration or model.current_iteration())
    model.save_model(str(ROOT/f'imp17_tree_d{day}.txt'),num_iteration=trees)
    raw=model.predict(xv,num_iteration=trees)
    assert np.isfinite(raw).all() and ((raw>0)&(raw<1)).all()
    np.savez_compressed(ROOT/f'imp17_tree_raw_day{day}.npz',y=ref['y'],events=ref['events'],raw=raw)
    y=ref['y'];base=ref['pred'];old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    br=rankdata(base,method='average')/(len(base)+1);rr=rankdata(raw,method='average')/(len(raw)+1)
    best=None;trials=[]
    for weight in ([chosen['weight']] if chosen else [.1,.25,.5,.75,1.]):
        pred=(1-weight)*br+weight*rr
        auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
        result=dict(day=day,trees=trees,weight=weight,auc=auc,reweighted=weighted,baseline_auc=old,baseline_reweighted=oldw,raw_auc=float(roc_auc_score(y,raw)))
        trials.append(result);print(json.dumps(result),flush=True)
        if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
            best=result.copy();se=np.linalg.norm(influence(y,pred)-influence(y,base))
            sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
            best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
            np.savez_compressed(ROOT/f'imp17_tree_day{day}.npz',y=y,events=ref['events'],pred=pred,raw=raw)
    (ROOT/f'imp17_tree_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    del model,data,validation,x,xv;gc.collect();path.unlink()
    return best


if __name__=='__main__':
    output=ROOT/'imp17_tree_report.json';assert not output.exists(),'Do not restart a completed experiment.'
    selected=fit(6);reports=[selected];output.write_text(json.dumps(reports,indent=2))
    if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
        reports.append(fit(5,selected));output.write_text(json.dumps(reports,indent=2))
    print('ALS feature learner complete; independent verification required.',flush=True)
