"""Test 63 leaves with the unchanged causal 354-feature representation."""
from imp6_warmcontext import *
from imp17_tree import combined as previous_combined, setup as previous_setup
from imp19_als_aligned import interactions
from imp10_leafwise import lgb,SETTINGS as BASE_SETTINGS
import hashlib
SETTINGS=dict(BASE_SETTINGS,num_leaves=63,num_threads=4)
from scipy.stats import rankdata

_base_combined=previous_combined


def combined(name,rows,histories,cut):
    return np.column_stack([_base_combined(name,rows,histories,cut),interactions(name,rows,histories,cut)])


def setup():
    previous_setup();engine.TAG='capacity63';engine.WIDTH=354;engine.aligned_features=combined


def load_baseline():
    path=ROOT/'imp20_baseline.json'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen=json.loads(path.read_text())
    for day in [6,5]:
        member=ROOT/f"{frozen['baseline']}_day{day}.npz"
        assert hashlib.sha256(member.read_bytes()).hexdigest()==frozen['sha256'][str(day)]
    audit_path=ROOT/frozen['audit_filename']
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==frozen['independent_audit_sha256']
    audit=json.loads(audit_path.read_text())
    assert audit['status']=='PASS' and audit['both_dates_improved']
    frozen['_manifest_sha256']=digest
    return frozen


def fit(day,frozen,chosen=None):
    assert hashlib.sha256((ROOT/'imp20_baseline.json').read_bytes()).hexdigest()==frozen['_manifest_sha256']
    setup();x,y,path=engine.assemble_aligned([day-3,day-2,day-1],f'd{day}')
    xv,prior,sw=engine.aligned_validation(day)
    baseline_path=ROOT/f"{frozen['baseline']}_day{day}.npz"
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==frozen['sha256'][str(day)]
    ref=np.load(baseline_path)
    assert np.array_equal(prior['events'],ref['events']) and np.array_equal(prior['y'],ref['y'])
    data=lgb.Dataset(x,label=y,free_raw_data=True)
    validation=lgb.Dataset(xv,label=ref['y'],reference=data,free_raw_data=True)
    callbacks=[lgb.log_evaluation(100)]
    if chosen is None:callbacks.append(lgb.early_stopping(120,first_metric_only=True))
    model=lgb.train(SETTINGS,data,num_boost_round=1600 if chosen is None else chosen['trees'],
                    valid_sets=[validation],valid_names=['fixed_validation'],callbacks=callbacks)
    trees=int(model.best_iteration or model.current_iteration())
    model.save_model(str(ROOT/f'imp20_tree_d{day}.txt'),num_iteration=trees)
    raw=model.predict(xv,num_iteration=trees)
    assert np.isfinite(raw).all() and ((raw>0)&(raw<1)).all()
    np.savez_compressed(ROOT/f'imp20_tree_raw_day{day}.npz',y=ref['y'],events=ref['events'],raw=raw)
    y=ref['y'];base=ref['pred'];old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    br=rankdata(base,method='average')/(len(base)+1);rr=rankdata(raw,method='average')/(len(raw)+1)
    best=None;trials=[]
    for weight in ([chosen['weight']] if chosen else [.1,.25,.5,.75,1.]):
        pred=(1-weight)*br+weight*rr
        auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
        result=dict(day=day,trees=trees,weight=weight,auc=auc,reweighted=weighted,baseline_auc=old,baseline_reweighted=oldw,raw_auc=float(roc_auc_score(y,raw)),baseline=frozen['baseline'],baseline_manifest_sha256=frozen['_manifest_sha256'])
        trials.append(result);print(json.dumps(result),flush=True)
        if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
            best=result.copy();se=np.linalg.norm(influence(y,pred)-influence(y,base))
            sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
            best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
            np.savez_compressed(ROOT/f'imp20_tree_day{day}.npz',y=y,events=ref['events'],pred=pred,raw=raw)
    (ROOT/f'imp20_tree_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    del model,data,validation,x,xv;gc.collect();path.unlink()
    return best


if __name__=='__main__':
    output=ROOT/'imp20_tree_report.json';assert not output.exists(),'Do not restart a completed experiment.'
    frozen=load_baseline()
    selected=fit(6,frozen);reports=[selected];output.write_text(json.dumps(reports,indent=2))
    if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
        reports.append(fit(5,frozen,selected));output.write_text(json.dumps(reports,indent=2))
    print('Larger tree capacity experiment complete; independent verification required.',flush=True)
