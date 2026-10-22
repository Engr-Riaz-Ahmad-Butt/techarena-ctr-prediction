"""Match training event selection to the fixed first-event validation cohort."""
from imp6_warmcontext import *


def first_indices(users):
    _,first=np.unique(users,return_index=True)
    return (np.sort(first)[:,None]*10+np.arange(10)).ravel()


def run():
    output=ROOT/'imp9_firstevents_report.json'
    assert not output.exists(),'Do not restart a completed experiment.'
    configure();engine.TAG='firstevents'
    reports=[];selected=None
    for day in [6,5]:
        days=[day-3,day-2,day-1]
        x,y,path=engine.assemble_aligned(days,f'd{day}')
        selected_rows=[];offset=0
        for d in days:
            a=np.asarray(train[(train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1))])
            indices=first_indices(a[::10,0])
            assert len(np.unique(a[indices[::10],0]))==len(indices)//10
            assert np.all(a[indices,6].reshape(-1,10)==a[indices[::10],6,None])
            selected_rows.append(offset+indices);offset+=len(a)
        selected_rows=np.concatenate(selected_rows)
        xf=np.asarray(x[selected_rows]);yf=y[selected_rows]
        np.save(ROOT/f'imp9_training_indices_day{day}.npy',selected_rows)
        print('First-event training rows',day,len(xf),'original',len(x),flush=True)
        del x,y;gc.collect();path.unlink()
        xv,_,sw=engine.aligned_validation(day)
        ref=np.load(ROOT/f'predictions_imp5_warmcontext_day{day}.npz')
        model=CatBoostClassifier(iterations=599,**engine.PARAMS,verbose=100)
        model.fit(xf,yf,eval_set=(xv,ref['y']),use_best_model=False)
        raw=model.predict_proba(xv)[:,1]
        model.save_model(str(ROOT/f'imp9_firstevents_d{day}.cbm'))
        old_component=ref['raw']
        def predict(mix):
            return candidate_v19(day,(1-mix)*old_component+mix*raw,.5)
        baseline=predict(0.)
        assert np.max(np.abs(baseline-ref['pred']))<1e-12
        y=ref['y'];old=float(roc_auc_score(y,baseline));oldw=float(roc_auc_score(y,baseline,sample_weight=sw))
        if selected is None:
            candidates=[]
            for mix in [.5,1.]:
                p=predict(mix);auc=float(roc_auc_score(y,p));weighted=float(roc_auc_score(y,p,sample_weight=sw))
                candidates.append(dict(mix=mix,auc=auc,reweighted=weighted))
            (ROOT/'imp9_firstevents_trials.json').write_text(json.dumps(candidates,indent=2))
            eligible=[c for c in candidates if c['auc']>old and c['reweighted']>oldw]
            if not eligible:
                output.write_text(json.dumps([dict(day=day,status='rejected_no_gain',baseline_auc=old,trials=candidates)],indent=2));return
            selected=max(eligible,key=lambda c:c['auc'])['mix']
        p=predict(selected);auc=float(roc_auc_score(y,p));weighted=float(roc_auc_score(y,p,sample_weight=sw))
        se=np.linalg.norm(influence(y,p)-influence(y,baseline));sew=np.linalg.norm(weighted_influence(y,p,sw)-weighted_influence(y,baseline,sw))
        report=dict(day=day,mix=selected,auc=auc,reweighted=weighted,baseline_auc=old,baseline_reweighted=oldw,delta=auc-old,delta_reweighted=weighted-oldw,
                    paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
        reports.append(report);output.write_text(json.dumps(reports,indent=2));print(json.dumps(report),flush=True)
        np.savez_compressed(ROOT/f'imp9_firstevents_day{day}.npz',y=y,events=ref['events'],pred=p,raw=raw)
        del xf,yf,xv,model;gc.collect()
        if auc<=old or weighted<=oldw or (day==6 and (report['paired_interval'][0]<=0 or report['weighted_paired_interval'][0]<=0)):return


if __name__=='__main__':run()
