"""64-dimensional choice model with stronger regularization, compared with nonlinear blend."""
from choice64_core import *
reports=[];chosen_epoch=None;chosen_weight=None
for day in [6,5]:
    a=np.asarray(train[train[:,2]<start+day*DAY]);users,items,history,target=event_data(a,hist);del a
    _,val=split(day);vu,vi,vh,_=event_data(val,hist)
    baseline=np.load(ROOT/f'predictions_nonlinear_day{day}.npz')
    assert np.array_equal(baseline['y'],val[:,5])
    old=baseline['pred'];old_auc=float(roc_auc_score(val[:,5],old))
    params,accum=initial(start+(day-2)*DAY)
    rng=np.random.default_rng(91);best=old_auc;best_params=None
    for epoch in range(1,(6 if day==6 else chosen_epoch)+1):
        begin=time.time()
        loss=train_epoch(rng.permutation(len(users)),users,items,history,target,*params,*accum,.03,.02)
        raw=predict_events(vu,vi,vh,*params).ravel()
        if day==6:
            for w in [.05,.1,.2,.3]:
                auc=float(roc_auc_score(val[:,5],(1-w)*old+w*raw))
                if auc>best:
                    best=auc;chosen_epoch=epoch;chosen_weight=w;best_params=[p.copy() for p in params]
        print('day',day,'epoch',epoch,'raw AUC',float(roc_auc_score(val[:,5],raw)),'best blend',best,'seconds',round(time.time()-begin,1),flush=True)
    if day==6:
        if best_params is None:
            (ROOT/'validation_choice64.json').write_text(json.dumps({'accepted':False,'reason':'No selection-date gain','baseline_auc':old_auc},indent=2))
            print('No improvement; retained nonlinear blend.',flush=True);sys.exit(0)
        params=best_params
    raw=predict_events(vu,vi,vh,*params).ravel();pred=(1-chosen_weight)*old+chosen_weight*raw
    auc=float(roc_auc_score(val[:,5],pred))
    reports.append(dict(day=day,baseline_auc=old_auc,candidate_auc=auc,improvement=auc-old_auc,epochs=chosen_epoch,weight=chosen_weight))
    print(reports[-1],flush=True)
    np.savez_compressed(ROOT/f'predictions_choice64_day{day}.npz',y=val[:,5],events=val[:,6],pred=pred)
    (ROOT/'validation_choice64.json').write_text(json.dumps(reports,indent=2))
    del users,items,history,target,val,vu,vi,vh,params,accum,baseline,old,best_params;gc.collect()
if not all(r['improvement']>0 for r in reports):
    print('Rejected on confirmation date; retained nonlinear blend.',flush=True);sys.exit(0)
params,accum=initial(start+6*DAY)
users,items,history,target=event_data(train,hist);rng=np.random.default_rng(91)
for epoch in range(1,chosen_epoch+1):
    loss=train_epoch(rng.permutation(len(users)),users,items,history,target,*params,*accum,.03,.02)
    print('Final epoch',epoch,'loss',loss,flush=True)
np.savez_compressed(ROOT/'model_choice64.npz',U=params[0],V=params[1],H=params[2],bu=params[3],bi=params[4])
raw=predict_events(evaluation[::10,0],evaluation[:,1].reshape(-1,10),ehist,*params).ravel()
with (ROOT/'submission_nonlinear.csv').open(newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==len(raw)==529390
with (ROOT/'submission_choice64.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['rowid','predicted_probability'])
    for row,p in zip(rows,raw,strict=True):
        value=(1-chosen_weight)*float(row['predicted_probability'])+chosen_weight*p
        assert np.isfinite(value) and 0<=value<=1
        w.writerow([row['rowid'],format(value,'.10g')])
print('Wrote submission_choice64.csv',flush=True)
