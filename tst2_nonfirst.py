"""Frozen, independent non-first-event backtest; run only after model selection.

Jan-26 first events were used for early stopping. All other Jan-26 events are
excluded from that selection and from pre-Jan-26 parameter fitting. Their
history remains observable at prediction time. V6 is the baseline here, not
the full V11 ensemble. This diagnostic does not match official first events.
"""
from choice64_core import *
import argparse

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--max-events',type=int,default=10000)
    ap.add_argument('--weight',type=float,required=True);args=ap.parse_args()
    checkpoint=Path(args.checkpoint)
    d=np.load(checkpoint);params=[d[k] for k in ['U','V','H','bu','bi']]
    rows=np.asarray(train[(train[:,2]>=start+6*DAY)&(train[:,2]<start+7*DAY)])
    selected=np.load(ROOT/'predictions_v11_day6.npz')['events'][::10]
    mask=~np.isin(rows[::10,6],selected);rows=rows[np.repeat(mask,10)]
    eligible_events=len(rows)//10
    if eligible_events>args.max_events:
        sample=np.linspace(0,eligible_events-1,args.max_events,dtype=int)
        rows=rows[(sample[:,None]*10+np.arange(10)).ravel()]
    assert len(rows) and not np.isin(rows[:,6],selected).any()
    with (ROOT/f'preference_stats_{start+4*DAY}.pkl').open('rb') as f:stats=pickle.load(f)
    model=CatBoostClassifier();model.load_model(str(ROOT/'validation_v6.cbm'))
    reference=np.load(ROOT/'predictions_v6_day6.npz');check_events=reference['events'][:100]
    check_rows=np.asarray(train[check_events*10+np.tile(np.arange(10),10)])
    check_pred=model.predict_proba(preference_features(check_rows,hist,stats))[:,1]
    assert np.max(np.abs(check_pred-reference['pred'][:100]))<1e-9,'V6 checkpoint/statistics provenance mismatch'
    base=[];raw=[]
    for off in range(0,len(rows),10000):
        b=rows[off:off+10000]
        base.append(model.predict_proba(preference_features(b,hist,stats))[:,1])
        u,i,h,_=event_data(b,hist);raw.append(predict_events(u,i,h,*params).ravel())
    base=np.concatenate(base);raw=np.concatenate(raw);y=rows[:,5]
    lengths=(hist[rows[:,6]]>=0).sum(axis=1)
    preds={'v6':base,'warm_raw':raw,'warm_frozen_blend':(1-args.weight)*base+args.weight*raw}
    report={'checkpoint':checkpoint.name,'rows':len(y),'events':len(y)//10,'eligible_events':eligible_events,'weight':args.weight,'sampling':'Deterministic evenly spaced eligible event indices; no labels inspected for sampling.','baseline':'V6 validation checkpoint, selected on Jan26 first events; trained before Jan26','auc':{k:float(roc_auc_score(y,p)) for k,p in preds.items()},'caution':'Independent diagnostic on non-first events; not a match to official event distribution or full V11.'}
    np.savez_compressed(ROOT/'tst2_nonfirst_predictions.npz',events=rows[:,6],y=y,base=base,raw=raw)
    (ROOT/'tst2_nonfirst.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
