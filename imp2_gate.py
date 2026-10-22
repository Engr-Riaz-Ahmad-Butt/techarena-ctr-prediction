"""Predeclared sparse-history gate for auxiliary warm-click component."""
from imp2_warm_choice import *

reports=[]
for day in [6,5]:
    _,val=split(day)
    h=np.asarray(hist[val[:,6]])
    gate=(h>=0).sum(axis=1)<=4
    path=ROOT/f'imp2_warm0.15_d32_pc0_day{day}.npz'
    if not path.exists():path=ROOT/f'imp2_warm0.15_day{day}.npz'
    d=np.load(path)
    old=np.load(ROOT/f'predictions_weighted_day{day}.npz')
    assert np.array_equal(d['events'],val[:,6]) and np.array_equal(d['y'],val[:,5])
    pred=old['pred'].copy();pred[gate]=.95*pred[gate]+.05*d['raw'][gate]
    eg=np.digitize((ehist>=0).sum(axis=1),[0,4,9,19],right=True)
    vg=np.digitize((h>=0).sum(axis=1),[0,4,9,19],right=True)
    ratios=(np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg))
    weights=ratios[vg]
    np.savez_compressed(ROOT/f'imp2_gated_day{day}.npz',events=val[:,6],y=val[:,5],pred=pred)
    r=dict(day=day,gate='history_length<=4',weight=.05,rows_gated=int(gate.sum()),
        baseline_auc=float(roc_auc_score(val[:,5],old['pred'])),auc=float(roc_auc_score(val[:,5],pred)),
        baseline_history_weighted_auc=float(roc_auc_score(val[:,5],old['pred'],sample_weight=weights)),
        history_weighted_auc=float(roc_auc_score(val[:,5],pred,sample_weight=weights)))
    print(r,flush=True);reports.append(r)
(ROOT/'imp2_gated_report.json').write_text(json.dumps(reports,indent=2))
