"""V11: tune a bounded log-odds ensemble grid on day 6, confirm on day 5."""
from ensemble_core import *

def blend(parts,weights):
    return competitive(sigmoid(np.array(weights)@logit(parts)),.5)

reports=[];selected=None
for day in [6,5]:
    y,parts=load(day)
    prior=np.load(ROOT/f'predictions_nonlinear_day{day}.npz')
    assert np.array_equal(y,prior['y']);base=prior['pred']
    if selected is None:
        trials=[(float(roc_auc_score(y,base)),[.45,.27,.18,.1])]
        for s in [.3,.45,.6]:
            for c in [.1,.2,.3]:
                for r in [0,.1,.2]:
                    v=1-s-c-r
                    if v<0:continue
                    w=[s,v,c,r]
                    trials.append((float(roc_auc_score(y,blend(parts,w))),w))
        trials.sort(reverse=True);selected=trials[0][1]
        print('Selected grid:',trials[:5],flush=True)
        (ROOT/'v11_weights_search.json').write_text(json.dumps(trials,indent=2))
    pred=blend(parts,selected)
    old=float(roc_auc_score(y,base));auc=float(roc_auc_score(y,pred));delta=auc-old
    se=float(np.sqrt(np.sum((influence(y,pred)-influence(y,base))**2)))
    report=dict(day=day,v10_auc=old,v11_auc=auc,improvement=delta,weights=selected,approximate_paired_95pct_interval=[delta-1.96*se,delta+1.96*se])
    reports.append(report);print(report,flush=True)
    reference=np.load(ROOT/f'predictions_v6_day{day}.npz')
    assert np.array_equal(reference['y'],y)
    np.savez_compressed(ROOT/f'predictions_weighted_day{day}.npz',
                        y=y,pred=pred,events=reference['events'])
(ROOT/'validation_v11.json').write_text(json.dumps(reports,indent=2))
if not all(r['improvement']>0 for r in reports):
    print('Rejected; V10 remains selected.',flush=True);sys.exit(0)
for day in [6,5]:
    verified=np.load(ROOT/f'predictions_weighted_day{day}.npz')
    np.savez_compressed(ROOT/f'predictions_v11_day{day}.npz',
                        y=verified['y'],pred=verified['pred'],events=verified['events'])
ids=None;parts=[]
for name in ['stack','v6','v8','v7']:
    with (ROOT/f'submission_{name}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    rid=[r['rowid'] for r in rows]
    if ids is None:ids=rid
    else:assert rid==ids
    parts.append(np.array([float(r['predicted_probability']) for r in rows]))
parts[2]=(parts[2]-.75*parts[1])/.25
pred=blend(np.array(parts),selected)
assert len(ids)==529390 and len(set(ids))==len(ids) and np.isfinite(pred).all()
with (ROOT/'submission_v11.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['rowid','predicted_probability'])
    w.writerows((i,format(p,'.10g')) for i,p in zip(ids,pred,strict=True))
print('Wrote submission_v11.csv',flush=True)
