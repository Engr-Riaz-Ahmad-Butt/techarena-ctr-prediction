"""V11: separate within-event ranking sharpness from event click propensity.
Select on January 26; apply fixed settings on January 25. No training/evaluation labels are mixed.
"""
import csv,json,sys
from pathlib import Path
from config import ARTIFACT_DIR as ROOT

import numpy as np
from sklearn.metrics import roc_auc_score

def transform(p,sharpness,mass_power):
    p=np.clip(p,1e-10,1-1e-10).reshape(-1,10)
    mass=p.sum(axis=1,keepdims=True)
    conditional=p**sharpness
    out=conditional/conditional.sum(axis=1,keepdims=True)*mass**mass_power
    return np.clip(out,0,1).ravel()

reports=[];settings=None
for day in [6,5]:
    data=np.load(ROOT/f'predictions_nonlinear_day{day}.npz')
    aligned=np.load(ROOT/f'predictions_v6_day{day}.npz')
    assert np.array_equal(data['y'],aligned['y'])
    ev=aligned['events'].reshape(-1,10);assert (ev==ev[:,:1]).all()
    y=data['y'];base=data['pred']
    if settings is None:
        trials=[]
        for sharp in [.7,1.,1.3]:
            for mass in [.5,.75,1.,1.25,1.5]:
                pred=transform(base,sharp,mass)
                trials.append((float(roc_auc_score(y,pred)),sharp,mass))
        trials.sort(reverse=True);settings=trials[0][1:]
        print('Top candidates:',trials[:5],flush=True)
        (ROOT/'v11_search.json').write_text(json.dumps(trials,indent=2))
    pred=transform(base,*settings)
    old=float(roc_auc_score(y,base));auc=float(roc_auc_score(y,pred))
    report=dict(day=day,v10_auc=old,v11_auc=auc,improvement=auc-old,sharpness=settings[0],mass_power=settings[1])
    reports.append(report);print(report,flush=True)
    np.savez_compressed(ROOT/f'predictions_event_propensity_day{day}.npz',y=y,pred=pred,events=aligned['events'])
(ROOT/'validation_event_propensity.json').write_text(json.dumps(reports,indent=2))
if not all(r['improvement']>0 for r in reports):
    print('Rejected; V10 remains selected.',flush=True);sys.exit(0)
with (ROOT/'submission_v10.csv').open(newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==529390
base=np.array([float(r['predicted_probability']) for r in rows])
pred=transform(base,*settings)
assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
with (ROOT/'submission_v11.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['rowid','predicted_probability'])
    w.writerows((r['rowid'],format(p,'.10g')) for r,p in zip(rows,pred,strict=True))
print('Wrote submission_v11.csv',flush=True)
