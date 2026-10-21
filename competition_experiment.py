"""Test event competition and ridge diversity; select on day 6, confirm on day 5."""
import sys,json,csv
from pathlib import Path
from config import ARTIFACT_DIR as ROOT

import numpy as np
from sklearn.metrics import roc_auc_score

def adjust(p, strength):
    # At most one candidate can click. Shrink independent odds using competing odds.
    p=np.clip(p,1e-9,1-1e-9).reshape(-1,10)
    odds=p/(1-p)
    competitive=odds/(1+odds.sum(axis=1,keepdims=True))
    return ((1-strength)*p+strength*competitive).ravel()

def load(day):
    data=[np.load(ROOT/f'predictions_{v}_day{day}.npz') for v in ['stack','v6','v8','v7']]
    for d in data[1:]:
        assert np.array_equal(d['events'],data[0]['events'])
        assert np.array_equal(d['y'],data[0]['y'])
    s,v6,v8,v7=[d['pred'] for d in data]
    baseline=.5*s+.3*v6+.2*((v8-.75*v6)/.25)
    return data[0]['y'],baseline,v7

reports=[];selected=None
for day in [6,5]:
    y,base,v7=load(day)
    if selected is None:
        candidates=[]
        for weight in [0,.1,.2]:
            for strength in [0,.25,.5,.75,1.]:
                pred=adjust((1-weight)*base+weight*v7,strength)
                candidates.append((float(roc_auc_score(y,pred)),weight,strength))
        best=max(candidates);selected=best[1:]
        print('Selected on day 6:',best,flush=True)
    weight,strength=selected
    pred=adjust((1-weight)*base+weight*v7,strength)
    old=float(roc_auc_score(y,base));new=float(roc_auc_score(y,pred))
    report=dict(day=day,baseline_auc=old,candidate_auc=new,improvement=new-old,ridge_weight=weight,competition_strength=strength)
    reports.append(report);print(report,flush=True)
(ROOT/'validation_competition.json').write_text(json.dumps(reports,indent=2))
if not all(r['improvement']>0 for r in reports):
    print('Rejected. Existing submission_blend.csv remains selected.');sys.exit(0)
with (ROOT/'submission_blend.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
with (ROOT/'submission_v7.csv').open(newline='') as f:
    ridge=list(csv.DictReader(f))
assert len(rows)==len(ridge)==529390
assert all(a['rowid']==b['rowid'] for a,b in zip(rows,ridge))
p=np.array([float(r['predicted_probability']) for r in rows])
r=np.array([float(r['predicted_probability']) for r in ridge])
pred=adjust((1-weight)*p+weight*r,strength)
with (ROOT/'submission_competition.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['rowid','predicted_probability'])
    w.writerows((r['rowid'],format(p,'.10g')) for r,p in zip(rows,pred))
print('Wrote submission_competition.csv',flush=True)
