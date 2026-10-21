"""Geometric/log-odds ensembles, selected on day 6 and frozen on day 5.

Recorded intervals are approximate paired event-cluster intervals, conditional on
chosen settings; they do not account for repeated validation-set selection.
"""
import csv,json,sys
from pathlib import Path
from config import ARTIFACT_DIR as ROOT

import numpy as np
from sklearn.metrics import roc_auc_score

WEIGHTS=np.array([.45,.27,.18,.10])

def logit(p):
    p=np.clip(p,1e-8,1-1e-8)
    return np.log(p)-np.log1p(-p)

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-35,35)))

def competitive(p,strength):
    p=np.clip(p,1e-8,1-1e-8).reshape(-1,10)
    odds=p/(1-p)
    return ((1-strength)*p+strength*odds/(1+odds.sum(axis=1,keepdims=True))).ravel()

def prediction(parts,settings):
    mode,amount,temperature,strength=settings
    arithmetic=WEIGHTS@parts
    if mode=='logit': alternative=sigmoid(WEIGHTS@logit(parts))
    elif mode=='geometric':alternative=np.exp(WEIGHTS@np.log(np.clip(parts,1e-8,1)))
    else:alternative=arithmetic
    p=(1-amount)*arithmetic+amount*alternative
    return competitive(sigmoid(logit(p)/temperature),strength)

def load(day):
    ds=[np.load(ROOT/f'predictions_{name}_day{day}.npz') for name in ['stack','v6','v8','v7']]
    for d in ds[1:]:
        assert np.array_equal(d['events'],ds[0]['events']) and np.array_equal(d['y'],ds[0]['y'])
    ev=ds[0]['events'].reshape(-1,10)
    assert (ev==ev[:,:1]).all()
    s,v6,v8,v7=[d['pred'] for d in ds]
    choice=(v8-.75*v6)/.25
    assert choice.min()>-1e-6 and choice.max()<1+1e-6
    return ds[0]['y'],np.array([s,v6,choice,v7])

def influence(y,p):
    pos=np.sort(p[y==1]);neg=np.sort(p[y==0]);auc=roc_auc_score(y,p)
    out=np.empty(len(y));mask=y==1
    out[mask]=((np.searchsorted(neg,p[mask],'left')+np.searchsorted(neg,p[mask],'right'))/(2*len(neg))-auc)/len(pos)
    out[~mask]=(1-(np.searchsorted(pos,p[~mask],'left')+np.searchsorted(pos,p[~mask],'right'))/(2*len(pos))-auc)/len(neg)
    return out.reshape(-1,10).sum(axis=1)

baseline_settings=('arithmetic',0,1.,.5)
reports=[];settings=None
for day in [6,5]:
    y,parts=load(day);base=prediction(parts,baseline_settings)
    if settings is None:
        trials=[]
        for mode in ['logit','geometric']:
            for amount in [.5,1.]:
                for temp in [.85,1.,1.15]:
                    for strength in [0,.5,1.]:
                        params=(mode,amount,temp,strength)
                        trials.append((float(roc_auc_score(y,prediction(parts,params))),params))
        trials.append((float(roc_auc_score(y,base)),baseline_settings))
        trials.sort(reverse=True);settings=trials[0][1]
        print('Top selection-date candidates:',trials[:5],flush=True)
        (ROOT/'nonlinear_search.json').write_text(json.dumps(trials,indent=2))
    pred=prediction(parts,settings)
    old=float(roc_auc_score(y,base));new=float(roc_auc_score(y,pred));delta=new-old
    diff=influence(y,pred)-influence(y,base)
    se=float(np.sqrt(np.sum(diff**2)))
    report=dict(day=day,baseline_auc=old,candidate_auc=new,improvement=delta,approximate_paired_95pct_interval=[delta-1.96*se,delta+1.96*se],settings=settings)
    reports.append(report);print(report,flush=True)
    np.savez_compressed(ROOT/f'predictions_nonlinear_day{day}.npz',y=y,pred=pred)
(ROOT/'validation_nonlinear.json').write_text(json.dumps(reports,indent=2))
if not all(r['improvement']>0 for r in reports):
    print('Rejected; existing competition submission retained.',flush=True);sys.exit(0)
ids=None;parts=[]
for name in ['stack','v6','v8','v7']:
    with (ROOT/f'submission_{name}.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    rowids=[r['rowid'] for r in rows]
    if ids is None:ids=rowids
    else:assert ids==rowids
    parts.append(np.array([float(r['predicted_probability']) for r in rows]))
parts[2]=(parts[2]-.75*parts[1])/.25
assert len(ids)==529390 and len(set(ids))==len(ids)
pred=prediction(np.array(parts),settings)
assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
with (ROOT/'submission_nonlinear.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['rowid','predicted_probability'])
    w.writerows((i,format(p,'.10g')) for i,p in zip(ids,pred,strict=True))
print('Wrote submission_nonlinear.csv',flush=True)
