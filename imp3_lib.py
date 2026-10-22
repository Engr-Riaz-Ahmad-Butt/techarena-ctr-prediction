"""Shared helpers for the imp3_ improvement pass (protocol_v2.md).

Everything here is label-safe: the history-length reweighting uses only unlabeled
evaluation histories, and all blend selection happens on day 6 only.
"""
import sys,json,csv,time,gc
from pathlib import Path
from config import ARTIFACT_DIR as ROOT

import numpy as np
from sklearn.metrics import roc_auc_score

PY=sys.executable
EDGES=[0,4,9,19]
V14={6:ROOT/'imp2_warm1.0_d16_pc1_day6.npz',5:ROOT/'imp2_warm1.0_d16_pc1_day5.npz'}
# Explicit baseline selection avoids a dependency on a not-yet-built later version.
import os
BASE_NAME=os.environ.get('CTR_BASELINE_VERSION','V17').upper()
BASE_PREFIXES={'V14':'imp2_warm1.0_d16_pc1_day',
               'V15':'predictions_imp3_reblend_day',
               'V16':'predictions_imp3_wf_day',
               'V17':'predictions_imp3_wf3_day'}
if BASE_NAME not in BASE_PREFIXES:raise ValueError('CTR_BASELINE_VERSION must be V14, V15, V16 or V17')
BASE={day:ROOT/f'{BASE_PREFIXES[BASE_NAME]}{day}.npz' for day in [6,5]}

def logit(p):
    p=np.clip(p,1e-8,1-1e-8);return np.log(p)-np.log1p(-p)

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-35,35)))

def competitive(p,strength=.5):
    """50% event-competition adjustment from nonlinear_experiment.py (rows grouped by 10)."""
    p=np.clip(p,1e-8,1-1e-8).reshape(-1,10);odds=p/(1-p)
    return ((1-strength)*p+strength*odds/(1+odds.sum(axis=1,keepdims=True))).ravel()

def history_weights(events):
    """evaluation share / validation share of the history-length bin, per validation row."""
    h=np.load(ROOT/'train_hist.npy',mmap_mode='r');eh=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
    eg=np.digitize((eh>=0).sum(1),EDGES,right=True);esh=np.bincount(eg,minlength=5)/len(eg)
    vg=np.digitize((np.asarray(h[events])>=0).sum(1),EDGES,right=True);vsh=np.bincount(vg,minlength=5)/len(vg)
    return (esh/vsh)[vg],vg

def history_bins_eval():
    eh=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
    return np.digitize((eh>=0).sum(1),EDGES,right=True)

def load_components(day,extra=()):
    """Return y, events, dict of member predictions (probability scale), V14 pred."""
    names=['stack','v6','v8','v7']
    ds={n:np.load(ROOT/f'predictions_{n}_day{day}.npz') for n in names}
    ref=ds['stack'];ev=ref['events'];y=ref['y']
    for d in ds.values():assert np.array_equal(d['events'],ev) and np.array_equal(d['y'],y)
    w=np.load(V14[day]);assert np.array_equal(w['events'],ev) and np.array_equal(w['y'],y)
    parts={'stack':ds['stack']['pred'],'v6':ds['v6']['pred'],'v7':ds['v7']['pred'],
           'choice':np.clip((ds['v8']['pred']-.75*ds['v6']['pred'])/.25,0,1),'warm16':w['raw']}
    for n in extra:
        d=np.load(ROOT/f'predictions_{n}_day{day}.npz');assert np.array_equal(d['events'],ev) and np.array_equal(d['y'],y)
        parts[n]=d['pred']
    assert (ev.reshape(-1,10)==ev.reshape(-1,10)[:,:1]).all()
    b=np.load(BASE[day]);assert np.array_equal(b['events'],ev) and np.array_equal(b['y'],y)
    return y,ev,parts,b['pred']

def scores(y,pred,sw):
    return float(roc_auc_score(y,pred)),float(roc_auc_score(y,pred,sample_weight=sw))

def bootstrap_delta(y,new,old,sw,n=1000,seed=3):
    """Paired event-cluster bootstrap of the reweighted AUC delta (new - old)."""
    rng=np.random.default_rng(seed);ne=len(y)//10;out=np.empty(n)
    for b in range(n):
        c=np.bincount(rng.integers(0,ne,ne),minlength=ne).astype(float)
        w=sw*np.repeat(c,10);m=w>0
        out[b]=roc_auc_score(y[m],new[m],sample_weight=w[m])-roc_auc_score(y[m],old[m],sample_weight=w[m])
    return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))],float(out.mean())

def accept(report):
    """Protocol rules 1-2 given a report with day6/day5 entries (and the bootstrap interval)."""
    r6,r5=report['day6'],report['day5']
    ok=(r6['reweighted']>r6['base_reweighted'] and r5['reweighted']>r5['base_reweighted']
        and r6['auc']>=r6['base_auc'] and r5['auc']>=r5['base_auc'] and report['bootstrap_95_day6'][0]>0)
    return bool(ok)

def save_predictions(name,day,y,pred,events,raw=None):
    kw=dict(y=y,pred=pred,events=events)
    if raw is not None:kw['raw']=raw
    np.savez_compressed(ROOT/f'predictions_imp3_{name}_day{day}.npz',**kw)

def read_submission(name):
    path=ROOT/(name if name.endswith('.csv') else f'submission_{name}.csv')
    ids=[];p=[]
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):ids.append(r['rowid']);p.append(float(r['predicted_probability']))
    return ids,np.array(p)

def next_version():
    import re;used={int(m.group(1)) for f in ROOT.glob('submission_v*.csv') for m in [re.match(r'submission_v(\d+)(_candidate)?\.csv$',f.name)] if m}
    return max(used)+1

def write_submission(name,ids,pred,versioned=False):
    """versioned=True writes submission_v<N>_candidate.csv (user naming rule); otherwise a component file."""
    assert len(ids)==529390==len(pred) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    path=ROOT/(f'submission_v{next_version()}_candidate.csv' if versioned else f'submission_imp3_{name}.csv')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['rowid','predicted_probability'])
        w.writerows((i,format(float(p),'.10g')) for i,p in zip(ids,pred,strict=True))
    return path

def queue_line(name,r,desc,script):
    line=f"{name} | {r['day6']['auc']:.6f} | {r['day6']['reweighted']:.6f} | {r['day5']['auc']:.6f} | {r['day5']['reweighted']:.6f} | {desc} | {script}\n"
    q=ROOT/'candidates3_queue.md'
    if not q.exists():q.write_text('name | day6 auc | day6 reweighted | day5 auc | day5 reweighted | description | script\n')
    with q.open('a') as f:f.write(line)

def stop_requested():return (ROOT/'STOP3').exists()
