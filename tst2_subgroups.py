"""Describe validation errors and evaluation support without fitting models."""
import sys,json
from pathlib import Path
from config import ARTIFACT_DIR as ROOT
import numpy as np
from sklearn.metrics import roc_auc_score
a=np.load(ROOT/'train.npy',mmap_mode='r');h=np.load(ROOT/'train_hist.npy',mmap_mode='r')
e=np.load(ROOT/'evaluation.npy',mmap_mode='r');eh=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
start=int(a[:,2].min());result={}
def describe(rows,history,prior,models=None,y=None):
    lengths=(history[rows[:,6]]>=0).sum(axis=1)
    uc=np.bincount(prior[:,0],minlength=int(max(a[:,0].max(),e[:,0].max()))+1)
    ic=np.bincount(prior[:,1],minlength=int(max(a[:,1].max(),e[:,1].max()))+1)
    groups={'history_length':(lengths,[-1,0,4,9,19,20]),'user_prior_impressions':(uc[rows[:,0]],[-1,0,10,50,100,250,500,10000]),'item_prior_impressions':(ic[rows[:,1]],[-1,0,50,100,150,200,300,100000])}
    out={}
    for key,(values,edges) in groups.items():
        out[key]=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            keep=(values>lo)&(values<=hi);n=int(keep.sum())
            r={'range':[lo+1,hi],'rows':n,'share':n/len(rows)}
            if y is not None and n and len(np.unique(y[keep]))==2:
                r['click_rate']=float(y[keep].mean());r['auc']={name:float(roc_auc_score(y[keep],p[keep])) for name,p in models.items()}
            out[key].append(r)
    return out
for day in [6,5]:
    d=np.load(ROOT/f'predictions_v11_day{day}.npz');events=d['events']
    ix=events*10+np.tile(np.arange(10),len(events)//10);rows=a[ix]
    models={'v11':d['pred']}
    candidate=ROOT/f'imp2_warm0.15_day{day}.npz'
    if candidate.exists():
        warm=np.load(candidate);assert np.array_equal(warm['events'],events);models['warm']=warm['pred']
    prior=a[a[:,2]<start+day*86400]
    result['day'+str(day)]=describe(rows,h,prior,models,d['y'])
result['evaluation']=describe(e,eh,a)
(ROOT/'tst2_subgroups.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
