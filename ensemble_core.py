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
