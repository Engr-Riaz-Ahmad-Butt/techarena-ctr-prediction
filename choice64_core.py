"""V8: directly trained user/history/item factor model with no-click softmax."""
import sys
from pathlib import Path
from config import ARTIFACT_DIR as ROOT

import numpy as np
from numba import njit
from core_v6 import *
import math

@njit
def predict_events(users,items,history,U,V,H,bu,bi):
    out=np.empty((len(users),10),dtype=np.float64)
    d=U.shape[1]
    for e in range(len(users)):
        u=users[e];z=U[u].copy();n=0
        for q in range(20):
            if history[e,q]>=0:n+=1
        scale=1/math.sqrt(max(n,1))
        for q in range(n):
            h=history[e,q]
            for k in range(d):z[k]+=scale*H[h,k]
        logits=np.empty(10);maximum=0.
        for j in range(10):
            item=items[e,j];value=-1.8+bu[u]+bi[item]
            for k in range(d):value+=z[k]*V[item,k]
            logits[j]=max(-30.,min(30.,value));maximum=max(maximum,logits[j])
        denom=math.exp(-maximum)
        for j in range(10):out[e,j]=math.exp(logits[j]-maximum);denom+=out[e,j]
        for j in range(10):out[e,j]/=denom
    return out

@njit
def train_epoch(order,users,items,history,target,U,V,H,bu,bi,au,av,ah,abu,abi,lr,reg):
    d=U.shape[1];total=0.
    for e in order:
        u=users[e];z=U[u].copy();n=0
        for q in range(20):
            if history[e,q]>=0:n+=1
        scale=1/math.sqrt(max(n,1))
        for q in range(n):
            h=history[e,q]
            for k in range(d):z[k]+=scale*H[h,k]
        logits=np.empty(10);maximum=0.
        for j in range(10):
            item=items[e,j];value=-1.8+bu[u]+bi[item]
            for k in range(d):value+=z[k]*V[item,k]
            logits[j]=max(-30.,min(30.,value));maximum=max(maximum,logits[j])
        p=np.empty(10);denom=math.exp(-maximum)
        for j in range(10):p[j]=math.exp(logits[j]-maximum);denom+=p[j]
        total+=math.log(denom)+maximum-(logits[target[e]] if target[e]<10 else 0.)
        gradient=np.zeros(d,dtype=np.float32);biasgrad=0.
        for j in range(10):
            item=items[e,j];error=p[j]/denom-(1. if target[e]==j else 0.)
            biasgrad+=error
            for k in range(d):
                gradient[k]+=error*V[item,k]
                g=error*z[k]+reg*V[item,k]
                av[item,k]+=g*g;V[item,k]-=lr*g/math.sqrt(av[item,k])
            g=error+reg*bi[item];abi[item]+=g*g;bi[item]-=lr*g/math.sqrt(abi[item])
        for k in range(d):
            g=gradient[k]+reg*U[u,k];au[u,k]+=g*g;U[u,k]-=lr*g/math.sqrt(au[u,k])
        g=biasgrad+reg*bu[u];abu[u]+=g*g;bu[u]-=lr*g/math.sqrt(abu[u])
        for q in range(n):
            h=history[e,q]
            for k in range(d):
                g=scale*gradient[k]+reg*H[h,k]
                ah[h,k]+=g*g;H[h,k]-=lr*g/math.sqrt(ah[h,k])
    return total/len(order)

def event_data(a,h):
    assert len(a)%10==0 and np.all(a[:,6].reshape(-1,10)==a[::10,6,None])
    y=a[:,5].reshape(-1,10)
    assert np.all(y.sum(axis=1)<=1)
    target=np.where(y.sum(axis=1)>0,y.argmax(axis=1),10).astype('int32')
    return a[::10,0].copy(),a[:,1].reshape(-1,10).copy(),np.asarray(h[a[::10,6]]).copy(),target

def initial(cutoff):
    with (ROOT/('preference_stats_'+str(cutoff)+'.pkl')).open('rb') as f:base=pickle.load(f)
    d=64
    V=normalize(base[3][:,:d]).astype('float32');H=.25*V.copy();U=.25*normalize(base[4][:,:d]).astype('float32')
    context=base[0][1];g=context['global']
    ur=(context['user_click']+50*g)/(context['user_impr']+50)
    ir=(context['item_click']+50*g)/(context['item_impr']+50)
    bu=np.clip(np.log(np.clip(ur*10,.02,.98)/(1-np.clip(ur*10,.02,.98))/10)+1.8,-2,2).astype('float32')
    bi=np.clip(np.log(ir/g),-2,2).astype('float32')
    params=[U,V,H,bu,bi]
    accum=[np.full_like(x,.1) for x in params]
    return params,accum
