"""Stable Bernoulli real-event and conditional warm NCE updates.

Gradients are computed from one immutable parameter snapshot per record.
Noise IDs may repeat and may equal the positive item.
"""
import numpy as np
from numba import njit
import math


@njit
def bernoulli(z,y):
    loss=max(z,0.)+math.log1p(math.exp(-abs(z)))-y*z
    if z>=0:p=1./(1.+math.exp(-z))
    else:
        e=math.exp(z);p=e/(1.+e)
    return loss,p-y


@njit
def real_grad(u,items,h,y,U,V,H,bu,bi,g,reg):
    d=U.shape[1];n=0
    for j in h:
        if j>=0:n+=1
    scale=1./math.sqrt(max(n,1));z=U[u].astype(np.float64)
    for j in h:
        if j>=0:z+=scale*H[j]
    gu=np.zeros(d);gv=np.zeros((len(items),d));gh=np.zeros((len(h),d))
    gb=np.zeros(len(items));gub=0.;gg=0.;loss=0.
    for j in range(len(items)):
        i=items[j];logit=g+bu[u]+bi[i]+np.dot(z,V[i].astype(np.float64))
        ll,e=bernoulli(logit,y[j]);loss+=ll;gb[j]=e+reg*bi[i]
        gu+=e*V[i];gv[j]=e*z+reg*V[i];gub+=e;gg+=e
        loss+=.5*reg*(np.dot(V[i],V[i])+bi[i]*bi[i])
    for j in range(len(h)):
        i=h[j]
        if i>=0:
            gh[j]=scale*gu+reg*H[i];loss+=.5*reg*np.dot(H[i],H[i])
    gu+=reg*U[u];gub+=reg*bu[u];gg+=reg*(g+1.8)
    loss+=.5*reg*(np.dot(U[u],U[u])+bu[u]*bu[u]+(g+1.8)**2)
    return loss,gu,gv,gh,gub,gb,gg


@njit
def warm_grad(u,items,U,V,aw,offset,reg):
    d=U.shape[1];gu=np.zeros(d);gv=np.zeros((len(items),d));ga=0.;loss=0.
    for j in range(len(items)):
        i=items[j];z=aw[u]+offset[i]+np.dot(U[u].astype(np.float64),V[i].astype(np.float64))
        ll,e=bernoulli(z,1. if j==0 else 0.);loss+=ll
        gu+=e*V[i];gv[j]=e*U[u]+reg*V[i];ga+=e
        loss+=.5*reg*np.dot(V[i],V[i])
    gu+=reg*U[u];ga+=reg*(aw[u]+math.log(9.))
    loss+=.5*reg*(np.dot(U[u],U[u])+(aw[u]+math.log(9.))**2)
    return loss,gu,gv,ga


@njit
def vector_update(ids,grad,param,acc,lr):
    # Aggregate duplicates before the single AdaGrad step for each unique ID.
    for j in range(len(ids)):
        i=ids[j]
        if i<0:continue
        seen=False
        for k in range(j):
            if ids[k]==i:seen=True;break
        if seen:continue
        for k in range(param.shape[1]):
            v=0.
            for m in range(j,len(ids)):
                if ids[m]==i:v+=grad[m,k]
            acc[i,k]+=v*v;param[i,k]-=lr*v/math.sqrt(acc[i,k])


@njit
def bias_update(ids,grad,param,acc,lr):
    for j in range(len(ids)):
        i=ids[j];seen=False
        for k in range(j):
            if ids[k]==i:seen=True;break
        if seen:continue
        v=0.
        for k in range(j,len(ids)):
            if ids[k]==i:v+=grad[k]
        acc[i]+=v*v;param[i]-=lr*v/math.sqrt(acc[i])


@njit
def real_epoch(order,users,items,histories,targets,U,V,H,bu,bi,g,aU,aV,aH,abu,abi,ag,lr,reg):
    loss=0.
    for q in order:
        u=users[q];ll,gu,gv,gh,gub,gb,gg=real_grad(u,items[q],histories[q],targets[q],U,V,H,bu,bi,g[0],reg)
        loss+=ll
        vector_update(items[q],gv,V,aV,lr);vector_update(histories[q],gh,H,aH,lr)
        bias_update(items[q],gb,bi,abi,lr)
        for k in range(U.shape[1]):
            aU[u,k]+=gu[k]*gu[k];U[u,k]-=lr*gu[k]/math.sqrt(aU[u,k])
        abu[u]+=gub*gub;bu[u]-=lr*gub/math.sqrt(abu[u])
        ag[0]+=gg*gg;g[0]-=lr*gg/math.sqrt(ag[0])
    return loss/max(len(order),1)


@njit
def warm_epoch(order,users,items,U,V,aw,aU,aV,aaw,offset,lr,reg):
    loss=0.
    for q in order:
        u=users[q];ll,gu,gv,ga=warm_grad(u,items[q],U,V,aw,offset,reg);loss+=ll
        vector_update(items[q],gv,V,aV,lr)
        for k in range(U.shape[1]):
            aU[u,k]+=gu[k]*gu[k];U[u,k]-=lr*gu[k]/math.sqrt(aU[u,k])
        aaw[u]+=ga*ga;aw[u]-=lr*ga/math.sqrt(aaw[u])
    return loss/max(len(order),1)


@njit
def predict(users,items,histories,U,V,H,bu,bi,g):
    out=np.empty(items.shape,np.float64)
    for q in range(len(users)):
        u=users[q];z=U[u].astype(np.float64);n=0
        for i in histories[q]:
            if i>=0:n+=1
        scale=1./math.sqrt(max(n,1))
        for i in histories[q]:
            if i>=0:z+=scale*H[i]
        for j in range(items.shape[1]):
            out[q,j]=g[0]+bu[u]+bi[items[q,j]]+np.dot(z,V[items[q,j]].astype(np.float64))
    return out.ravel()
