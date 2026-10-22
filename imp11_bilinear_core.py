"""Small differentiable residual model; no data reads or fitting at import."""
import numpy as np


def loss_gradient(theta,u,h,v,offset,extra,target,penalty=.01):
    """Mean event softmax loss plus ridge; outside option has fixed logit zero."""
    d=u.shape[1];a=theta[:d*d].reshape(d,d);c=theta[d*d:2*d*d].reshape(d,d)
    b=theta[2*d*d:]
    logits=offset+np.einsum('ed,ecd->ec',u@a+h@c,v)+np.einsum('ecf,f->ec',extra,b)
    maximum=np.maximum(0,logits.max(1,keepdims=True))
    exp=np.exp(logits-maximum);denom=np.exp(-maximum)+exp.sum(1,keepdims=True)
    p=exp/denom
    selected=np.zeros(len(u),dtype=logits.dtype);clicked=target<10
    selected[clicked]=logits[np.flatnonzero(clicked),target[clicked]]
    loss=np.mean(np.log(denom[:,0])+maximum[:,0]-selected)+.5*penalty*np.dot(theta,theta)
    error=p.copy();error[np.flatnonzero(clicked),target[clicked]]-=1
    projected=np.einsum('ec,ecd->ed',error,v)
    ga=u.T@projected/len(u);gc=h.T@projected/len(u)
    gb=np.einsum('ecf,ec->f',extra,error)/len(u)
    gradient=np.concatenate([ga.ravel(),gc.ravel(),gb])+penalty*theta
    return float(loss),gradient,p


def adam_step(theta,gradient,first,second,step,rate=.01):
    gradient=gradient/max(1.,np.linalg.norm(gradient)/5.)
    first*=.9;first+=.1*gradient
    second*=.999;second+=.001*gradient*gradient
    theta-=rate*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
