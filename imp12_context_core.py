"""Shared dense bilinear calibration plus eight regularized context deviations."""
import numpy as np
from imp11_bilinear_core import loss_gradient,adam_step


def context_loss_gradient(theta,u,h,v,offset,extra,target,groups):
    assert theta.shape==(9,515)
    assert len(groups)==len(u) and np.all((groups>=0)&(groups<8))
    penalty=np.full(9,.002,dtype=theta.dtype);penalty[0]=.01
    gradient=penalty[:,None]*theta
    loss=.5*np.sum(penalty[:,None]*theta*theta)
    prediction=np.empty(offset.shape,dtype=theta.dtype)
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group)
        value,derivative,p=loss_gradient(theta[0]+theta[group+1],u[ix],h[ix],v[ix],offset[ix],extra[ix],target[ix],penalty=0.)
        share=len(ix)/len(u)
        loss+=share*value
        gradient[0]+=share*derivative;gradient[group+1]+=share*derivative
        prediction[ix]=p
    return float(loss),gradient,prediction
