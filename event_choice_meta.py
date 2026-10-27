"""Convex two-stage click-anywhere and conditional item-choice calibration."""
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

def event_features(x):
 q=x.reshape(-1,10,x.shape[1]);std=q.std(1);std[std<1e-10]=0
 return np.column_stack([q.mean(1),q.max(1),std])
def choice_loss_gradient(theta,x,target,weight,penalty=.001):
 scores=(x.reshape(-1,x.shape[-1])@theta).reshape(-1,10);p=softmax(scores,axis=1);denom=weight.sum();loss=float(np.dot(weight,logsumexp(scores,axis=1)-scores[np.arange(len(target)),target])/denom+.5*penalty*np.dot(theta,theta))
 p[np.arange(len(target)),target]-=1;p*=weight[:,None]/denom
 gradient=x.reshape(-1,x.shape[-1]).T@p.ravel()+penalty*theta
 return loss,gradient
def train_model(x,y,groups,row_selected,target):
 selected=row_selected[::10];assert np.all(row_selected.reshape(-1,10)==selected[:,None]);event_y=y.reshape(-1,10).sum(1);assert (event_y<=1).all();group=groups[::10]
 source=np.bincount(group[selected],minlength=5);source=source/source.sum();weight=np.divide(target,source,out=np.zeros(5),where=source>0)[group];weight/=weight[selected].mean()
 xe=event_features(x);es=StandardScaler();ef=es.fit_transform(xe[selected]);em=LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,tol=1e-6)
 with threadpool_limits(limits=1):em.fit(ef,event_y[selected],sample_weight=weight[selected])
 assert em.n_iter_[0]<1000
 cs=StandardScaler();cs.fit(x[row_selected]);clicked=selected&(event_y>0);click_rows=np.repeat(clicked,10);xx=cs.transform(x[click_rows]).reshape(-1,10,x.shape[1]);choice_target=y.reshape(-1,10)[clicked].argmax(1);cw=weight[clicked]
 with threadpool_limits(limits=1):fit=minimize(choice_loss_gradient,np.zeros(x.shape[1]),args=(xx,choice_target,cw,.001),method='L-BFGS-B',jac=True,options=dict(maxiter=500,ftol=1e-10,gtol=1e-6))
 assert fit.success,fit.message
 return dict(event_model=em,event_scaler=es,choice_scaler=cs,theta=fit.x,choice_iterations=int(fit.nit),choice_gradient_max=float(np.max(np.abs(fit.jac))))
def probability(model,x):
 with threadpool_limits(limits=1):
  any_click=model['event_model'].predict_proba(model['event_scaler'].transform(event_features(x)))[:,1]
  logits=(model['choice_scaler'].transform(x)@model['theta']).reshape(-1,10);choice=softmax(logits,axis=1);p=any_click[:,None]*choice
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.max(np.abs(p.sum(1)-any_click))<1e-12
 return p.ravel()
def check_gradient():
 rng=np.random.default_rng(5401);x=rng.normal(size=(4,10,7));theta=rng.normal(size=7)*.1;target=np.array([0,2,5,9]);w=np.array([.5,1.,2.,3.]);loss,g=choice_loss_gradient(theta,x,target,w);fd=[]
 for j in range(len(theta)):
  a=theta.copy();b=theta.copy();a[j]+=1e-5;b[j]-=1e-5;fd.append((choice_loss_gradient(a,x,target,w)[0]-choice_loss_gradient(b,x,target,w)[0])/2e-5)
 error=float(np.max(np.abs(g-fd)));assert error<1e-7
 return dict(choice_gradient_max_error=error)
