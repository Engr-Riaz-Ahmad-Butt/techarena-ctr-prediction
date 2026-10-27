"""Expose strictly pre-cutoff propensity and supplied history summaries to V50."""
import numpy as np
from scipy.special import logit
from config import ARTIFACT_DIR as R
from v49_expert_calibration import features as prior_features,source_paths
from crossfit_calibration import run

def rate(c,n,global_rate,strength):return (c+strength*global_rate)/(n+strength)
def features(day,t,h,ht,spline):
 z,x,users,groups,names=prior_features(day,t,h,ht,spline);row_indices=z['events']*10+np.tile(np.arange(10),len(z['events'])//10);rows=np.asarray(t[row_indices]);start=int(t[:,2].min());cut=start+day*86400
 prior=np.asarray(t[t[:,2]<cut]);assert int(prior[:,2].max())<cut<=int(rows[:,2].min());nu=int(max(t[:,0].max(),users.max()))+1;ni=int(max(t[:,1].max(),rows[:,1].max()))+1;u=rows[:,0];i=rows[:,1];bucket=rows[:,3];g=float(prior[:,5].mean());event_g=10*g
 head=prior[::10];cy=prior[:,5].reshape(-1,10).sum(1);parts=[x];added=[]
 def add(label,values):
  values=np.asarray(values);assert values.shape==(len(rows),) and np.isfinite(values).all();parts.append(values[:,None]);added.append(label)
 def counts(a,key,size,weights=None):return np.bincount(a[:,key],weights=weights,minlength=size)
 un=np.bincount(head[:,0],minlength=nu);uc=np.bincount(head[:,0],weights=cy,minlength=nu);inn=np.bincount(prior[:,1],minlength=ni);ic=np.bincount(prior[:,1],weights=prior[:,5],minlength=ni)
 add('log_user_prior_events',np.log1p(un[u]));add('log_user_prior_clicks',np.log1p(uc[u]));add('log_item_prior_exposures',np.log1p(inn[i]))
 for strength in [5,20,100]:
  ur=logit(np.clip(rate(uc[u],un[u],event_g,strength),1e-6,1-1e-6));add(f'user_propensity_{strength}',ur);add(f'user_propensity_{strength}_score_interaction',ur*z['pred'])
 for strength in [50,200]:add(f'item_ctr_{strength}',logit(np.clip(rate(ic[i],inn[i],g,strength),1e-6,1-1e-6)))
 ub=head[:,0]*4+head[:,3];ubn=np.bincount(ub,minlength=nu*4);ubc=np.bincount(ub,weights=cy,minlength=nu*4);ix=u*4+bucket
 add('user_bucket_propensity',logit(np.clip((ubc[ix]+10*rate(uc[u],un[u],event_g,20))/(ubn[ix]+10),1e-6,1-1e-6)));add('user_bucket_support',np.log1p(ubn[ix]))
 ib=prior[:,1]*4+prior[:,3];ibn=np.bincount(ib,minlength=ni*4);ibc=np.bincount(ib,weights=prior[:,5],minlength=ni*4);ix=i*4+bucket
 add('item_bucket_ctr',logit(np.clip((ibc[ix]+100*rate(ic[i],inn[i],g,200))/(ibn[ix]+100),1e-6,1-1e-6)))
 for window in [1,3]:
  keep=prior[:,2]>=cut-window*86400;a=prior[keep];ev=a[::10];click=a[:,5].reshape(-1,10).sum(1)
  wn=np.bincount(ev[:,0],minlength=nu);wc=np.bincount(ev[:,0],weights=click,minlength=nu);pn=np.bincount(a[:,1],minlength=ni);pc=np.bincount(a[:,1],weights=a[:,5],minlength=ni)
  add(f'user_recent{window}_support',np.log1p(wn[u]));add(f'user_recent{window}_rate',logit(np.clip((wc[u]+5*rate(uc[u],un[u],event_g,20))/(wn[u]+5),1e-6,1-1e-6)))
  add(f'item_recent{window}_support',np.log1p(pn[i]));add(f'item_recent{window}_ctr',logit(np.clip((pc[i]+50*rate(ic[i],inn[i],g,200))/(pn[i]+50),1e-6,1-1e-6)))
 eh=rows[::10];history=np.asarray(h[eh[:,6]]);stamps=np.asarray(ht[eh[:,6]]);valid=history>=0;length=valid.sum(1);assert np.all((stamps<eh[:,2,None])|~valid)
 age=np.maximum((eh[:,2,None]-stamps)/86400,0)*valid;mean=age.sum(1)/np.maximum(length,1);std=np.sqrt(np.maximum((age*age).sum(1)/np.maximum(length,1)-mean*mean,0))
 add('history_mean_age',np.repeat(np.log1p(mean),10));add('history_std_age',np.repeat(np.log1p(std),10))
 for window in [1,3,7]:add(f'history_recent{window}_fraction',np.repeat(((age<window)&valid).sum(1)/np.maximum(length,1),10))
 aligned=((stamps%86400//21600)==eh[:,3,None])&valid;add('history_bucket_fraction',np.repeat(aligned.sum(1)/np.maximum(length,1),10))
 match=(history[:,None,:]==rows[:,1].reshape(-1,10)[:,:,None])&valid[:,None,:];add('candidate_prior_click_count',match.sum(2).ravel())
 out=np.column_stack(parts);assert np.isfinite(out).all();return z,out,users,groups,names+added

if __name__=='__main__':run('v53_causal_propensity',features,source_paths(5)+source_paths(6)+[R/'train.npy',R/'train_hist.npy',R/'train_times.npy'],'v50_crossfit_experts',['v53_causal_propensity.py'])
