"""Lever 2 screen: session-structure and candidate-recency timing features added to the V53 head.

Everything V53 uses stays identical (same 190 features, same head, same 0.75/0.25 rank blend, same
cross-fit protocol); only strictly-past history timestamps and the event time add 12 columns.
Comparison baseline is V53 itself, exactly as V53 was compared with V50.
"""
import numpy as np
from config import ARTIFACT_DIR as R
from v49_expert_calibration import source_paths
from v53_causal_propensity import features as v53_features
from crossfit_calibration import run

HOUR=3600.;DAY=86400.;GAP=4*HOUR  # a new session starts after four hours without a click

def features(day,t,h,ht,spline):
 z,x,users,groups,names=v53_features(day,t,h,ht,spline)
 row_indices=z['events']*10+np.tile(np.arange(10),len(z['events'])//10);rows=np.asarray(t[row_indices]);eh=rows[::10]
 start=int(t[:,2].min());cut=start+day*86400;assert cut<=int(rows[:,2].min())
 history=np.asarray(h[eh[:,6]]);stamps=np.asarray(ht[eh[:,6]]).astype(np.float64);valid=history>=0;length=valid.sum(1)
 assert np.all((stamps<eh[:,2,None])|~valid),'history must precede its event'
 # Histories are oldest -> newest with the valid entries at the front; positions >= length are padding.
 now=eh[:,2].astype(np.float64);age=np.where(valid,(now[:,None]-stamps)/HOUR,np.nan)  # hours since each click
 last=np.where(length>0,np.nanmin(np.where(valid,age,np.inf),axis=1),np.inf);first=np.where(length>0,np.nanmax(np.where(valid,age,-np.inf),axis=1),0.)
 span_h=np.where(length>0,first-np.where(np.isinf(last),0,last),0.)
 gaps=np.where(valid[:,1:]&valid[:,:-1],(stamps[:,1:]-stamps[:,:-1])/HOUR,np.nan)  # between consecutive clicks
 n_sessions=1+np.nansum(gaps>GAP/HOUR,axis=1);n_sessions=np.where(length>0,n_sessions,0)
 # Last session: trailing run of clicks with gaps <= GAP. Walk from the newest valid click backwards.
 last_session_len=np.zeros(len(eh),np.int64);last_session_mask=np.zeros_like(valid)
 for r in range(len(eh)):
  n=length[r]
  if n==0:continue
  k=n-1;last_session_mask[r,k]=True
  while k>0 and (stamps[r,k]-stamps[r,k-1])<=GAP:k-=1;last_session_mask[r,k]=True
  last_session_len[r]=n-k
 mean_gap=np.where(length>1,np.nanmean(gaps,axis=1),0.);last_gap=np.where(length>1,gaps[np.arange(len(eh)),np.maximum(length-2,0)],0.);last_gap=np.where(length>1,last_gap,0.)
 rate=length/np.maximum(span_h/24.,1/24.)  # clicks per day of history span
 cand=rows[:,1].reshape(-1,10);match=(history[:,None,:]==cand[:,:,None])&valid[:,None,:]          # event x slot x history
 cand_age=np.where(match,age[:,None,:],np.inf).min(axis=2)                                          # hours since this candidate was last clicked
 cand_in_last_session=(match&last_session_mask[:,None,:]).any(axis=2)
 cand_decay=np.where(match,np.exp(-np.nan_to_num(age[:,None,:],nan=1e9)/24.),0.).sum(axis=2)        # one-day decayed repeat score
 added=[('session_count',np.log1p(n_sessions)),('last_session_len',np.log1p(last_session_len)),('last_session_share',last_session_len/np.maximum(length,1)),
        ('mean_gap_h',np.log1p(mean_gap)),('last_gap_h',np.log1p(last_gap)),('history_span_d',np.log1p(span_h/24.)),('click_rate_per_day',np.log1p(rate)),
        ('last_click_age_h',np.log1p(np.where(np.isinf(last),24*30.,last)))]
 parts=[x];labels=[]
 for label,v in added:
  v=np.repeat(np.asarray(v,np.float64),10);assert v.shape==(len(rows),) and np.isfinite(v).all(),label;parts.append(v[:,None]);labels.append(label)
 for label,v in [('candidate_last_click_age_h',np.log1p(np.where(np.isinf(cand_age),24*30.,cand_age)).ravel()),('candidate_in_last_session',cand_in_last_session.astype(np.float64).ravel()),
                 ('candidate_decayed_repeat',cand_decay.ravel()),('candidate_decayed_repeat_x_score',cand_decay.ravel()*z['pred'])]:
  assert v.shape==(len(rows),) and np.isfinite(v).all(),label;parts.append(np.asarray(v,np.float64)[:,None]);labels.append(label)
 out=np.column_stack(parts).astype(np.float64);assert np.isfinite(out).all();return z,out,users,groups,names+labels

if __name__=='__main__':run('v70_session_timing',features,source_paths(5)+source_paths(6)+[R/'train.npy',R/'train_hist.npy',R/'train_times.npy'],'v53_causal_propensity',['v70_session_timing.py','v53_causal_propensity.py'])
