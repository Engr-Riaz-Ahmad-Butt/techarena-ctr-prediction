"""Read-only cohort/error diagnosis; no model selection or fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
DAY=86400

def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None

def main():
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r')
 e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');et=np.load(R/'evaluation_times.npy',mmap_mode='r')
 start=int(t[:,2].min()); result={'official':{'version':'V18','auc':.809189,'newer_uploads':False},'local':{},'note':'Diagnostic subgroups are descriptive, not proof of causality or an untouched holdout.'}
 def features(rows,hist,times,cut):
  head=rows[::10]; hh=np.asarray(hist[head[:,6]]);tt=np.asarray(times[head[:,6]])
  valid=hh>=0;n=valid.sum(1); last=np.where(valid,tt,-1).max(1)
  age=np.where(n>0,(head[:,2]-last)/DAY,-1.)
  fit=np.asarray(t[t[:,2]<cut]);users=np.bincount(fit[::10,0],minlength=int(max(t[:,0].max(),e[:,0].max()))+1)
  items=np.bincount(fit[:,1],minlength=int(max(t[:,1].max(),e[:,1].max()))+1)
  repeat=(hh[:,None,:]==rows[:,1].reshape(-1,10)[:,:,None])&valid[:,None,:]
  return {'history':np.repeat(np.select([n==0,n<=4,n<=9,n<=19],['empty','1-4','5-9','10-19'],default='20'),10),
          'prior_user_events':np.repeat(np.select([users[head[:,0]]==0,users[head[:,0]]<=2,users[head[:,0]]<=9],['0','1-2','3-9'],default='10+'),10),
          'last_click_age':np.repeat(np.select([n==0,age<=1,age<=3,age<=7],['empty','0-1d','1-3d','3-7d'],default='7d+'),10),
          'item_exposure':np.select([items[rows[:,1]]<100,items[rows[:,1]]<500,items[rows[:,1]]<2000],['<100','100-499','500-1999'],default='2000+'),
          'repeat_item':np.where(repeat.any(2).ravel(),'repeat','new'),
          'time_bucket':rows[:,3].astype(str)}
 eval_features=features(e,eh,et,start+7*DAY)
 result['evaluation_distribution']={name:{str(k):float(np.mean(v==k)) for k in np.unique(v)} for name,v in eval_features.items()}
 for d in (6,5):
  z=np.load(R/f'v23_outside_day{d}.npz'); later=np.asarray(t[(t[:,2]>=start+d*DAY)&(t[:,2]<start+(d+1)*DAY)])
  _,first=np.unique(later[::10,0],return_index=True);rows=later[(first[:,None]*10+np.arange(10)).ravel()]
  assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
  fs=features(rows,h,ht,start+d*DAY);y=z['y'];p=z['pred'];base=np.load(R/f'imp19_tree_day{d}.npz')['pred'];cohorts={}
  for name,values in fs.items():
   cohorts[name]={}
   for key in np.unique(values):
    mask=values==key
    cohorts[name][str(key)]={'rows':int(mask.sum()),'share':float(mask.mean()),'click_rate':float(y[mask].mean()),'auc':auc(y[mask],p[mask]),'v20_auc':auc(y[mask],base[mask]),'evaluation_share':result['evaluation_distribution'][name].get(str(key),0)}
  yy=y.reshape(-1,10); pp=p.reshape(-1,10);clicked=yy.sum(1)>0;target=yy.argmax(1);chosen=pp.argmax(1)
  true_scores=pp[np.arange(len(pp)),target]
  ranks=1+(pp>true_scores[:,None]).sum(1)
  raw=z['raw'].reshape(-1,10)
  row={'auc':auc(y,p),'cohorts':cohorts,'clicked_event_count':int(clicked.sum()),'event_click_rate':float(clicked.mean()),'top1_on_clicked_events':float((chosen[clicked]==target[clicked]).mean()),'mean_clicked_rank':float(ranks[clicked].mean()),'within_clicked_events_auc':auc(yy[clicked].ravel(),pp[clicked].ravel()),'event_any_click_auc':auc(clicked,raw.sum(1)), 'prediction_sha256':hashlib.sha256((R/f'v23_outside_day{d}.npz').read_bytes()).hexdigest()}
  result['local'][str(d)]=row
 out=Path('results/ctr_error_diagnosis.json');out.write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
