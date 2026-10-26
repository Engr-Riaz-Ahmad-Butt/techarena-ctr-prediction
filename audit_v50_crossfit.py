"""Independent logistic replay, user-OOF checks and fixed-model event bootstrap."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v49_expert_calibration import features
from v39_regularized_als32 import rank,weighted_auc
from bootstrap_v39 import weighted_auc_sorted,differences

def main():
 set_num_threads(4);frozen=json.loads((R/'v50_crossfit_experts_frozen.json').read_text())
 for filename,key in [('v50_crossfit_experts.py','source_sha256'),('v49_expert_calibration.py','expert_feature_source_sha256'),('v48_history_calibration.py','base_feature_source_sha256')]:assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==frozen[key]
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 with (R/'v50_crossfit_expert_models.pkl').open('rb') as f:bundle=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');reports=[]
 day5_users=None
 for day in (5,6):
  parent,x,users,groups,names=features(day,t,h,ht,bundle['spline']);z=np.load(R/f'v50_crossfit_experts_day{day}.npz');assert np.array_equal(parent['events'],z['events']) and np.array_equal(parent['y'],z['y'])
  if day==5:
   day5_users=np.unique(users);prob=np.zeros(len(users));coverage=np.zeros(len(users),dtype='uint8')
   for model in bundle['folds']:
    selected=~np.isin(users,model['fit_users']);assert not np.intersect1d(users[selected],model['fit_users']).size
    with threadpool_limits(limits=1):prob[selected]=expit(model['scaler'].transform(x[selected])@model['model'].coef_[0]+model['model'].intercept_[0])
    coverage[selected]+=1
   assert np.all(coverage==1)
  else:
   model=bundle['full'];assert np.array_equal(model['fit_users'],day5_users)
   # Calibrator labels come exclusively from day5, hence before every day6 event.
   start=int(t[:,2].min());assert int(t[np.load(R/'v39_regularized_als32_day5.npz')['events'][::10]*10,2].max())<start+6*86400<=int(t[z['events'][::10]*10,2].min())
   with threadpool_limits(limits=1):prob=expit(model['scaler'].transform(x)@model['model'].coef_[0]+model['model'].intercept_[0])
  error=float(np.max(np.abs(prob-z['calibrated'])));assert error<1e-10
  pred=rank(.75*rank(parent['pred'])+.25*rank(prob));assert np.max(np.abs(pred-z['pred']))<1e-10
  y=z['y'];p=z['pred'];old=z['baseline'];events=z['events'];n=len(y)//10;ei=np.arange(len(y))//10
  assert np.unique(users[::10]).size==n
  vg=np.digitize((h[events]>=0).sum(1),[0,4,9,19],right=True);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);w=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
  def ordered(values):
   ix=np.argsort(values,kind='stable');return y[ix],values[ix],w[ix],ei[ix]
  a,b=ordered(old),ordered(p);ones=np.ones(n,dtype=np.uint16)
  for arr,values in [(a,old),(b,p)]:assert abs(weighted_auc_sorted(*arr,ones)-roc_auc_score(y,values,sample_weight=w))<1e-10
  rng=np.random.default_rng(5001+day);deltas=[];distribution=np.full(n,1/n)
  for block in range(10):
   counts=np.array([rng.multinomial(n,distribution) for _ in range(100)],dtype=np.uint16);deltas.extend(differences(*a,*b,counts).tolist())
   if block in [4,9]:print(json.dumps(dict(day=day,bootstrap_completed=(block+1)*100)),flush=True)
  interval=np.quantile(deltas,[.025,.975]);reports.append(dict(day=day,protocol='user crossfit' if day==5 else 'future temporal date',saved_logistic_replay_max_error=error,events=n,ordinary_auc=float(roc_auc_score(y,p)),history_adjusted_auc=weighted_auc(z,p),history_adjusted_gain=float(roc_auc_score(y,p,sample_weight=w)-roc_auc_score(y,old,sample_weight=w)),bootstrap_replicates=1000,seed=5001+day,paired_event_interval=interval.tolist(),passes=bool(interval[0]>0)))
 result=dict(status='PASS',reports=reports,promotable=all(r['passes'] for r in reports),official_auc=None,csv_ready=False,note='Fixed-model bootstrap excludes calibrator-fitting and repeated-selection uncertainty. Day5 is user-OOF, not a strictly future temporal fold; weekday target mismatch remains.')
 (R/'v50_crossfit_experts_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
