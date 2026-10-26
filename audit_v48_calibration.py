"""Audit user partition, saved calibrator replay, metrics and paired bootstrap."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from numba import set_num_threads
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from scipy.special import expit
from config import ARTIFACT_DIR as R
from v48_history_calibration import features
from v39_regularized_als32 import rank,weighted_auc
from bootstrap_v39 import weighted_auc_sorted,differences

def main():
 set_num_threads(4);frozen=json.loads((R/'v48_history_calibration_frozen.json').read_text());repair=json.loads((R/'v48_source_repair.json').read_text());assert hashlib.sha256(Path('v48_history_calibration.py').read_bytes()).hexdigest()==repair['repaired_source_sha256']
 assert frozen['source_sha256']==repair['original_source_sha256']==hashlib.sha256((R/'v48_original_source.py').read_bytes()).hexdigest()
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 with (R/'v48_history_calibrator.pkl').open('rb') as f:saved=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');reports=[]
 for day in (5,6):
  base=np.load(R/f'v39_regularized_als32_day{day}.npz');candidate=np.load(R/f'v48_history_calibration_day{day}.npz');x,users,group=features(base,t,h,ht,saved['spline'])
  selected=~np.isin(users,saved['fit_users']) if day==5 else np.ones(len(users),bool)
  if day==5:
   unique=np.unique(users);expected=np.random.default_rng(4801).permutation(unique)[:len(unique)//2];assert np.array_equal(saved['fit_users'],expected)
   assert not np.intersect1d(users[selected],saved['fit_users']).size
  assert np.array_equal(candidate['events'],base['events'][selected]) and np.array_equal(candidate['y'],base['y'][selected])
  # Independent direct logistic linear predictor; no predict_proba call.
  with threadpool_limits(limits=1):prob=expit(saved['scaler'].transform(x[selected])@saved['model'].coef_[0]+saved['model'].intercept_[0])
  error=float(np.max(np.abs(prob-candidate['calibrated'])));assert error<1e-10
  recipe=rank(.75*rank(candidate['baseline'])+.25*rank(prob));assert np.max(np.abs(recipe-candidate['pred']))<1e-10
  ev=candidate['events'];y=candidate['y'];p=candidate['pred'];old=candidate['baseline'];assert np.array_equal(ev.reshape(-1,10),np.repeat(ev[::10,None],10,axis=1))
  vg=np.digitize((h[ev]>=0).sum(1),[0,4,9,19],right=True);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);w=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
  idx=np.arange(len(y))//10;n=len(y)//10
  def ordered(a):
   ix=np.argsort(a,kind='stable');return y[ix],a[ix],w[ix],idx[ix]
  a,b=ordered(old),ordered(p);ones=np.ones(n,dtype=np.uint16)
  for arr,values in [(a,old),(b,p)]:assert abs(weighted_auc_sorted(*arr,ones)-roc_auc_score(y,values,sample_weight=w))<1e-10
  rng=np.random.default_rng(4801+day);deltas=[];distribution=np.full(n,1/n)
  for block in range(10):
   counts=np.array([rng.multinomial(n,distribution) for _ in range(100)],dtype=np.uint16);deltas.extend(differences(*a,*b,counts).tolist())
   if block in [4,9]:print(json.dumps(dict(day=day,bootstrap_completed=(block+1)*100)),flush=True)
  interval=np.quantile(deltas,[.025,.975]);observed=float(roc_auc_score(y,p,sample_weight=w)-roc_auc_score(y,old,sample_weight=w))
  reports.append(dict(day=day,replay_max_error=error,user_partition_verified=day==5,events=n,ordinary_auc=float(roc_auc_score(y,p)),history_adjusted_auc=weighted_auc(candidate,p),history_adjusted_gain=observed,bootstrap_replicates=1000,bootstrap_seed=4801+day,paired_event_percentile_interval=interval.tolist(),passes=bool(interval[0]>0)))
 result=dict(status='PASS',reports=reports,promotable=all(r['passes'] for r in reports),csv_ready=False,official_auc=None,note='Conditional on fixed calibrator and repeatedly reused parent holdouts; bootstrap does not remove selection uncertainty or calendar distribution shift.')
 (R/'v48_history_calibration_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
