"""Fixed protocol for candidate calibration features; no validation tuning."""
import json,hashlib,pickle,time
from pathlib import Path
import numpy as np
from config import ARTIFACT_DIR as R
from v50_crossfit_experts import train_model,probability,good
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

def run(prefix,feature_function,input_paths,baseline_prefix,source_files):
 tic=time.monotonic();files=list(dict.fromkeys(input_paths+[R/'v48_history_calibrator.pkl']+[R/f'{baseline_prefix}_day{d}.npz' for d in (5,6)]))
 frozen=dict(C=.01,solver='lbfgs',max_iter=1000,tol=1e-6,blend_weight=.25,folds=2,seed=4801,baseline=baseline_prefix,selection='No tuning. Positive ordinary gain and positive adjusted lower interval on day5 user-crossfit, then day6 future cohort.',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files+['crossfit_calibration.py','v50_crossfit_experts.py','v49_expert_calibration.py','v48_history_calibration.py']})
 (R/f'{prefix}_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with (R/'v48_history_calibrator.pkl').open('rb') as f:prior=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r')
 z,x,users,groups,names=feature_function(5,t,h,ht,prior['spline']);first=np.isin(users,prior['fit_users']);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);target=np.bincount(eg,minlength=5)/len(eg)
 baseline=np.load(R/f'{baseline_prefix}_day5.npz');assert np.array_equal(z['events'],baseline['events']) and np.array_equal(z['y'],baseline['y'])
 models=[];prob=np.full(len(users),np.nan)
 for fold,selected in enumerate([first,~first]):
  assert not np.intersect1d(users[selected],users[~selected]).size
  model=train_model(x,z['y'],groups,selected,target);model['fit_users']=np.unique(users[selected]);prob[~selected]=probability(model,x[~selected]);models.append(model)
  print(json.dumps(dict(stage='crossfit',experiment=prefix,fold=fold,features=x.shape[1],iterations=int(model['model'].n_iter_[0]),elapsed_s=time.monotonic()-tic)),flush=True)
 assert np.isfinite(prob).all();pred=rank(.75*rank(z['pred'])+.25*rank(prob));m=measure(baseline,pred);m.update(day=5,protocol='full-day5 user crossfit',weighted_auc=weighted_auc(baseline,pred),baseline_weighted_auc=weighted_auc(baseline,baseline['pred']));reports=[m]
 np.savez_compressed(R/f'{prefix}_day5.npz',y=z['y'],events=z['events'],pred=pred,calibrated=prob,baseline=baseline['pred']);print(json.dumps(m),flush=True)
 bundle=dict(folds=models,full=None,spline=prior['spline'],expert_names=names)
 if good(m):
  model=train_model(x,z['y'],groups,np.ones(len(users),bool),target);model['fit_users']=np.unique(users);bundle['full']=model
  print(json.dumps(dict(stage='full_meta_fit',experiment=prefix,iterations=int(model['model'].n_iter_[0]),elapsed_s=time.monotonic()-tic)),flush=True)
  z,x,users,groups,_=feature_function(6,t,h,ht,prior['spline']);prob=probability(model,x);pred=rank(.75*rank(z['pred'])+.25*rank(prob));baseline=np.load(R/f'{baseline_prefix}_day6.npz');assert np.array_equal(z['events'],baseline['events']) and np.array_equal(z['y'],baseline['y'])
  m=measure(baseline,pred);m.update(day=6,protocol='future date; fit labels day5 only',weighted_auc=weighted_auc(baseline,pred),baseline_weighted_auc=weighted_auc(baseline,baseline['pred']),elapsed_s=time.monotonic()-tic);reports.append(m)
  np.savez_compressed(R/f'{prefix}_day6.npz',y=z['y'],events=z['events'],pred=pred,calibrated=prob,baseline=baseline['pred']);print(json.dumps(m),flush=True)
 with (R/f'{prefix}_models.pkl').open('wb') as f:pickle.dump(bundle,f)
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 result=dict(reports=reports,accepted=len(reports)==2 and all(good(r) for r in reports),csv_ready=False,official_auc=None,note='Day5 user-crossfit is not a future temporal fold. Day6 is future. Repeatedly reused dates; confidence intervals exclude fitting/selection uncertainty.')
 (R/f'{prefix}_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
