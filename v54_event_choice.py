"""Fit two-stage event/choice meta-model with V53 features and fixed user folds."""
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from config import ARTIFACT_DIR as R
from v53_causal_propensity import features,source_paths
from event_choice_meta import train_model,probability,check_gradient
from v50_crossfit_experts import good
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

def main():
 tic=time.monotonic();files=list(dict.fromkeys(source_paths(5)+source_paths(6)+[R/'v48_history_calibrator.pkl',R/'train.npy',R/'train_hist.npy',R/'train_times.npy']+[R/f'v53_causal_propensity_day{d}.npz' for d in (5,6)]))
 frozen=dict(event_C=.1,choice_mean_loss_l2=.001,event_max_iter=1000,choice_max_iter=500,blend_weight=.25,folds=2,seed=4801,selection='Fixed two-stage likelihood; require gain against V53 on user-crossfit day5 and future day6',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['v54_event_choice.py','event_choice_meta.py','v53_causal_propensity.py','v49_expert_calibration.py','v48_history_calibration.py']})
 (R/'v54_event_choice_frozen.json').open('x').write(json.dumps(frozen,indent=2));gradient=check_gradient();print(json.dumps(gradient),flush=True)
 with (R/'v48_history_calibrator.pkl').open('rb') as f:prior=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');z,x,users,groups,names=features(5,t,h,ht,prior['spline']);first=np.isin(users,prior['fit_users']);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);target=np.bincount(eg,minlength=5)/len(eg)
 baseline=np.load(R/'v53_causal_propensity_day5.npz');assert np.array_equal(z['events'],baseline['events']) and np.array_equal(z['y'],baseline['y']);prob=np.full(len(users),np.nan);models=[]
 for fold,selected in enumerate([first,~first]):
  assert not np.intersect1d(users[selected],users[~selected]).size
  model=train_model(x,z['y'],groups,selected,target);model['fit_users']=np.unique(users[selected]);prob[~selected]=probability(model,x[~selected]);models.append(model)
  print(json.dumps(dict(stage='two_stage_crossfit',fold=fold,event_iterations=int(model['event_model'].n_iter_[0]),choice_iterations=model['choice_iterations'],elapsed_s=time.monotonic()-tic)),flush=True)
 pred=rank(.75*rank(z['pred'])+.25*rank(prob));m=measure(baseline,pred);m.update(day=5,weighted_auc=weighted_auc(baseline,pred),baseline_weighted_auc=weighted_auc(baseline,baseline['pred']));reports=[m]
 np.savez_compressed(R/'v54_event_choice_day5.npz',y=z['y'],events=z['events'],pred=pred,calibrated=prob,baseline=baseline['pred']);print(json.dumps(m),flush=True);bundle=dict(folds=models,full=None,spline=prior['spline'],expert_names=names)
 if good(m):
  model=train_model(x,z['y'],groups,np.ones(len(users),bool),target);model['fit_users']=np.unique(users);bundle['full']=model
  z,x,users,groups,_=features(6,t,h,ht,prior['spline']);prob=probability(model,x);pred=rank(.75*rank(z['pred'])+.25*rank(prob));baseline=np.load(R/'v53_causal_propensity_day6.npz');assert np.array_equal(z['events'],baseline['events']) and np.array_equal(z['y'],baseline['y']);m=measure(baseline,pred);m.update(day=6,weighted_auc=weighted_auc(baseline,pred),baseline_weighted_auc=weighted_auc(baseline,baseline['pred']),elapsed_s=time.monotonic()-tic);reports.append(m)
  np.savez_compressed(R/'v54_event_choice_day6.npz',y=z['y'],events=z['events'],pred=pred,calibrated=prob,baseline=baseline['pred']);print(json.dumps(m),flush=True)
 with (R/'v54_event_choice_models.pkl').open('wb') as f:pickle.dump(bundle,f)
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 result=dict(reports=reports,gradient_check=gradient,accepted=len(reports)==2 and all(good(r) for r in reports),csv_ready=False,official_auc=None,note='Day5 user-OOF, day6 future. Reused dates and fixed-model uncertainty; no official-score forecast.')
 (R/'v54_event_choice_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
