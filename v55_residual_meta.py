"""Shallow, strongly regularized logit residual trees above the saved V53 model."""
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v53_causal_propensity import features,source_paths
from v50_crossfit_experts import probability,good
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

PARAMS=dict(metric='None',num_leaves=7,max_depth=3,learning_rate=.03,min_data_in_leaf=500,lambda_l2=20.,max_bin=63,feature_fraction=1.,bagging_fraction=1.,num_threads=4,verbosity=-1,deterministic=True,force_col_wise=True,seed=5501,boost_from_average=False)
def fit_tree(x,y,weight,offset,tag):
 first=[True]
 def objective(pred,dataset):
  if first[0]:assert np.max(np.abs(pred))<1e-12;first[0]=False
  p=expit(offset+pred);return (p-y)*weight,np.maximum(p*(1-p),1e-6)*weight
 model=lgb.train(dict(PARAMS,objective=objective),lgb.Dataset(x,label=y,free_raw_data=True),num_boost_round=100)
 path=R/f'v55_residual_{tag}.txt';model.save_model(str(path));assert not first[0];return model
def main():
 tic=time.monotonic();files=list(dict.fromkeys(source_paths(5)+source_paths(6)+[R/'v53_causal_propensity_models.pkl',R/'train.npy',R/'train_hist.npy',R/'train_times.npy']+[R/f'v53_causal_propensity_day{d}.npz' for d in (5,6)]))
 frozen=dict(params=PARAMS,trees=100,blend_weight=.25,folds=2,baseline='v53_causal_propensity',objective='Weighted Bernoulli residual: sigmoid(frozen base logit + tree score)',selection='No tuning; positive ordinary gain and positive adjusted lower interval on day5 OOF then future day6',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['v55_residual_meta.py','v53_causal_propensity.py','v49_expert_calibration.py','v48_history_calibration.py','v50_crossfit_experts.py']})
 (R/'v55_residual_meta_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with (R/'v53_causal_propensity_models.pkl').open('rb') as f:parent=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');z,x,users,groups,names=features(5,t,h,ht,parent['spline']);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);target=np.bincount(eg,minlength=5)/len(eg);out=np.full(len(users),np.nan);metadata=[]
 for fold,model in enumerate(parent['folds']):
  selected=np.isin(users,model['fit_users']);assert not np.intersect1d(users[~selected],model['fit_users']).size
  source=np.bincount(groups[selected],minlength=5);source=source/source.sum();w=np.divide(target,source,out=np.zeros(5),where=source>0)[groups[selected]];w/=w.mean()
  offset=logit(np.clip(probability(model,x[selected]),1e-7,1-1e-7));tree=fit_tree(x[selected],z['y'][selected],w,offset,f'day5_fold{fold}');outside=logit(np.clip(probability(model,x[~selected]),1e-7,1-1e-7));out[~selected]=expit(outside+tree.predict(x[~selected],raw_score=True,num_threads=4))
  metadata.append(dict(fold=fold,fit_users=model['fit_users'].tolist(),model_file=f'v55_residual_day5_fold{fold}.txt'));print(json.dumps(dict(stage='residual_crossfit',fold=fold,elapsed_s=time.monotonic()-tic)),flush=True)
 assert np.isfinite(out).all();base=np.load(R/'v53_causal_propensity_day5.npz');assert np.array_equal(z['events'],base['events']) and np.array_equal(z['y'],base['y']);pred=rank(.75*rank(z['pred'])+.25*rank(out));m=measure(base,pred);m.update(day=5,weighted_auc=weighted_auc(base,pred),baseline_weighted_auc=weighted_auc(base,base['pred']));reports=[m]
 np.savez_compressed(R/'v55_residual_meta_day5.npz',y=z['y'],events=z['events'],pred=pred,calibrated=out,baseline=base['pred']);print(json.dumps(m),flush=True)
 if good(m):
  model=parent['full'];assert np.array_equal(model['fit_users'],np.unique(users));source=np.bincount(groups,minlength=5)/len(groups);w=np.divide(target,source,out=np.zeros(5),where=source>0)[groups];w/=w.mean();offset=logit(np.clip(probability(model,x),1e-7,1-1e-7));tree=fit_tree(x,z['y'],w,offset,'full')
  z,x,users,groups,_=features(6,t,h,ht,parent['spline']);offset=logit(np.clip(probability(model,x),1e-7,1-1e-7));out=expit(offset+tree.predict(x,raw_score=True,num_threads=4));pred=rank(.75*rank(z['pred'])+.25*rank(out));base=np.load(R/'v53_causal_propensity_day6.npz');assert np.array_equal(z['events'],base['events']) and np.array_equal(z['y'],base['y']);m=measure(base,pred);m.update(day=6,weighted_auc=weighted_auc(base,pred),baseline_weighted_auc=weighted_auc(base,base['pred']),elapsed_s=time.monotonic()-tic);reports.append(m)
  np.savez_compressed(R/'v55_residual_meta_day6.npz',y=z['y'],events=z['events'],pred=pred,calibrated=out,baseline=base['pred']);print(json.dumps(m),flush=True)
 (R/'v55_residual_meta_models.json').write_text(json.dumps(dict(folds=metadata,full='v55_residual_full.txt' if len(reports)==2 else None),indent=2))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(good(r) for r in reports),official_auc=None,csv_ready=False,note='Parent and residual share each fitting user fold. No held-out labels fit either. Day5 user-OOF, day6 future; reused dates and selection uncertainty remain.')
 (R/'v55_residual_meta_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
