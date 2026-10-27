"""Reload residual tree files and replay the held-out user predictions."""
import json,hashlib,pickle
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v53_causal_propensity import features
from v50_crossfit_experts import probability
from v39_regularized_als32 import rank,weighted_auc

def main():
 frozen=json.loads((R/'v55_residual_meta_frozen.json').read_text())
 for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 with (R/'v53_causal_propensity_models.pkl').open('rb') as f:parent=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');p,x,users,groups,_=features(5,t,h,ht,parent['spline']);z=np.load(R/'v55_residual_meta_day5.npz');out=np.zeros(len(users));coverage=np.zeros(len(users),dtype='uint8')
 for fold,model in enumerate(parent['folds']):
  selected=~np.isin(users,model['fit_users']);assert not np.intersect1d(users[selected],model['fit_users']).size;tree=lgb.Booster(model_file=str(R/f'v55_residual_day5_fold{fold}.txt'));assert tree.num_trees()==100
  original=logit(np.clip(probability(model,x[selected]),1e-7,1-1e-7));out[selected]=expit(original+tree.predict(x[selected],raw_score=True,num_threads=4));coverage[selected]+=1
 assert np.all(coverage==1) and np.array_equal(z['events'],p['events']) and np.array_equal(z['y'],p['y']);error=float(np.max(np.abs(out-z['calibrated'])));assert error<1e-10
 recipe=rank(.75*rank(p['pred'])+.25*rank(out));assert np.array_equal(recipe,z['pred']);report=json.loads((R/'v55_residual_meta_report.json').read_text());m=report['reports'][0];assert abs(roc_auc_score(z['y'],recipe)-m['auc'])<1e-12 and abs(weighted_auc(z,recipe)-m['weighted_auc'])<1e-12;assert report['accepted'] is False
 result=dict(status='PASS',saved_tree_replay_max_error=error,partition_and_recipe=True,metrics=True,rejection_verified=True);(R/'v55_residual_meta_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
