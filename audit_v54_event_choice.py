"""Direct two-stage predictor replay and user partition/rejection verification."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v53_causal_propensity import features
from v39_regularized_als32 import rank,weighted_auc
from event_choice_meta import event_features,check_gradient

def main():
 frozen=json.loads((R/'v54_event_choice_frozen.json').read_text())
 for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 with (R/'v54_event_choice_models.pkl').open('rb') as f:bundle=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');parent,x,users,groups,_=features(5,t,h,ht,bundle['spline']);z=np.load(R/'v54_event_choice_day5.npz');prob=np.zeros(len(users));coverage=np.zeros(len(users),dtype='uint8')
 with threadpool_limits(limits=1):
  for m in bundle['folds']:
   selected=~np.isin(users,m['fit_users']);assert not np.intersect1d(users[selected],m['fit_users']).size
   v=x[selected];pe=expit(m['event_scaler'].transform(event_features(v))@m['event_model'].coef_[0]+m['event_model'].intercept_[0]);logits=(m['choice_scaler'].transform(v)@m['theta']).reshape(-1,10);e=np.exp(logits-logits.max(1)[:,None]);choice=e/e.sum(1)[:,None];p=pe[:,None]*choice
   assert np.max(np.abs(p.sum(1)-pe))<1e-12 and ((p>=0)&(p<=1)).all();prob[selected]=p.ravel();coverage[selected]+=1
 assert np.all(coverage==1) and np.array_equal(z['events'],parent['events']) and np.array_equal(z['y'],parent['y'])
 error=float(np.max(np.abs(prob-z['calibrated'])));assert error<1e-10;recipe=rank(.75*rank(parent['pred'])+.25*rank(prob));assert np.max(np.abs(recipe-z['pred']))<1e-10
 report=json.loads((R/'v54_event_choice_report.json').read_text());m=report['reports'][0];assert abs(roc_auc_score(z['y'],z['pred'])-m['auc'])<1e-12 and abs(weighted_auc(z,z['pred'])-m['weighted_auc'])<1e-12;assert not report['accepted'] and bundle['full'] is None
 result=dict(status='PASS',predictor_replay_max_error=error,probability_sum_and_partition=True,rejection_verified=True,gradient_check=check_gradient());(R/'v54_event_choice_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
