"""Saved predictor/partition/metric audit for a fixed crossfit experiment."""
import argparse,hashlib,importlib,json,pickle
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v39_regularized_als32 import rank,weighted_auc

def main():
 parser=argparse.ArgumentParser();parser.add_argument('prefix');parser.add_argument('module');args=parser.parse_args();prefix=args.prefix;feature_function=importlib.import_module(args.module).features
 frozen=json.loads((R/f'{prefix}_frozen.json').read_text())
 for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 result=json.loads((R/f'{prefix}_report.json').read_text())
 with (R/f'{prefix}_models.pkl').open('rb') as f:bundle=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');reports=[];day5_users=None
 for r in result['reports']:
  day=r['day'];parent,x,users,groups,_=feature_function(day,t,h,ht,bundle['spline']);z=np.load(R/f'{prefix}_day{day}.npz');assert np.array_equal(parent['events'],z['events']) and np.array_equal(parent['y'],z['y'])
  baseline=np.load(R/f"{frozen['baseline']}_day{day}.npz");assert np.array_equal(z['baseline'],baseline['pred'])
  if day==5:
   day5_users=np.unique(users);prob=np.zeros(len(users));coverage=np.zeros(len(users),dtype='uint8')
   for model in bundle['folds']:
    selected=~np.isin(users,model['fit_users']);assert not np.intersect1d(users[selected],model['fit_users']).size
    with threadpool_limits(limits=1):prob[selected]=expit(model['scaler'].transform(x[selected])@model['model'].coef_[0]+model['model'].intercept_[0])
    coverage[selected]+=1
   assert np.all(coverage==1)
  else:
   model=bundle['full'];assert np.array_equal(model['fit_users'],day5_users)
   with threadpool_limits(limits=1):prob=expit(model['scaler'].transform(x)@model['model'].coef_[0]+model['model'].intercept_[0])
   assert int(t[np.load(R/'v39_regularized_als32_day5.npz')['events'][::10]*10,2].max())<int(t[z['events'][::10]*10,2].min())
  error=float(np.max(np.abs(prob-z['calibrated'])));assert error<1e-10;pred=rank(.75*rank(parent['pred'])+.25*rank(prob));assert np.max(np.abs(pred-z['pred']))<1e-10
  assert abs(roc_auc_score(z['y'],pred)-r['auc'])<1e-12 and abs(weighted_auc(z,pred)-r['weighted_auc'])<1e-12
  reports.append(dict(day=day,predictor_replay_max_error=error,partition_and_recipe=True,metrics=True))
 audit=dict(status='PASS',reports=reports,experiment_accepted=result['accepted'],official_auc=None,note='Replay validates implementation and reported metrics; it does not establish target attainment or remove reuse/selection uncertainty.')
 (R/f'{prefix}_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
