"""Two-fold user-cross-fitted expert calibration, then future-date validation.

Fixed V49 feature recipe and hyperparameters. Day5 predictions are OOF by user;
the future-day6 calibrator fits all day5 causal parent predictions/labels.
"""
import json,hashlib,pickle,time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v49_expert_calibration import features,source_paths
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

def train_model(x,y,groups,selected,target):
 scaler=StandardScaler();xf=scaler.fit_transform(x[selected]);source=np.bincount(groups[selected],minlength=5);source=source/source.sum();w=np.divide(target,source,out=np.zeros(5),where=source>0)[groups[selected]];w/=w.mean()
 model=LogisticRegression(C=.01,solver='lbfgs',max_iter=1000,tol=1e-6)
 with threadpool_limits(limits=1):model.fit(xf,y[selected],sample_weight=w)
 assert model.n_iter_[0]<1000;return dict(model=model,scaler=scaler)
def probability(saved,x):
 with threadpool_limits(limits=1):return saved['model'].predict_proba(saved['scaler'].transform(x))[:,1]
def good(m):return m['gain']>=0 and m['weighted_interval'][0]>0
def main():
 tic=time.monotonic();files=list(dict.fromkeys(sum([source_paths(d) for d in (5,6)],[])+[R/'v48_history_calibrator.pkl',R/'v48_history_calibration_day6.npz']))
 frozen=dict(seed=4801,C=.01,solver='lbfgs',max_iter=1000,tol=1e-6,blend_weight=.25,folds=2,selection='No tuning. Positive ordinary gain and positive adjusted interval on crossfit day5 vs V39; then future day6 vs V48.',protocol='Two complementary user folds on day5. Fit all day5 labels only after crossfit gate; predict future day6.',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),expert_feature_source_sha256=hashlib.sha256(Path('v49_expert_calibration.py').read_bytes()).hexdigest(),base_feature_source_sha256=hashlib.sha256(Path('v48_history_calibration.py').read_bytes()).hexdigest())
 (R/'v50_crossfit_experts_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with (R/'v48_history_calibrator.pkl').open('rb') as f:prior=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r')
 z,x,users,groups,names=features(5,t,h,ht,prior['spline']);first=np.isin(users,prior['fit_users']);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);target=np.bincount(eg,minlength=5)/len(eg)
 prob=np.full(len(users),np.nan);models=[]
 for fold,selected in enumerate([first,~first]):
  assert not np.intersect1d(users[selected],users[~selected]).size
  saved=train_model(x,z['y'],groups,selected,target);saved['fit_users']=np.unique(users[selected]);prob[~selected]=probability(saved,x[~selected]);models.append(saved)
  print(json.dumps(dict(stage='crossfit',fold=fold,fit_users=len(saved['fit_users']),iterations=int(saved['model'].n_iter_[0]),elapsed_s=time.monotonic()-tic)),flush=True)
 assert np.isfinite(prob).all()
 pred=rank(.75*rank(z['pred'])+.25*rank(prob));m=measure(z,pred);m.update(day=5,protocol='full-day5 user crossfit, not a temporal fold',weighted_auc=weighted_auc(z,pred),baseline_weighted_auc=weighted_auc(z,z['pred']))
 np.savez_compressed(R/'v50_crossfit_experts_day5.npz',y=z['y'],events=z['events'],pred=pred,calibrated=prob,baseline=z['pred'],fold=first.astype('uint8'))
 print(json.dumps(m),flush=True);reports=[m]
 saved_bundle=dict(folds=models,full=None,spline=prior['spline'],expert_names=names)
 if good(m):
  full=train_model(x,z['y'],groups,np.ones(len(users),bool),target);full['fit_users']=np.unique(users);saved_bundle['full']=full
  print(json.dumps(dict(stage='full_meta_fit',iterations=int(full['model'].n_iter_[0]),elapsed_s=time.monotonic()-tic)),flush=True)
  z,x,users,groups,_=features(6,t,h,ht,prior['spline']);p6=probability(full,x);pred=rank(.75*rank(z['pred'])+.25*rank(p6));base=np.load(R/'v48_history_calibration_day6.npz');assert np.array_equal(z['events'],base['events']) and np.array_equal(z['y'],base['y'])
  m=measure(base,pred);m.update(day=6,protocol='future date; all calibration fitting labels on day5',weighted_auc=weighted_auc(base,pred),baseline_weighted_auc=weighted_auc(base,base['pred']),elapsed_s=time.monotonic()-tic);reports.append(m)
  np.savez_compressed(R/'v50_crossfit_experts_day6.npz',y=base['y'],events=base['events'],pred=pred,calibrated=p6,baseline=base['pred']);print(json.dumps(m),flush=True)
 with (R/'v50_crossfit_expert_models.pkl').open('wb') as f:pickle.dump(saved_bundle,f)
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(good(r) for r in reports),csv_ready=False,official_auc=None,note='Day5 OOF by user, day6 temporal confirmation. Repeatedly reused dates and model settings derived after V49; uncertainty includes selection limitations.')
 (R/'v50_crossfit_experts_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
