"""Learn a fixed low-capacity combination of causal expert scores on day5 users."""
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v48_history_calibration import features as base_features
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

SOURCES=[('imp19_tree','pred'),('v23_outside','raw'),('v36_recent_context','raw'),('v36_recent_context','old'),('v39_regularized_als32','raw'),('imp15_als','raw')]
def source_paths(day):
 return list(dict.fromkeys([R/f'{name}_day{day}.npz' for name,key in SOURCES]+[R/f'v21_fastseeds_day{day}_s{s}_ep1_raw.npz' for s in (0,1,2)]))
def features(day,t,h,ht,spline):
 z=np.load(R/f'v39_regularized_als32_day{day}.npz');x,users,groups=base_features(z,t,h,ht,spline);experts=[];names=[]
 for name,key in SOURCES:
  with np.load(R/f'{name}_day{day}.npz') as a:
   assert np.array_equal(a['events'],z['events']) and np.array_equal(a['y'],z['y'])
   experts.append(a[key].copy());names.append(name+'_'+key)
 seeds=[]
 for seed in (0,1,2):
  with np.load(R/f'v21_fastseeds_day{day}_s{seed}_ep1_raw.npz') as a:
   assert np.array_equal(a['events'],z['events']) and np.array_equal(a['y'],z['y']);seeds.append(rank(a['raw']))
 experts.append(np.mean(seeds,axis=0));names.append('seed_mean_rank')
 one=np.eye(5)[groups];parts=[x]
 for values in experts:
  r=rank(values);q=r.reshape(-1,10);center=r-np.repeat(q.mean(1),10);l=logit(np.clip(r,.001,.999))
  parts.extend([np.column_stack([r,l,center,np.repeat(q.max(1),10),np.repeat(q.std(1),10)]),one*r[:,None],one*center[:,None]])
 x=np.column_stack(parts);assert np.isfinite(x).all();return z,x,users,groups,names

def main():
 tic=time.monotonic();files=list(dict.fromkeys(sum([source_paths(d) for d in (5,6)],[])+[R/f'v48_history_calibration_day{d}.npz' for d in (5,6)]+[R/'v48_history_calibrator.pkl']))
 frozen=dict(fit='Same day5 training users as V48; remaining day5 users then future day6',C=.01,solver='lbfgs',max_iter=1000,tol=1e-6,blend_weight=.25,experts=SOURCES,seed_ensemble=[0,1,2],selection='Fixed settings; require positive ordinary gain and positive adjusted interval against V48 on both splits',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_feature_source_sha256=hashlib.sha256(Path('v48_history_calibration.py').read_bytes()).hexdigest())
 (R/'v49_expert_calibration_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with (R/'v48_history_calibrator.pkl').open('rb') as f:old_model=pickle.load(f)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r')
 z,x,users,groups,names=features(5,t,h,ht,old_model['spline']);fit=np.isin(users,old_model['fit_users']);heldout=~fit
 assert not np.intersect1d(users[fit],users[heldout]).size and np.all(fit.reshape(-1,10)==fit[::10,None])
 scaler=StandardScaler();xf=scaler.fit_transform(x[fit]);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);source=np.bincount(groups[fit],minlength=5);source=source/source.sum();target=np.bincount(eg,minlength=5)/len(eg);weights=np.divide(target,source,out=np.zeros(5),where=source>0)[groups[fit]];weights/=weights.mean()
 model=LogisticRegression(C=.01,solver='lbfgs',max_iter=1000,tol=1e-6)
 with threadpool_limits(limits=1):model.fit(xf,z['y'][fit],sample_weight=weights)
 assert model.n_iter_[0]<1000
 with (R/'v49_expert_calibrator.pkl').open('wb') as f:pickle.dump(dict(model=model,scaler=scaler,spline=old_model['spline'],fit_users=old_model['fit_users'],expert_names=names),f)
 print(json.dumps(dict(stage='meta_fit_complete',features=x.shape[1],iterations=int(model.n_iter_[0]),elapsed_s=time.monotonic()-tic)),flush=True)
 reports=[]
 for day in (5,6):
  if day==6:z,x,users,groups,_=features(day,t,h,ht,old_model['spline']);heldout=np.ones(len(users),bool)
  prior=np.load(R/f'v48_history_calibration_day{day}.npz');assert np.array_equal(z['events'][heldout],prior['events']) and np.array_equal(z['y'][heldout],prior['y'])
  with threadpool_limits(limits=1):prob=model.predict_proba(scaler.transform(x[heldout]))[:,1]
  pred=rank(.75*rank(z['pred'][heldout])+.25*rank(prob));m=measure(prior,pred);m.update(day=day,weighted_auc=weighted_auc(prior,pred),baseline_weighted_auc=weighted_auc(prior,prior['pred']),elapsed_s=time.monotonic()-tic)
  reports.append(m);np.savez_compressed(R/f'v49_expert_calibration_day{day}.npz',y=prior['y'],events=prior['events'],pred=pred,calibrated=prob,baseline=prior['pred']);print(json.dumps(m),flush=True)
  if m['gain']<0 or m['weighted_interval'][0]<=0:break
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(m['gain']>=0 and m['weighted_interval'][0]>0 for m in reports),official_auc=None,csv_ready=False,note='Same reused V48 partition and dates. Day5 user holdout is not a second temporal fold. No model or blend settings selected on these validation labels.')
 (R/'v49_expert_calibration_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
