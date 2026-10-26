"""Fit a small conditional calibrator on causal day5 OOF scores.

Partition users before fitting; check remaining day5 users, then all of day6.
No day6 or validation-partition labels are available to the calibrator fit.
These dates were previously reused for model selection; they are not untouched.
"""
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

def features(z,t,h,ht,spline):
 events=z['events'];row_indices=events*10+np.tile(np.arange(10),len(events)//10);rows=np.asarray(t[row_indices]);assert np.array_equal(rows[:,6],events)
 length=(h[events]>=0).sum(1);groups=np.digitize(length,[0,4,9,19],right=True);one=np.eye(5)[groups]
 score=np.asarray(z['pred']);basis=spline.transform(score[:,None]);parts=[basis,one,(one[:,:,None]*basis[:,None,:]).reshape(len(score),-1)]
 scores=score.reshape(-1,10);mass=np.column_stack([scores.mean(1),scores.max(1),scores.std(1)])
 last=np.where(h[events]>=0,ht[events],-1).max(1);age=np.where(length>0,(rows[:,2]-last)/86400,30.)
 parts.extend([np.repeat(mass,10,axis=0),np.eye(4)[rows[:,3]],(length/20)[:,None],np.log1p(np.maximum(age,0))[:,None]])
 x=np.column_stack(parts);assert np.isfinite(x).all();return x,rows[:,0],groups

def main():
 tic=time.monotonic();files=[R/f'v39_regularized_als32_day{d}.npz' for d in (5,6)]
 frozen=dict(calibration_fit='Random half of day5 users; all parent scores are causal OOF for day5',seed=4801,knots=[0,.2,.4,.6,.8,1],degree=3,C=.1,blend_weight=.25,solver='lbfgs',max_iter=500,tol=1e-7,selection='No hyperparameter search. Require ordinary gain and positive adjusted interval on held-out day5 users, then day6.',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 manifest=R/'v48_history_calibration_frozen.json'
 if manifest.exists():
  prior=json.loads(manifest.read_text());assert hashlib.sha256((R/'v48_original_source.py').read_bytes()).hexdigest()==prior['source_sha256']
  assert {k:v for k,v in prior.items() if k!='source_sha256'}=={k:v for k,v in frozen.items() if k!='source_sha256'}
  (R/'v48_source_repair.json').write_text(json.dumps(dict(original_source_sha256=prior['source_sha256'],repaired_source_sha256=frozen['source_sha256'],change='Correct event-to-candidate row indexing; initial assertion stopped before model fitting. No settings or partition changed.'),indent=2))
 else:manifest.open('x').write(json.dumps(frozen,indent=2))
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r')
 spline=SplineTransformer(knots=np.array(frozen['knots'])[:,None],degree=3,include_bias=False,extrapolation='linear');spline.fit(np.array([[0.],[1.]]))
 z=np.load(files[0]);x,users,groups=features(z,t,h,ht,spline);unique=np.unique(users);rng=np.random.default_rng(4801);fit_users=rng.permutation(unique)[:len(unique)//2];fit=np.isin(users,fit_users);heldout=~fit
 assert not np.intersect1d(users[fit],users[heldout]).size and np.all(fit.reshape(-1,10)==fit[::10,None])
 scaler=StandardScaler();xf=scaler.fit_transform(x[fit]);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);source=np.bincount(groups[fit],minlength=5);source=source/source.sum();target=np.bincount(eg,minlength=5)/len(eg);weight=np.divide(target,source,out=np.zeros(5),where=source>0)[groups[fit]];weight/=weight.mean()
 model=LogisticRegression(C=.1,solver='lbfgs',max_iter=500,tol=1e-7)
 with threadpool_limits(limits=1):model.fit(xf,z['y'][fit],sample_weight=weight)
 assert model.n_iter_[0]<500
 with (R/'v48_history_calibrator.pkl').open('wb') as f:pickle.dump(dict(model=model,scaler=scaler,spline=spline,fit_users=fit_users),f)
 reports=[]
 for day in (5,6):
  if day==6:z=np.load(files[1]);x,users,groups=features(z,t,h,ht,spline);heldout=np.ones(len(users),bool)
  with threadpool_limits(limits=1):prob=model.predict_proba(scaler.transform(x))[:,1]
  base={k:z[k][heldout] for k in ['y','events','pred']};cal=prob[heldout];p=rank(.75*rank(base['pred'])+.25*rank(cal));m=measure(base,p)
  m.update(day=day,evaluation_users=len(np.unique(users[heldout])),calibrator_fit_users=len(fit_users),weighted_auc=weighted_auc(base,p),baseline_weighted_auc=weighted_auc(base,base['pred']),calibrator_iterations=int(model.n_iter_[0]),elapsed_s=time.monotonic()-tic)
  reports.append(m);np.savez_compressed(R/f'v48_history_calibration_day{day}.npz',y=base['y'],events=base['events'],pred=p,calibrated=cal,baseline=base['pred'])
  print(json.dumps(m),flush=True)
  if m['gain']<0 or m['weighted_interval'][0]<=0:break
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(m['gain']>=0 and m['weighted_interval'][0]>0 for m in reports),official_auc=None,csv_ready=False,note='Calibrator trained only on partition of prior-date OOF labels. Reused dates; no official-score claim. Day5 subset score cannot be directly compared with full-day5 headline.')
 (R/'v48_history_calibration_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
