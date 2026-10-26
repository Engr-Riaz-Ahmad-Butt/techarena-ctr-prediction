"""Capacity experiment: exposure-aware ALS32, judged on evaluation-history AUC."""
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp15_als import prepare,augmented,score,solve_rows,train,hist,split,cutoff_of,ni,nu
from v24_mass import measure
DIM=32;ITERATIONS=6
rank=lambda x:rankdata(x,method='average')/(len(x)+1)

def weighted_auc(z,p):
 h=np.load(R/'train_hist.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r')
 vg=np.digitize((h[z['events']]>=0).sum(1),[0,4,9,19],right=True);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
 w=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
 return float(roc_auc_score(z['y'],p,sample_weight=w))

def fit(day,selected=None):
 started=time.monotonic();print(json.dumps(dict(stage='prepare_ALS32',day=day)),flush=True)
 w,t,wt,tt,g,up,ip,confidence,exposures=prepare(day)
 with (R/f'preference_stats_{cutoff_of(day)}.pkl').open('rb') as f:stats=pickle.load(f)
 V=stats[3][:,:DIM].astype(np.float64);del stats
 V/=np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-12)
 assert V.shape==(ni,DIM);U=np.zeros((nu,DIM));bu=up.copy();bi=ip.copy()
 for iteration in range(1,ITERATIONS+1):
  tic=time.monotonic();x=augmented(V)
  bu,U=solve_rows(w.indptr,w.indices,w.data,t.data,V,bi,x.T@x,-x.T@(g+bi),confidence,g,up)
  x=augmented(U)
  bi,V=solve_rows(wt.indptr,wt.indices,wt.data,tt.data,U,bu,x.T@(confidence[:,None]*x),-x.T@(confidence*(g+bu)),np.ones(ni),g,ip)
  assert all(np.isfinite(a).all() for a in [U,V,bu,bi])
  np.savez_compressed(R/f'v37_als32_day{day}_checkpoint.npz',U=U,V=V,bu=bu,bi=bi,g=g,exposures=exposures,cutoff=cutoff_of(day),iteration=iteration)
  print(json.dumps(dict(stage='ALS32',day=day,iteration=iteration,seconds=time.monotonic()-tic,elapsed_s=time.monotonic()-started)),flush=True)
 _,rows=split(day);z=np.load(R/f'v36_recent_context_day{day}.npz');old=np.load(R/f'imp15_als_day{day}.npz')
 assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
 assert np.array_equal(old['events'],z['events']) and np.array_equal(old['y'],z['y'])
 raw=score(rows,U,V,bu,bi,g,exposures);assert np.isfinite(raw).all()
 reference=rank(z['pred']);rr=rank(raw);prior=rank(old['raw']);trials=[]
 recipes=[(selected['mode'],selected['weight'])] if selected else [(mode,weight) for mode in ['replace_component','blend'] for weight in [.05,.1,.2]]
 for mode,weight in recipes:
  p=rank(reference+weight*(rr-prior)) if mode=='replace_component' else rank((1-weight)*reference+weight*rr)
  auc=float(roc_auc_score(z['y'],p));wa=weighted_auc(z,p);trials.append(dict(mode=mode,weight=weight,auc=auc,weighted_auc=wa))
 chosen=max(trials,key=lambda a:a['weighted_auc'])
 mode=chosen['mode'];weight=chosen['weight'];p=rank(reference+weight*(rr-prior)) if mode=='replace_component' else rank((1-weight)*reference+weight*rr)
 metrics=measure(z,p);metrics.update(day=day,mode=mode,weight=weight,weighted_auc=chosen['weighted_auc'],baseline_weighted_auc=weighted_auc(z,z['pred']),raw_auc=float(roc_auc_score(z['y'],raw)),old_raw_auc=float(roc_auc_score(z['y'],old['raw'])),cutoff=cutoff_of(day),elapsed_s=time.monotonic()-started)
 np.savez_compressed(R/f'v37_als32_day{day}.npz',y=z['y'],events=z['events'],pred=p,raw=raw)
 (R/f'v37_als32_day{day}_report.json').write_text(json.dumps(dict(metrics=metrics,trials=trials),indent=2));print(json.dumps(metrics),flush=True)
 return metrics

def main():
 set_num_threads(4)
 files=[R/f'preference_stats_{cutoff_of(d)}.pkl' for d in (6,5)]+[R/f'v36_recent_context_day{d}.npz' for d in (6,5)]+[R/f'imp15_als_day{d}.npz' for d in (6,5)]
 frozen=dict(dim=DIM,iterations=ITERATIONS,latent_regularization=1.,bias_regularization=20.,threads=4,weight_grid=[.05,.1,.2],modes=['replace_component','blend'],selection_metric='evaluation_history_weighted_auc',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v37_als32_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with threadpool_limits(limits=1):
  first=fit(6);reports=[first];(R/'v37_als32_selection.json').open('x').write(json.dumps(first,indent=2))
  if first['weighted_interval'][0]>0 and first['gain']>=0:reports.append(fit(5,first))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 accepted=len(reports)==2 and all(a['gain']>=0 and a['weighted_interval'][0]>0 for a in reports)
 result=dict(reports=reports,accepted=accepted,evaluation_csv_ready=False,note='Reused local dates; intervals conditional on settings and history weighting, excluding selection uncertainty. Official V35 score is 0.816623. Local history weighting is an imperfect proxy, not an official forecast.')
 (R/'v37_als32_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
