"""Predeclared ALS64 capacity test against V39, without changing validation weights."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp15_als import prepare,augmented,solve_rows,split,cutoff_of,ni,nu
from v39_regularized_als32 import score,rank,weighted_auc
from v24_mass import measure

def fit(day):
 tic=time.monotonic();w,t,wt,tt,g,up,ip,confidence,exposures=prepare(day)
 with np.load(R/f'v39_regularized_als32_day{day}_checkpoint.npz') as old:
  assert int(old['cutoff'])==cutoff_of(day)
  original=old['V'].copy();bu=old['bu'].copy();bi=old['bi'].copy()
 rng=np.random.default_rng(4501+day);V=np.column_stack([original,rng.normal(size=(ni,32))*np.std(original)*.25]);U=np.zeros((nu,64))
 for iteration in range(1,7):
  x=augmented(V);bu,U=solve_rows(w.indptr,w.indices,w.data,t.data,V,bi,x.T@x,-x.T@(g+bi),confidence,g,up,latent_reg=4.)
  x=augmented(U);bi,V=solve_rows(wt.indptr,wt.indices,wt.data,tt.data,U,bu,x.T@(confidence[:,None]*x),-x.T@(confidence*(g+bu)),np.ones(ni),g,ip,latent_reg=4.)
  print(json.dumps(dict(day=day,iteration=iteration,seconds=time.monotonic()-tic)),flush=True)
 assert all(np.isfinite(a).all() for a in [U,V,bu,bi])
 np.savez_compressed(R/f'v45_capacity64_day{day}_checkpoint.npz',U=U,V=V,bu=bu,bi=bi,g=g,exposures=exposures,cutoff=cutoff_of(day),iteration=6)
 _,rows=split(day);z=np.load(R/f'v39_regularized_als32_day{day}.npz');assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
 raw=score(rows,U,V,bu,bi,g,exposures);p=rank(rank(z['pred'])+.05*(rank(raw)-rank(z['raw'])))
 metrics=measure(z,p);metrics.update(day=day,weighted_auc=weighted_auc(z,p),baseline_weighted_auc=weighted_auc(z,z['pred']))
 np.savez_compressed(R/f'v45_capacity64_day{day}.npz',y=z['y'],events=z['events'],raw=raw,pred=p)
 (R/f'v45_capacity64_day{day}_report.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True);return metrics

def main():
 set_num_threads(4);files=[R/f'v39_regularized_als32_day{d}{suffix}.npz' for d in (6,5) for suffix in ['', '_checkpoint']]
 frozen=dict(dim=64,iterations=6,latent_regularization=4.,bias_regularization=20.,initialization='V39 ALS32 plus 32 Gaussian columns at .25 scale',seed=4501,weight=.05,recipe='rank(rank(V39)+.05*(rank(ALS64)-rank(ALS32)))',threads=4,blas_threads=1,promotion='Positive ordinary gain and positive lower history-adjusted interval on both dates; no weight search',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v45_capacity64_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with threadpool_limits(limits=1):
  reports=[fit(6)]
  if reports[0]['gain']>=0 and reports[0]['weighted_interval'][0]>0:reports.append(fit(5))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(m['gain']>=0 and m['weighted_interval'][0]>0 for m in reports),official_auc=None,csv_ready=False,target_history_adjusted_auc=.82,note='Reused validation dates, conditional intervals exclude repeated-selection uncertainty. No official-score guarantee.')
 (R/'v45_capacity64_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
