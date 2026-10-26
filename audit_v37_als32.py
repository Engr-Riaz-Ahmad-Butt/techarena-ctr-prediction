"""Audit ALS32 equations, factor cutoff, saved scores and weighted blend."""
import json,hashlib
from pathlib import Path
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp15_als import solve_rows,score,split,cutoff_of
from v24_mass import measure
from v37_als32 import weighted_auc,rank

def equations_check():
 rng=np.random.default_rng(37);V=rng.normal(size=(9,32));bias=rng.normal(size=9)*.02;g=.06
 pointers=np.array([0,3,5,5]);indices=np.array([0,2,7,1,8]);weights=np.array([1.,2.,1.,1.,3.]);targets=np.array([0.,1.,1.,0.,1.]);background=np.array([.02,.03,.01]);prior=np.array([.01,-.02,0.])
 X=np.column_stack([np.ones(9),V]);gram=X.T@X;rhs=-X.T@(g+bias)
 bu,U=solve_rows(pointers,indices,weights,targets,V,bias,gram,rhs,background,g,prior)
 expected=[]
 for row in range(3):
  ix=indices[pointers[row]:pointers[row+1]];w=weights[pointers[row]:pointers[row+1]];y=targets[pointers[row]:pointers[row+1]]
  a=background[row]*gram+np.diag([20.]+[1.]*32)
  b=background[row]*rhs;b[0]+=20*prior[row]
  a+=X[ix].T@(w[:,None]*X[ix]);b+=X[ix].T@(y-w*(g+bias[ix]))
  expected.append(np.linalg.solve(a,b))
 error=float(np.max(np.abs(np.column_stack([bu,U])-np.asarray(expected))));assert error<1e-10;return error

def main():
 set_num_threads(2)
 with threadpool_limits(limits=1):
  equation_error=equations_check()
  frozen=json.loads((R/'v37_als32_frozen.json').read_text())
  assert hashlib.sha256(Path('v37_als32.py').read_bytes()).hexdigest()==frozen['source_sha256']
  for name,h in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
  report=json.loads((R/'v37_als32_report.json').read_text());selection=json.loads((R/'v37_als32_selection.json').read_text());out=[]
  for entry in report['reports']:
   d=entry['day'];z=np.load(R/f'v37_als32_day{d}.npz');ref=np.load(R/f'v36_recent_context_day{d}.npz');prior=np.load(R/f'imp15_als_day{d}.npz')
   assert np.array_equal(z['y'],ref['y']) and np.array_equal(z['events'],ref['events'])
   assert entry['mode']==selection['mode'] and entry['weight']==selection['weight']
   base=rank(ref['pred']);rr=rank(z['raw']);old=rank(prior['raw']);w=entry['weight']
   p=rank(base+w*(rr-old)) if entry['mode']=='replace_component' else rank((1-w)*base+w*rr)
   assert np.max(np.abs(p-z['pred']))<1e-12
   m=measure(ref,p);m.update(day=d,weighted_auc=weighted_auc(ref,p),baseline_weighted_auc=weighted_auc(ref,ref['pred']))
   assert abs(m['auc']-entry['auc'])<1e-12 and abs(m['weighted_auc']-entry['weighted_auc'])<1e-12
   _,val=split(d);assert np.array_equal(val[:,6],z['events']) and np.array_equal(val[:,5],z['y'])
   f=np.load(R/f'v37_als32_day{d}_checkpoint.npz');assert int(f['cutoff'])==cutoff_of(d) and int(f['iteration'])==6 and f['V'].shape[1]==32
   raw=score(val[:1000],f['U'],f['V'],f['bu'],f['bi'],float(f['g']),f['exposures'])
   error=float(np.max(np.abs(raw-z['raw'][:1000])));assert error<1e-10;m['replay_max_error']=error;out.append(m)
  accepted=len(out)==2 and all(m['gain']>=0 and m['weighted_interval'][0]>0 for m in out)
  result=dict(accepted=accepted,reports=out,equation_max_error=equation_error,unchanged_hashes=True,note='Evaluation-history adjusted AUC is the selection metric. Reused local dates and conditional intervals; no official-score forecast.')
  (R/'v37_als32_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
