"""Check posterior normal equations, empty histories and replay of V40 scores."""
import hashlib,json
from pathlib import Path
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp15_als import split,cutoff_of
from v40_recent_profiles import adapted_scores,rank
from v39_regularized_als32 import weighted_auc
from v24_mass import measure

def main():
 set_num_threads(2)
 with threadpool_limits(limits=1):
  frozen=json.loads((R/'v40_recent_profiles_frozen.json').read_text());assert hashlib.sha256(Path('v40_recent_profiles.py').read_bytes()).hexdigest()==frozen['source_sha256']
  for name,h in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
  report=json.loads((R/'v40_recent_profiles_report.json').read_text());selection=json.loads((R/'v40_recent_profiles_selection.json').read_text());out=[]
  histories=np.load(R/'train_hist.npy',mmap_mode='r');times=np.load(R/'train_times.npy',mmap_mode='r')
  for entry in report['reports']:
   day=entry['day'];z=np.load(R/f'v40_recent_profiles_day{day}.npz');ref=np.load(R/f'v39_regularized_als32_day{day}.npz');model=np.load(R/f'v39_regularized_als32_day{day}_checkpoint.npz')
   assert int(model['cutoff'])==cutoff_of(day) and np.array_equal(z['events'],ref['events']) and np.array_equal(z['y'],ref['y']) and entry['weight']==selection['weight']
   pred=rank(rank(ref['pred'])+entry['weight']*(rank(z['raw'])-rank(ref['raw'])));assert np.array_equal(pred,z['pred']) and np.isfinite(pred).all()
   m=measure(ref,pred);m.update(day=day,weighted_auc=weighted_auc(ref,pred));assert abs(m['auc']-entry['auc'])<1e-12 and abs(m['weighted_auc']-entry['weighted_auc'])<1e-12
   _,rows=split(day);assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
   n=(histories[rows[::10,6]]>=0).sum(1);selected=list(range(min(30,len(n))))
   for condition in [n==0,n<=4,n==20]:
    matches=np.flatnonzero(condition)
    if len(matches):selected.append(int(matches[0]))
   selected=np.unique(selected);indices=(selected[:,None]*10+np.arange(10)).ravel();sample=rows[indices]
   raw,details=adapted_scores(sample,model,histories,times,return_details=True);error=float(np.max(np.abs(raw-z['raw'][indices])));assert error<1e-10
   anchors,bias,newu,newb,h,ts,gram,rhs=details;dim=model['V'].shape[1];penalties=np.r_[20.,np.full(dim,4.)];maxeq=0.
   for row in range(len(h)):
    valid=np.flatnonzero(h[row]>=0);unique=[];seen=set()
    for q in valid:
     key=(int(h[row,q]),int(ts[row,q]))
     if key not in seen:unique.append(int(q));seen.add(key)
    if not unique:
     assert np.array_equal(newu[row],anchors[row]) and newb[row]==bias[row];continue
    ids=h[row,unique];age=(sample[row*10,2]-ts[row,unique])/86400.;w=np.exp(-np.log(2)*age/2);confidence=9*w.sum()/len(model['V'])
    X=np.column_stack([np.ones(len(ids)),model['V'][ids]]);A=confidence*gram+np.diag(penalties)+X.T@(w[:,None]*X)
    b=confidence*rhs+penalties*np.r_[bias[row],anchors[row]]+X.T@(w*(1-float(model['g'])-model['bi'][ids]))
    expected=np.linalg.solve(A,b);maxeq=max(maxeq,float(np.max(np.abs(expected-np.r_[newb[row],newu[row]]))))
   assert maxeq<1e-10;m.update(replay_max_error=error,equations_max_error=maxeq);out.append(m)
  accepted=len(out)==2 and all(m['gain']>=0 and m['weighted_interval'][0]>0 for m in out)
  result=dict(accepted=accepted,reports=out,hashes_unchanged=True,note='Conditional on selected recipe and history weights; repeated-validation uncertainty remains. No official score prediction.')
  (R/'v40_recent_profiles_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
