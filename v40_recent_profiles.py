"""Test recent-click posterior adaptation for known and cold users alike."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from numba import njit,prange,set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp15_als import split,cutoff_of
from v39_regularized_als32 import score as baseline_score,weighted_auc,rank
from v24_mass import measure

@njit(parallel=True)
def update_profiles(anchors,biases,histories,stamps,events,V,bi,g,gram,rhs):
 n,dim=anchors.shape;new_u=anchors.copy();new_b=biases.copy();ni=len(V)
 for row in prange(n):
  total=0.
  for q in range(20):
   item=histories[row,q]
   if item<0:continue
   duplicate=False
   for previous in range(q):
    if histories[row,previous]==item and stamps[row,previous]==stamps[row,q]:duplicate=True
   if not duplicate:total+=np.exp(-np.log(2.)*(events[row]-stamps[row,q])/(2*86400.))
  if total==0:continue
  confidence=9.*total/ni
  a=confidence*gram.copy();b=confidence*rhs.copy()
  a[0,0]+=20.;b[0]+=20.*biases[row]
  for j in range(dim):a[j+1,j+1]+=4.;b[j+1]+=4.*anchors[row,j]
  vector=np.empty(dim+1);vector[0]=1.
  for q in range(20):
   item=histories[row,q]
   if item<0:continue
   duplicate=False
   for previous in range(q):
    if histories[row,previous]==item and stamps[row,previous]==stamps[row,q]:duplicate=True
   if duplicate:continue
   weight=np.exp(-np.log(2.)*(events[row]-stamps[row,q])/(2*86400.))
   for j in range(dim):vector[j+1]=V[item,j]
   for j in range(dim+1):
    b[j]+=weight*(1.-g-bi[item])*vector[j]
    for k in range(dim+1):a[j,k]+=weight*vector[j]*vector[k]
  result=np.linalg.solve(a,b);new_b[row]=result[0];new_u[row]=result[1:]
 return new_b,new_u

def adapted_scores(rows,model,histories,history_times,return_details=False):
 head=rows[::10];h=np.asarray(histories[head[:,6]]);ts=np.asarray(history_times[head[:,6]])
 assert np.all((ts<head[:,2,None])|(h<0))
 u,b=baseline_score(rows,model['U'],model['V'],model['bu'],model['bi'],float(model['g']),model['exposures'],return_profile=True,histories=histories,history_times=history_times)
 V=model['V'];bi=model['bi'];g=float(model['g']);x=np.column_stack([np.ones(len(V)),V]);gram=x.T@x;rhs=-x.T@(g+bi)
 new_b,new_u=update_profiles(u,b,h,ts,head[:,2],V,bi,g,gram,rhs)
 items=rows[:,1].reshape(-1,10);raw=(g+new_b[:,None]+bi[items]+np.einsum('ed,ecd->ec',new_u,V[items])).ravel()
 assert np.isfinite(raw).all()
 if return_details:return raw,(u,b,new_u,new_b,h,ts,gram,rhs)
 return raw

def run(day,selection=None):
 started=time.monotonic();print(json.dumps(dict(stage='recent_profile_adaptation',day=day)),flush=True)
 _,rows=split(day);ref=np.load(R/f'v39_regularized_als32_day{day}.npz');model=np.load(R/f'v39_regularized_als32_day{day}_checkpoint.npz')
 assert int(model['cutoff'])==cutoff_of(day) and np.array_equal(rows[:,6],ref['events']) and np.array_equal(rows[:,5],ref['y'])
 histories=np.load(R/'train_hist.npy',mmap_mode='r');times=np.load(R/'train_times.npy',mmap_mode='r');raw=adapted_scores(rows,model,histories,times)
 recipes=[selection['weight']] if selection is not None else [.05,.1,.2]
 base=rank(ref['pred']);delta=rank(raw)-rank(ref['raw']);trials=[]
 for w in recipes:
  p=rank(base+w*delta);m=measure(ref,p);m.update(day=day,weight=w,weighted_auc=weighted_auc(ref,p),baseline_weighted_auc=weighted_auc(ref,ref['pred']))
  trials.append(m)
 chosen=max(trials,key=lambda a:a['weighted_auc']);w=chosen['weight'];pred=rank(base+w*delta)
 np.savez_compressed(R/f'v40_recent_profiles_day{day}.npz',y=ref['y'],events=ref['events'],raw=raw,pred=pred)
 chosen['elapsed_s']=time.monotonic()-started
 (R/f'v40_recent_profiles_day{day}_report.json').write_text(json.dumps(dict(selected=chosen,trials=trials),indent=2));print(json.dumps(chosen),flush=True)
 return chosen

def main():
 set_num_threads(4)
 inputs=[R/f'v39_regularized_als32_day{d}{suffix}' for d in (6,5) for suffix in ('.npz','_checkpoint.npz')]
 frozen=dict(history_half_life_days=2,latent_anchor_penalty=4.,bias_anchor_penalty=20.,background_negatives_per_click=9.,grid=[.05,.1,.2],threads=4,inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v40_recent_profiles_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with threadpool_limits(limits=1):
  first=run(6);reports=[first];(R/'v40_recent_profiles_selection.json').open('x').write(json.dumps(first,indent=2))
  if first['gain']>=0 and first['weighted_interval'][0]>0:reports.append(run(5,first))
 for p in inputs:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 accepted=len(reports)==2 and all(v['gain']>=0 and v['weighted_interval'][0]>0 for v in reports)
 result=dict(reports=reports,accepted=accepted,evaluation_csv_ready=False,note='Recent observations are deliberately reweighted relative to frozen anchors, including known-user histories. Reused local dates; conditional intervals exclude selection uncertainty. Official V35 0.816623; no newer official score confirmed.')
 (R/'v40_recent_profiles_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
