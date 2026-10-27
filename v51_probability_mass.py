"""Add actual expert probabilities and query mass/entropy to V50's rank inputs."""
import numpy as np
from scipy.special import logit
from config import ARTIFACT_DIR as R
from v49_expert_calibration import features as prior_features,source_paths as prior_paths
from crossfit_calibration import run

def source_paths(day):return prior_paths(day)+[R/f'imp19_tree_raw_day{day}.npz']
def features(day,t,h,ht,spline):
 z,x,users,groups,names=prior_features(day,t,h,ht,spline);values=[];extra_names=[]
 for name,key in [('imp19_tree_raw','raw'),('v23_outside','raw'),('v36_recent_context','raw'),('v36_recent_context','old')]:
  with np.load(R/f'{name}_day{day}.npz') as a:
   assert np.array_equal(a['events'],z['events']) and np.array_equal(a['y'],z['y']);values.append(a[key].copy());extra_names.append(name+'_'+key+'_probability')
 seeds=[]
 for seed in (0,1,2):
  with np.load(R/f'v21_fastseeds_day{day}_s{seed}_ep1_raw.npz') as a:seeds.append(a['raw'].copy())
 values.append(np.mean(seeds,axis=0));extra_names.append('seed_mean_probability')
 one=np.eye(5)[groups];parts=[x]
 for p in values:
  assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  q=p.reshape(-1,10);mass=q.sum(1);conditional=q/np.maximum(mass[:,None],1e-12);entropy=-(conditional*np.log(np.maximum(conditional,1e-12))).sum(1);lp=np.log(np.maximum(p,1e-7))
  parts.extend([np.column_stack([lp,logit(np.clip(p,1e-7,1-1e-7)),conditional.ravel(),np.repeat(np.log(np.maximum(mass,1e-7)),10),np.repeat(entropy,10),np.repeat(conditional.max(1),10)]),one*lp[:,None]])
 out=np.column_stack(parts);assert np.isfinite(out).all();return z,out,users,groups,names+extra_names
if __name__=='__main__':run('v51_probability_mass',features,source_paths(5)+source_paths(6),'v50_crossfit_experts',['v51_probability_mass.py'])
