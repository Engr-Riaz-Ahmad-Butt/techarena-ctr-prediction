"""Exact paired event resampling for the fixed ALS39 weighted-AUC recipe."""
import hashlib,json
import numpy as np
from numba import njit,prange,set_num_threads
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R

@njit
def weighted_auc_sorted(y,scores,weight,event_index,counts):
 numerator=0.;positives=0.;negatives=0.;gp=0.;gn=0.;last=scores[0]
 for j in range(len(y)):
  if scores[j]!=last:
   numerator+=gp*(negatives+.5*gn);positives+=gp;negatives+=gn;gp=0.;gn=0.;last=scores[j]
  w=weight[j]*counts[event_index[j]]
  if y[j]>0:gp+=w
  else:gn+=w
 numerator+=gp*(negatives+.5*gn);positives+=gp;negatives+=gn
 return numerator/(positives*negatives)

@njit(parallel=True)
def differences(y0,p0,w0,e0,y1,p1,w1,e1,counts):
 out=np.empty(len(counts))
 for b in prange(len(counts)):
  out[b]=weighted_auc_sorted(y1,p1,w1,e1,counts[b])-weighted_auc_sorted(y0,p0,w0,e0,counts[b])
 return out

def main():
 set_num_threads(4);day=6;baseline=np.load(R/'v36_recent_context_day6.npz');candidate=np.load(R/'v39_regularized_als32_day6.npz')
 assert np.array_equal(baseline['events'],candidate['events']) and np.array_equal(baseline['y'],candidate['y'])
 h=np.load(R/'train_hist.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r')
 vg=np.digitize((h[baseline['events']]>=0).sum(1),[0,4,9,19],right=True);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
 w=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
 y=baseline['y'];events=np.arange(len(y))//10;nevents=len(y)//10
 def ordered(p):
  ix=np.argsort(p,kind='stable');return y[ix],p[ix],w[ix],events[ix]
 a=ordered(baseline['pred']);b=ordered(candidate['pred']);ones=np.ones(nevents,dtype=np.uint16)
 for arrays,p in [(a,baseline['pred']),(b,candidate['pred'])]:assert abs(weighted_auc_sorted(*arrays,ones)-roc_auc_score(y,p,sample_weight=w))<1e-10
 rng=np.random.default_rng(3901);deltas=[];probabilities=np.full(nevents,1/nevents)
 for block in range(10):
  counts=np.empty((100,nevents),dtype=np.uint16)
  for i in range(100):counts[i]=rng.multinomial(nevents,probabilities)
  deltas.extend(differences(*a,*b,counts).tolist());print(json.dumps(dict(stage='paired_event_bootstrap',completed=(block+1)*100,total=1000)),flush=True)
 interval=np.quantile(deltas,[.025,.975]);observed=float(roc_auc_score(y,candidate['pred'],sample_weight=w)-roc_auc_score(y,baseline['pred'],sample_weight=w))
 report=dict(day=6,replicates=1000,seed=3901,events=nevents,metric='evaluation_history_weighted_auc',observed_gain=observed,paired_event_percentile_interval=interval.tolist(),passes=bool(interval[0]>0),candidate_sha256=hashlib.sha256((R/'v39_regularized_als32_day6.npz').read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256((R/'v36_recent_context_day6.npz').read_bytes()).hexdigest(),note='Exact multinomial event resampling with fixed history weights. Conditional on chosen recipe; does not correct repeated selection or evaluation-distribution mismatch.')
 (R/'v39_regularized_als32_bootstrap.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
