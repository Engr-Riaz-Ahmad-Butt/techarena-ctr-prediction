"""Exact paired user/event bootstrap for a saved candidate with baseline arrays."""
import argparse,hashlib,json
import numpy as np
from sklearn.metrics import roc_auc_score
from numba import set_num_threads
from config import ARTIFACT_DIR as R
from bootstrap_v39 import weighted_auc_sorted,differences

def main():
 p=argparse.ArgumentParser();p.add_argument('prefix');p.add_argument('--seed',type=int,required=True);args=p.parse_args();set_num_threads(4)
 h=np.load(R/'train_hist.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');t=np.load(R/'train.npy',mmap_mode='r');reports=[];source=json.loads((R/f'{args.prefix}_report.json').read_text())
 for report in source['reports']:
  day=report['day'];path=R/f'{args.prefix}_day{day}.npz';z=np.load(path);y=z['y'];p=z['pred'];old=z['baseline'];events=z['events'];n=len(y)//10;ei=np.arange(len(y))//10
  assert np.unique(t[events[::10]*10,0]).size==n and np.array_equal(events.reshape(-1,10),np.repeat(events[::10,None],10,axis=1))
  vg=np.digitize((h[events]>=0).sum(1),[0,4,9,19],right=True);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);w=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
  def ordered(values):
   ix=np.argsort(values,kind='stable');return y[ix],values[ix],w[ix],ei[ix]
  a,b=ordered(old),ordered(p);ones=np.ones(n,dtype=np.uint16)
  for arr,values in [(a,old),(b,p)]:assert abs(weighted_auc_sorted(*arr,ones)-roc_auc_score(y,values,sample_weight=w))<1e-10
  rng=np.random.default_rng(args.seed+day);deltas=[];prob=np.full(n,1/n)
  for block in range(10):
   counts=np.array([rng.multinomial(n,prob) for _ in range(100)],dtype=np.uint16);deltas.extend(differences(*a,*b,counts).tolist())
   if block in [4,9]:print(json.dumps(dict(day=day,completed=(block+1)*100)),flush=True)
  interval=np.quantile(deltas,[.025,.975]);reports.append(dict(day=day,events=n,replicates=1000,seed=args.seed+day,observed_gain=float(roc_auc_score(y,p,sample_weight=w)-roc_auc_score(y,old,sample_weight=w)),paired_event_interval=interval.tolist(),passes=bool(interval[0]>0),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 result=dict(reports=reports,passes=all(r['passes'] for r in reports),note='Fixed-model event bootstrap; one user per event in these cohorts. Does not include fitting or repeated-selection uncertainty, nor calendar distribution shift.')
 (R/f'{args.prefix}_bootstrap.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
