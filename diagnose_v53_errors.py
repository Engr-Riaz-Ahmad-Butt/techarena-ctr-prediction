"""Labelled validation error decomposition, never a submission predictor."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R

def auc(y,p,w):
 return float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None

def main():
 h=np.load(R/'train_hist.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');target=np.bincount(np.digitize((eh>=0).sum(1),[0,4,9,19],right=True),minlength=5)/len(eh);out=[]
 for day in [6,5]:
  z=np.load(R/f'v53_causal_propensity_day{day}.npz');y=z['y'];p=z['pred'];events=z['events'];g=np.digitize((h[events]>=0).sum(1),[0,4,9,19],right=True);source=np.bincount(g,minlength=5)/len(g);w=(target/source)[g]
  ey=y.reshape(-1,10).sum(1);assert (ey<=1).all();clicked=np.repeat(ey>0,10);positive=y>0;groups=[];reconstructed=0.
  for name,negative in [('negative_in_clicked_event',~positive&clicked),('negative_in_no_click_event',~clicked)]:
   keep=positive|negative;value=auc(y[keep],p[keep],w[keep]);share=float(w[negative].sum()/w[~positive].sum());reconstructed+=share*value
   groups.append(dict(name=name,negative_weight_share=share,auc_against_all_positives=value,weighted_global_error_contribution=share*(1-value)))
  adjusted=auc(y,p,w);assert abs(reconstructed-adjusted)<1e-12
  history=[]
  for k in range(5):
   m=g==k;history.append(dict(group=k,rows=int(m.sum()),target_share=float(target[k]),local_share=float(source[k]),auc=auc(y[m],p[m],w[m]),click_rate=float(y[m].mean())))
  ep=p.reshape(-1,10);ew=w[::10];cy=y.reshape(-1,10);correct=(ep[ey>0]*cy[ey>0]).sum(1);other=ep[ey>0];wins=((correct[:,None]>other)+.5*(correct[:,None]==other)).sum(1)-.5;within=float(np.average(wins/9,weights=ew[ey>0]));top1=float(np.average(np.argmax(ep[ey>0],1)==np.argmax(cy[ey>0],1),weights=ew[ey>0]))
  score=dict(day=day,ordinary_auc=auc(y,p,np.ones(len(y))),history_adjusted_auc=adjusted,negative_source_decomposition=groups,history_groups=history,weighted_click_event_fraction=float(np.average(ey,weights=ew)),event_auc_using_mean_rank=auc(ey,ep.mean(1),ew),event_auc_using_max_rank=auc(ey,ep.max(1),ew),conditional_within_event_pair_auc=within,conditional_top1=top1)
  out.append(score);print(json.dumps(score),flush=True)
 result=dict(reports=out,note='Diagnostics use validation labels. Group AUCs and error shares are descriptive, not causal attribution, achievable score bounds, new candidates, or official forecasts.')
 (R/'v53_error_decomposition.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
