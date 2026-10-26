"""Replay calendar benchmark with every validation history frozen at holdout cutoff."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v36_recent_context import Context as Linear,pack as lp,predict as predict_linear,DAY
from v41_neural_context import Context as Neural,pack as npack,predict as predict_neural

def main():
 torch.set_num_threads(4)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');cut=int(t[:,2].min())+4*DAY
 future=np.asarray(t[t[:,2]>=cut]);_,ix=np.unique(future[::10,0],return_index=True);rows=future[(ix[:,None]*10+np.arange(10)).ravel()];head=rows[::10]
 saved=np.load(R/'v42_calendar_holdout.npz');assert np.array_equal(saved['events'],rows[:,6])
 # Only remove future clicks; retain original order and timestamps for valid clicks.
 histories=np.asarray(h).copy();remove=(histories>=0)&(ht>=cut);removed=int(remove[head[:,6]].sum());histories[remove]=-1
 with np.load(R/'imp3_model_warm16_cut2.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 outputs={}
 for name,cls,packer in [('linear',Linear,lp),('neural',Neural,npack)]:
  model=cls(16);model.load_state_dict(torch.load(R/f'v42_calendar_{name}.pt',weights_only=True));data=packer(rows,histories,ht,factors)
  outputs[name]=predict_linear(model,data,factors)[0] if name=='linear' else predict_neural(model,data,factors)
 vg=np.digitize((histories[rows[:,6]]>=0).sum(1),[0,4,9,19],right=True)*2+rows[:,4]
 eg=np.digitize((eh[e[::10,6]]>=0).sum(1),[0,4,9,19],right=True)*2+e[::10,4]
 source=np.bincount(vg,minlength=10)/len(vg);target=np.bincount(eg,minlength=10)/len(eg);w=np.divide(target,source,out=np.zeros(10),where=source>0)[vg]
 report=dict(status='PASS',removed_post_cutoff_history_entries=removed,all_histories_before_cutoff=True,ordinary_auc={k:float(roc_auc_score(rows[:,5],v)) for k,v in outputs.items()},joint_adjusted_auc={k:float(roc_auc_score(rows[:,5],v,sample_weight=w)) for k,v in outputs.items()},common_support_target_mass=float(target[source>0].sum()),scope='Expert components only, not full-ensemble or official scores',supersedes='V42 original allowed post-cutoff history for delayed first events; use these frozen-history results instead.')
 np.savez_compressed(R/'v42_frozen_history.npz',y=rows[:,5],events=rows[:,6],weights=w,**outputs)
 (R/'v42_frozen_history_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
