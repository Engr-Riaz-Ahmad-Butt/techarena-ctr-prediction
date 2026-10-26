"""Fresh first-future-event benchmark with calendar/history distribution diagnostics.

This compares two expert components only, not the full V39 ensemble or leaderboard.
"""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v36_recent_context import Context as Linear,pack as linear_pack,batches, predict as linear_predict,DAY
from v41_neural_context import Context as Neural,pack as neural_pack,predict as neural_predict
from tst2_candidate import influence,weighted_influence

def main():
 torch.set_num_threads(4);start_time=time.monotonic()
 frozen=dict(cutoff_day=4,factor_cutoff_day=2,fit_days=[2,3],holdout='each user first event at/after day4, through end of training',epochs=3,linear_lr=.003,linear_penalty=.001,neural_lr=.001,neural_weight_decay=1e-4,neural_residual_penalty=.01,neural_dropout=.1,neural_hidden=[128,64],seed=4101,selection_metric='joint_history_daytype_adjusted_AUC',scope='Expert-component comparison only; no full-ensemble or official-score claim.',factor_sha256=hashlib.sha256((R/'imp3_model_warm16_cut2.npz').read_bytes()).hexdigest())
 (R/'v42_calendar_holdout_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min())
 fit=np.asarray(t[(t[:,2]>=start+2*DAY)&(t[:,2]<start+4*DAY)]);future=np.asarray(t[t[:,2]>=start+4*DAY]);_,first=np.unique(future[::10,0],return_index=True);rows=future[(first[:,None]*10+np.arange(10)).ravel()]
 assert len(np.unique(rows[::10,0]))==len(rows)//10 and fit[:,2].max()<start+4*DAY<=rows[:,2].min()
 with np.load(R/'imp3_model_warm16_cut2.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 outputs={}
 for name,cls,packer in [('linear',Linear,linear_pack),('neural',Neural,neural_pack)]:
  torch.manual_seed(4101);td=packer(fit,h,ht,factors);vd=packer(rows,h,ht,factors);model=cls(factors['V'].shape[1]);lr=.003 if name=='linear' else .001;wd=0. if name=='linear' else 1e-4;opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd);rng=np.random.default_rng(4105)
  for epoch in range(1,4):
   model.train()
   for z,v,b,k,base,target in batches(td,factors,rng.permutation(td['n'])):
    opt.zero_grad();logits=model(z,v,b,k,base);loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(z),1)],1),target)
    if name=='linear':penalty=.001*(model.bucket.square().sum()+model.kind.square().sum()+model.recent.square().sum())
    else:penalty=.01*(logits-base).square().mean()
    (loss+penalty).backward();opt.step()
   print(json.dumps(dict(stage='fresh_calendar_holdout',expert=name,epoch=epoch,elapsed_s=time.monotonic()-start_time)),flush=True)
  torch.save(model.state_dict(),R/f'v42_calendar_{name}.pt')
  outputs[name]=linear_predict(model,vd,factors)[0] if name=='linear' else neural_predict(model,vd,factors)
 y=rows[:,5].astype(float);vg=np.digitize((h[rows[:,6]]>=0).sum(1),[0,4,9,19],right=True)*2+rows[:,4]
 eg=np.digitize((eh[e[::10,6]]>=0).sum(1),[0,4,9,19],right=True)*2+e[::10,4]
 source=np.bincount(vg,minlength=10)/len(vg);target=np.bincount(eg,minlength=10)/len(eg);ratio=np.divide(target,source,out=np.zeros(10),where=source>0);w=ratio[vg]
 a=outputs['linear'];b=outputs['neural'];ordinary_gain=float(roc_auc_score(y,b)-roc_auc_score(y,a));gain=float(roc_auc_score(y,b,sample_weight=w)-roc_auc_score(y,a,sample_weight=w));se=float(np.linalg.norm(weighted_influence(y,b,w)-weighted_influence(y,a,w)));ordinary_se=float(np.linalg.norm(influence(y,b)-influence(y,a)))
 np.savez_compressed(R/'v42_calendar_holdout.npz',y=y,events=rows[:,6],linear=a,neural=b,weights=w)
 report=dict(scope=frozen['scope'],events=len(rows)//10,cutoff=start+4*DAY,heldout_last_timestamp=int(rows[:,2].max()),heldout_weekday_share=float(np.mean(rows[::10,4]==0)),evaluation_weekday_share=float(np.mean(e[::10,4]==0)),common_support_target_mass=float(target[source>0].sum()),joint_source_shares=source.tolist(),joint_target_shares=target.tolist(),ordinary_auc={'linear':float(roc_auc_score(y,a)),'neural':float(roc_auc_score(y,b))},joint_adjusted_auc={'linear':float(roc_auc_score(y,a,sample_weight=w)),'neural':float(roc_auc_score(y,b,sample_weight=w))},joint_adjusted_gain=gain,conditional_interval=[gain-1.96*se,gain+1.96*se],ordinary_gain=ordinary_gain,ordinary_interval=[ordinary_gain-1.96*ordinary_se,ordinary_gain+1.96*ordinary_se],note='New cutoff for this expert comparison. Calendar/history weighting does not fix all covariate shifts; follow-up recipe choices would reuse this holdout. Component AUC must not be presented as full-model or official AUC.')
 (R/'v42_calendar_holdout_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
