"""Matched candidate-set residual versus identical network with zero set inputs.

No identifier arithmetic: all candidate-set information uses causal learned
item embeddings. Whole events remain together; all fitting labels are earlier
than the validation cutoff. No change to the V53 baseline or upload threshold.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v36_recent_context import Context,pack,rank,DAY
from v24_mass import measure
from v57_sequence_residual import digest,weights

PREFIX='v58_slate_residual'

def save(name,value):
 (R/f'{PREFIX}_{name}.json').write_text(json.dumps(value,indent=2))

def prepare(rows,h,ht,f):
 parts=[]
 for off in range(0,len(rows),20000):
  a=rows[off:off+20000];p=pack(a,h,ht,f);p['length']=(h[a[::10,6]]>=0).sum(1).astype('float32');parts.append(p)
 out={key:np.concatenate([p[key] for p in parts]) for key in parts[0] if key!='n'};out['n']=sum(p['n'] for p in parts);return out

class SlateResidual(torch.nn.Module):
 def __init__(self,f,context,slate):
  super().__init__();self.context=context;self.slate=slate;d=f['V'].shape[1]
  for p in context.parameters():p.requires_grad_(False)
  self.register_buffer('V',torch.from_numpy(f['V']));self.register_buffer('H',torch.from_numpy(f['H']))
  self.net=torch.nn.Sequential(torch.nn.LayerNorm(9*d+6),torch.nn.Linear(9*d+6,64),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(64,1))
  torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,z,items,bucket,kind,base,length):
  v=self.V[items];hv=self.H[items];d=v.shape[-1];profile=z[:,:d];recent=z[:,d:]
  original=self.context(z,v,bucket,kind,base)
  local=torch.cat([profile[:,None,:].expand_as(v),recent[:,None,:].expand_as(v),v,profile[:,None,:]*v,recent[:,None,:]*v],-1)
  other_v=(v.sum(1,keepdim=True)-v)/9;other_h=(hv.sum(1,keepdim=True)-hv)/9
  std=v.std(1,unbiased=False)[:,None,:].expand_as(v)
  extra=torch.cat([other_v,other_h,std,v*other_v],-1)
  if not self.slate:extra=torch.zeros_like(extra)
  scalars=torch.cat([original[:,:,None],torch.log1p(length)[:,None,None].expand(-1,10,1)/3,torch.nn.functional.one_hot(bucket.long(),4).float()[:,None,:].expand(-1,10,-1)],-1)
  delta=self.net(torch.cat([local,extra,scalars],-1)).squeeze(-1)
  return original+delta,delta,original

def batches(data,f,order):
 for off in range(0,len(order),1024):
  ix=order[off:off+1024];z=torch.from_numpy(data['z'][ix]);items=torch.from_numpy(data['items'][ix]);v=torch.from_numpy(f['V'][data['items'][ix]])
  base=(-1.8+torch.from_numpy(data['bias'][ix])[:,None]+torch.from_numpy(f['bi'][data['items'][ix]])+(z[:,:v.shape[-1]][:,None,:]*v).sum(-1)).clamp(-30,30)
  inputs=(z,items,torch.from_numpy(data['bucket'][ix]),torch.from_numpy(data['kind'][ix]),base,torch.from_numpy(data['length'][ix]))
  yield inputs,torch.from_numpy(data['target'][ix])

def predict(model,data,f):
 model.eval();out=[];old=[]
 with torch.no_grad():
  for inputs,_ in batches(data,f,np.arange(data['n'])):
   logits,_,original=model(*inputs);zero=torch.zeros(len(logits),1)
   out.append(torch.softmax(torch.cat([logits,zero],1),1)[:,:10].numpy());old.append(torch.softmax(torch.cat([original,zero],1),1)[:,:10].numpy())
 return np.concatenate(out).ravel(),np.concatenate(old).ravel()

def equivariance_probe(model,data,f):
 inputs,_=next(batches(data,f,np.arange(min(16,data['n']))));z,items,bucket,kind,base,length=inputs;perm=torch.tensor([9,2,4,0,8,1,7,5,3,6]);model.eval()
 with torch.no_grad():
  a=model(*inputs)[0];b=model(z,items[:,perm],bucket,kind,base[:,perm],length)[0]
 error=float((b-a[:,perm]).abs().max());assert error<2e-6,error;return error

def fit(day,choice=None):
 tic=time.monotonic();t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min())
 with np.load(R/f'imp3_model_warm16_cut{day-2}.npz') as bundle:f={k:bundle[k].astype('float32') for k in ['U','V','H','bu','bi']}
 baseline=np.load(R/f'v53_causal_propensity_day{day}.npz');prior=np.load(R/f'v36_recent_context_day{day}.npz');assert np.array_equal(baseline['events'],prior['events']) and np.array_equal(baseline['y'],prior['y'])
 ix=baseline['events']*10+np.tile(np.arange(10),len(baseline['events'])//10);rows=np.asarray(t[ix]);assert np.array_equal(rows[:,5],baseline['y']) and rows[:,2].min()>=start+day*DAY;val=prepare(rows,h,ht,f);del rows,ix
 rows=np.asarray(t[(t[:,2]>=start+(day-2)*DAY)&(t[:,2]<start+day*DAY)]);last=int(rows[:,2].max());assert last<start+day*DAY;train=prepare(rows,h,ht,f);del rows
 base=rank(baseline['pred']);w=weights(baseline,h,eh);trials=[];initial=None
 for arm in ['control','slate']:
  torch.manual_seed(5801+day);rng=np.random.default_rng(5801+day)
  context=Context(f['V'].shape[1]);context.load_state_dict(torch.load(R/f'v36_recent_context_day{day}.pt',weights_only=True));model=SlateResidual(f,context,arm=='slate')
  if initial is None:initial={k:v.clone() for k,v in model.state_dict().items()}
  else:
   for k,v in model.state_dict().items():assert torch.equal(v,initial[k]),k
  old,_=predict(model,val,f);replay=float(np.max(np.abs(old-prior['raw'])));assert replay<2e-7;oldrank=rank(old)
  equiv=equivariance_probe(model,val,f);opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
  epochs=choice['epoch'] if choice else 4;checks=[epochs] if choice else [1,2,4]
  for epoch in range(1,epochs+1):
   model.train();total=0.;count=0
   for inputs,target in batches(train,f,rng.permutation(train['n'])):
    opt.zero_grad();logits,delta,_=model(*inputs);loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(logits),1)],1),target)+.01*delta.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(target);count+=len(target)
   print(json.dumps(dict(day=day,arm=arm,epoch=epoch,loss=total/count,elapsed_s=time.monotonic()-tic)),flush=True)
   if epoch not in checks:continue
   raw,_=predict(model,val,f);equiv=equivariance_probe(model,val,f);torch.save(model.state_dict(),R/f'{PREFIX}_{arm}_day{day}_epoch{epoch}.pt');np.save(R/f'{PREFIX}_{arm}_day{day}_epoch{epoch}_raw.npy',raw)
   for weight in ([choice['weight']] if choice else [.1,.25]):
    pred=rank(base+weight*(rank(raw)-oldrank));trial=dict(arm=arm,epoch=epoch,weight=weight,auc=float(roc_auc_score(baseline['y'],pred)),weighted_auc=float(roc_auc_score(baseline['y'],pred,sample_weight=w)),equivariance_error=equiv);trials.append(trial);print(json.dumps(dict(day=day,**trial)),flush=True)
   save(f'day{day}_progress',dict(trials=trials,csv_ready=False))
  del model,opt
 best=max([r for r in trials if r['arm']=='slate'],key=lambda r:r['weighted_auc']) if choice is None else next(r for r in trials if r['arm']=='slate')
 predictions={};raws={}
 for arm in ['control','slate']:
  raw=np.load(R/f'{PREFIX}_{arm}_day{day}_epoch{best["epoch"]}_raw.npy');raws[arm]=raw;predictions[arm]=rank(base+best['weight']*(rank(raw)-oldrank))
 metrics=measure(baseline,predictions['slate']);metrics.update(best);metrics.update(day=day,fit_last=last,cutoff=start+day*DAY,baseline_weighted_auc=float(roc_auc_score(baseline['y'],baseline['pred'],sample_weight=w)))
 vs_control=measure(dict(y=baseline['y'],events=baseline['events'],pred=predictions['control']),predictions['slate'])
 result=dict(metrics=metrics,vs_matched_control=vs_control,trials=trials)
 np.savez_compressed(R/f'{PREFIX}_day{day}.npz',y=baseline['y'],events=baseline['events'],pred=predictions['slate'],raw=raws['slate'],control=predictions['control'],control_raw=raws['control']);save(f'day{day}_report',result);print(json.dumps(result),flush=True);return result

def good(r):return r['metrics']['gain']>0 and r['metrics']['weighted_interval'][0]>0 and r['vs_matched_control']['weighted_interval'][0]>0

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 paths=[R/name for name in ['train.npy','train_hist.npy','train_times.npy','evaluation_hist.npy']]
 for day in [6,5]:paths.extend([R/f'imp3_model_warm16_cut{day-2}.npz',R/f'v36_recent_context_day{day}.pt',R/f'v36_recent_context_day{day}.npz',R/f'v53_causal_propensity_day{day}.npz'])
 frozen=dict(arms=['control','slate'],epochs=[1,2,4],weights=[.1,.25],lr=.001,weight_decay=.001,residual_penalty=.01,batch_size=1024,threads=4,seed=5801,hidden=64,selection='Select slate setting by main day6 adjusted AUC; require positive ordinary gain and positive adjusted lower interval vs V53 and matched control; freeze before day5. Reused dates, intervals exclude selection uncertainty.',inputs={str(p):digest(p) for p in paths},sources={name:digest(name) for name in [__file__,'v36_recent_context.py','v57_sequence_residual.py','v24_mass.py','tst2_candidate.py']})
 with (R/f'{PREFIX}_frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
 reports=[fit(6)];save('selection',reports[0]['metrics'])
 if good(reports[0]):reports.append(fit(5,reports[0]['metrics']))
 for path,sha in frozen['inputs'].items():assert digest(path)==sha
 for path,sha in frozen['sources'].items():assert digest(path)==sha
 result=dict(reports=reports,accepted=len(reports)==2 and all(good(r) for r in reports),csv_ready=False,official_auc=None,target_achieved=False,training_running=False);save('report',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
