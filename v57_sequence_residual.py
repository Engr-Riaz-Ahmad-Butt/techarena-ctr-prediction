"""Bounded-memory right-padded, no-click-aware GRU residual experiment.

Day6 chooses among six preregistered settings; day5 confirms unchanged.
Repeatedly reused dates do not provide an untouched generalization estimate.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
 os.environ[name]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from history_sequence import pack_event_histories
from v36_recent_context import Context,pack,rank
from v24_mass import measure

PREFIX='v57_sequence_residual'
DAY=86400

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(name,value):
 (R/f'{PREFIX}_{name}.json').write_text(json.dumps(value,indent=2))

def prepare(rows,h,ht,f):
 parts=[]
 for offset in range(0,len(rows),20000):
  block=rows[offset:offset+20000];p=pack(block,h,ht,f);head=block[::10]
  tokens,lengths,real=pack_event_histories(head[:,6],h,len(f['H']))
  valid=tokens<len(f['H']);gaps=np.maximum((head[:,2,None]-ht[head[:,6]])/DAY,0)
  p.update(tokens=tokens,lengths=lengths,real=real,gaps=(np.log1p(gaps)*valid).astype('float32'))
  parts.append(p)
 out={k:np.concatenate([p[k] for p in parts]) for k in parts[0] if k!='n'}
 out['n']=sum(p['n'] for p in parts)
 return out

class SequenceResidual(torch.nn.Module):
 def __init__(self,f,context):
  super().__init__();dim=f['H'].shape[1]
  self.context=context
  for p in self.context.parameters():p.requires_grad_(False)
  self.register_buffer('H',torch.from_numpy(np.r_[f['H'],np.zeros((1,dim),np.float32)]))
  self.register_buffer('V',torch.from_numpy(np.r_[f['V'],np.zeros((1,dim),np.float32)]))
  self.project=torch.nn.Linear(2*dim+1,32);self.gru=torch.nn.GRU(32,32,batch_first=True)
  self.head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(32+5,dim+1))
  torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
 def forward(self,z,items,bucket,kind,base,tokens,lengths,real,gaps):
  v=self.V[items];original=self.context(z,v,bucket,kind,base)
  x=torch.cat([self.H[tokens],self.V[tokens],gaps[:,:,None]],-1)
  x=torch.tanh(self.project(x))
  packed=torch.nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
  _,last=self.gru(packed)
  extras=torch.cat([torch.nn.functional.one_hot(bucket.long(),4).float(),torch.log1p(real.float())[:,None]],1)
  delta=self.head(torch.cat([last[-1],extras],1))*(real>0)[:,None]
  correction=(delta[:,:-1,None].transpose(1,2)*v).sum(-1)+delta[:,-1,None]
  return original+correction,correction,original

def batches(data,f,order):
 for offset in range(0,len(order),512):
  ix=order[offset:offset+512];z=torch.from_numpy(data['z'][ix]);items=torch.from_numpy(data['items'][ix]);v=torch.from_numpy(f['V'][data['items'][ix]])
  base=(-1.8+torch.from_numpy(data['bias'][ix])[:,None]+torch.from_numpy(f['bi'][data['items'][ix]])+(z[:,:v.shape[-1]][:,None,:]*v).sum(-1)).clamp(-30,30)
  inputs=(z,items,torch.from_numpy(data['bucket'][ix]),torch.from_numpy(data['kind'][ix]),base)
  inputs+=tuple(torch.from_numpy(data[k][ix]) for k in ['tokens','lengths','real','gaps'])
  yield inputs,torch.from_numpy(data['target'][ix])

def predict(model,data,f):
 pred=[];old=[];model.eval()
 with torch.no_grad():
  for inputs,_ in batches(data,f,np.arange(data['n'])):
   logits,_,original=model(*inputs);zeros=torch.zeros(len(logits),1)
   pred.append(torch.softmax(torch.cat([logits,zeros],1),1)[:,:10].numpy())
   old.append(torch.softmax(torch.cat([original,zeros],1),1)[:,:10].numpy())
 return np.concatenate(pred).ravel(),np.concatenate(old).ravel()

def weights(z,h,eh):
 g=np.digitize((h[z['events']]>=0).sum(1),[0,4,9,19],right=True)
 e=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
 return ((np.bincount(e,minlength=5)/len(e))/(np.bincount(g,minlength=5)/len(g)))[g]

def fit(day,choice=None):
 tic=time.monotonic();torch.manual_seed(5701+day);rng=np.random.default_rng(5701+day)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min())
 with np.load(R/f'imp3_model_warm16_cut{day-2}.npz') as z:f={k:z[k].astype('float32') for k in ['U','V','H','bu','bi']}
 baseline=np.load(R/f'v53_causal_propensity_day{day}.npz');prior=np.load(R/f'v36_recent_context_day{day}.npz')
 assert np.array_equal(baseline['events'],prior['events']) and np.array_equal(baseline['y'],prior['y'])
 ix=baseline['events']*10+np.tile(np.arange(10),len(baseline['events'])//10);rows=np.asarray(t[ix])
 assert np.array_equal(rows[:,5],baseline['y']);assert rows[:,2].min()>=start+day*DAY
 val=prepare(rows,h,ht,f);del rows,ix
 rows=np.asarray(t[(t[:,2]>=start+(day-2)*DAY)&(t[:,2]<start+day*DAY)])
 assert rows[:,2].max()<start+day*DAY
 train=prepare(rows,h,ht,f);del rows
 context=Context(f['V'].shape[1]);context.load_state_dict(torch.load(R/f'v36_recent_context_day{day}.pt',weights_only=True));model=SequenceResidual(f,context)
 raw,old=predict(model,val,f);replay_error=float(np.max(np.abs(old-prior['raw'])))
 assert replay_error<2e-7 and np.array_equal(raw,old),replay_error
 print(json.dumps(dict(day=day,stage='prepared',train_events=train['n'],validation_events=val['n'],zero_residual_replay_error=replay_error)),flush=True)
 optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-4)
 wts=weights(baseline,h,eh);base=rank(baseline['pred']);oldrank=rank(old);trials=[]
 epochs=choice['epoch'] if choice else 3
 for epoch in range(1,epochs+1):
  model.train();loss_sum=0.;count=0
  for inputs,target in batches(train,f,rng.permutation(train['n'])):
   optimizer.zero_grad();logits,delta,_=model(*inputs)
   loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(logits),1)],1),target)+.01*delta.square().mean()
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();loss_sum+=float(loss.detach())*len(target);count+=len(target)
  torch.save(model.state_dict(),R/f'{PREFIX}_day{day}_epoch{epoch}.pt')
  if not choice or epoch==epochs:
   raw,old=predict(model,val,f)
   for weight in ([choice['weight']] if choice else [.05,.1]):
    pred=rank(base+weight*(rank(raw)-oldrank));m=dict(epoch=epoch,weight=weight,auc=float(roc_auc_score(baseline['y'],pred)),weighted_auc=float(roc_auc_score(baseline['y'],pred,sample_weight=wts)))
    trials.append(m);print(json.dumps(dict(day=day,elapsed_s=time.monotonic()-tic,loss=loss_sum/count,**m)),flush=True)
   np.save(R/f'{PREFIX}_day{day}_epoch{epoch}_raw.npy',raw)
  else:print(json.dumps(dict(day=day,epoch=epoch,loss=loss_sum/count,elapsed_s=time.monotonic()-tic)),flush=True)
  save(f'day{day}_progress',dict(trials=trials,training_running=epoch<epochs,csv_ready=False))
 chosen=max(trials,key=lambda a:a['weighted_auc']) if not choice else trials[-1]
 raw=np.load(R/f'{PREFIX}_day{day}_epoch{chosen["epoch"]}_raw.npy');pred=rank(base+chosen['weight']*(rank(raw)-oldrank))
 metrics=measure(baseline,pred);metrics.update(chosen);metrics.update(day=day,baseline_weighted_auc=float(roc_auc_score(baseline['y'],baseline['pred'],sample_weight=wts)),zero_residual_replay_error=replay_error)
 np.savez_compressed(R/f'{PREFIX}_day{day}.npz',events=baseline['events'],y=baseline['y'],pred=pred,raw=raw)
 save(f'day{day}_report',dict(metrics=metrics,trials=trials));return metrics

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 paths=[R/name for name in ['train.npy','train_hist.npy','train_times.npy','evaluation_hist.npy']]
 for day in [6,5]:paths.extend([R/f'imp3_model_warm16_cut{day-2}.npz',R/f'v36_recent_context_day{day}.pt',R/f'v36_recent_context_day{day}.npz',R/f'v53_causal_propensity_day{day}.npz'])
 frozen=dict(epochs=[1,2,3],weights=[.05,.1],lr=.001,weight_decay=.0001,residual_penalty=.01,batch_size=512,threads=4,seed=5701,gru_dim=32,selection='Day6 adjusted AUC, freeze best choice before day5 confirmation. Reused validation dates; intervals exclude selection uncertainty.',inputs={str(p):digest(p) for p in paths},sources={name:digest(name) for name in [__file__,'history_sequence.py','v36_recent_context.py','v24_mass.py','tst2_candidate.py']})
 with (R/f'{PREFIX}_frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
 first=fit(6);save('selection',first);reports=[first]
 if first['gain']>0 and first['weighted_interval'][0]>0:reports.append(fit(5,first))
 for path,sha in frozen['inputs'].items():assert digest(path)==sha
 for path,sha in frozen['sources'].items():assert digest(path)==sha
 accepted=len(reports)==2 and all(r['gain']>0 and r['weighted_interval'][0]>0 for r in reports)
 result=dict(reports=reports,accepted=accepted,csv_ready=False,official_auc=None,target_achieved=False,training_running=False)
 save('report',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
