"""Nonlinear preference residual on strictly later events than the factor cutoff."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from config import ARTIFACT_DIR as R
from v36_recent_context import pack as recent_pack,batches,rank,DAY
from v39_regularized_als32 import weighted_auc
from v24_mass import measure

class Context(torch.nn.Module):
 def __init__(self,dim):
  super().__init__();self.dim=dim;self.bucket=torch.nn.Embedding(4,8);self.kind=torch.nn.Embedding(2,4)
  self.net=torch.nn.Sequential(torch.nn.Linear(5*dim+15,128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
  torch.nn.init.normal_(self.bucket.weight,std=.05);torch.nn.init.normal_(self.kind.weight,std=.05);torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,z,v,b,k,base):
  profile=z[:,:self.dim][:,None,:]
  features=torch.cat([z[:,None,:].expand(-1,10,-1),v,profile*v,torch.abs(profile-v),self.bucket(b)[:,None,:].expand(-1,10,-1),self.kind(k)[:,None,:].expand(-1,10,-1),base[:,:,None]],dim=-1).clamp(-10,10)
  return base+self.net(features).squeeze(-1)

def pack(rows,h,ht,factors):
 data=recent_pack(rows,h,ht,factors);head=rows[::10];history=np.asarray(h[head[:,6]]);ts=np.asarray(ht[head[:,6]]);valid=history>=0
 length=valid.sum(1);latest=np.where(valid,ts,-1).max(1);age=np.where(length>0,(head[:,2]-latest)/DAY,60.)
 data['z']=np.column_stack([data['z'],np.log1p(length)/3,np.log1p(age)/3]).astype('float32');return data

def predict(model,data,factors):
 model.eval();result=[]
 with torch.no_grad():
  for z,v,b,k,base,_ in batches(data,factors,np.arange(data['n'])):
   logits=model(z,v,b,k,base);result.append(torch.softmax(torch.cat([logits,torch.zeros(len(z),1)],1),1)[:,:10].numpy())
 return np.concatenate(result).ravel()

def fit(day,chosen=None):
 started=time.monotonic();torch.manual_seed(4101);t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min())
 with np.load(R/f'imp3_model_warm16_cut{day-2}.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 trainrows=np.asarray(t[(t[:,2]>=start+(day-2)*DAY)&(t[:,2]<start+day*DAY)])
 td=pack(trainrows,h,ht,factors);first=int(trainrows[:,2].min());last=int(trainrows[:,2].max());del trainrows
 rows=np.asarray(t[(t[:,2]>=start+day*DAY)&(t[:,2]<start+(day+1)*DAY)]);_,firsts=np.unique(rows[::10,0],return_index=True);rows=rows[(firsts[:,None]*10+np.arange(10)).ravel()]
 ref=np.load(R/f'v39_regularized_als32_day{day}.npz');prior=np.load(R/f'v36_recent_context_day{day}.npz')
 assert np.array_equal(rows[:,6],ref['events']) and np.array_equal(rows[:,5],ref['y']) and np.array_equal(prior['events'],ref['events'])
 vd=pack(rows,h,ht,factors);model=Context(factors['V'].shape[1]);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(4101+day)
 for epoch in range(1,4):
  total=0.;count=0;model.train()
  for z,v,b,k,base,target in batches(td,factors,rng.permutation(td['n'])):
   opt.zero_grad();residual=model(z,v,b,k,base)-base;logits=base+residual
   loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(z),1)],1),target)+.01*residual.square().mean()
   loss.backward();opt.step();total+=float(loss.detach())*len(z);count+=len(z)
  print(json.dumps(dict(stage='nonlinear_residual',day=day,epoch=epoch,loss=total/count,elapsed_s=time.monotonic()-started)),flush=True)
 torch.save(model.state_dict(),R/f'v41_neural_context_day{day}.pt');raw=predict(model,vd,factors)
 base=rank(ref['pred']);delta=rank(raw)-rank(prior['raw']);trials=[]
 for w in ([chosen['weight']] if chosen else [.05,.1,.2]):
  p=rank(base+w*delta);m=measure(ref,p);m.update(day=day,weight=w,weighted_auc=weighted_auc(ref,p),baseline_weighted_auc=weighted_auc(ref,ref['pred']));trials.append(m)
 selected=max(trials,key=lambda a:a['weighted_auc']);p=rank(base+selected['weight']*delta);selected.update(first_training_timestamp=first,last_training_timestamp=last,factor_cutoff=start+(day-2)*DAY,validation_cutoff=start+day*DAY)
 np.savez_compressed(R/f'v41_neural_context_day{day}.npz',y=ref['y'],events=ref['events'],raw=raw,pred=p)
 (R/f'v41_neural_context_day{day}_report.json').write_text(json.dumps(dict(selected=selected,trials=trials),indent=2));print(json.dumps(selected),flush=True)
 return selected

def main():
 torch.set_num_threads(4)
 inputs=[R/f'imp3_model_warm16_cut{d-2}.npz' for d in (6,5)]+[R/f'{prefix}_day{d}.npz' for d in (6,5) for prefix in ['v39_regularized_als32','v36_recent_context']]
 frozen=dict(epochs=3,lr=.001,weight_decay=1e-4,residual_penalty=.01,dropout=.1,hidden=[128,64],seed=4101,threads=4,grid=[.05,.1,.2],selection_metric='evaluation_history_weighted_auc',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v41_neural_context_frozen.json').open('x').write(json.dumps(frozen,indent=2));selected=fit(6);reports=[selected];(R/'v41_neural_context_selection.json').open('x').write(json.dumps(selected,indent=2))
 if selected['gain']>=0 and selected['weighted_interval'][0]>0:reports.append(fit(5,selected))
 for p in inputs:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 accepted=len(reports)==2 and all(v['gain']>=0 and v['weighted_interval'][0]>0 for v in reports)
 (R/'v41_neural_context_report.json').write_text(json.dumps(dict(accepted=accepted,reports=reports,evaluation_csv_ready=False,note='Reused local dates; conditional history-weighted evidence, not an official-score forecast.'),indent=2))
if __name__=='__main__':main()
