"""Joint warm-factor/context fine tuning with train-only anchored embeddings.

All fitting observations precede validation cutoff. Unlike separated residual
experiments this intentionally fine tunes the factor model on the same fitting
prefix, never on validation labels. Final fitting can represent both day types.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from config import ARTIFACT_DIR as R
from v36_recent_context import rank,DAY
from v39_regularized_als32 import weighted_auc
from v24_mass import measure

class Joint(torch.nn.Module):
 def __init__(self,f):
  super().__init__();self.dim=f['V'].shape[1]
  for k in ['U','V','H','bu','bi']:
   x=torch.as_tensor(f[k],dtype=torch.float32);setattr(self,k,torch.nn.Parameter(x.clone()));self.register_buffer('anchor_'+k,x.clone())
  d=self.dim;ni=len(f['V'])
  self.bucket=torch.nn.Parameter(torch.zeros(4,d,d));self.kind=torch.nn.Parameter(torch.zeros(2,d,d));self.recent=torch.nn.Parameter(torch.zeros(d,d))
  self.item_bucket=torch.nn.Parameter(torch.zeros(ni,4));self.item_kind=torch.nn.Parameter(torch.zeros(ni,2))
 def forward(self,u,items,history,age,bucket,kind):
  valid=history>=0;hi=history.clamp(min=0);hv=self.H[hi]*valid[:,:,None]
  profile=self.U[u]+hv.sum(1)/valid.sum(1).clamp(min=1).sqrt()[:,None]
  weights=torch.exp(-np.log(2)*age/2)*valid
  recent=(hv*weights[:,:,None]).sum(1)/weights.sum(1).clamp(min=1).sqrt()[:,None]
  change=torch.bmm(profile[:,None,:],self.bucket[bucket]+self.kind[kind]).squeeze(1)+recent@self.recent
  v=self.V[items]
  logits=-1.8+self.bu[u,None]+self.bi[items]+((profile+change)[:,None,:]*v).sum(-1)
  logits=logits+self.item_bucket[items,bucket[:,None]]+self.item_kind[items,kind[:,None]]
  return logits.clamp(-30,30)
 def penalty(self,u,items,history):
  hi=history.clamp(min=0);valid=history>=0
  anchor=(self.U[u]-self.anchor_U[u]).square().sum(1).mean()+(self.V[items]-self.anchor_V[items]).square().sum(-1).mean()
  anchor=anchor+((self.H[hi]-self.anchor_H[hi]).square().sum(-1)*valid).sum()/valid.sum().clamp(min=1)
  anchor=anchor+(self.bu[u]-self.anchor_bu[u]).square().mean()+(self.bi[items]-self.anchor_bi[items]).square().mean()
  return .01*anchor+.001*(self.bucket.square().sum()+self.kind.square().sum()+self.recent.square().sum())+.01*(self.item_bucket[items].square().mean()+self.item_kind[items].square().mean())

def pack(rows,h,ht):
 e=rows[::10];history=np.asarray(h[e[:,6]]);stamps=np.asarray(ht[e[:,6]]);assert np.all((stamps<e[:,2,None])|(history<0))
 y=rows[:,5].reshape(-1,10);assert (y.sum(1)<=1).all()
 return dict(u=e[:,0],items=rows[:,1].reshape(-1,10),history=history,age=np.maximum((e[:,2,None]-stamps)/DAY,0).astype('float32'),bucket=e[:,3],kind=e[:,4],target=np.where(y.sum(1)>0,y.argmax(1),10),n=len(e))
def batches(d,order):
 for off in range(0,len(order),2048):
  ix=order[off:off+2048]
  yield tuple(torch.as_tensor(d[k][ix],dtype=torch.float32 if k=='age' else torch.long) for k in ['u','items','history','age','bucket','kind','target'])
def predict(model,d):
 model.eval();out=[]
 with torch.no_grad():
  for *args,y in batches(d,np.arange(d['n'])):
   logits=model(*args);out.append(torch.softmax(torch.cat([logits,torch.zeros(len(logits),1)],1),1)[:,:10].numpy())
 return np.concatenate(out).ravel()

def fit(day,chosen=None):
 tic=time.monotonic();torch.set_num_threads(4);torch.manual_seed(4601)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min());cut=start+day*DAY
 source=R/f'imp2_model_warm1.0_d16_pc1_day{day}.npz'
 with np.load(source) as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 rows=np.asarray(t[t[:,2]<cut]);td=pack(rows,h,ht);last=int(rows[:,2].max());del rows
 rows=np.asarray(t[(t[:,2]>=cut)&(t[:,2]<cut+DAY)]);_,first=np.unique(rows[::10,0],return_index=True);rows=rows[(first[:,None]*10+np.arange(10)).ravel()];vd=pack(rows,h,ht)
 z=np.load(R/f'v39_regularized_als32_day{day}.npz');assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
 model=Joint(factors);original=predict(model,vd)
 context=[model.bucket,model.kind,model.recent,model.item_bucket,model.item_kind];emb=[getattr(model,k) for k in ['U','V','H','bu','bi']]
 opt=torch.optim.Adam([dict(params=emb,lr=.0005),dict(params=context,lr=.003)])
 rng=np.random.default_rng(4601+day);epochs=chosen['epoch'] if chosen else 6;trials=[];best=None
 for epoch in range(1,epochs+1):
  model.train();total=0.
  for u,i,hi,age,b,k,y in batches(td,rng.permutation(td['n'])):
   opt.zero_grad();logits=model(u,i,hi,age,b,k);loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(logits),1)],1),y)+model.penalty(u,i,hi);loss.backward();opt.step();total+=float(loss.detach())*len(y)
  print(json.dumps(dict(day=day,epoch=epoch,loss=total/td['n'],elapsed_s=time.monotonic()-tic)),flush=True)
  if epoch not in ([chosen['epoch']] if chosen else [1,3,6]):continue
  raw=predict(model,vd)
  for w in ([chosen['weight']] if chosen else [.1,.2,.4]):
   p=rank(rank(z['pred'])+w*(rank(raw)-rank(original)));wa=weighted_auc(z,p);trial=dict(epoch=epoch,weight=w,weighted_auc=wa);trials.append(trial)
   if best is None or wa>best['weighted_auc']:
    best=trial;np.savez_compressed(R/f'v46_joint_context_day{day}.npz',y=z['y'],events=z['events'],pred=p,raw=raw,old=original);torch.save(model.state_dict(),R/f'v46_joint_context_day{day}.pt')
 saved=np.load(R/f'v46_joint_context_day{day}.npz');m=measure(z,saved['pred']);m.update(best);m.update(day=day,fit_last=last,cutoff=cut,training_kind_counts=np.bincount(td['kind'],minlength=2).tolist(),baseline_weighted_auc=weighted_auc(z,z['pred']))
 (R/f'v46_joint_context_day{day}_report.json').write_text(json.dumps(dict(metrics=m,trials=trials),indent=2));print(json.dumps(m),flush=True);return m

def main():
 files=[R/f'imp2_model_warm1.0_d16_pc1_day{d}.npz' for d in (6,5)]+[R/f'v39_regularized_als32_day{d}.npz' for d in (6,5)]
 frozen=dict(epochs=[1,3,6],weights=[.1,.2,.4],seed=4601,threads=4,embedding_lr=.0005,context_lr=.003,anchor_penalty=.01,context_penalty=.001,item_context_penalty=.01,initialization='same-cutoff warm16, fine tune on fitting labels only',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v46_joint_context_frozen.json').open('x').write(json.dumps(frozen,indent=2));first=fit(6);reports=[first]
 (R/'v46_joint_context_selection.json').open('x').write(json.dumps(first,indent=2))
 if first['gain']>=0 and first['weighted_interval'][0]>0:reports.append(fit(5,first))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(m['gain']>=0 and m['weighted_interval'][0]>0 for m in reports),official_auc=None,csv_ready=False,note='Repeatedly reused holdouts. Settings selected on day6, frozen before day5. Local AUC is not official AUC.')
 (R/'v46_joint_context_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
