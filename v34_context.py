"""Learn time-conditioned bilinear residuals on frozen causal preference factors."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v24_mass import measure
DAY=86400
rank=lambda a:rankdata(a,method='average')/(len(a)+1)

class Context(torch.nn.Module):
 def __init__(self,dim):
  super().__init__();self.bucket=torch.nn.Parameter(torch.zeros(4,dim,dim));self.kind=torch.nn.Parameter(torch.zeros(2,dim,dim))
 def forward(self,z,v,b,k,base):
  matrices=self.bucket[b]+self.kind[k]
  transformed=torch.bmm(z[:,None,:],matrices).squeeze(1)
  return base+(transformed[:,None,:]*v).sum(-1)

def pack(rows,h,ht,f):
 head=rows[::10];history=np.asarray(h[head[:,6]]);valid=history>=0;assert np.all((ht[head[:,6]]<head[:,2,None])|~valid)
 z=f['U'][head[:,0]].copy()+(f['H'][np.maximum(history,0)]*valid[:,:,None]).sum(1)/np.sqrt(np.maximum(valid.sum(1)[:,None],1))
 y=rows[:,5].reshape(-1,10);assert (y.sum(1)<=1).all()
 return dict(z=z.astype('float32'),items=rows[:,1].reshape(-1,10),bias=f['bu'][head[:,0]],bucket=head[:,3],kind=head[:,4],target=np.where(y.sum(1)>0,y.argmax(1),10),n=len(head))

def batches(data,f,indices):
 for offset in range(0,len(indices),1024):
  ix=indices[offset:offset+1024];z=torch.from_numpy(data['z'][ix]);v=torch.from_numpy(f['V'][data['items'][ix]])
  base=(-1.8+torch.from_numpy(data['bias'][ix])[:,None]+torch.from_numpy(f['bi'][data['items'][ix]])+(z[:,None,:]*v).sum(-1)).clamp(-30,30)
  yield z,v,torch.from_numpy(data['bucket'][ix]),torch.from_numpy(data['kind'][ix]),base,torch.from_numpy(data['target'][ix])

def predict(model,data,f):
 new=[];old=[];model.eval()
 with torch.no_grad():
  for z,v,b,k,base,_ in batches(data,f,np.arange(data['n'])):
   zeros=torch.zeros(len(z),1)
   new.append(torch.softmax(torch.cat([model(z,v,b,k,base),zeros],1),1)[:,:10].numpy())
   old.append(torch.softmax(torch.cat([base,zeros],1),1)[:,:10].numpy())
 return np.concatenate(new).ravel(),np.concatenate(old).ravel()

def fit(day,weight=None):
 started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(3401)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min())
 source=R/f'imp2_model_warm1.0_d16_pc1_day{day}.npz'
 with np.load(source) as f: factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 rows=np.asarray(t[t[:,2]<start+day*DAY]);train=pack(rows,h,ht,factors);last=int(rows[:,2].max());del rows
 later=np.asarray(t[(t[:,2]>=start+day*DAY)&(t[:,2]<start+(day+1)*DAY)])
 _,first=np.unique(later[::10,0],return_index=True);later=later[(first[:,None]*10+np.arange(10)).ravel()]
 ref=np.load(R/f'v23_outside_day{day}.npz');assert np.array_equal(later[:,6],ref['events']) and np.array_equal(later[:,5],ref['y'])
 val=pack(later,h,ht,factors);model=Context(factors['V'].shape[1]);optimizer=torch.optim.Adam(model.parameters(),lr=.003);rng=np.random.default_rng(3401+day)
 for epoch in range(1,4):
  model.train();total=0.;count=0
  for z,v,b,k,base,target in batches(train,factors,rng.permutation(train['n'])):
   optimizer.zero_grad();logits=model(z,v,b,k,base)
   loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(z),1)],1),target)
   penalty=.001*(model.bucket.square().sum()+model.kind.square().sum())
   (loss+penalty).backward();optimizer.step();total+=float(loss.detach())*len(z);count+=len(z)
  print(json.dumps(dict(day=day,epoch=epoch,loss=total/count,elapsed_s=time.monotonic()-started)),flush=True)
 torch.save(model.state_dict(),R/f'v34_context_day{day}.pt')
 raw,old=predict(model,val,factors);delta=rank(raw)-rank(old);base=rank(ref['pred']);trials=[]
 for w in ([weight] if weight is not None else [.05,.1,.2]):
  p=rank(base+w*delta);trials.append(dict(weight=w,auc=float(roc_auc_score(ref['y'],p))))
 selected=max(trials,key=lambda a:a['auc']);w=selected['weight'];pred=rank(base+w*delta);metrics=measure(ref,pred)
 metrics.update(day=day,weight=w,raw_auc=float(roc_auc_score(ref['y'],raw)),old_raw_auc=float(roc_auc_score(ref['y'],old)),last_training_timestamp=last,cutoff=start+day*DAY)
 assert last<metrics['cutoff']
 np.savez_compressed(R/f'v34_context_day{day}.npz',y=ref['y'],events=ref['events'],raw=raw,old=old,pred=pred)
 (R/f'v34_context_day{day}_report.json').write_text(json.dumps(dict(metrics=metrics,trials=trials),indent=2));print(json.dumps(metrics),flush=True)
 return metrics

def main():
 inputs=[R/f'{prefix}{d}{suffix}' for d in (6,5) for prefix,suffix in [('imp2_model_warm1.0_d16_pc1_day','.npz'),('v23_outside_day','.npz')]]
 frozen=dict(epochs=3,lr=.003,regularization=.001,seed=3401,threads=4,weights=[.05,.1,.2],inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v34_context_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 selected=fit(6);reports=[selected];(R/'v34_context_selection.json').open('x').write(json.dumps(selected,indent=2))
 if selected['conditional_interval'][0]>0 and selected['weighted_interval'][0]>0:reports.append(fit(5,selected['weight']))
 for p in inputs:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 accepted=len(reports)==2 and all(r['conditional_interval'][0]>0 and r['weighted_interval'][0]>0 for r in reports)
 (R/'v34_context_report.json').write_text(json.dumps(dict(reports=reports,accepted=accepted,evaluation_csv_ready=False,note='Reused local validation; confidence intervals exclude selection uncertainty.'),indent=2))
if __name__=='__main__':main()
