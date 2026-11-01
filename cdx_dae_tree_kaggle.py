"""Fixed catalog-DAE features inside the main 354-feature tree; Kaggle only.

This tests feature ingestion, not a new AE or a score/head/blend search.
All five AE checkpoints are fitted before their own scoring blocks.
"""
from pathlib import Path
import gc, hashlib, json, math, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import lightgbm as lgb

SEED=8711; EPOCHS=20; BATCH=256; TREES=735
SETTINGS=dict(objective='binary',metric='auc',num_leaves=31,learning_rate=.04,
 min_data_in_leaf=200,feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,
 lambda_l2=2.,max_bin=127,force_col_wise=True,deterministic=True,
 num_threads=4,seed=42,verbosity=-1)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

class CatalogAE(nn.Module):
 def __init__(self,initial,clicks):
  super().__init__();d=16;ni=50000
  self.encoder=nn.EmbeddingBag(ni,d,mode='sum',include_last_offset=True)
  self.mean=nn.Linear(d,d);self.logvar=nn.Linear(d,d);self.decoder=nn.Linear(d,ni)
  with torch.no_grad():
   self.encoder.weight.copy_(torch.tensor(initial['H']));self.mean.weight.copy_(torch.eye(d));self.mean.bias.zero_()
   self.logvar.weight.zero_();self.logvar.bias.fill_(-4.)
   self.decoder.weight.copy_(torch.tensor(initial['V']))
   p=np.asarray(clicks.sum(0)).ravel()+1
   self.decoder.bias.copy_(torch.tensor(np.log(p/p.sum())))
 def encode(self,items,offsets,weights,rng=None):
  if self.training:weights=weights*(torch.rand(weights.shape,generator=rng)>=.5)*2
  return self.mean(torch.tanh(self.encoder(items,offsets,per_sample_weights=weights)))

def batch(x,rows):
 sub=x[rows].tocsr();norm=np.sqrt(np.asarray(sub.multiply(sub).sum(1)).ravel())
 owner=np.repeat(np.arange(len(rows)),np.diff(sub.indptr));weight=sub.data/np.maximum(norm[owner],1e-8)
 return (torch.tensor(sub.indices,dtype=torch.long),torch.tensor(sub.indptr,dtype=torch.long),
  torch.tensor(weight,dtype=torch.float32),torch.tensor(owner,dtype=torch.long),torch.tensor(sub.data,dtype=torch.float32))

def fit_ae(folder,cut,out):
 x=sp.load_npz(folder/f'clicks_cut{cut}.npz');initial=dict(np.load(folder/f'initial_cut{cut}.npz'))
 active=np.flatnonzero(np.asarray(x.sum(1)).ravel()>0);torch.manual_seed(SEED)
 model=CatalogAE(initial,x);opt=torch.optim.Adam(model.parameters(),lr=.001)
 order_rng=np.random.default_rng(SEED);noise=torch.Generator(device='cuda').manual_seed(SEED+1);logs=[]
 for epoch in range(EPOCHS):
  st=time.monotonic();model.train();order=order_rng.permutation(active);losses=[]
  for off in range(0,len(order),BATCH):
   items,ptr,w,owner,count=batch(x,order[off:off+BATCH]);opt.zero_grad(set_to_none=True)
   logits=model.decoder(model.encode(items,ptr,w,noise))
   loss=-(torch.log_softmax(logits,1)[owner,items]*count).sum()/len(ptr[:-1])
   assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
  rec=dict(cut=cut,epoch=epoch+1,nll=float(np.mean(losses)),seconds=time.monotonic()-st);logs.append(rec)
  print('AE_FIT',json.dumps(rec),flush=True)
  if epoch==0:
   estimate=rec['seconds']*EPOCHS*5*1.5+600
   print('AE_RUNTIME_PILOT_SECONDS',estimate,flush=True)
   assert estimate<1800,'Stop: projected job exceeds30minutes'
 torch.save(model.state_dict(),out/f'dae_cut{cut}.pt');model.eval()
 return model,initial,logs

@torch.no_grad()
def ae_features(model,history,items,item_bias):
 """Ten invariant scalar features; no fitting or labels in this function."""
 ne=len(history);valid=history>=0;owner=np.broadcast_to(np.arange(ne)[:,None],history.shape)[valid]
 counts=sp.csr_matrix((np.ones(int(valid.sum()),np.float32),(owner,history[valid])),shape=(ne,50000))
 result=np.empty((ne*10,10),np.float32)
 for off in range(0,ne,512):
  stop=min(off+512,ne);z=model.encode(*batch(counts,np.arange(off,stop))[:3])
  ids=torch.tensor(items[off:stop],dtype=torch.long);v=model.decoder.weight[ids]
  dot=(z[:,None,:]*v).sum(2);logit=dot+torch.tensor(item_bias[items[off:stop]])
  catalog=dot+model.decoder.bias[ids];zn=z.norm(dim=1)[:,None];vn=v.norm(dim=2)
  cosine=dot/torch.clamp(zn*vn,min=1e-8)
  normalizer=torch.logsumexp(model.decoder(z),dim=1)[:,None]
  values=[dot,logit,catalog,cosine,torch.log1p(vn),torch.log1p(zn).expand_as(dot),
   logit-logit.mean(1,keepdim=True),logit-logit.max(1,keepdim=True).values,
   torch.softmax(logit,1),catalog-normalizer]
  result[off*10:stop*10]=torch.stack(values,2).reshape(-1,10).cpu().numpy()
 assert np.isfinite(result).all();return result

def rank(x):return rankdata(np.asarray(x).ravel(),method='average')/(np.size(x)+1)

def assess(y,b,p,w,seed=8941):
 y=y.reshape(-1);b=b.reshape(-1);p=p.reshape(-1);w=np.repeat(w,10)
 # Weighted ROC AUC bootstrap with fixed ordering: resample whole users/events.
 def compiled(s):
  ix=np.argsort(s,kind='stable');yy=y[ix];ww=w[ix]
  end=np.r_[s[ix][1:]!=s[ix][:-1],True];starts=np.r_[0,np.flatnonzero(end)[:-1]+1]
  def value(event_count):
   ww2=ww*np.repeat(event_count,10)[ix]
   pos=np.add.reduceat(ww2*yy,starts);neg=np.add.reduceat(ww2*(1-yy),starts)
   total_pos=pos.sum();total_neg=neg.sum()
   return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(total_pos*total_neg))
  return value
 f,g=compiled(b),compiled(p);n=len(y)//10;ones=np.ones(n);before=f(ones);after=g(ones)
 assert abs(before-roc_auc_score(y,b,sample_weight=w))<1e-11
 rng=np.random.default_rng(seed);d=[]
 for _ in range(200):
  count=np.bincount(rng.integers(0,n,n),minlength=n);d.append(g(count)-f(count))
 return dict(events=n,positive_rows=int(y.sum()),before=before,after=after,gain=after-before,
  paired_95=np.quantile(d,[.025,.975]).tolist(),event_weight_ess=float(w.sum()**2/(10*np.sum(w*w))))

def run(folder,output='/kaggle/working/cdx_dae_tree_results'):
 folder=Path(folder);out=Path(output);out.mkdir(exist_ok=False)
 manifest=json.loads((folder/'upload_manifest.json').read_text())
 for name,digest in manifest['hashes'].items():assert sha(folder/name)==digest,name
 info=json.loads((folder/'data_info.json').read_text());tic=time.monotonic()
 assert torch.cuda.is_available(),'GPU required for AE; no local/CPU training fallback'
 torch.set_default_device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 print('ACCELERATOR',torch.cuda.get_device_name(0),flush=True);training={}
 for cut in [2,3,4,5,6]:
  assert info[str(cut)]['max_fit_timestamp']<info[str(cut)]['cutoff']
  model,initial,logs=fit_ae(folder,cut,out);training[str(cut)]=logs
  if cut<=5:
   block=np.load(folder/f'block{cut}.npz')
   z=ae_features(model,block['history'],block['items'],initial['bi']);np.save(out/f'new_block{cut}.npy',z)
  if cut>=5:
   block=np.load(folder/f'validation{cut}.npz')
   z=ae_features(model,block['history'],block['items'],initial['bi']);np.save(out/f'new_val{cut}.npy',z)
  del model,initial;gc.collect();torch.cuda.empty_cache()
 torch.set_default_device('cpu');fits={};predictions={}
 for day in [5,6]:
  x=np.concatenate([np.load(folder/f'x_block{d}.npy') for d in range(day-3,day)])
  y=np.concatenate([np.load(folder/f'block{d}.npz')['y'].ravel() for d in range(day-3,day)])
  z=np.concatenate([np.load(out/f'new_block{d}.npy') for d in range(day-3,day)])
  xv=np.load(folder/f'x_val{day}.npy');zv=np.load(out/f'new_val{day}.npy')
  assert x.shape[1]==xv.shape[1]==354 and z.shape[1]==zv.shape[1]==10
  fits[str(day)]={};predictions[day]={}
  for arm in ['control','treatment']:
   xx=x if arm=='control' else np.column_stack([x,z]);vv=xv if arm=='control' else np.column_stack([xv,zv])
   data=lgb.Dataset(xx,label=y,params=SETTINGS,free_raw_data=False);setup=time.monotonic();data.construct();setup=time.monotonic()-setup;st=time.monotonic()
   # First20trees are a training-only timing pilot, continued without reset.
   model=lgb.train(SETTINGS,data,num_boost_round=20,keep_training_booster=True)
   elapsed=time.monotonic()-st;estimate=(elapsed*TREES/20+setup)*4*1.4
   print('TREE_RUNTIME_PILOT_SECONDS',json.dumps(dict(day=day,arm=arm,first20_seconds=elapsed,estimate=estimate)),flush=True)
   assert estimate+time.monotonic()-tic<1800,'Stop before long fit; user approval required'
   model=lgb.train(SETTINGS,data,num_boost_round=TREES-20,init_model=model,
    callbacks=[lgb.log_evaluation(200)])
   assert model.current_iteration()==TREES
   model.save_model(str(out/f'tree_{arm}_day{day}.txt'))
   predictions[day][arm]=model.predict(vv,num_threads=4)
   fits[str(day)][arm]=dict(seconds=time.monotonic()-st,rows=len(y),columns=xx.shape[1],trees=TREES)
   del model,data,xx,vv;gc.collect()
  parent=np.load(folder/f'validation{day}.npz')
  # Joint ranks across all allowed rows, followed by screen slicing only.
  # Shared raw rows allow a matched, fixed original0.5 tree replacement.
  correction=.5*(rank(predictions[day]['treatment'])-rank(predictions[day]['control']))
  predictions[day]['candidate']=parent['parent'].ravel()+correction
  predictions[day]['parent']=parent['parent'].ravel()
  np.savez_compressed(out/f'predictions_day{day}.npz',**predictions[day],events=parent['events'])
  del x,y,z,xv,zv;gc.collect()
 # Complete fixed models and predictions precede any scored-day outcome access.
 screen=[]
 for day in [5,6]:
  f=np.load(folder/f'validation{day}.npz');y=np.load(folder/f'labels_val{day}.npy');take=f['sample']
  parent=predictions[day]['parent'].reshape(-1,10)[take];candidate=predictions[day]['candidate'].reshape(-1,10)[take]
  for weight in ['3bin','5bin']:
   result=assess(y,parent,candidate,f['w3' if weight=='3bin' else 'w5'])
   result.update(day=day,weight=weight,
    control_tree_auc=float(roc_auc_score(y.ravel(),predictions[day]['control'].reshape(-1,10)[take].ravel(),sample_weight=np.repeat(f['w3' if weight=='3bin' else 'w5'],10))),
    treatment_tree_auc=float(roc_auc_score(y.ravel(),predictions[day]['treatment'].reshape(-1,10)[take].ravel(),sample_weight=np.repeat(f['w3' if weight=='3bin' else 'w5'],10))))
   screen.append(result);print('SCREEN',json.dumps(result),flush=True)
 passed=all(x['gain']>=.0002 for x in screen)
 report=dict(status='screen_complete',seconds=time.monotonic()-tic,screen=screen,screen_passed=passed,
  full_confirmation_run=False,training_logs=training,tree_fits=fits,manifest=manifest,
  labels_accessed_only_after_all_predictions=True,holdout_scored=False,evaluation_read=False,submission_created=False,
  caveat='20%screen only; not full V53 reconstruction. Existing H1day5uses user crossfit. Trees on20%past events; identical settings. Ranks joint within complete allowed daily cohorts before screen slicing, diagnostic not deployment probabilities.')
 (out/'report.json').write_text(json.dumps(report,indent=2));print('CDX_DAE_TREE_COMPLETE',json.dumps(report),flush=True)
 return report
