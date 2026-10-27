"""Matched importance-weighted likelihood versus global pairwise ranking.

Same frozen causal context, residual architecture, minibatches and optimizer.
Pairwise arm ranks positives against sampled negatives across whole events.
Sampling uses its own RNG so network dropout and event orders stay matched.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json,time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from v58_slate_residual import R,DAY,digest,prepare,SlateResidual,Context,predict,rank,weights,batches
from v24_mass import measure

PREFIX='v59_pairwise_residual'

def save(name,value):
 (R/f'{PREFIX}_{name}.json').write_text(json.dumps(value,indent=2))

def pair_loss(scores,pos,neg,weight):
 p=scores[pos[:,None]];n=scores[neg];w=weight[pos[:,None]]*weight[neg]
 return (torch.nn.functional.softplus(n-p)*w).sum()/w.sum()

def check_gradient():
 scores=torch.tensor([-.7,.4,.1,1.3,-1.2],dtype=torch.float64,requires_grad=True);pos=torch.tensor([1,3]);neg=torch.tensor([[0,2,4],[2,4,0]]);w=torch.tensor([.5,2.,1.,.7,3.],dtype=torch.float64);loss=pair_loss(scores,pos,neg,w);loss.backward();actual=scores.grad.numpy().copy();fd=[]
 for j in range(len(scores)):
  a=scores.detach().clone();b=a.clone();a[j]+=1e-5;b[j]-=1e-5;fd.append(float((pair_loss(a,pos,neg,w)-pair_loss(b,pos,neg,w))/2e-5))
 error=float(np.max(np.abs(actual-fd)));assert error<1e-8;assert abs(actual.sum())<1e-12
 return dict(max_gradient_error=error,translation_gradient_sum=float(actual.sum()))

def fit(day,choice=None):
 tic=time.monotonic();t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min())
 with np.load(R/f'imp3_model_warm16_cut{day-2}.npz') as bundle:f={k:bundle[k].astype('float32') for k in ['U','V','H','bu','bi']}
 baseline=np.load(R/f'v53_causal_propensity_day{day}.npz');prior=np.load(R/f'v36_recent_context_day{day}.npz');assert np.array_equal(baseline['events'],prior['events']) and np.array_equal(baseline['y'],prior['y'])
 ix=baseline['events']*10+np.tile(np.arange(10),len(baseline['events'])//10);rows=np.asarray(t[ix]);assert np.array_equal(rows[:,5],baseline['y']) and rows[:,2].min()>=start+day*DAY;val=prepare(rows,h,ht,f);del rows,ix
 rows=np.asarray(t[(t[:,2]>=start+(day-2)*DAY)&(t[:,2]<start+day*DAY)]);last=int(rows[:,2].max());assert last<start+day*DAY;train=prepare(rows,h,ht,f);del rows
 tg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);target=np.bincount(tg,minlength=5)/len(tg);source=np.bincount(np.digitize(train['length'],[0,4,9,19],right=True),minlength=5)/train['n'];ratios=np.clip(target/np.maximum(source,1e-6),.25,4).astype('float32');ratio=torch.tensor(ratios)
 w=weights(baseline,h,eh);base=rank(baseline['pred']);trials=[];initial=None;pair_counts={}
 for arm in ['likelihood','pairwise']:
  torch.manual_seed(5901+day);rng=np.random.default_rng(5901+day);generator=torch.Generator().manual_seed(5911+day)
  context=Context(f['V'].shape[1]);context.load_state_dict(torch.load(R/f'v36_recent_context_day{day}.pt',weights_only=True));model=SlateResidual(f,context,False)
  if initial is None:initial={k:v.clone() for k,v in model.state_dict().items()}
  else:
   for key,value in model.state_dict().items():assert torch.equal(value,initial[key])
  old,_=predict(model,val,f);assert np.max(np.abs(old-prior['raw']))<2e-7;oldrank=rank(old);opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
  epochs=choice['epoch'] if choice else 4;checks=[epochs] if choice else [1,2,4];pair_count=0
  for epoch in range(1,epochs+1):
   model.train();total=0.;count=0
   for inputs,label in batches(train,f,rng.permutation(train['n'])):
    opt.zero_grad();logits,delta,_=model(*inputs);full=torch.cat([logits,torch.zeros(len(logits),1)],1);group=torch.bucketize(inputs[-1],torch.tensor([0.,4.,9.,19.]),right=False);ew=ratio[group];ce=(torch.nn.functional.cross_entropy(full,label,reduction='none')*ew).sum()/ew.sum()
    if arm=='pairwise':
     y=torch.nn.functional.one_hot(label,11)[:,:10].reshape(-1);scores=torch.nn.functional.log_softmax(full,1)[:,:10].reshape(-1);positive=torch.where(y==1)[0];negative=torch.where(y==0)[0]
     if len(positive) and len(negative):
      neg=negative[torch.randint(len(negative),(len(positive),8),generator=generator)];ranking=pair_loss(scores,positive,neg,ew.repeat_interleave(10));loss=ranking+.1*ce;pair_count+=int(neg.numel())
     else:loss=ce
    else:loss=ce
    loss=loss+.01*delta.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(label);count+=len(label)
   print(json.dumps(dict(day=day,arm=arm,epoch=epoch,loss=total/count,pairs=pair_count,elapsed_s=time.monotonic()-tic)),flush=True)
   if epoch not in checks:continue
   raw,_=predict(model,val,f);torch.save(model.state_dict(),R/f'{PREFIX}_{arm}_day{day}_epoch{epoch}.pt');np.save(R/f'{PREFIX}_{arm}_day{day}_epoch{epoch}_raw.npy',raw)
   for weight in ([choice['weight']] if choice else [.1,.25]):
    pred=rank(base+weight*(rank(raw)-oldrank));trial=dict(arm=arm,epoch=epoch,weight=weight,auc=float(roc_auc_score(baseline['y'],pred)),weighted_auc=float(roc_auc_score(baseline['y'],pred,sample_weight=w)));trials.append(trial);print(json.dumps(dict(day=day,**trial)),flush=True)
   save(f'day{day}_progress',dict(trials=trials,csv_ready=False))
  pair_counts[arm]=pair_count;del model,opt
 best=max([r for r in trials if r['arm']=='pairwise'],key=lambda r:r['weighted_auc']) if choice is None else next(r for r in trials if r['arm']=='pairwise');predictions={};raws={}
 for arm in ['likelihood','pairwise']:
  raw=np.load(R/f'{PREFIX}_{arm}_day{day}_epoch{best["epoch"]}_raw.npy');raws[arm]=raw;predictions[arm]=rank(base+best['weight']*(rank(raw)-oldrank))
 metrics=measure(baseline,predictions['pairwise']);metrics.update(best);metrics.update(day=day,fit_last=last,cutoff=start+day*DAY,baseline_weighted_auc=float(roc_auc_score(baseline['y'],baseline['pred'],sample_weight=w)))
 vs_control=measure(dict(y=baseline['y'],events=baseline['events'],pred=predictions['likelihood']),predictions['pairwise']);result=dict(metrics=metrics,vs_matched_control=vs_control,trials=trials,importance_ratios=ratios.tolist(),sampled_pair_counts=pair_counts)
 np.savez_compressed(R/f'{PREFIX}_day{day}.npz',y=baseline['y'],events=baseline['events'],pred=predictions['pairwise'],raw=raws['pairwise'],control=predictions['likelihood'],control_raw=raws['likelihood']);save(f'day{day}_report',result);print(json.dumps(result),flush=True);return result

def good(r):return r['metrics']['gain']>0 and r['metrics']['weighted_interval'][0]>0 and r['vs_matched_control']['weighted_interval'][0]>0

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 paths=[R/name for name in ['train.npy','train_hist.npy','train_times.npy','evaluation_hist.npy']]
 for day in [6,5]:paths.extend([R/f'imp3_model_warm16_cut{day-2}.npz',R/f'v36_recent_context_day{day}.pt',R/f'v36_recent_context_day{day}.npz',R/f'v53_causal_propensity_day{day}.npz'])
 frozen=dict(arms=['likelihood','pairwise'],epochs=[1,2,4],weights=[.1,.25],lr=.001,weight_decay=.001,residual_penalty=.01,batch_size=1024,threads=4,seed=5901,negative_sampling_seed=5911,negative_pairs_per_positive=8,pairwise_ce_weight=.1,importance_clip=[.25,4],selection='Select pairwise setting by main day6 adjusted AUC; require positive ordinary gain and positive adjusted lower interval vs V53 and matched likelihood control; freeze before day5. Reused dates, intervals exclude selection uncertainty.',inputs={str(p):digest(p) for p in paths},sources={name:digest(name) for name in [__file__,'v58_slate_residual.py','v36_recent_context.py','v57_sequence_residual.py','v24_mass.py','tst2_candidate.py']})
 with (R/f'{PREFIX}_frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
 gradient=check_gradient();print(json.dumps(gradient),flush=True);reports=[fit(6)];save('selection',reports[0]['metrics'])
 if good(reports[0]):reports.append(fit(5,reports[0]['metrics']))
 for path,sha in frozen['inputs'].items():assert digest(path)==sha
 for path,sha in frozen['sources'].items():assert digest(path)==sha
 result=dict(reports=reports,gradient_check=gradient,accepted=len(reports)==2 and all(good(r) for r in reports),csv_ready=False,official_auc=None,target_achieved=False,training_running=False);save('report',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
