"""Saved-model, independent NumPy forward, cohort and metric audit of V58."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from v58_slate_residual import R,PREFIX,DAY,digest,prepare,SlateResidual,Context,predict,rank,weights,batches,equivariance_probe,good
from v24_mass import measure

def oracle(model,inputs):
 z,items,bucket,kind,base,length=[a.numpy() for a in inputs];state={k:v.numpy() for k,v in model.state_dict().items()};v=state['V'][items];hv=state['H'][items];d=v.shape[-1];profile=z[:,:d];recent=z[:,d:]
 transformed=np.einsum('bd,bdh->bh',profile,state['context.bucket'][bucket]+state['context.kind'][kind])+recent@state['context.recent']
 original=base+np.einsum('bd,bcd->bc',transformed,v)
 local=np.concatenate([np.broadcast_to(profile[:,None,:],v.shape),np.broadcast_to(recent[:,None,:],v.shape),v,profile[:,None,:]*v,recent[:,None,:]*v],-1)
 other_v=(v.sum(1,keepdims=True)-v)/9;other_h=(hv.sum(1,keepdims=True)-hv)/9;std=np.broadcast_to(v.std(1)[:,None,:],v.shape);extra=np.concatenate([other_v,other_h,std,v*other_v],-1)
 if not model.slate:extra=np.zeros_like(extra)
 scalars=np.concatenate([original[:,:,None],np.broadcast_to(np.log1p(length)[:,None,None]/3,(len(z),10,1)),np.broadcast_to(np.eye(4,dtype='float32')[bucket][:,None,:],(len(z),10,4))],-1)
 x=np.concatenate([local,extra,scalars],-1);x=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5);x=x*state['net.0.weight']+state['net.0.bias'];x=np.maximum(x@state['net.1.weight'].T+state['net.1.bias'],0);delta=(x@state['net.4.weight'].T+state['net.4.bias']).squeeze(-1)
 with torch.no_grad():actual=model(*inputs)[0].numpy()
 error=float(np.max(np.abs(actual-original-delta)));assert error<3e-6,error;return error

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 frozen=json.loads((R/f'{PREFIX}_frozen.json').read_text())
 for path,sha in frozen['inputs'].items():assert digest(path)==sha,path
 for path,sha in frozen['sources'].items():assert digest(path)==sha,path
 report=json.loads((R/f'{PREFIX}_report.json').read_text());selection=json.loads((R/f'{PREFIX}_selection.json').read_text());audits=[]
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min())
 for recorded in report['reports']:
  m=recorded['metrics'];day=m['day'];assert m['epoch']==selection['epoch'] and m['weight']==selection['weight'];assert m['fit_last']<start+day*DAY==m['cutoff']
  baseline=np.load(R/f'v53_causal_propensity_day{day}.npz');candidate=np.load(R/f'{PREFIX}_day{day}.npz');prior=np.load(R/f'v36_recent_context_day{day}.npz');assert np.array_equal(candidate['events'],baseline['events']) and np.array_equal(candidate['y'],baseline['y'])
  with np.load(R/f'imp3_model_warm16_cut{day-2}.npz') as bundle:f={k:bundle[k].astype('float32') for k in ['U','V','H','bu','bi']}
  arms={};arm_audits=[]
  for arm,key in [('control','control_raw'),('slate','raw')]:
   context=Context(f['V'].shape[1]);context.load_state_dict(torch.load(R/f'v36_recent_context_day{day}.pt',weights_only=True));model=SlateResidual(f,context,arm=='slate');path=R/f'{PREFIX}_{arm}_day{day}_epoch{m["epoch"]}.pt';state=torch.load(path,weights_only=True)
   for name,value in context.state_dict().items():assert torch.equal(value,state['context.'+name])
   for name in ['V','H']:assert np.array_equal(f[name],state[name].numpy())
   model.load_state_dict(state);model.eval();raw=[];old=[];oracle_error=0.;permutation_error=0.
   # Match 1024-event inference batch boundaries. Otherwise last partial
   # batches can change float32 rounding and break exact rank-score replay.
   for off in range(0,len(baseline['events']),20480):
    ids=baseline['events'][off:off+20480];rows=np.asarray(t[ids*10+np.tile(np.arange(10),len(ids)//10)]);assert np.array_equal(rows[:,5],baseline['y'][off:off+20480]);assert rows[:,2].min()>=start+day*DAY
    data=prepare(rows,h,ht,f);p,q=predict(model,data,f);raw.append(p);old.append(q)
    if off==0:
     inputs,_=next(batches(data,f,np.arange(16)));oracle_error=oracle(model,inputs);permutation_error=equivariance_probe(model,data,f)
   raw=np.concatenate(raw);old=np.concatenate(old);raw_error=float(np.max(np.abs(raw-candidate[key])));old_error=float(np.max(np.abs(old-prior['raw'])));assert raw_error<2e-7 and old_error<2e-7
   arms[arm]=rank(rank(baseline['pred'])+m['weight']*(rank(raw)-rank(old)));saved_key='control' if arm=='control' else 'pred';assert np.max(np.abs(arms[arm]-candidate[saved_key]))<1e-12
   arm_audits.append(dict(arm=arm,raw_replay_error=raw_error,context_replay_error=old_error,numpy_forward_logit_error=oracle_error,candidate_permutation_error=permutation_error,checkpoint_sha256=digest(path)))
  metrics=measure(baseline,arms['slate']);vs_control=measure(dict(y=baseline['y'],events=baseline['events'],pred=arms['control']),arms['slate']);adjusted=float(roc_auc_score(baseline['y'],arms['slate'],sample_weight=weights(baseline,h,eh)))
  assert abs(adjusted-m['weighted_auc'])<1e-12
  for observed,expected in [(metrics,m),(vs_control,recorded['vs_matched_control'])]:
   for key in ['auc','gain','weighted_gain','conditional_interval','weighted_interval']:assert np.allclose(observed[key],expected[key],atol=1e-12,rtol=0),key
  trials=recorded['trials'];assert len(trials)==(12 if day==6 else 2)
  if day==6:
   best=max([r for r in trials if r['arm']=='slate'],key=lambda a:a['weighted_auc']);assert best['epoch']==selection['epoch'] and best['weight']==selection['weight']
  audits.append(dict(day=day,rows=len(baseline['y']),history_adjusted_auc=adjusted,arms=arm_audits));print(json.dumps(audits[-1]),flush=True)
 assert report['accepted']==(len(report['reports'])==2 and all(good(r) for r in report['reports']))
 result=dict(status='PASS',audits=audits,accepted=report['accepted'],csv_ready=False,official_auc=None,note='Verifies implementation and scores; reused validation dates and conditional selection uncertainty remain.')
 (R/f'{PREFIX}_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
