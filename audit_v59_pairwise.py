"""Replay both V59 arms, independently check logits, sampling and weighted scores."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from v59_pairwise_residual import R,PREFIX,DAY,digest,prepare,SlateResidual,Context,predict,rank,weights,batches,good,check_gradient
from audit_v58_slate import oracle
from v24_mass import measure

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 frozen=json.loads((R/f'{PREFIX}_frozen.json').read_text())
 for path,sha in frozen['inputs'].items():assert digest(path)==sha,path
 for path,sha in frozen['sources'].items():assert digest(path)==sha,path
 gradient=check_gradient();report=json.loads((R/f'{PREFIX}_report.json').read_text());selected=json.loads((R/f'{PREFIX}_selection.json').read_text());audits=[]
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min())
 for record in report['reports']:
  m=record['metrics'];day=m['day'];assert m['epoch']==selected['epoch'] and m['weight']==selected['weight']
  train=np.asarray(t[(t[:,2]>=start+(day-2)*DAY)&(t[:,2]<start+day*DAY)]);assert int(train[:,2].max())==m['fit_last']<m['cutoff']==start+day*DAY
  group=np.digitize((h[train[::10,6]]>=0).sum(1),[0,4,9,19],right=True);source=np.bincount(group,minlength=5)/len(group);eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True);target=np.bincount(eg,minlength=5)/len(eg);ratios=np.clip(target/np.maximum(source,1e-6),.25,4).astype('float32');assert np.array_equal(ratios,np.array(record['importance_ratios'],dtype='float32'))
  epochs=4 if day==6 else selected['epoch'];assert int(train[:,5].sum())*8*epochs==record['sampled_pair_counts']['pairwise'];assert record['sampled_pair_counts']['likelihood']==0;del train
  baseline=np.load(R/f'v53_causal_propensity_day{day}.npz');candidate=np.load(R/f'{PREFIX}_day{day}.npz');prior=np.load(R/f'v36_recent_context_day{day}.npz');assert np.array_equal(candidate['events'],baseline['events']) and np.array_equal(candidate['y'],baseline['y'])
  with np.load(R/f'imp3_model_warm16_cut{day-2}.npz') as bundle:f={k:bundle[k].astype('float32') for k in ['U','V','H','bu','bi']}
  predictions={};arm_audits=[]
  for arm,key in [('likelihood','control_raw'),('pairwise','raw')]:
   context=Context(f['V'].shape[1]);context.load_state_dict(torch.load(R/f'v36_recent_context_day{day}.pt',weights_only=True));model=SlateResidual(f,context,False);path=R/f'{PREFIX}_{arm}_day{day}_epoch{m["epoch"]}.pt';state=torch.load(path,weights_only=True)
   for name,value in context.state_dict().items():assert torch.equal(value,state['context.'+name])
   for name in ['V','H']:assert np.array_equal(f[name],state[name].numpy())
   model.load_state_dict(state);model.eval();raw=[];old=[];oracle_error=0.
   for off in range(0,len(baseline['events']),20480):
    ids=baseline['events'][off:off+20480];rows=np.asarray(t[ids*10+np.tile(np.arange(10),len(ids)//10)]);assert np.array_equal(rows[:,5],baseline['y'][off:off+20480]) and rows[:,2].min()>=start+day*DAY
    data=prepare(rows,h,ht,f);p,q=predict(model,data,f);raw.append(p);old.append(q)
    if off==0:inputs,_=next(batches(data,f,np.arange(16)));oracle_error=oracle(model,inputs)
   raw=np.concatenate(raw);old=np.concatenate(old);raw_error=float(np.max(np.abs(raw-candidate[key])));old_error=float(np.max(np.abs(old-prior['raw'])));assert raw_error==0 and old_error==0
   pred=rank(rank(baseline['pred'])+m['weight']*(rank(raw)-rank(old)));saved='control' if arm=='likelihood' else 'pred';assert np.array_equal(pred,candidate[saved]);predictions[arm]=pred
   arm_audits.append(dict(arm=arm,raw_replay_error=raw_error,context_replay_error=old_error,numpy_forward_logit_error=oracle_error,checkpoint_sha256=digest(path)))
  metrics=measure(baseline,predictions['pairwise']);control=measure(dict(y=baseline['y'],events=baseline['events'],pred=predictions['likelihood']),predictions['pairwise']);adjusted=float(roc_auc_score(baseline['y'],predictions['pairwise'],sample_weight=weights(baseline,h,eh)));assert abs(adjusted-m['weighted_auc'])<1e-12
  for actual,expected in [(metrics,m),(control,record['vs_matched_control'])]:
   for key in ['auc','gain','weighted_gain','conditional_interval','weighted_interval']:assert np.allclose(actual[key],expected[key],atol=1e-12,rtol=0)
  assert len(record['trials'])==(12 if day==6 else 2)
  if day==6:
   best=max([r for r in record['trials'] if r['arm']=='pairwise'],key=lambda r:r['weighted_auc']);assert best['epoch']==selected['epoch'] and best['weight']==selected['weight']
  audits.append(dict(day=day,rows=len(baseline['y']),history_adjusted_auc=adjusted,arms=arm_audits));print(json.dumps(audits[-1]),flush=True)
 assert report['accepted']==(len(report['reports'])==2 and all(good(r) for r in report['reports']))
 result=dict(status='PASS',audits=audits,gradient_check=gradient,accepted=report['accepted'],csv_ready=False,official_auc=None,note='Saved-model/score checks do not remove validation reuse or model-selection uncertainty.')
 (R/f'{PREFIX}_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
