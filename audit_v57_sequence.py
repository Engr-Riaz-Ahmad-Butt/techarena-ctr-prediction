"""Independent saved-checkpoint replay and score/selection audit for V57."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from v57_sequence_residual import R,PREFIX,DAY,digest,prepare,SequenceResidual,Context,predict,rank,weights
from v24_mass import measure

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 frozen=json.loads((R/f'{PREFIX}_frozen.json').read_text())
 for path,sha in frozen['inputs'].items():assert digest(path)==sha,path
 for path,sha in frozen['sources'].items():assert digest(path)==sha,path
 report=json.loads((R/f'{PREFIX}_report.json').read_text());selection=json.loads((R/f'{PREFIX}_selection.json').read_text())
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min());audits=[]
 for recorded in report['reports']:
  day=recorded['day'];baseline=np.load(R/f'v53_causal_propensity_day{day}.npz');candidate=np.load(R/f'{PREFIX}_day{day}.npz');prior=np.load(R/f'v36_recent_context_day{day}.npz')
  assert recorded['epoch']==selection['epoch'] and recorded['weight']==selection['weight']
  assert np.array_equal(candidate['events'],baseline['events']) and np.array_equal(candidate['y'],baseline['y'])
  with np.load(R/f'imp3_model_warm16_cut{day-2}.npz') as bundle:f={k:bundle[k].astype('float32') for k in ['U','V','H','bu','bi']}
  context=Context(f['V'].shape[1]);context.load_state_dict(torch.load(R/f'v36_recent_context_day{day}.pt',weights_only=True));model=SequenceResidual(f,context)
  checkpoint=R/f'{PREFIX}_day{day}_epoch{recorded["epoch"]}.pt';state=torch.load(checkpoint,weights_only=True)
  for name,value in context.state_dict().items():assert torch.equal(value,state['context.'+name])
  model.load_state_dict(state);model.eval();raw=[];old=[];empty_max=0.;latest_history=0
  events=baseline['events'];y=baseline['y']
  for offset in range(0,len(events),20000):
   ids=events[offset:offset+20000];rows=np.asarray(t[ids*10+np.tile(np.arange(10),len(ids)//10)])
   assert np.array_equal(rows[:,5],y[offset:offset+20000]);assert rows[:,2].min()>=start+day*DAY
   data=prepare(rows,h,ht,f);assert np.array_equal(data['tokens'][data['tokens']<len(f['H'])],h[rows[::10,6]][h[rows[::10,6]]>=0])
   p,q=predict(model,data,f);raw.append(p);old.append(q)
   empty=np.repeat(data['real']==0,10)
   if empty.any():empty_max=max(empty_max,float(np.max(np.abs(p[empty]-q[empty]))))
   valid=h[rows[::10,6]]>=0
   if valid.any():latest_history=max(latest_history,int(ht[rows[::10,6]][valid].max()))
  raw=np.concatenate(raw);old=np.concatenate(old)
  raw_error=float(np.max(np.abs(raw-candidate['raw'])));context_error=float(np.max(np.abs(old-prior['raw'])))
  assert raw_error<2e-7 and context_error<2e-7 and empty_max==0
  p=rank(rank(baseline['pred'])+recorded['weight']*(rank(raw)-rank(old)))
  assert np.max(np.abs(p-candidate['pred']))<1e-12
  w=weights(baseline,h,eh);auc=float(roc_auc_score(y,p));adjusted=float(roc_auc_score(y,p,sample_weight=w));metrics=measure(baseline,p)
  assert abs(auc-recorded['auc'])<1e-12 and abs(adjusted-recorded['weighted_auc'])<1e-12
  for key in ['gain','weighted_gain','conditional_interval','weighted_interval']:assert np.allclose(metrics[key],recorded[key],atol=1e-12,rtol=0)
  trials=json.loads((R/f'{PREFIX}_day{day}_report.json').read_text())['trials']
  if day==6:
   assert len(trials)==6 and max(trials,key=lambda a:a['weighted_auc'])['epoch']==selection['epoch']
   assert max(trials,key=lambda a:a['weighted_auc'])['weight']==selection['weight']
  else:assert len(trials)==1
  audits.append(dict(day=day,rows=len(y),ordinary_auc=auc,history_adjusted_auc=adjusted,raw_replay_error=raw_error,frozen_context_replay_error=context_error,empty_history_probability_difference=empty_max,checkpoint_sha256=digest(checkpoint),latest_supplied_history_timestamp=latest_history))
  print(json.dumps(audits[-1]),flush=True)
 accepted=len(report['reports'])==2 and all(m['gain']>0 and m['weighted_interval'][0]>0 for m in report['reports'])
 assert report['accepted']==accepted and not report['csv_ready']
 result=dict(status='PASS',audits=audits,accepted=accepted,official_auc=None,csv_ready=False,note='Repeated validation use and selection uncertainty remain; no official score inferred.')
 (R/f'{PREFIX}_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
