"""Replay time-interaction model and validate signed residual blend."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from v36_recent_context import R,DAY,Context,pack,predict,rank
from v24_mass import measure

def main():
 torch.set_num_threads(2)
 frozen=json.loads((R/'v36_recent_context_frozen.json').read_text())
 assert hashlib.sha256(Path('v36_recent_context.py').read_bytes()).hexdigest()==frozen['source_sha256']
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 reports=json.loads((R/'v36_recent_context_report.json').read_text())['reports'];selected=json.loads((R/'v36_recent_context_selection.json').read_text());out=[]
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min())
 for entry in reports:
  d=entry['day'];z=np.load(R/f'v36_recent_context_day{d}.npz');base=np.load(R/f'v23_outside_day{d}.npz')
  assert np.array_equal(z['y'],base['y']) and np.array_equal(z['events'],base['events'])
  assert entry['weight']==selected['weight'] and entry['last_training_timestamp']<entry['cutoff']
  assert entry['first_training_timestamp']>=entry['factor_cutoff']
  p=rank(rank(base['pred'])+entry['weight']*(rank(z['raw'])-rank(z['old'])))
  assert np.max(np.abs(p-z['pred']))<1e-12
  for a in [z['raw'],z['old']]:assert np.isfinite(a).all() and (a>=0).all() and (a.reshape(-1,10).sum(1)<=1.000001).all()
  rows=np.asarray(t[(t[:,2]>=start+d*DAY)&(t[:,2]<start+(d+1)*DAY)]);_,ix=np.unique(rows[::10,0],return_index=True);rows=rows[(ix[:,None]*10+np.arange(10)).ravel()]
  assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
  with np.load(R/f'imp3_model_warm16_cut{d-2}.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
  data=pack(rows[:1000],h,ht,factors);model=Context(factors['V'].shape[1]);zero_raw,old=predict(model,data,factors);assert np.max(np.abs(zero_raw-old))<1e-7
  model.load_state_dict(torch.load(R/f'v36_recent_context_day{d}.pt',weights_only=True));raw,old=predict(model,data,factors)
  error=max(float(np.max(np.abs(raw-z['raw'][:1000]))),float(np.max(np.abs(old-z['old'][:1000]))));assert error<1e-6
  comparison=np.load(R/f'v35_temporal_context_day{d}.npz')
  metrics=measure(comparison,p);assert abs(metrics['auc']-entry['auc'])<1e-12
  metrics.update(day=d,replay_max_error=error);out.append(metrics)
 accepted=len(out)==2 and all(m['conditional_interval'][0]>0 and m['weighted_interval'][0]>0 for m in out)
 result=dict(accepted=accepted,reports=out,hashes_unchanged=True,note='Reused local validation. Intervals exclude model-selection uncertainty. No official score or export claim.')
 (R/'v36_recent_context_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
