"""Independent initialization, saved-prediction, cutoff and metric checks for V46."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v46_joint_context import Joint,pack,predict
from v39_regularized_als32 import rank,weighted_auc

def main():
 torch.set_num_threads(1)
 frozen=json.loads((R/'v46_joint_context_frozen.json').read_text());assert hashlib.sha256(Path('v46_joint_context.py').read_bytes()).hexdigest()==frozen['source_sha256']
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');out=[]
 for day in (6,5):
  path=R/f'v46_joint_context_day{day}.npz'
  if not path.exists():continue
  z=np.load(path);baseline=np.load(R/f'v39_regularized_als32_day{day}.npz');assert np.array_equal(z['y'],baseline['y']) and np.array_equal(z['events'],baseline['events'])
  with np.load(R/f'imp2_model_warm1.0_d16_pc1_day{day}.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
  events=z['events'][::10];rng=np.random.default_rng(4607);sample=np.unique(np.r_[np.arange(min(16,len(events))),rng.choice(len(events),min(128,len(events)),replace=False)])
  selected=events[sample];rows=np.asarray(t[(selected[:,None]*10+np.arange(10)).ravel()]);assert np.array_equal(rows[::10,6],selected)
  d=pack(rows,h,ht);head=rows[::10];history=h[head[:,6]];valid=history>=0
  profile=factors['U'][head[:,0]]+(factors['H'][np.maximum(history,0)]*valid[:,:,None]).sum(1)/np.maximum(valid.sum(1),1)[:,None]**.5
  logits=(-1.8+factors['bu'][head[:,0],None]+factors['bi'][rows[:,1].reshape(-1,10)]+(profile[:,None,:]*factors['V'][rows[:,1].reshape(-1,10)]).sum(-1)).clip(-30,30)
  expected=softmax(np.column_stack([logits,np.zeros(len(head))]),axis=1)[:,:10].ravel();model=Joint(factors)
  init=predict(model,d);init_error=float(np.max(np.abs(init-expected)));assert init_error<1e-6
  state=torch.load(R/f'v46_joint_context_day{day}.pt',weights_only=True);model.load_state_dict(state);raw=predict(model,d);ix=(sample[:,None]*10+np.arange(10)).ravel();replay_error=float(np.max(np.abs(raw-z['raw'][ix])));assert replay_error<1e-6
  report=json.loads((R/f'v46_joint_context_day{day}_report.json').read_text())['metrics'];recipe=rank(rank(baseline['pred'])+report['weight']*(rank(z['raw'])-rank(z['old'])));assert np.array_equal(recipe,z['pred'])
  assert abs(roc_auc_score(z['y'],recipe)-report['auc'])<1e-12 and abs(weighted_auc(z,recipe)-report['weighted_auc'])<1e-12
  assert report['fit_last']<report['cutoff']<=int(rows[:,2].min())
  out.append(dict(day=day,initialization_max_error=init_error,saved_replay_max_error=replay_error,recipe_and_metrics=True,fit_before_validation=True))
 assert out
 result=dict(status='PASS',reports=out,source_and_inputs_unchanged=True,scope='Model replay and metric validity; does not validate target achievement or eliminate repeated-selection uncertainty.')
 (R/'v46_joint_context_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
