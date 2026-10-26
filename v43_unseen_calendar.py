"""Fixed inference ablation: borrow observed calendar matrix for an unseen level."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v36_recent_context import Context, pack, predict, rank, DAY
from v39_regularized_als32 import weighted_auc
from v24_mass import measure

def main():
 torch.set_num_threads(4)
 sources=[R/'v36_recent_context_day5.pt',R/'v36_recent_context_day5.npz',R/'v39_regularized_als32_day5.npz',R/'imp3_model_warm16_cut3.npz']
 frozen=dict(rule='Copy observed weekday matrix to unseen weekend matrix; no fit or weight search',day=5,weight=.1,input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v43_unseen_calendar_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min())
 rows=np.asarray(t[(t[:,2]>=start+5*DAY)&(t[:,2]<start+6*DAY)])
 _,first=np.unique(rows[::10,0],return_index=True);rows=rows[(first[:,None]*10+np.arange(10)).ravel()]
 with np.load(sources[3]) as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 z=np.load(sources[2]);old=np.load(sources[1]);assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
 model=Context(16);model.load_state_dict(torch.load(sources[0],weights_only=True));norms=model.kind.detach().norm(dim=(1,2)).tolist();assert norms[0]>0 and norms[1]==0
 val=pack(rows,h,ht,factors);original,_=predict(model,val,factors);assert np.max(np.abs(original-old['raw']))<1e-7
 with torch.no_grad():model.kind[1].copy_(model.kind[0])
 raw,_=predict(model,val,factors)
 p=rank(rank(z['pred'])+.1*(rank(raw)-rank(original)))
 metrics=measure(z,p);metrics.update(auc=float(roc_auc_score(z['y'],p)),weighted_auc=weighted_auc(z,p),baseline_weighted_auc=weighted_auc(z,z['pred']),raw_auc=float(roc_auc_score(z['y'],raw)),baseline_raw_auc=float(roc_auc_score(z['y'],original)),original_kind_norms=norms)
 np.savez_compressed(R/'v43_unseen_calendar_day5.npz',y=z['y'],events=z['events'],pred=p,raw=raw)
 result=dict(metrics=metrics,scope='Single unseen-calendar ablation, on previously reused day5. No official-score claim or automatic promotion.',supported=metrics['gain']>0 and metrics['weighted_interval'][0]>0)
 (R/'v43_unseen_calendar_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
