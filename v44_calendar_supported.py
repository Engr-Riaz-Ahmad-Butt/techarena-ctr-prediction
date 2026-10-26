"""Three-day causal adaptation with both calendar levels represented at export."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v36_recent_context import Context,pack,batches,predict,rank,DAY
from v39_regularized_als32 import weighted_auc
from v24_mass import measure

def fit(day):
 start_time=time.monotonic();torch.set_num_threads(4);torch.manual_seed(3401)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min())
 with np.load(R/f'imp3_model_warm16_cut{day-3}.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 rows=np.asarray(t[(t[:,2]>=start+(day-3)*DAY)&(t[:,2]<start+day*DAY)]);td=pack(rows,h,ht,factors)
 later=np.asarray(t[(t[:,2]>=start+day*DAY)&(t[:,2]<start+(day+1)*DAY)]);_,first=np.unique(later[::10,0],return_index=True);later=later[(first[:,None]*10+np.arange(10)).ravel()]
 vd=pack(later,h,ht,factors);z=np.load(R/f'v39_regularized_als32_day{day}.npz');old=np.load(R/f'v36_recent_context_day{day}.npz');assert np.array_equal(z['events'],later[:,6]) and np.array_equal(z['y'],later[:,5])
 model=Context(16);opt=torch.optim.Adam(model.parameters(),lr=.003);rng=np.random.default_rng(3401+day)
 for epoch in range(1,4):
  model.train()
  for a,v,b,k,base,target in batches(td,factors,rng.permutation(td['n'])):
   opt.zero_grad();logits=model(a,v,b,k,base);loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(a),1)],1),target)
   penalty=.001*(model.bucket.square().sum()+model.kind.square().sum()+model.recent.square().sum());(loss+penalty).backward();opt.step()
  print(json.dumps(dict(day=day,epoch=epoch,seconds=time.monotonic()-start_time)),flush=True)
 torch.save(model.state_dict(),R/f'v44_calendar_supported_day{day}.pt');raw,base=predict(model,vd,factors)
 # Fixed correction with separately ranked residuals, no blend search.
 p=rank(rank(z['pred'])+.1*((rank(raw)-rank(base))-(rank(old['raw'])-rank(old['old']))))
 m=measure(z,p);m.update(day=day,weighted_auc=weighted_auc(z,p),baseline_weighted_auc=weighted_auc(z,z['pred']),auc=float(roc_auc_score(z['y'],p)),kind_norms=model.kind.detach().norm(dim=(1,2)).tolist(),training_kind_counts=np.bincount(td['kind'],minlength=2).tolist(),factor_cutoff=start+(day-3)*DAY,fit_last=int(rows[:,2].max()),validation_first=int(later[:,2].min()))
 assert m['fit_last']<start+day*DAY<=m['validation_first']
 np.savez_compressed(R/f'v44_calendar_supported_day{day}.npz',y=z['y'],events=z['events'],pred=p,raw=raw,old=base)
 (R/f'v44_calendar_supported_day{day}_report.json').write_text(json.dumps(m,indent=2));print(json.dumps(m),flush=True);return m

def main():
 files=[R/f'imp3_model_warm16_cut{d-3}.npz' for d in (6,5)]+[R/f'{prefix}_day{d}.npz' for d in (6,5) for prefix in ['v39_regularized_als32','v36_recent_context']]
 frozen=dict(factor_lag_days=3,fit_window_days=3,epochs=3,lr=.003,regularization=.001,seed=3401,weight=.1,selection='No weight search; require ordinary nonfall and positive history-adjusted interval on both old dates',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'v44_calendar_supported_frozen.json').open('x').write(json.dumps(frozen,indent=2));reports=[fit(6)]
 if reports[0]['gain']>=0 and reports[0]['weighted_interval'][0]>0:reports.append(fit(5))
 accepted=len(reports)==2 and all(a['gain']>=0 and a['weighted_interval'][0]>0 for a in reports)
 result=dict(reports=reports,accepted=accepted,official_auc=None,evaluation_csv_ready=False,note='Reused weekend validation dates. Calendar support is improved at final fit, but official gain unproven.')
 (R/'v44_calendar_supported_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
