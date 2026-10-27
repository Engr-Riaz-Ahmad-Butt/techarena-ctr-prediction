"""Matched warm conditional-choice continuation: 16 versus expanded 32 factors.

Same negative draws, event orders, optimizer reset and epochs in both arms.
Choose expansion settings on day6 only, require improvement over its matched
control, then confirm unchanged on day5. No held-out labels enter fitting.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp2_warm_choice import event_data,warm_data,warm_epoch,train_epoch,predict_events,train,hist,split,ni,nu,start,DAY
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

def fit(day,chosen=None):
 tic=time.monotonic();rows=np.asarray(train[train[:,2]<start+day*DAY]);users,items,history,target=event_data(rows,hist);wu,wi=warm_data(rows);seen=np.bincount(rows[:,0],minlength=nu)>0;last=int(rows[:,2].max());del rows
 _,val=split(day);vu,vi,vh,_=event_data(val,hist);base=np.load(R/f'v50_crossfit_experts_day{day}.npz');assert np.array_equal(val[:,6],base['events']) and np.array_equal(val[:,5],base['y'])
 with np.load(R/f'imp2_model_warm1.0_d16_pc1_day{day}.npz') as old:initial=[old[k].copy() for k in ['U','V','H','bu','bi']]
 original=predict_events(vu,vi,vh,*initial).ravel();counts=np.bincount(wi,minlength=ni)+1;pop_bias=np.log(counts/counts.mean()).astype('float32');outputs={};trials=[]
 epochs=chosen['epoch'] if chosen else 6;checks=[epochs] if chosen else [2,4,6]
 for dim in [16,32]:
  params=[p.copy() for p in initial]
  if dim==32:
   initrng=np.random.default_rng(5201+day)
   for j in range(3):
    extra=(initrng.normal(size=params[j].shape)*float(np.std(params[j]))*.25).astype('float32')
    if j==0:extra[~seen]=0
    params[j]=np.column_stack([params[j],extra]);assert np.array_equal(params[j][:,:16],initial[j])
  accum=[np.full_like(p,.1) for p in params];rng=np.random.default_rng(5211+day)
  for epoch in range(1,epochs+1):
   neg=rng.integers(0,ni,size=(len(wu),11),dtype=np.int32);collision=neg==wi[:,None]
   while collision.any():neg[collision]=rng.integers(0,ni,size=int(collision.sum()),dtype=np.int32);collision=neg==wi[:,None]
   warm_loss=warm_epoch(rng.permutation(len(wu)),wu,wi,neg,params[0],params[1],accum[0],accum[1],.015,.002,1.,pop_bias)
   loss=train_epoch(rng.permutation(len(users)),users,items,history,target,*params,*accum,.015,.002)
   print(json.dumps(dict(day=day,dim=dim,epoch=epoch,warm_loss=float(warm_loss),loss=float(loss),elapsed_s=time.monotonic()-tic)),flush=True)
   if epoch not in checks:continue
   raw=predict_events(vu,vi,vh,*params).ravel();assert np.isfinite(raw).all()
   np.savez_compressed(R/f'v52_choice_day{day}_d{dim}_ep{epoch}_model.npz',U=params[0],V=params[1],H=params[2],bu=params[3],bi=params[4],cutoff=start+day*DAY,epoch=epoch,dim=dim)
   for weight in ([chosen['weight']] if chosen else [.05,.1]):
    p=rank(rank(base['pred'])+weight*(rank(raw)-rank(original)));wa=weighted_auc(base,p);outputs[(dim,epoch,weight)]=(raw,p);trials.append(dict(dim=dim,epoch=epoch,weight=weight,weighted_auc=wa,ordinary_auc=float(roc_auc_score(base['y'],p)),raw_auc=float(roc_auc_score(base['y'],raw))))
   np.savez_compressed(R/f'v52_choice_day{day}_d{dim}_ep{epoch}.npz',y=base['y'],events=base['events'],raw=raw)
 best=max([r for r in trials if r['dim']==32],key=lambda r:r['weighted_auc']);epoch=best['epoch'];weight=best['weight'];raw,p=outputs[(32,epoch,weight)];cr,cp=outputs[(16,epoch,weight)]
 metric=measure(base,p);metric.update(best);metric.update(day=day,fit_last=last,cutoff=start+day*DAY,baseline_weighted_auc=weighted_auc(base,base['pred']),initial_raw_auc=float(roc_auc_score(base['y'],original)))
 control={'y':base['y'],'events':base['events'],'pred':cp};vs_control=measure(control,p)
 np.savez_compressed(R/f'v52_choice_capacity_day{day}.npz',y=base['y'],events=base['events'],pred=p,raw=raw,control=cp,control_raw=cr,old=original)
 result=dict(metrics=metric,vs_matched_control=vs_control,trials=trials);(R/f'v52_choice_capacity_day{day}_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result

def good(r):return r['metrics']['gain']>=0 and r['metrics']['weighted_interval'][0]>0 and r['vs_matched_control']['weighted_interval'][0]>0
def main():
 files=[R/f'imp2_model_warm1.0_d16_pc1_day{d}.npz' for d in (6,5)]+[R/f'v50_crossfit_experts_day{d}.npz' for d in (6,5)]
 frozen=dict(epochs=[2,4,6],weights=[.05,.1],arms=[16,32],lr=.015,regularization=.002,auxiliary_weight=1.,pop_correct=True,accumulator_reset=.1,extra_scale=.25,initialization_seed=5201,batch_seed=5211,blas_threads=1,initialization='Converged same-cutoff warm16 plus 16 small Gaussian columns; unseen-user extra columns zero',selection='Expanded arm chosen on day6 adjusted AUC; require gain and positive adjusted interval against baseline and matched control; then unchanged day5',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['v52_choice_capacity.py','imp2_warm_choice.py','choice64_core.py']})
 (R/'v52_choice_capacity_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with threadpool_limits(limits=1):
  reports=[fit(6)]
  (R/'v52_choice_capacity_selection.json').open('x').write(json.dumps(reports[0]['metrics'],indent=2))
  if good(reports[0]):reports.append(fit(5,reports[0]['metrics']))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(good(r) for r in reports),csv_ready=False,official_auc=None,note='Repeated validation dates and primary selection; conditional intervals exclude selection uncertainty. No official-score forecast.')
 (R/'v52_choice_capacity_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
