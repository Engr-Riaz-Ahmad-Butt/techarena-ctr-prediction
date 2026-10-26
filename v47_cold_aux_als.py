"""Matched ALS continuation with optional pre-cutoff histories of unseen users.

Only supplied history items/timestamps from unlabeled held-out events enter the
auxiliary fitting observations. Held-out clicked labels do not enter fitting.
Control and treatment each continue the same V39 checkpoint for six iterations.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from imp15_als import prepare,augmented,solve_rows,split,cutoff_of,ni,nu
from v39_regularized_als32 import score,rank,weighted_auc
from v24_mass import measure

def fit(day):
 tic=time.monotonic();w,t,wt,tt,g,up,ip,confidence,exposures=prepare(day)
 _,rows=split(day);h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');head=rows[::10];cold=head[exposures[head[:,0]]==0]
 history=np.asarray(h[cold[:,6]]);stamps=np.asarray(ht[cold[:,6]]);users=np.repeat(cold[:,0],20);items=history.ravel();stamps=stamps.ravel();valid=(items>=0)&(stamps<cutoff_of(day))
 keys=np.rec.fromarrays([users[valid],items[valid],stamps[valid]],names='user,item,timestamp');_,unique=np.unique(keys,return_index=True);users=users[valid][unique];items=items[valid][unique];stamps=stamps[valid][unique]
 assert (stamps<cutoff_of(day)).all() and (exposures[users]==0).all()
 coverage=dict(day=day,cold_validation_users=len(cold),auxiliary_clicks=len(users),auxiliary_users=len(np.unique(users)),all_auxiliary_before_cutoff=True,auxiliary_last_timestamp=int(stamps.max()) if len(stamps) else None)
 print(json.dumps(coverage),flush=True)
 (R/f'v47_cold_aux_day{day}_coverage.json').write_text(json.dumps(coverage,indent=2))
 if len(users)==0:raise ValueError('No auxiliary information available')
 extra=coo_matrix((np.ones(len(users)),(users,items)),shape=(nu,ni)).tocsr();counts=np.bincount(users,minlength=nu)
 z=np.load(R/f'v39_regularized_als32_day{day}.npz');assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
 with np.load(R/f'v39_regularized_als32_day{day}_checkpoint.npz') as old:initial={k:old[k].copy() for k in ['U','V','bu','bi']}
 predictions={};raws={};metrics={}
 for arm in ['control','cold_aux']:
  aw=w if arm=='control' else (w+extra).tocsr();at=t if arm=='control' else (t+extra).tocsr();aw.sort_indices();at.sort_indices()
  # Sparse addition discards explicit target zeros; restore identical support.
  if not (np.array_equal(aw.indptr,at.indptr) and np.array_equal(aw.indices,at.indices)):
   rr=np.repeat(np.arange(nu),np.diff(aw.indptr));values=np.asarray(at[rr,aw.indices]).ravel()
   at=csr_matrix((values,aw.indices.copy(),aw.indptr.copy()),shape=aw.shape)
  assert np.array_equal(aw.indptr,at.indptr) and np.array_equal(aw.indices,at.indices)
  awt=aw.T.tocsr();att=at.T.tocsr();conf=confidence if arm=='control' else confidence+9.*counts/ni
  U,V,bu,bi=[initial[k].copy() for k in ['U','V','bu','bi']]
  checkpoint=R/f'v47_cold_aux_day{day}_{arm}_checkpoint.npz'
  completed=False
  if checkpoint.exists():
   with np.load(checkpoint) as c:
    assert int(c['cutoff'])==cutoff_of(day) and int(c['iteration'])==12
    U,V,bu,bi=[c[k].copy() for k in ['U','V','bu','bi']];assert np.array_equal(c['exposures'],exposures)
   completed=True
  for iteration in ([] if completed else range(1,7)):
   x=augmented(V);bu,U=solve_rows(aw.indptr,aw.indices,aw.data,at.data,V,bi,x.T@x,-x.T@(g+bi),conf,g,up,latent_reg=4.)
   x=augmented(U);bi,V=solve_rows(awt.indptr,awt.indices,awt.data,att.data,U,bu,x.T@(conf[:,None]*x),-x.T@(conf*(g+bu)),np.ones(ni),g,ip,latent_reg=4.)
   print(json.dumps(dict(day=day,arm=arm,iteration=iteration,elapsed_s=time.monotonic()-tic)),flush=True)
  np.savez_compressed(R/f'v47_cold_aux_day{day}_{arm}_checkpoint.npz',U=U,V=V,bu=bu,bi=bi,g=g,exposures=exposures,cutoff=cutoff_of(day),iteration=12)
  raw=score(rows,U,V,bu,bi,g,exposures);p=rank(rank(z['pred'])+.05*(rank(raw)-rank(z['raw'])))
  raws[arm]=raw;predictions[arm]=p;metrics[arm]=measure(z,p);metrics[arm].update(weighted_auc=weighted_auc(z,p))
 comparison={'y':z['y'],'events':z['events'],'pred':predictions['control']};vs_control=measure(comparison,predictions['cold_aux'])
 np.savez_compressed(R/f'v47_cold_aux_day{day}.npz',y=z['y'],events=z['events'],pred=predictions['cold_aux'],control=predictions['control'],raw=raws['cold_aux'],control_raw=raws['control'])
 report=dict(day=day,coverage=coverage,arms=metrics,treatment_vs_control=vs_control,baseline_weighted_auc=weighted_auc(z,z['pred']))
 (R/f'v47_cold_aux_day{day}_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return report

def good(r):
 m=r['arms']['cold_aux'];c=r['treatment_vs_control'];return m['gain']>=0 and m['weighted_interval'][0]>0 and c['weighted_gain']>0
def main():
 set_num_threads(4);files=[R/f'v39_regularized_als32_day{d}{suffix}.npz' for d in (6,5) for suffix in ['', '_checkpoint']]
 frozen=dict(dim=32,extra_iterations=6,latent_regularization=4.,bias_regularization=20.,weight=.05,auxiliary_weight=1.,background_per_aux_click=9.,threads=4,arms=['control','cold_aux'],auxiliary_source='Only supplied histories of users with no training exposure; timestamps strictly before fitting cutoff; labels excluded',selection='Fixed recipe; treatment must improve baseline and control; then unchanged confirmation day5',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 manifest=R/'v47_cold_aux_frozen.json'
 if manifest.exists():
  prior=json.loads(manifest.read_text());archived=R/'v47_original_source.py'
  assert hashlib.sha256(archived.read_bytes()).hexdigest()==prior['source_sha256']
  assert {k:v for k,v in prior.items() if k!='source_sha256'}=={k:v for k,v in frozen.items() if k!='source_sha256'}
  repair=dict(original_source_sha256=prior['source_sha256'],repaired_source_sha256=frozen['source_sha256'],change='Restore explicit zero target support after sparse addition; reuse completed matched-control checkpoint. No model settings changed.')
  (R/'v47_sparse_repair.json').write_text(json.dumps(repair,indent=2))
 else:manifest.open('x').write(json.dumps(frozen,indent=2))
 with threadpool_limits(limits=1):
  reports=[fit(6)]
  if good(reports[0]):reports.append(fit(5))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(reports=reports,accepted=len(reports)==2 and all(good(r) for r in reports),official_auc=None,csv_ready=False,note='Transductive use of supplied unlabeled history, never evaluation clicked labels. Reused local validation; official score unknown.')
 (R/'v47_cold_aux_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
