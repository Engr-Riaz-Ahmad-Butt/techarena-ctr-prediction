"""Export the approved ALS32 correction over the verified V36 submission."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp15_als import prepare,augmented,solve_rows,cutoff_of,score as legacy_score,ni,nu,evaluation,ehist
from v39_regularized_als32 import score,rank

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 set_num_threads(4)
 with threadpool_limits(limits=1):
  audit=json.loads((R/'v39_regularized_als32_audit.json').read_text());boot=json.loads((R/'v39_regularized_als32_bootstrap.json').read_text());assert audit['accepted'] and boot['passes']
  output=Path('submission_v39.csv');assert not output.exists()
  parent=Path('submission_v36.csv');checks=json.loads((R/'v36_submission_checks.json').read_text());assert sha(parent)==checks['csv_sha256']
  paths=[parent,R/'v36_final_pred.npy',R/'imp17_als_cut7.npz',R/'imp15_als_final_raw.npy']
  frozen=dict(input_sha256={str(p.resolve()):sha(p) for p in paths},dim=32,iterations=6,latent_regularization=4.,bias_regularization=20.,new_dimension_scale=.25,seed=3808,cutoff=cutoff_of(7),mode='replace_component',weight=.05,local_adjusted_auc=audit['reports'][0]['weighted_auc'],local_ordinary_auc=audit['reports'][0]['auc'],note='Local history-adjusted AUC is a diagnostic proxy, not an official-score forecast. Reuses the verified V36 final backbone, including its recorded earlier-parent provenance.')
  (R/'v39_export_frozen.json').open('x').write(json.dumps(frozen,indent=2))
  times=np.load(R/'evaluation_times.npy',mmap_mode='r')
  with np.load(R/'imp17_als_cut7.npz') as f:
   assert int(f['cutoff'])==cutoff_of(7) and int(f['iteration'])==6
   U0,V0,bu0,bi0,exposures0=[f[k].copy() for k in ['U','V','bu','bi','exposures']];g0=float(f['g'])
  prior=legacy_score(np.asarray(evaluation),U0,V0,bu0,bi0,g0,exposures0,histories=ehist,history_times=times)
  saved=np.load(R/'imp15_als_final_raw.npy');error=float(np.max(np.abs(prior-saved)));assert error<1e-10,error
  print(json.dumps(dict(stage='prior_final_model_replay',maximum_error=error)),flush=True)
  w,t,wt,tt,g,up,ip,confidence,exposures=prepare(7)
  assert abs(g-g0)<1e-15
  rng=np.random.default_rng(3808);extra=rng.normal(size=(ni,16))*float(np.std(V0))*.25
  V=np.column_stack([V0,extra]);U=np.zeros((nu,32));bu=bu0.copy();bi=bi0.copy()
  for iteration in range(1,7):
   x=augmented(V);bu,U=solve_rows(w.indptr,w.indices,w.data,t.data,V,bi,x.T@x,-x.T@(g+bi),confidence,g,up,latent_reg=4.)
   x=augmented(U);bi,V=solve_rows(wt.indptr,wt.indices,wt.data,tt.data,U,bu,x.T@(confidence[:,None]*x),-x.T@(confidence*(g+bu)),np.ones(ni),g,ip,latent_reg=4.)
   np.savez_compressed(R/'v39_als32_final.npz',U=U,V=V,bu=bu,bi=bi,g=g,exposures=exposures,cutoff=cutoff_of(7),iteration=iteration,latent_regularization=4.,bias_regularization=20.)
   print(json.dumps(dict(stage='final_ALS32',iteration=iteration,total=6)),flush=True)
  raw=score(np.asarray(evaluation),U,V,bu,bi,g,exposures,histories=ehist,history_times=times)
  base=np.load(R/'v36_final_pred.npy');pred=rank(rank(base)+.05*(rank(raw)-rank(prior)))
  assert len(pred)==529390 and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
  np.save(R/'v39_final_raw.npy',raw);np.save(R/'v39_final_pred.npy',pred)
  temporary=output.with_suffix('.csv.tmp');ids=set();ih=hashlib.sha256();n=0
  with parent.open(newline='',encoding='utf-8') as src,temporary.open('w',newline='',encoding='utf-8') as dest:
   writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
   for row,p in zip(csv.DictReader(src),pred,strict=True):
    assert row['rowid'] not in ids;ids.add(row['rowid']);ih.update((row['rowid']+'\n').encode());writer.writerow([row['rowid'],format(float(p),'.15g')]);n+=1
  assert n==529390;temporary.replace(output)
  for name,h in frozen['input_sha256'].items():assert sha(Path(name))==h
  manifest=dict(output=str(output.resolve()),rows=n,csv_sha256=sha(output),rowids_sha256=ih.hexdigest(),local_adjusted_auc=frozen['local_adjusted_auc'],local_ordinary_auc=frozen['local_ordinary_auc'],official_auc=None,cutoff=cutoff_of(7),artifact_sha256={p.name:sha(p) for p in [R/'v39_als32_final.npz',R/'v39_final_raw.npy',R/'v39_final_pred.npy']},note=frozen['note'])
  (R/'v39_submission_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
