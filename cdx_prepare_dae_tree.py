"""Data preparation only. All fitting happens in private Kaggle notebook."""
from pathlib import Path
import ast, hashlib, json, shutil, time, zipfile
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

B=Path(__file__).resolve().parent;A=B/'artifacts';R=A/'v54_horizon/cdx_dae_tree_001';P=R/'kaggle_payload'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
 return h.hexdigest()

def main():
 tic=time.monotonic();R.mkdir(exist_ok=True);P.mkdir(exist_ok=True);assert not (R/'frozen.json').exists()
 plan=dict(date='2026-10-31',hypothesis='The verified catalog reconstruction signal may require nonlinear interactions with existing features inside the main tree; a fixed direct rank correction already failed and is not repeated.',
  novelty='Main354-feature learner with causal new DAEfeatures, not a shallow score stacker, factor variant, blend sweep or AE retuning.',
  fixed_ae='Original16D20epochs50%corruptionAdam.001batch256seed8711; same-day warm16seed7 initial factors. Refit unchanged at cut2..6 only for causal block features.',
  fixed_trees='735trees; exact imp19/imp10 SETTINGS except4KaggleCPUthreads; control354columns, treatment364; unweighted training, same rows and seed42.',
  extra_features=['DAE personal dot','dot plus original item bias','catalog reconstruction logit','cosine','log1p item norm','log1p profile norm','logit minus event mean','logit minus event max','conditional item probability','catalog log probability'],
  train_sample='20%events using eventindex*2654435761mod100<20; rolling blocks2/3/4->day5 and3/4/5->day6',
  screen_sample='Existing20%userhash, allowed daily first events only; no reserved day6 events',
  bridge='Frozen strongest H1/V53 rank plus .5*(rank(treatment tree)-rank(control tree)); .5is original imp19 tree replacement setting; no search.',
  ranking='Joint across entire ALLOWED DAILY cohort before20%screen slicing; never separately by history group. Full daily rows have no labels in feature construction.',
  gate='Screen+.0002BOTHdatesBOTHweightings3binandhistorical5bin; only survivor may score fullallowedcohort with+.0005BOTHdatesBOTHweightings.',
  baseline_caveat='Inherited H1day5usercrossfit;day6earlierday5fit. This is main-tree feature screening, not full causal V53 rebuild or true multiarrival V53 drift curve.',
  prior_result='DirectDAEstrongintegration rejected; no model/seed/epoch/blend tweaks authorized in this recipe',
  no_holdout=True,no_evaluation=True,no_submission=True,no_external_code=True)
 # Freeze the recipe before any new fit or metric.
 plan['source_hashes']={p.name:sha(p) for p in [Path(__file__),B/'cdx_dae_tree_kaggle.py',B/'imp19_tree.py',B/'imp10_leafwise.py',B/'imp3_warmfeat.py']}
 (R/'frozen.json').open('x').write(json.dumps(plan,indent=2));shutil.copyfile(R/'frozen.json',P/'frozen.json')
 t=np.load(A/'train.npy',mmap_mode='r');h=np.load(A/'train_hist.npy',mmap_mode='r');ht=np.load(A/'train_times.npy',mmap_mode='r');head=t[::10];start=int(head[0,2]);info={};hashes={}
 def record(p):hashes[str(p.relative_to(B))]=sha(p)
 def features(name,idx):
  cut=int(name[3:]);files=[f'stack_{name}_c{cut}.npy',f'imp3_wf_{name}.npy']+[f'{tag}_{name}.npy' for tag in ['imp4_decompose','imp5_aligned','imp6_context','imp17_als','imp19_als_aligned']]
  cols=[]
  for filename in files:
   path=A/'features'/filename;source=np.load(path,mmap_mode='r');cols.append(np.asarray(source[idx]));record(path)
  x=np.column_stack(cols);assert x.shape==(len(idx),354) and np.isfinite(x).all();return x
 for cut in [2,3,4,5,6]:
  cutoff=start+cut*86400;n=int(np.searchsorted(head[:,2],cutoff));past=np.asarray(t[:n*10]);e=np.asarray(head[:n]);_,first=np.unique(e[:,0],return_index=True);e=e[first]
  hh=np.asarray(h[e[:,6]]);tt=np.asarray(ht[e[:,6]]);valid=hh>=0;assert np.all((tt<e[:,2,None])|~valid)
  click=past[past[:,5]==1];u=np.r_[click[:,0],np.broadcast_to(e[:,0,None],hh.shape)[valid]];ii=np.r_[click[:,1],hh[valid]];ts=np.r_[click[:,2],tt[valid]]
  order=np.lexsort((ts,ii,u));u,ii,ts=u[order],ii[order],ts[order];keep=np.r_[True,(u[1:]!=u[:-1])|(ii[1:]!=ii[:-1])|(ts[1:]!=ts[:-1])];assert ts.max()<cutoff
  initial=A/(f'imp3_model_warm16_s7_cut{cut}.npz' if cut<5 else f'imp3_model_warm16_s7_day{cut}.npz');record(initial);m=np.load(initial)
  x=sp.csr_matrix((np.ones(int(keep.sum()),np.float32),(u[keep],ii[keep])),shape=(len(m['U']),50000));sp.save_npz(P/f'clicks_cut{cut}.npz',x);shutil.copyfile(initial,P/f'initial_cut{cut}.npz')
  info[str(cut)]=dict(cutoff=cutoff,max_fit_timestamp=int(ts.max()),training_events=n,active_profiles=int((np.asarray(x.sum(1)).ravel()>0).sum()),deduplicated_clicks=int(keep.sum()))
  if cut<=5:
   stop=int(np.searchsorted(head[:,2],cutoff+86400));events=np.arange(n,stop);take=(events.astype(np.uint64)*np.uint64(2654435761)%100)<20;local=np.flatnonzero(take);events=events[take];rows=events[:,None]*10+np.arange(10)
   hist=np.asarray(h[events]);stamp=np.asarray(ht[events]);assert np.all((stamp<head[events,2,None])|(hist<0))
   xx=features(f'day{cut}',(local[:,None]*10+np.arange(10)).ravel());assert np.array_equal(xx[:,4],t[rows.ravel(),3]) and np.array_equal(xx[:,5],t[rows.ravel(),4])
   np.save(P/f'x_block{cut}.npy',xx);np.savez_compressed(P/f'block{cut}.npz',y=np.asarray(t[rows.ravel(),5]).reshape(-1,10).astype(np.int8),events=events,history=hist,items=np.asarray(t[rows,1]),timestamps=np.asarray(head[events,2]))
   info[str(cut)]['block_events']=len(events);del xx
  if cut>=5:
   old=A/'v54_horizon/cdx_dae_strong_001/kaggle_payload';f=np.load(old/f'features_day{cut}.npz');ev=f['events'];allowed=np.load(A/f'selection_day{cut}.npz');idx=allowed['full_indices'] if cut==6 else np.arange(len(ev)*10)
   assert np.array_equal(allowed['events'][::10],ev);rows=ev[:,None]*10+np.arange(10)
   xx=features(f'val{cut}',idx);assert np.array_equal(xx[:,4],t[rows.ravel(),3])
   # All ALLOWED cohort features are supplied without its outcomes. Joint ranks
   # before sample selection are essential to reproduce the frozen comparison.
   np.save(P/f'x_val{cut}.npy',xx);np.savez_compressed(P/f'validation{cut}.npz',history=f['history'],items=f['items'],parent=f['control'],events=ev,sample=f['sample'],w3=f['sw3'],w5=f['w5'][f['sample']],history_group=f['history_group'])
   np.save(P/f'labels_val{cut}.npy',allowed['y'].reshape(-1,10)[f['sample']].astype(np.int8))
   record(A/f'selection_day{cut}.npz');record(old/f'features_day{cut}.npz');info[str(cut)].update(validation_events=len(ev),screen_events=int(f['sample'].sum()));del xx
  print('PREPARED',cut,json.dumps(info[str(cut)]),flush=True)
 (R/'input_hashes.json').write_text(json.dumps(hashes,indent=2));(P/'data_info.json').write_text(json.dumps(info,indent=2));shutil.copyfile(B/'cdx_dae_tree_kaggle.py',P/'cdx_dae_tree_kaggle.py')
 manifest=dict(hashes={p.name:sha(p) for p in P.iterdir() if p.name!='upload_manifest.json'},no_holdout=True,no_evaluation=True,no_submissions=True);(P/'upload_manifest.json').write_text(json.dumps(manifest,indent=2))
 path=R/'cdx_dae_tree_kaggle.zip'
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in P.iterdir():z.write(p,p.name)
 cell="""from pathlib import Path
import importlib.util,zipfile
archive=list(Path('/kaggle/input').rglob('cdx_dae_tree_kaggle.zip'));assert len(archive)==1
target=Path('/kaggle/working/cdx_dae_tree_inputs');target.mkdir(exist_ok=True)
with zipfile.ZipFile(archive[0]) as z:z.extractall(target)
spec=importlib.util.spec_from_file_location('cdx_dae_tree',target/'cdx_dae_tree_kaggle.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.run(target)
"""
 display="""import pandas as pd
display(pd.DataFrame(result['screen'])[['day','weight','events','before','after','gain','paired_95']])
print('Screen passed:',result['screen_passed'],'; No holdout, evaluation or submission.')
"""
 nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Catalog DAE features inside the main tree\nFixed causal reconstruction signal, matched354/364feature trees. No AE/blend/seed search. Private20%screen.\n'])]+[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=c.splitlines(keepends=True)) for c in [cell,display]],metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=4)
 (R/'cdx_dae_tree_kaggle.ipynb').write_text(json.dumps(nb,indent=2))
 protected=json.loads((A/'v54_horizon/protected_before.json').read_text());assert all(sha(B/k)==v['sha256'] for k,v in protected.items())
 (R/'preflight.json').write_text(json.dumps(dict(protected_unchanged=len(protected),bundle_bytes=path.stat().st_size,seconds=time.monotonic()-tic,metadata=info,validation_metrics_accessed=False),indent=2))
 print('READY',str(path),path.stat().st_size/1e6,flush=True)

if __name__=='__main__':main()
