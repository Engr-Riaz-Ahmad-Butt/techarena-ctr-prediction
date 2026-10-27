"""Package only the prepared causal inputs needed for cloud experiments."""
import hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]
R=BASE/'artifacts'
OUT=R/'kaggle_ready'
PREFIXES=['stack','imp3_wf','imp4_decompose','imp5_aligned','imp6_context','imp17_als','imp19_als_aligned']

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def feature_paths(name,cut):
 return [R/'features'/(f'stack_{name}_c{cut}.npy' if p=='stack' else f'{p}_{name}.npy') for p in PREFIXES]

def build():
 OUT.mkdir(exist_ok=True);stage=OUT/'metadata';stage.mkdir(exist_ok=True);archive=OUT/'techarena_kaggle_data.zip';assert not archive.exists(),'Bundle already exists; verify it rather than overwrite it'
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min());target=np.bincount(np.digitize((eh>=0).sum(1),[0,4,9,19],right=True),minlength=5)/len(eh)
 files={};manifest=dict(schema=1,feature_width=354,train={},validation={},files={},baseline='V53 main local day6 history-adjusted AUC 0.8171690462350286; official V35 0.816623',requested_target=.834786,minimum_before_upload=.82,notes='Prepared causal caches, not raw CSV preprocessing. No final submission export. Validation dates reused; no official-score forecast.')
 def add(source,name):
  source=Path(source);assert source.is_file();files[name]=source;return name
 for day in [2,3,4,5]:
  rows=np.asarray(t[(t[:,2]>=start+day*86400)&(t[:,2]<start+(day+1)*86400)]);p=stage/f'day{day}_labels.npy';np.save(p,rows[:,5].astype('float32'));meta=stage/f'day{day}.json';meta.write_text(json.dumps(dict(rows=len(rows),first_timestamp=int(rows[:,2].min()),last_timestamp=int(rows[:,2].max()),feature_cutoff=start+day*86400)))
  paths=feature_paths(f'day{day}',day);assert all(len(np.load(p,mmap_mode='r'))==len(rows) for p in paths)
  manifest['train'][str(day)]=dict(labels=add(p,f'labels/day{day}.npy'),metadata=add(meta,f'metadata/day{day}.json'),features=[add(p,'features/'+p.name) for p in paths]);del rows
 for day in [5,6]:
  z=np.load(R/f'v53_causal_propensity_day{day}.npz');events=z['events'];rows=np.asarray(t[events*10+np.tile(np.arange(10),len(events)//10)]);assert np.array_equal(rows[:,5],z['y']) and rows[:,2].min()>=start+day*86400;assert np.all((ht[rows[::10,6]]<rows[::10,2,None])|(h[rows[::10,6]]<0));group=np.digitize((h[events]>=0).sum(1),[0,4,9,19],right=True);source=np.bincount(group,minlength=5)/len(group);weights=(target/source)[group]
  paths=feature_paths(f'val{day}',day);x=np.load(paths[0],mmap_mode='r');assert len(x)==len(rows) and np.array_equal(x[:,4],rows[:,3]) and np.array_equal(x[:,5],rows[:,4]);p=stage/f'validation_day{day}.npz';np.savez_compressed(p,y=z['y'],events=events,baseline=z['pred'],weights=weights)
  manifest['validation'][str(day)]=dict(cutoff=start+day*86400,data=add(p,f'validation/day{day}.npz'),features=[add(p,'features/'+p.name) for p in paths]);del rows,x
 cb=np.load(R/'v56_weekday/v56_predictions.npz');ctx=np.load(R/'v42_calendar_holdout.npz');assert np.array_equal(cb['events'],ctx['events']) and np.array_equal(cb['y'],ctx['y']);p=stage/'weekday.npz';np.savez_compressed(p,y=cb['y'],events=cb['events'],weights=cb['weights'],catboost=cb['baseline'],neural=ctx['neural']);manifest['weekday']=dict(cutoff=start+4*86400,data=add(p,'validation/weekday.npz'),features=[add(R/'v56_weekday/v56_future_features.npy','features/weekday_future.npy')])
 add(BASE/'kaggle/runner.py','runner.py')
 for name,path in files.items():manifest['files'][name]=dict(bytes=path.stat().st_size,sha256=sha(path))
 manifest_path=stage/'bundle_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2));raw_bytes=sum(p.stat().st_size for p in files.values());assert shutil.disk_usage(OUT).free>raw_bytes,'Not enough free space for the upload archive'
 temporary=archive.with_suffix('.partial.zip')
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  z.write(manifest_path,'bundle_manifest.json')
  for n,(name,path) in enumerate(files.items(),1):
   z.write(path,name);assert sha(path)==manifest['files'][name]['sha256'],'Input changed during packaging';print(json.dumps(dict(packaged=n,total=len(files),file=name)),flush=True)
 temporary.replace(archive);report=dict(archive=str(archive),bytes=archive.stat().st_size,uncompressed_bytes=raw_bytes,sha256=sha(archive),files=len(files),manifest_sha256=sha(manifest_path));(OUT/'bundle_build.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':build()
