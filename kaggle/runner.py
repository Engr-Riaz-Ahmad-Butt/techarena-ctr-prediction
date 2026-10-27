"""Portable CPU/GPU XGBoost experiments over checksum-verified causal caches."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PARAMS=dict(objective='binary:logistic',tree_method='hist',max_bin=63,max_depth=6,eta=.05,min_child_weight=50,subsample=.8,colsample_bytree=.9,reg_lambda=5.,reg_alpha=.1,nthread=4,seed=6001,max_cached_hist_node=32)
rank=lambda p:rankdata(p,method='average')/(len(p)+1)

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def load_bundle(root,verify=True):
 root=Path(root).resolve();manifest=json.loads((root/'bundle_manifest.json').read_text())
 assert manifest['schema']==1 and manifest['feature_width']==354
 if verify:
  for i,(name,info) in enumerate(manifest['files'].items()):
   path=(root/name).resolve();assert path.is_relative_to(root),name
   if not path.is_file() or path.stat().st_size!=info['bytes'] or digest(path)!=info['sha256']:raise ValueError('Bundle integrity check failed: '+name)
   if i%10==0:print(f'Checked {i+1}/{len(manifest["files"])} input files',flush=True)
 return manifest

def choose_device(requested='auto'):
 if requested not in ['auto','cpu','cuda']:raise ValueError('DEVICE must be auto, cpu or cuda')
 if requested=='cpu':return 'cpu',None
 try:
  import cupy as cp
  assert cp.cuda.runtime.getDeviceCount()>0
  return 'cuda',cp
 except Exception as error:
  if requested=='cuda':raise RuntimeError('Enable a Kaggle GPU accelerator, or select DEVICE="cpu"') from error
  return 'cpu',None

class FeatureBlocks:
 def __init__(self,root,groups):
  self.root=Path(root);self.groups=groups;self.sizes=[]
  for group in groups:
   arrays=[np.load(self.root/p,mmap_mode='r') for p in group];n=len(arrays[0]);assert all(len(a)==n for a in arrays) and sum(a.shape[1] for a in arrays)==354;self.sizes.append(n)
   for a in arrays:a._mmap.close()
  self.ends=np.cumsum(self.sizes);self.starts=np.r_[0,self.ends[:-1]]
 def __len__(self):return int(self.ends[-1])
 def block(self,start,end):
  parts=[]
  for left,right,group in zip(self.starts,self.ends,self.groups):
   a=max(start,int(left));b=min(end,int(right))
   if a<b:
    arrays=[np.load(self.root/p,mmap_mode='r') for p in group];parts.append(np.column_stack([p[a-left:b-left] for p in arrays]).astype('float32',copy=False))
    for p in arrays:p._mmap.close()
  out=parts[0] if len(parts)==1 else np.concatenate(parts);assert out.shape==(end-start,354) and np.isfinite(out).all();return out

class BatchIterator(xgb.DataIter):
 def __init__(self,features,y,cp=None,batch_size=100000):
  self.features=features;self.y=y;self.cp=cp;self.batch=batch_size;self.position=0;super().__init__(release_data=True)
 def reset(self):self.position=0
 def next(self,input_data):
  if self.position>=len(self.y):return False
  end=min(self.position+self.batch,len(self.y));x=self.features.block(self.position,end);y=self.y[self.position:end]
  if self.cp is not None:x=self.cp.asarray(x);y=self.cp.asarray(y)
  input_data(data=x,label=y);self.position=end;return True

class Progress(xgb.callback.TrainingCallback):
 def __init__(self,path,total):self.path=Path(path);self.total=total;self.started=time.monotonic()
 def after_iteration(self,model,epoch,evals_log):
  if epoch==0 or (epoch+1)%25==0:
   z=dict(rounds_done=epoch+1,total_rounds=self.total,elapsed_seconds=round(time.monotonic()-self.started,1));(self.path/'progress.json').write_text(json.dumps(z));print(z,flush=True)
  if (epoch+1)%100==0:model.save_model(self.path/f'checkpoint_{epoch+1}.ubj')
  return False

def weighted_influence(y,p,w):
 def cdf(values,weight,query):
  ix=np.argsort(values);values=values[ix];weight=weight[ix];cs=np.r_[0.,np.cumsum(weight)]
  return (cs[np.searchsorted(values,query,'left')]+cs[np.searchsorted(values,query,'right')])/(2*weight.sum())
 pos=y==1;auc=roc_auc_score(y,p,sample_weight=w);out=np.empty(len(y));out[pos]=w[pos]/w[pos].sum()*(cdf(p[~pos],w[~pos],p[pos])-auc);out[~pos]=w[~pos]/w[~pos].sum()*(1-cdf(p[pos],w[pos],p[~pos])-auc)
 return out.reshape(-1,10).sum(1)

def metrics(y,p,baseline,w):
 gain=float(roc_auc_score(y,p,sample_weight=w)-roc_auc_score(y,baseline,sample_weight=w));se=float(np.linalg.norm(weighted_influence(y,p,w)-weighted_influence(y,baseline,w)));ordinary=float(roc_auc_score(y,p)-roc_auc_score(y,baseline))
 return dict(ordinary_auc=float(roc_auc_score(y,p)),baseline_ordinary_auc=float(roc_auc_score(y,baseline)),history_adjusted_auc=float(roc_auc_score(y,p,sample_weight=w)),baseline_history_adjusted_auc=float(roc_auc_score(y,baseline,sample_weight=w)),ordinary_gain=ordinary,adjusted_gain=gain,adjusted_gain_interval=[gain-1.96*se,gain+1.96*se],positive_gate=bool(ordinary>0 and gain-1.96*se>0))

def experiment(root,manifest,output,day=6,device='auto',rounds=700,mode='main'):
 if tuple(int(x) for x in xgb.__version__.split('.')[:2])<(3,0):raise RuntimeError('XGBoost >=3.0 is required. Install the notebook dependency cell with Internet enabled.')
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);dev,cp=choose_device(device)
 if mode=='main':days=list(range(day-3,day));validation=manifest['validation'][str(day)]
 elif mode=='weekday':days=[2,3];validation=manifest['weekday']
 else:raise ValueError('mode must be main or weekday')
 specs=[manifest['train'][str(d)] for d in days];labels=[]
 for spec in specs:
  meta=json.loads((root/spec['metadata']).read_text());assert meta['last_timestamp']<validation['cutoff'];labels.append(np.load(root/spec['labels']))
 y=np.concatenate(labels).astype('float32');features=FeatureBlocks(root,[s['features'] for s in specs]);assert len(y)==len(features)
 params=dict(PARAMS,device=dev);recipe=dict(params=params,rounds=rounds,mode=mode,day=day,fit_days=days,blend_weight=.1 if mode=='main' else .2,input_manifest_sha256=digest(root/'bundle_manifest.json'),library=xgb.__version__,device=dev)
 recipe_path=output/'recipe.json';model_path=output/'model.ubj'
 if model_path.exists():
  assert json.loads(recipe_path.read_text())==recipe,'Use a new RUN_NAME for a changed recipe/device.';done=json.loads((output/'trained.json').read_text());assert digest(model_path)==done['model_sha256'];model=xgb.Booster();model.load_model(model_path);model.set_param({'device':dev,'nthread':4});print('Reusing verified completed model',flush=True)
 else:
  if recipe_path.exists() and json.loads(recipe_path.read_text())!=recipe:raise ValueError('Output contains a different recipe. Choose a new RUN_NAME.')
  recipe_path.write_text(json.dumps(recipe,indent=2));iterator=BatchIterator(features,y,cp);data=xgb.QuantileDMatrix(iterator,nthread=4,max_bin=63);assert data.num_row()==len(y) and data.num_col()==354
  model=xgb.train(params,data,num_boost_round=rounds,callbacks=[Progress(output,rounds)]);temporary=output/'model.pending.ubj';model.save_model(temporary);temporary.replace(model_path);(output/'trained.json').write_text(json.dumps(dict(model_sha256=digest(model_path),rounds=model.num_boosted_rounds())))
  del data,iterator
 val=np.load(root/validation['data']);xv=FeatureBlocks(root,[validation['features']]);assert len(xv)==len(val['y']);raw=[]
 for off in range(0,len(xv),25000):
  x=xv.block(off,min(off+25000,len(xv)));p=model.inplace_predict(cp.asarray(x) if cp is not None else x);raw.append(cp.asnumpy(p) if cp is not None else p)
 raw=np.concatenate(raw);assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
 if mode=='main':baseline=val['baseline'];pred=rank(.9*rank(baseline)+.1*rank(raw))
 else:baseline=rank(.9*rank(val['catboost'])+.1*rank(val['neural']));pred=rank(.7*rank(val['catboost'])+.2*rank(raw)+.1*rank(val['neural']))
 m=metrics(val['y'],pred,baseline,val['weights']);m.update(mode=mode,day=day,metric_scope='history adjusted' if mode=='main' else 'joint history/calendar adjusted; different cohort from main',rows=len(raw),device=dev,csv_ready=False,official_auc=None,target_achieved=False,notes='Local diagnostic only. Reused validation cohorts; intervals exclude fitting/selection uncertainty. No official score forecast.')
 np.savez_compressed(output/'predictions.npz',y=val['y'],events=val['events'],raw=raw,pred=pred,baseline=baseline,weights=val['weights']);(output/'report.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2),flush=True);return m

def run(root,output,device='auto',mode='main',rounds=700,verify=True):
 manifest=load_bundle(root,verify);output=Path(output);reports=[experiment(root,manifest,output/'day6' if mode=='main' else output/'weekday',6,device,rounds,mode)]
 if mode=='main' and reports[0]['positive_gate']:reports.append(experiment(root,manifest,output/'day5',5,device,rounds,mode))
 summary=dict(reports=reports,eligible_for_review=mode=='main' and len(reports)==2 and all(m['positive_gate'] for m in reports),minimum_upload_threshold=.82,requested_target=.834786,target_achieved=False,csv_ready=False,official_auc=None,note='No competition CSV export. Main acceptance and evaluation refit/export require review; a weekday score cannot replace the main metric.')
 (output/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def archive_results(output,destination):
 output=Path(output).resolve();destination=Path(destination).resolve();assert not destination.is_relative_to(output),'Put the archive outside the results directory'
 with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in output.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(output))
 return destination
