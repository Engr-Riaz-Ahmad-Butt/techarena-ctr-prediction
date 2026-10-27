"""Check serialized categorical training data agrees with raw-data inference."""
import os,json,gc
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from catboost import Pool,CatBoostClassifier,CatBoostError
from v56_memory_workflow import frame,sha
R=Path(__file__).resolve().parent/'artifacts'/'v56_weekday'
def main():
 model=CatBoostClassifier();model.load_model(str(R/'v56_categorical.cbm'));q=Pool('quantized://'+str(R/'v56_categorical_quantized.bin'));q.set_feature_names([f'x{i}' for i in range(354)]+['user','item','user_bucket','item_bucket']);assert q.get_cat_feature_indices()==[354,355,356,357]
 try:p=model.predict_proba(q,thread_count=4)[:,1]
 except CatBoostError as error:
  limitation=str(error);del q;gc.collect()
  import pandas as pd
  rng=np.random.default_rng(5609);category=rng.integers(0,128,4096);y=(category%3==0).astype('int32');w=np.where(y>0,1.,5.);w/=w.mean();f=pd.DataFrame({'n0':rng.normal(size=len(y)),'n1':rng.normal(size=len(y)),'category':np.char.add('c',category.astype(str))})
  a=Pool(f,label=y,weight=w,cat_features=['category']);a.quantize(border_count=63,random_seed=5601,used_ram_limit='750mb');path=R/'v56_serialization_training_probe.bin';a.save(str(path));b=Pool('quantized://'+str(path))
  params=dict(iterations=20,depth=4,learning_rate=.1,random_seed=5601,random_strength=0.,bootstrap_type='No',thread_count=4,allow_writing_files=False,verbose=False)
  ma=CatBoostClassifier(**params);mb=CatBoostClassifier(**params);ma.fit(a);mb.fit(b);raw=Pool(f,cat_features=['category']);difference=float(np.max(np.abs(ma.predict_proba(raw,thread_count=4)-mb.predict_proba(raw,thread_count=4))));importance=float(ma.get_feature_importance(thread_count=4)[2]);assert difference<1e-12 and importance>0
  result=dict(status='LIMITED',direct_quantized_categorical_prediction='Installed library raises an internal error on this API path',error=limitation,serialization_probe_rows=len(y),serialization_training_raw_prediction_max_error=difference,probe_category_importance=importance,scope='Probe verifies save/reload categorical training on independent toy data. It does not replace full real-data per-object quantized-vs-raw equivalence, which remains unavailable. Full saved-model raw inference is checked separately.')
  (R/'v56_quantized_prediction_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
 n=len(p);del q;gc.collect()
 starts=np.linspace(0,n-512,10,dtype=int);sample=np.concatenate([np.arange(s,s+512) for s in starts])
 with np.load(R/'v56_fit_metadata.npz') as m:rows=m['rows'][sample]
 x=np.load(R/'v56_fit_fortran.npy',mmap_mode='r');values=np.asarray(x[sample]);raw=model.predict_proba(Pool(frame(values,rows),cat_features=['user','item','user_bucket','item_bucket']),thread_count=4)[:,1];error=float(np.max(np.abs(raw-p[sample])));x._mmap.close();assert error<1e-12,error
 result=dict(status='PASS',sampled_training_rows=len(sample),quantized_vs_raw_prediction_max_error=error,model_sha256=sha(R/'v56_categorical.cbm'),pool_sha256=sha(R/'v56_categorical_quantized.bin'),note='Checks raw inference is consistent with serialized quantized training features; does not assert generalization gain.')
 (R/'v56_quantized_prediction_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
