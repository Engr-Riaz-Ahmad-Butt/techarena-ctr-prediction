"""Resume the unchanged V56 comparison in separate memory-bounded processes.

Preparation, quantization, fitting and scoring are separate invocations. The
completed numeric model is retained. No observations/features/learning settings
are removed; only storage layout and the categorical CTR cache budget change.
"""
import os,json,hashlib,argparse,gc,ctypes
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='4'
BASE=Path(__file__).resolve().parent;A=BASE/'artifacts';R=A/'v56_weekday'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(stage,**extra):
 path=R/'v56_memory_workflow.json';data=json.loads(path.read_text()) if path.exists() else dict(source_sha256=sha(Path(__file__)),numeric_model_sha256=sha(R/'v56_numeric.cbm'),purpose='Same frozen rows/features/learning settings; serialized quantized pool and 750mb cache budget; numeric control not refitted')
 assert data['source_sha256']==sha(Path(__file__)) and data['numeric_model_sha256']==sha(R/'v56_numeric.cbm')
 data[stage]=extra;path.write_text(json.dumps(data,indent=2));print(json.dumps(dict(stage=stage,pid=os.getpid(),**extra)),flush=True)
def affinity():
 if os.name=='nt':
  k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=ctypes.c_void_p;handle=k.GetCurrentProcess();pm=ctypes.c_size_t();sm=ctypes.c_size_t();k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
  assert k.GetProcessAffinityMask(handle,ctypes.byref(pm),ctypes.byref(sm));bits=[1<<i for i in range(pm.value.bit_length()) if pm.value&(1<<i)][:4];assert k.SetProcessAffinityMask(handle,sum(bits))
def metadata():
 import numpy as np
 t=np.load(R/'train.npy',mmap_mode='r');start=int(t[:,2].min());rng=np.random.default_rng(5601);parts=[]
 for day in [2,3]:
  a=np.asarray(t[(t[:,2]>=start+day*86400)&(t[:,2]<start+(day+1)*86400)]);keep=(a[:,5]==1)|(rng.random(len(a))<.2);parts.append(a[keep])
 rows=np.concatenate(parts);weight=np.where(rows[:,5]>0,1.,5.);weight/=weight.mean();return rows,weight
def frame(x,rows):
 import pandas as pd
 f=pd.DataFrame(x,columns=[f'x{i}' for i in range(354)],copy=False)
 for name,v in [('user',rows[:,0]),('item',rows[:,1]),('user_bucket',rows[:,0]*4+rows[:,3]),('item_bucket',rows[:,1]*4+rows[:,3])]:f[name]=v.astype(str)
 return f
def stage():
 import numpy as np
 rows,w=metadata();x=np.load(R/'v56_fit_features.npy',mmap_mode='r');assert x.shape==(len(rows),354);path=R/'v56_fit_fortran.npy';out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=x.shape,fortran_order=True);logical=hashlib.sha256()
 for off in range(0,len(x),10000):
  block=np.asarray(x[off:off+10000]);out[off:off+len(block)]=block;assert np.array_equal(out[off:off+len(block)],block);logical.update(block.tobytes(order='C'))
 out.flush();np.savez_compressed(R/'v56_fit_metadata.npz',rows=rows,weight=w);record('staged',rows=len(rows),features=354,logical_values_sha256=logical.hexdigest(),same_logical_values=True,metadata_sha256=sha(R/'v56_fit_metadata.npz'))
def quantize():
 import numpy as np
 from catboost import Pool
 with np.load(R/'v56_fit_metadata.npz') as m:rows=m['rows'];w=m['weight']
 x=np.load(R/'v56_fit_fortran.npy',mmap_mode='r');assert x.flags.f_contiguous and len(x)==len(rows);f=frame(x,rows);pool=Pool(f,label=rows[:,5],weight=w,cat_features=['user','item','user_bucket','item_bucket']);pool.quantize(border_count=63,random_seed=5601,used_ram_limit='750mb');path=R/'v56_categorical_quantized.bin';pool.save(str(path));pool.save_quantization_borders(str(R/'v56_categorical_borders.txt'));del pool,f,x;gc.collect()
 q=Pool('quantized://'+str(path));assert q.num_row()==len(rows) and q.num_col()==358 and q.get_cat_feature_indices()==[354,355,356,357];assert np.array_equal(np.asarray(q.get_label(),dtype='float32'),rows[:,5]) and np.max(np.abs(np.asarray(q.get_weight())-w))<1e-6
 record('quantized',rows=q.num_row(),columns=q.num_col(),category_indices=q.get_cat_feature_indices(),labels_and_weights_preserved=True,pool_sha256=sha(path),border_sha256=sha(R/'v56_categorical_borders.txt'))
def train_model():
 import numpy as np
 from catboost import Pool,CatBoostClassifier
 frozen=json.loads((R/'v56_frozen.json').read_text());q=Pool('quantized://'+str(R/'v56_categorical_quantized.bin'));q.set_feature_names([f'x{i}' for i in range(354)]+['user','item','user_bucket','item_bucket'])
 model=CatBoostClassifier(**dict(frozen['params'],used_ram_limit='750mb'));model.fit(q,verbose=100);assert model.tree_count_==600 and list(map(str,model.classes_)) in [['0.0','1.0'],['0','1']];model.save_model(str(R/'v56_categorical.cbm'));record('trained',trees=600,model_sha256=sha(R/'v56_categorical.cbm'),numeric_control_unchanged=True)
def score():
 import numpy as np
 from catboost import Pool,CatBoostClassifier
 from sklearn.metrics import roc_auc_score
 from tst2_candidate import weighted_influence
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');cut=int(t[:,2].min())+4*86400;future=np.asarray(t[t[:,2]>=cut]);_,ix=np.unique(future[::10,0],return_index=True);rows=future[(ix[:,None]*10+np.arange(10)).ravel()];del future
 valid=h[rows[::10,6]]>=0;assert np.all((ht[rows[::10,6]]<cut)|~valid);x=np.load(R/'v56_future_features.npy',mmap_mode='r');model=CatBoostClassifier();model.load_model(str(R/'v56_categorical.cbm'));out=[]
 for off in range(0,len(rows),25000):out.append(model.predict_proba(Pool(frame(x[off:off+25000],rows[off:off+25000]),cat_features=['user','item','user_bucket','item_bucket']),thread_count=4)[:,1])
 b=np.concatenate(out);x._mmap.close();del x,out,model;gc.collect();a=np.load(R/'v56_numeric_pred.npy');assert len(a)==len(b)==len(rows);np.save(R/'v56_categorical_pred.npy',b)
 vg=np.digitize((h[rows[:,6]]>=0).sum(1),[0,4,9,19],right=True)*2+rows[:,4];eg=np.digitize((eh[e[::10,6]]>=0).sum(1),[0,4,9,19],right=True)*2+e[::10,4];source=np.bincount(vg,minlength=10)/len(vg);target=np.bincount(eg,minlength=10)/len(eg);assert target[source==0].sum()==0;w=np.divide(target,source,out=np.zeros(10),where=source>0)[vg];y=rows[:,5];gain=float(roc_auc_score(y,b,sample_weight=w)-roc_auc_score(y,a,sample_weight=w));se=float(np.linalg.norm(weighted_influence(y,b,w)-weighted_influence(y,a,w)))
 frozen=json.loads((R/'v56_frozen.json').read_text())
 for relative,digest in frozen['input_hashes'].items():assert sha(A/relative)==digest==sha(R/relative)
 meta,unused=metadata();np.savez_compressed(R/'v56_predictions.npz',y=y,events=rows[:,6],baseline=a,pred=b,weights=w)
 report=dict(scope=frozen['scope'],holdout=dict(events=len(rows)//10,weekday_share=float((rows[::10,4]==0).mean()),evaluation_weekday_share=float((e[::10,4]==0).mean()),history_before_cutoff=True,event_ids_sha256=hashlib.sha256(rows[:,6].tobytes()).hexdigest()),fit_rows=len(meta),fit_last_timestamp=int(meta[:,2].max()),cutoff=cut,ordinary_auc={'numeric':float(roc_auc_score(y,a)),'categorical':float(roc_auc_score(y,b))},joint_adjusted_auc={'numeric':float(roc_auc_score(y,a,sample_weight=w)),'categorical':float(roc_auc_score(y,b,sample_weight=w))},joint_adjusted_gain=gain,conditional_interval=[gain-1.96*se,gain+1.96*se],provisional_gain=bool(gain-1.96*se>0),original_inputs_unchanged=True,masked_holdout_labels_for_features=True,source_shares=source.tolist(),target_shares=target.tolist(),official_auc=None,memory_workflow=sha(R/'v56_memory_workflow.json'),note='Unchanged rows/features/learning settings. Quantized storage and 750mb CTR cache budget. Previously used V42 cohort, truncated three-day future horizon; standalone scores not comparable to V53. No official target achievement.')
 (R/'v56_report.json').open('x').write(json.dumps(report,indent=2));record('scored',report_sha256=sha(R/'v56_report.json'));print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['stage','quantize','train','score']);a=p.parse_args();affinity();print(json.dumps(dict(stage='starting',mode=a.mode,pid=os.getpid(),logical_cpus=4)),flush=True);globals()[{'stage':'stage','quantize':'quantize','train':'train_model','score':'score'}[a.mode]]()
