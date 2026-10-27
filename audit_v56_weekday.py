"""Full saved-model replay, split/cache checks and paired weekday bootstrap."""
import hashlib,json,gc,os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier,Pool
from numba import set_num_threads
from v56_weekday_categories import R,CANONICAL,BASE,frame,PREFIXES
from bootstrap_v39 import weighted_auc_sorted,differences
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 set_num_threads(4);frozen=json.loads((R/'v56_frozen.json').read_text());repair=json.loads((R/'v56_feature_repair.json').read_text())
 for name,digest in frozen['sources'].items():
  if name=='v56_weekday_categories.py':assert digest==repair['original_source_sha256']==sha(R/'v56_initial_source.py') and sha(BASE/name)==repair['repaired_source_sha256']
  else:assert sha(BASE/name)==digest
 for relative,digest in frozen['input_hashes'].items():assert sha(CANONICAL/relative)==digest==sha(R/relative)
 print(json.dumps(dict(stage='inputs_verified')),flush=True)
 workflow=json.loads((R/'v56_memory_workflow.json').read_text());assert sha(BASE/'v56_memory_workflow.py')==workflow['source_sha256'];assert sha(R/'v56_numeric.cbm')==workflow['numeric_model_sha256'];assert sha(R/'v56_categorical.cbm')==workflow['trained']['model_sha256'];assert sha(R/'v56_categorical_quantized.bin')==workflow['quantized']['pool_sha256'];assert workflow['staged']['same_logical_values'] and workflow['quantized']['labels_and_weights_preserved']
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');cut=int(t[:,2].min())+4*86400;future=np.asarray(t[t[:,2]>=cut]);nu=int(t[:,0].max())+1
 earliest=np.full(nu,np.iinfo(np.int64).max);np.minimum.at(earliest,future[::10,0],future[::10,6]);_,ix=np.unique(future[::10,0],return_index=True);rows=future[(ix[:,None]*10+np.arange(10)).ravel()];assert np.array_equal(rows[::10,6],earliest[rows[::10,0]]);del future;gc.collect()
 valid=h[rows[::10,6]]>=0;assert np.all((ht[rows[::10,6]]<cut)|~valid)
 z=np.load(R/'v56_predictions.npz');assert np.array_equal(z['events'],rows[:,6]) and np.array_equal(z['y'],rows[:,5]);report=json.loads((R/'v56_report.json').read_text())
 assert workflow['scored']['report_sha256']==sha(R/'v56_report.json')
 before_score={k:v for k,v in workflow.items() if k!='scored'};newline='\r\n' if b'\r\n' in (R/'v56_memory_workflow.json').read_bytes() else '\n';snapshot=json.dumps(before_score,indent=2).replace('\n',newline).encode('utf-8');assert hashlib.sha256(snapshot).hexdigest()==report['memory_workflow']
 rng=np.random.default_rng(5601);fit_x=np.load(R/'v56_fit_features.npy',mmap_mode='r');offset=0;fit_last=0
 for day in [2,3]:
  a=np.asarray(t[(t[:,2]>=int(t[:,2].min())+day*86400)&(t[:,2]<int(t[:,2].min())+(day+1)*86400)]);selected=np.flatnonzero((a[:,5]==1)|(rng.random(len(a))<.2));sample=np.linspace(0,len(selected)-1,128,dtype=int);parts=[]
  for prefix in PREFIXES:
   name=f'stack_day{day}_c{day}.npy' if prefix=='stack' else f'{prefix}_day{day}.npy';cache=np.load(R/'features'/name,mmap_mode='r');parts.append(cache[selected[sample]])
  assert np.array_equal(np.column_stack(parts),fit_x[offset+sample]);offset+=len(selected);fit_last=max(fit_last,int(a[selected,2].max()))
 assert offset==len(fit_x)==report['fit_rows'] and fit_last==report['fit_last_timestamp']<cut
 fit_x._mmap.close();del fit_x
 xv=np.load(R/'v56_future_features.npy',mmap_mode='r');assert xv.shape==(len(rows),354) and np.array_equal(xv[:,4],rows[:,3]) and np.array_equal(xv[:,5],rows[:,4]);xv._mmap.close();del xv;gc.collect();models={};replayed={};print(json.dumps(dict(stage='split_and_cache_join_verified')),flush=True)
 for name,categorical in [('numeric',False),('categorical',True)]:
  m=CatBoostClassifier();m.load_model(str(R/f'v56_{name}.cbm'));assert m.tree_count_==600 and len(m.feature_names_)==(358 if categorical else 354);assert m.get_cat_feature_indices()==([354,355,356,357] if categorical else [])
  values=[]
  for off in range(0,len(rows),25000):
   source=np.load(R/'v56_future_features.npy',mmap_mode='r');block=np.array(source[off:off+25000],copy=True);source._mmap.close();del source;f,cats=frame(block,rows[off:off+25000],categorical);values.append(m.predict_proba(Pool(f,cat_features=cats),thread_count=4)[:,1]);del f,block
  p=np.concatenate(values);saved=z['pred'] if categorical else z['baseline'];error=float(np.max(np.abs(p-saved)));assert error<1e-12;replayed[name]=p;models[name]=dict(full_saved_replay_max_error=error,model_sha256=sha(R/f'v56_{name}.cbm'),category_feature_importance=m.get_feature_importance(thread_count=4)[-4:].tolist() if categorical else [])
  params=m.get_all_params();assert params['depth']==7 and abs(params['learning_rate']-.06)<1e-7 and params['l2_leaf_reg']==10 and params['random_seed']==5601
  del m;gc.collect();print(json.dumps(dict(stage='full_model_replay_verified',model=name)),flush=True)
 vg=np.digitize((h[rows[:,6]]>=0).sum(1),[0,4,9,19],right=True)*2+rows[:,4];eg=np.digitize((eh[e[::10,6]]>=0).sum(1),[0,4,9,19],right=True)*2+e[::10,4];source=np.bincount(vg,minlength=10)/len(vg);target=np.bincount(eg,minlength=10)/len(eg);w=np.divide(target,source,out=np.zeros(10),where=source>0)[vg];assert np.array_equal(w,z['weights']) and abs(w.mean()-1)<1e-12
 for name,p in replayed.items():assert abs(roc_auc_score(z['y'],p)-report['ordinary_auc'][name])<1e-12 and abs(roc_auc_score(z['y'],p,sample_weight=w)-report['joint_adjusted_auc'][name])<1e-12
 n=len(rows)//10;y=z['y'];events=np.arange(len(y))//10
 def ordered(p):
  ix=np.argsort(p,kind='stable');return y[ix],p[ix],w[ix],events[ix]
 a,b=ordered(z['baseline']),ordered(z['pred']);ones=np.ones(n,dtype=np.uint16)
 for arr,p in [(a,z['baseline']),(b,z['pred'])]:assert abs(weighted_auc_sorted(*arr,ones)-roc_auc_score(y,p,sample_weight=w))<1e-10
 rng=np.random.default_rng(5607);deltas=[];distribution=np.full(n,1/n)
 for block in range(10):
  counts=np.array([rng.multinomial(n,distribution) for _ in range(100)],dtype=np.uint16);deltas.extend(differences(*a,*b,counts).tolist())
  if block in [4,9]:print(json.dumps(dict(bootstrap_completed=(block+1)*100)),flush=True)
 interval=np.quantile(deltas,[.025,.975]);ordinary_gain=report['ordinary_auc']['categorical']-report['ordinary_auc']['numeric'];result=dict(status='PASS',models=models,split_first_future_event=True,cached_training_join_sampled_rows=256,original_and_staged_inputs_unchanged=True,memory_workflow_verified=True,ordinary_gain=ordinary_gain,joint_adjusted_gain=report['joint_adjusted_gain'],replicates=1000,seed=5607,paired_event_interval=interval.tolist(),supports_main_protocol_test=bool(interval[0]>0 and ordinary_gain>=0),target_achieved=False,official_auc=None,note='Separate weekday model comparison on previously used V42 cohort. Quantized categorical pool and runtime cache budget were required for RAM; logical values and scientific learning settings preserved. Truncated horizon and fixed-model/reuse uncertainty remain; absolute score not comparable to V53.')
 (R/'v56_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
