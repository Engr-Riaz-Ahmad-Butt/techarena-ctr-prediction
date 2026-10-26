"""Joint event-choice tree with an explicit fixed no-click alternative."""
import gc,hashlib,json,time
import numpy as np
import lightgbm as lgb
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v24_mass import measure
F=R/'features';DAY=86400
PARAMS=dict(objective="custom_event_choice",metric="None",num_leaves=31,learning_rate=.05,min_data_in_leaf=100,max_bin=63,feature_fraction=.9,lambda_l2=5.,num_threads=4,seed=3301,verbosity=-1,deterministic=True,force_col_wise=True)

def choice_objective(predictions, dataset):
    logits=predictions.reshape(-1,10)
    full=np.column_stack([logits,np.zeros(len(logits))])
    p=softmax(full,axis=1)[:,:10].ravel()
    y=dataset.get_label()
    return p-y,np.maximum(p*(1-p),1e-6)

rank=lambda x:rankdata(x,method='average')/(len(x)+1)
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def cached(name,day,rows,indices):
 paths=[F/f'stack_{name}_c{day}.npy']+[F/f'{prefix}_{name}.npy' for prefix in ['imp3_wf','imp4_decompose','imp5_aligned','imp6_context','imp17_als','imp19_als_aligned']]
 pieces=[]
 for path in paths:
  a=np.load(path,mmap_mode='r');assert len(a)==len(rows),(path,len(a),len(rows));pieces.append(np.asarray(a[indices],dtype=np.float32))
 x=np.column_stack(pieces);assert x.shape[1]==354 and np.isfinite(x).all()
 assert np.array_equal(x[:,4],rows[indices,3]) and np.array_equal(x[:,5],rows[indices,4])
 return x

def run(day,chosen=None):
 start=time.monotonic();t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');epoch=int(t[:,2].min());rng=np.random.default_rng(3301+day)
 parts=[];targets=[];metadata=[]
 for d in range(day-3,day):
  rows=np.asarray(t[(t[:,2]>=epoch+d*DAY)&(t[:,2]<epoch+(d+1)*DAY)])
  yy=rows[:,5].reshape(-1,10);assert (yy.sum(1)<=1).all();clicked=np.arange(len(yy))
  length=(h[rows[::10,6]]>=0).sum(1)
  sparse=clicked[length[clicked]<=9];short=rng.choice(sparse,min(10000,len(sparse)),replace=False)
  remaining=np.setdiff1d(clicked,short);other=rng.choice(remaining,min(20000-len(short),len(remaining)),replace=False)
  selected=np.sort(np.r_[short,other]);idx=(selected[:,None]*10+np.arange(10)).ravel();x=cached(f'day{d}',d,rows,idx)
  parts.append(x);targets.append(rows[idx,5].astype(np.int32));metadata.append(dict(day=d,events=len(selected),sparse_events=int((length[selected]<=9).sum()),features_sha256=digest(x),event_ids_sha256=digest(rows[idx,6]),last_timestamp=int(rows[idx,2].max())))
  print(json.dumps(dict(stage='features',validation_day=day,fit_day=d,rows=len(x))),flush=True)
 x=np.vstack(parts);y=np.concatenate(targets);del parts;gc.collect()
 assert all(m['last_timestamp']<epoch+day*DAY for m in metadata)
 (R/f'v33_choice_tree_day{day}_training.json').open('x').write(json.dumps(metadata,indent=2))
 def progress(env):
  if (env.iteration+1)%50==0:print(json.dumps(dict(stage='training',day=day,trees=env.iteration+1,elapsed_s=time.monotonic()-start)),flush=True)
 params=dict(PARAMS,objective=choice_objective)
 model=lgb.train(params,lgb.Dataset(x,label=y,free_raw_data=True),num_boost_round=500,callbacks=[progress])
 model.save_model(str(R/f'v33_choice_tree_day{day}.txt'));del x,y;gc.collect()
 rows=np.asarray(t[(t[:,2]>=epoch+day*DAY)&(t[:,2]<epoch+(day+1)*DAY)])
 _,first=np.unique(rows[::10,0],return_index=True);rows=rows[(first[:,None]*10+np.arange(10)).ravel()]
 z=np.load(R/f'v23_outside_day{day}.npz');assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
 xv=cached(f'val{day}',day,rows,np.arange(len(rows)));logits=model.predict(xv,num_threads=4)
 raw=softmax(np.column_stack([logits.reshape(-1,10),np.zeros(len(logits)//10)]),axis=1)[:,:10].ravel()
 trials=[]
 for w in ([chosen] if chosen is not None else [.05,.1,.2]):
  pred=(1-w)*rank(z['pred'])+w*rank(raw);trials.append(dict(weight=w,auc=float(roc_auc_score(z['y'],pred))))
 best=max(trials,key=lambda v:v['auc']);w=best['weight'];p=(1-w)*rank(z['pred'])+w*rank(raw)
 metrics=measure(z,p);metrics.update(day=day,weight=w,raw_auc=float(roc_auc_score(z['y'],raw)),elapsed_s=time.monotonic()-start)
 np.savez_compressed(R/f'v33_choice_tree_day{day}.npz',y=z['y'],events=z['events'],pred=p,raw=raw,logits=logits)
 result=dict(metrics=metrics,trials=trials,validation_feature_sha256=digest(xv));(R/f'v33_choice_tree_day{day}_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 return metrics

def main():
 frozen=dict(params=PARAMS,trees=500,events_per_day=20000,sparse_budget_per_day=10000,blend_grid=[.05,.1,.2],note='Existing causal feature caches; selection on reused Jan26 then fixed Jan25. No official-score claim.')
 (R/'v33_choice_tree_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 first=run(6);reports=[first];(R/'v33_choice_tree_selection.json').open('x').write(json.dumps(first,indent=2))
 if first['conditional_interval'][0]>0 and first['weighted_interval'][0]>0:reports.append(run(5,first['weight']))
 accepted=len(reports)==2 and all(r['conditional_interval'][0]>0 and r['weighted_interval'][0]>0 for r in reports)
 (R/'v33_choice_tree_report.json').write_text(json.dumps(dict(reports=reports,accepted=accepted,evaluation_csv_ready=False,note=frozen['note']),indent=2))
if __name__=='__main__':main()
