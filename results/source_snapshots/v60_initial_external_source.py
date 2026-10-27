"""Causal complete weekday ensemble benchmark with a full-data XGBoost expert.

This constructs a new three-member predictor, not a replay of V53. Its absolute
scores cannot replace the main weekend score or establish official improvement.
All fitting labels precede cutoff4; whole first-future events are scored.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE=Path(__file__).resolve().parent
R=BASE/'artifacts'
OUT=R/'v60_weekday'
PREFIXES=['stack','imp3_wf','imp4_decompose','imp5_aligned','imp6_context','imp17_als','imp19_als_aligned']
PARAMS=dict(objective='binary:logistic',tree_method='hist',device='cpu',max_bin=63,max_depth=6,eta=.05,min_child_weight=50,subsample=.8,colsample_bytree=.9,reg_lambda=5.,reg_alpha=.1,nthread=4,seed=6001,max_cached_hist_node=32)
ROUNDS=700
DAY=86400
rank=lambda x:rankdata(x,method='average')/(len(x)+1)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def paths(day):
 return [R/'features'/(f'stack_day{day}_c{day}.npy' if p=='stack' else f'{p}_day{day}.npy') for p in PREFIXES]

def write(name,value):
 (OUT/f'{name}.json').write_text(json.dumps(value,indent=2))

class Features:
 def __init__(self,days):
  self.parts=[[np.load(p,mmap_mode='r') for p in paths(day)] for day in days]
  self.sizes=np.array([len(a[0]) for a in self.parts]);self.ends=np.cumsum(self.sizes);self.starts=np.r_[0,self.ends[:-1]]
  for parts,n in zip(self.parts,self.sizes):assert all(len(a)==n for a in parts) and sum(a.shape[1] for a in parts)==354
 def __len__(self):return int(self.ends[-1])
 def block(self,start,stop):
  blocks=[]
  for left,right,parts in zip(self.starts,self.ends,self.parts):
   a=max(start,int(left));b=min(stop,int(right))
   if a<b:blocks.append(np.column_stack([p[a-left:b-left] for p in parts]).astype('float32',copy=False))
  result=blocks[0] if len(blocks)==1 else np.concatenate(blocks);assert result.shape==(stop-start,354) and np.isfinite(result).all();return result

class Iterator(xgb.DataIter):
 def __init__(self,features,y,prefix,batch=100000):
  self.features=features;self.y=y;self.batch=batch;self.position=0;super().__init__(cache_prefix=str(prefix),release_data=True)
 def reset(self):self.position=0
 def next(self,input_data):
  if self.position>=len(self.y):return False
  end=min(self.position+self.batch,len(self.y));x=self.features.block(self.position,end)
  input_data(data=x,label=self.y[self.position:end]);self.position=end;return True

class Progress(xgb.callback.TrainingCallback):
 def __init__(self):self.started=time.monotonic()
 def after_iteration(self,model,epoch,evals_log):
  if epoch==0 or (epoch+1)%25==0:
   write('progress',dict(stage='training',rounds_done=epoch+1,total_rounds=ROUNDS,elapsed_s=time.monotonic()-self.started))
   print(json.dumps(dict(stage='training',rounds_done=epoch+1,total_rounds=ROUNDS,elapsed_s=time.monotonic()-self.started)),flush=True)
  if (epoch+1)%100==0:model.save_model(OUT/f'checkpoint_{epoch+1}.ubj')
  return False

def inputs():
 return [R/n for n in ['train.npy','train_hist.npy','train_times.npy','evaluation.npy','evaluation_hist.npy','v42_calendar_holdout.npz','v42_calendar_holdout_report.json']]+sum([paths(d) for d in [2,3]],[])+[R/'v56_weekday'/n for n in ['v56_future_features.npy','v56_predictions.npz','v56_numeric.cbm','v56_audit.json']]

def preflight():
 class Toy(xgb.DataIter):
  def __init__(self,prefix):
   self.position=0;super().__init__(cache_prefix=str(prefix))
  def reset(self):self.position=0
  def next(self,input_data):
   if self.position==4:return False
   ix=slice(self.position*256,(self.position+1)*256);input_data(data=x[ix],label=y[ix]);self.position+=1;return True
 rng=np.random.default_rng(6001);x=rng.normal(size=(1024,8)).astype('float32');y=(x[:,0]+x[:,1]>.2).astype('float32');data=xgb.ExtMemQuantileDMatrix(Toy(OUT/'toy_cache'),max_bin=63,nthread=4);model=xgb.train(dict(PARAMS,max_depth=2),data,num_boost_round=10);p=model.inplace_predict(x);assert np.isfinite(p).all() and roc_auc_score(y,p)>.9
 write('preflight',dict(status='PASS',version=xgb.__version__,cpu_external_memory_api=True,toy_auc=float(roc_auc_score(y,p)),note='Synthetic API check, not competition model score.'));del data,model;gc.collect()

def train():
 OUT.mkdir(exist_ok=True);assert not (OUT/'frozen.json').exists(),'Do not restart an existing recipe'
 assert json.loads((R/'v56_weekday/v56_audit.json').read_text())['status']=='PASS'
 frozen=dict(scope='New complete weekday three-member ensemble, not V53 backbone or official forecast',params=PARAMS,rounds=ROUNDS,batch_rows=100000,train_days=[2,3],negative_sampling=False,cutoff_day=4,holdout='Each user first actual timestamp at/after cutoff4 through end day7',control_weights=dict(catboost_numeric=.9,context_neural=.1),treatment_weights=dict(catboost_numeric=.7,xgboost=.2,context_neural=.1),selection='No tuning: fixed new expert and ensemble weights; require positive ordinary gain and positive joint adjusted lower interval before translating to main day6/day5 protocol. Weekday cohort was already used by V42/V56.',library_version=xgb.__version__,inputs={str(p):sha(p) for p in inputs()},source_sha256=sha(__file__))
 (OUT/'frozen.json').open('x').write(json.dumps(frozen,indent=2));preflight();t=np.load(R/'train.npy',mmap_mode='r');start=int(t[:,2].min());rows=np.concatenate([np.asarray(t[(t[:,2]>=start+d*DAY)&(t[:,2]<start+(d+1)*DAY)]) for d in [2,3]]);features=Features([2,3]);assert len(rows)==len(features) and int(rows[:,2].max())<start+4*DAY
 # Verify cache alignment from row context and independently sampled blocks.
 rng=np.random.default_rng(6011)
 for ix in rng.integers(0,len(rows),size=256):
  x=features.block(int(ix),int(ix)+1);assert x[0,4]==rows[ix,3] and x[0,5]==rows[ix,4]
 y=rows[:,5].astype('float32');write('fit_metadata',dict(rows=len(rows),positive_rows=int(y.sum()),first_timestamp=int(rows[:,2].min()),last_timestamp=int(rows[:,2].max()),label_sha256=hashlib.sha256(y.tobytes()).hexdigest(),row_metadata_sha256=hashlib.sha256(rows.tobytes()).hexdigest()));del rows
 print(json.dumps(dict(stage='constructing_full_data_quantile_matrix',rows=len(y),features=354)),flush=True)
 iterator=Iterator(features,y,OUT/'train_cache');data=xgb.ExtMemQuantileDMatrix(iterator,max_bin=63,nthread=4)
 assert data.num_row()==len(y) and data.num_col()==354 and np.array_equal(data.get_label(),y)
 print(json.dumps(dict(stage='full_data_quantile_matrix_ready',rows=data.num_row(),columns=data.num_col())),flush=True)
 model=xgb.train(PARAMS,data,num_boost_round=ROUNDS,callbacks=[Progress()]);model.save_model(OUT/'model.ubj');write('trained',dict(status='complete',rounds=model.num_boosted_rounds(),model_sha256=sha(OUT/'model.ubj')))
 for path,digest in frozen['inputs'].items():assert sha(path)==digest,path
 assert sha(__file__)==frozen['source_sha256'];print(json.dumps(dict(stage='fit_complete',rounds=model.num_boosted_rounds())),flush=True)

def score():
 from tst2_candidate import influence,weighted_influence
 frozen=json.loads((OUT/'frozen.json').read_text());assert sha(__file__)==frozen['source_sha256'];trained=json.loads((OUT/'trained.json').read_text());assert sha(OUT/'model.ubj')==trained['model_sha256']
 for path,digest in frozen['inputs'].items():assert sha(path)==digest,path
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min());cut=start+4*DAY;future=np.asarray(t[t[:,2]>=cut]);head=future[::10]
 # Use actual minimum timestamps, preserving saved benchmark user order.
 users,first=np.unique(head[:,0],return_index=True)
 for user,ix in zip(users,first):assert head[ix,2]==head[head[:,0]==user,2].min()
 rows=future[(first[:,None]*10+np.arange(10)).ravel()];del future;assert np.all((ht[rows[::10,6]]<cut)|(h[rows[::10,6]]<0))
 old=np.load(R/'v56_weekday/v56_predictions.npz');context=np.load(R/'v42_calendar_holdout.npz');assert np.array_equal(rows[:,6],old['events']) and np.array_equal(rows[:,5],old['y']);assert np.array_equal(old['events'],context['events']) and np.array_equal(old['y'],context['y']) and np.array_equal(old['weights'],context['weights'])
 xv=np.load(R/'v56_weekday/v56_future_features.npy',mmap_mode='r');assert len(xv)==len(rows);model=xgb.Booster();model.load_model(OUT/'model.ubj');assert model.num_boosted_rounds()==ROUNDS
 raw=np.concatenate([model.inplace_predict(np.asarray(xv[off:off+25000])) for off in range(0,len(xv),25000)]);cb=old['baseline'];neural=context['neural'];baseline=rank(.9*rank(cb)+.1*rank(neural));pred=rank(.7*rank(cb)+.2*rank(raw)+.1*rank(neural));y=old['y'];w=old['weights'];gain=float(roc_auc_score(y,pred,sample_weight=w)-roc_auc_score(y,baseline,sample_weight=w));se=float(np.linalg.norm(weighted_influence(y,pred,w)-weighted_influence(y,baseline,w)));ordinary=float(roc_auc_score(y,pred)-roc_auc_score(y,baseline));ose=float(np.linalg.norm(influence(y,pred)-influence(y,baseline)))
 report=dict(scope=frozen['scope'],ordinary_auc=dict(control=float(roc_auc_score(y,baseline)),treatment=float(roc_auc_score(y,pred)),xgboost_expert=float(roc_auc_score(y,raw))),joint_adjusted_auc=dict(control=float(roc_auc_score(y,baseline,sample_weight=w)),treatment=float(roc_auc_score(y,pred,sample_weight=w)),xgboost_expert=float(roc_auc_score(y,raw,sample_weight=w))),ordinary_gain=ordinary,ordinary_interval=[ordinary-1.96*ose,ordinary+1.96*ose],adjusted_gain=gain,adjusted_interval=[gain-1.96*se,gain+1.96*se],eligible_for_main_protocol=bool(ordinary>0 and gain-1.96*se>0),rows=len(y),csv_ready=False,official_auc=None,target_achieved=False,note='Previously reused weekday benchmark; conditional intervals exclude training/selection uncertainty. Different cohort/horizon/training from V53. No main or official score inferred.')
 np.savez_compressed(OUT/'predictions.npz',events=old['events'],y=y,weights=w,raw=raw,pred=pred,baseline=baseline);write('report',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','score']);args=parser.parse_args();train() if args.stage=='train' else score()
