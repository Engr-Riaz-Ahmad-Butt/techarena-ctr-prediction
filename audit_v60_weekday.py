"""Independent raw tree traversal, complete saved ensemble and weekday split audit."""
import json
import numpy as np
import xgboost as xgb
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from v60_weekday_xgb import R,OUT,DAY,ROUNDS,PARAMS,sha,paths,rank
from tst2_candidate import influence,weighted_influence

def tree_oracle(model,x):
 config=json.loads(model.save_config());intercept=json.loads(config['learner']['learner_model_param']['base_score']);intercept=intercept[0] if isinstance(intercept,list) else intercept
 total=np.full(len(x),float(logit(intercept)));trees=[json.loads(d) for d in model.get_dump(dump_format='json')];max_depth=0
 for tree in trees:
  for row,values in enumerate(x):
   node=tree;depth=0
   while 'leaf' not in node:
    index=int(node['split'].removeprefix('f'));child=node['yes'] if values[index]<node['split_condition'] else node['no'];node=next(n for n in node['children'] if n['nodeid']==child);depth+=1
   total[row]+=node['leaf'];max_depth=max(max_depth,depth)
 actual=model.inplace_predict(x,predict_type='margin');error=float(np.max(np.abs(total-actual)));assert error<5e-6,error;assert max_depth<=PARAMS['max_depth']
 return dict(rows=len(x),trees=len(trees),maximum_logit_difference=error,maximum_observed_tree_depth=max_depth)

def main():
 frozen=json.loads((OUT/'frozen.json').read_text());trained=json.loads((OUT/'trained.json').read_text());report=json.loads((OUT/'report.json').read_text());fit=json.loads((OUT/'fit_metadata.json').read_text());assert sha(R.parent/'v60_weekday_xgb.py')==frozen['source_sha256']
 for path,digest in frozen['inputs'].items():assert sha(path)==digest,path
 assert sha(OUT/'model.ubj')==trained['model_sha256'];t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');start=int(t[:,2].min());cut=start+4*DAY
 import hashlib
 labels=hashlib.sha256();metadata=hashlib.sha256();count=0;positives=0;rng=np.random.default_rng(6017);cache_checks=0
 for day in [2,3]:
  a=np.asarray(t[(t[:,2]>=start+day*DAY)&(t[:,2]<start+(day+1)*DAY)]);assert int(a[:,2].max())<cut;labels.update(a[:,5].astype('float32').tobytes());metadata.update(a.tobytes());count+=len(a);positives+=int(a[:,5].sum());blocks=[np.load(p,mmap_mode='r') for p in paths(day)];assert all(len(b)==len(a) for b in blocks) and sum(b.shape[1] for b in blocks)==354
  ix=np.r_[0,len(a)-1,rng.integers(0,len(a),128)];assert np.array_equal(blocks[0][ix,4],a[ix,3]) and np.array_equal(blocks[0][ix,5],a[ix,4]);cache_checks+=len(ix);del blocks,a
 assert labels.hexdigest()==fit['label_sha256'] and metadata.hexdigest()==fit['row_metadata_sha256'];assert count==fit['rows'] and positives==fit['positive_rows'] and fit['last_timestamp']<cut
 future=np.asarray(t[t[:,2]>=cut]);head=future[::10];users,first,inverse=np.unique(head[:,0],return_index=True,return_inverse=True);minimum=np.full(len(users),np.iinfo('int64').max,dtype='int64');np.minimum.at(minimum,inverse,head[:,2]);assert np.array_equal(head[first,2],minimum)
 rows=future[(first[:,None]*10+np.arange(10)).ravel()];assert len(np.unique(rows[::10,0]))==len(rows)//10;assert np.all((ht[rows[::10,6]]<cut)|(h[rows[::10,6]]<0))
 saved=np.load(OUT/'predictions.npz');cb=np.load(R/'v56_weekday/v56_predictions.npz');ctx=np.load(R/'v42_calendar_holdout.npz');assert np.array_equal(rows[:,6],saved['events']) and np.array_equal(rows[:,5],saved['y'])
 for source in [cb,ctx]:assert np.array_equal(source['events'],saved['events']) and np.array_equal(source['y'],saved['y'])
 groups=np.digitize((h[rows[:,6]]>=0).sum(1),[0,4,9,19],right=True)*2+rows[:,4];eg=np.digitize((eh[e[::10,6]]>=0).sum(1),[0,4,9,19],right=True)*2+e[::10,4];source=np.bincount(groups,minlength=10)/len(groups);target=np.bincount(eg,minlength=10)/len(eg);assert not np.any(target[source==0]>0);weights=np.divide(target,source,out=np.zeros(10),where=source>0)[groups];assert np.array_equal(weights,saved['weights'])
 xv=np.load(R/'v56_weekday/v56_future_features.npy',mmap_mode='r');assert np.array_equal(xv[:,4],rows[:,3]) and np.array_equal(xv[:,5],rows[:,4]);model=xgb.Booster();model.load_model(OUT/'model.ubj');model.set_param({'nthread':4});assert model.num_boosted_rounds()==ROUNDS and model.num_features()==354
 raw=np.concatenate([model.inplace_predict(np.asarray(xv[i:i+25000])) for i in range(0,len(xv),25000)]);raw_error=float(np.max(np.abs(raw-saved['raw'])));assert raw_error==0;oracle=tree_oracle(model,np.asarray(xv[rng.integers(0,len(xv),256)]))
 baseline=rank(.9*rank(cb['baseline'])+.1*rank(ctx['neural']));pred=rank(.7*rank(cb['baseline'])+.2*rank(raw)+.1*rank(ctx['neural']));assert np.array_equal(baseline,saved['baseline']) and np.array_equal(pred,saved['pred'])
 y=saved['y'];ordinary=float(roc_auc_score(y,pred)-roc_auc_score(y,baseline));adjusted=float(roc_auc_score(y,pred,sample_weight=weights)-roc_auc_score(y,baseline,sample_weight=weights));se=float(np.linalg.norm(weighted_influence(y,pred,weights)-weighted_influence(y,baseline,weights)));ose=float(np.linalg.norm(influence(y,pred)-influence(y,baseline)))
 assert abs(report['ordinary_gain']-ordinary)<1e-12 and abs(report['adjusted_gain']-adjusted)<1e-12;assert np.allclose(report['adjusted_interval'],[adjusted-1.96*se,adjusted+1.96*se],atol=1e-12,rtol=0);assert np.allclose(report['ordinary_interval'],[ordinary-1.96*ose,ordinary+1.96*ose],atol=1e-12,rtol=0);assert report['eligible_for_main_protocol']==bool(ordinary>0 and adjusted-1.96*se>0)
 result=dict(status='PASS',fit_rows=count,fit_positive_rows=positives,cache_context_checks=cache_checks,validation_rows=len(y),actual_first_timestamp_check=True,raw_prediction_error=raw_error,tree_traversal_oracle=oracle,complete_fixed_ensemble_replay=True,eligible_for_main_protocol=report['eligible_for_main_protocol'],main_candidate_promoted=False,main_score_unchanged=True,csv_ready=False,official_auc=None,note='New weekday complete predictor, not V53 replay. Passing an implementation audit does not establish main target or official score.')
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
