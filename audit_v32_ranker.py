"""Replay saved ranker and independently verify blend, rows and score gates."""
import json,hashlib
import numpy as np
import lightgbm as lgb
from scipy.special import softmax
from scipy.stats import rankdata
from config import ARTIFACT_DIR as R
from v32_ranker import cached,DAY
from v24_mass import measure
rank=lambda a:rankdata(a,method='average')/(len(a)+1)
def main():
 report=json.loads((R/'v32_ranker_report.json').read_text());selected=json.loads((R/'v32_ranker_selection.json').read_text());t=np.load(R/'train.npy',mmap_mode='r');epoch=int(t[:,2].min());out=[]
 for entry in report['reports']:
  day=entry['day'];z=np.load(R/f'v32_ranker_day{day}.npz');prior=np.load(R/f'v23_outside_day{day}.npz')
  assert np.array_equal(z['y'],prior['y']) and np.array_equal(z['events'],prior['events'])
  weights=entry['weight'];assert weights==selected['weight']
  expected_raw=(softmax(z['logits'].reshape(-1,10),axis=1)*prior['raw'].reshape(-1,10).sum(1)[:,None]).ravel()
  assert np.max(np.abs(expected_raw-z['raw']))<1e-12
  pred=(1-weights)*rank(prior['pred'])+weights*rank(z['raw']);assert np.max(np.abs(pred-z['pred']))<1e-12
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
  metrics=measure(prior,pred);assert abs(metrics['auc']-entry['auc'])<1e-12
  rows=np.asarray(t[(t[:,2]>=epoch+day*DAY)&(t[:,2]<epoch+(day+1)*DAY)])
  _,first=np.unique(rows[::10,0],return_index=True);rows=rows[(first[:,None]*10+np.arange(10)).ravel()]
  assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
  model=lgb.Booster(model_file=str(R/f'v32_ranker_day{day}.txt'));assert model.num_feature()==354 and model.num_trees()==350
  replay=model.predict(cached(f'val{day}',day,rows,np.arange(1000)),num_threads=1)
  error=float(np.max(np.abs(replay-z['logits'][:1000])));assert error<1e-10
  training=json.loads((R/f'v32_ranker_day{day}_training.json').read_text());assert all(m['last_timestamp']<epoch+day*DAY for m in training)
  metrics.update(day=day,replay_max_error=error,model_sha256=hashlib.sha256((R/f'v32_ranker_day{day}.txt').read_bytes()).hexdigest());out.append(metrics)
 accepted=len(out)==2 and all(m['conditional_interval'][0]>0 and m['weighted_interval'][0]>0 for m in out)
 result=dict(accepted=accepted,reports=out,note='Reused local dates; conditional intervals do not account for repeated selection. Existing causal feature caches reused. No official score or CSV-readiness claim.')
 (R/'v32_ranker_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
