from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def weighted_auc(y,p,w):
    order=np.argsort(p,kind='stable'); y=y[order];p=p[order];w=w[order]
    starts=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
    below=np.cumsum(neg)-neg
    return float(np.dot(pos,below+.5*neg)/(pos.sum()*neg.sum()))

paths=list(Path('/kaggle/input').rglob('trees6/predictions.npz'))
assert len(paths)==1,paths
path=paths[0];z=np.load(path);report=json.loads((path.parent/'report.json').read_text())
assert len(z['y'])==351380 and np.isfinite(z['weights']).all() and (z['weights']>0).all()
assert np.all(z['events'].reshape(-1,10)==z['events'].reshape(-1,10)[:,:1])
rank=lambda p:rankdata(np.asarray(p,dtype='float64'),method='average')/(len(p)+1)
result={'source_prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'new_training':False,'fresh_holdout':False,'official_score':None,'metrics':{},'blend_replays':{},'segments':{}}
for arm in ['baseline','control','treatment','control_raw','treatment_raw']:
 p=z[arm];assert np.isfinite(p).all()
 ordinary=weighted_auc(z['y'],p,np.ones(len(p)));adjusted=weighted_auc(z['y'],p,z['weights'])
 assert abs(ordinary-roc_auc_score(z['y'],p))<1e-12
 assert abs(adjusted-roc_auc_score(z['y'],p,sample_weight=z['weights']))<1e-12
 result['metrics'][arm]={'ordinary_auc':ordinary,'adjusted_auc':adjusted}
 if arm in report:
  assert abs(ordinary-report[arm]['ordinary_auc'])<1e-12
  assert abs(adjusted-report[arm]['adjusted_auc'])<1e-12
for arm in ['control','treatment']:
 expected=.9*rank(z['baseline'])+.1*rank(z[arm+'_raw'])
 error=float(np.max(np.abs(expected-z[arm])));assert error==0
 result['blend_replays'][arm]=error
# Diagnostic raw-tree improvements must be interpreted on the reused cohort.
print(json.dumps(result,indent=2),flush=True)
Path('/kaggle/working/oof_tree_independent_audit.json').write_text(json.dumps(result,indent=2))
