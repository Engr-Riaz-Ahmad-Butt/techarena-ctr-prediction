"""Direct three-expert rank blend; primary selection frozen before confirmation."""
import json, hashlib
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as ROOT
from v24_mass import measure

def rank(x): return rankdata(x,method='average')/(len(x)+1)
def load(day):
 z=np.load(ROOT/f'v23_outside_day{day}.npz')
 tree=np.load(ROOT/f'imp19_tree_day{day}.npz')
 assert np.array_equal(tree['y'],z['y']) and np.array_equal(tree['events'],z['events'])
 neural=[]
 for seed in (0,1,2):
  s=np.load(ROOT/f'v21_fastseeds_day{day}_s{seed}_ep1_raw.npz')
  assert np.array_equal(s['y'],z['y']) and np.array_equal(s['events'],z['events'])
  neural.append(rank(s['raw']))
 return z,rank(tree['pred']),np.mean(neural,axis=0),rank(z['raw'])
def main():
 paths=[ROOT/f'v23_outside_day{d}.npz' for d in (6,5)]
 paths += [ROOT/f'imp19_tree_day{d}.npz' for d in (6,5)]
 paths += [ROOT/f'v21_fastseeds_day{d}_s{s}_ep1_raw.npz' for d in (6,5) for s in (0,1,2)]
 grid=[(d,o) for d in (.1,.2,.3) for o in (.05,.1,.15,.2)]
 frozen=dict(grid=grid,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},note='Repeatedly reused local validation. No official-score claim.')
 (ROOT/'v25_direct_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 z,t,n,o=load(6); trials=[]
 for dw,ow in grid:
  p=(1-dw-ow)*t+dw*n+ow*o
  trials.append(dict(neural_weight=dw,outside_weight=ow,auc=float(roc_auc_score(z['y'],p))))
 best=max(trials,key=lambda r:r['auc'])
 (ROOT/'v25_direct_selection.json').open('x').write(json.dumps(best,indent=2))
 reports=[]
 for day in (6,5):
  z,t,n,o=load(day);dw=best['neural_weight'];ow=best['outside_weight'];p=(1-dw-ow)*t+dw*n+ow*o
  r=measure(z,p);r['day']=day;reports.append(r);print(json.dumps(r),flush=True)
  np.savez_compressed(ROOT/f'v25_direct_day{day}.npz',y=z['y'],events=z['events'],pred=p)
  if r['conditional_interval'][0]<=0 or r['weighted_interval'][0]<=0:break
 for path in paths: assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['sha256'][path.name]
 result=dict(selected=best,trials=trials,reports=reports,accepted=len(reports)==2 and all(r['conditional_interval'][0]>0 and r['weighted_interval'][0]>0 for r in reports),note=frozen['note'],evaluation_csv_ready=False)
 (ROOT/'v25_direct_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
