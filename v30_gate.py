"""History-length gated expert blend with frozen day6 selection and day5 confirmation."""
import json,hashlib
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as ROOT
from v24_mass import measure

def rank(x):return rankdata(x,method='average')/(len(x)+1)
def load(day):
 z=np.load(ROOT/f'v23_outside_day{day}.npz')
 h=np.load(ROOT/'train_hist.npy',mmap_mode='r')
 groups=np.digitize((h[z['events']]>=0).sum(1),[0,4],right=True)
 return z,groups,rank(z['base']),rank(z['raw'])
def predict(g,b,r,weights):
 w=np.asarray(weights)[g];return (1-w)*b+w*r

def main():
 paths=[ROOT/f'v23_outside_day{d}.npz' for d in (6,5)]
 frozen={'groups':['empty','1-4','5-20'],'grid':[0,.05,.1,.2],'passes':1,'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
 manifest=ROOT/'v30_gate_frozen.json'
 if manifest.exists(): assert json.loads(manifest.read_text())==frozen
 else: manifest.open('x').write(json.dumps(frozen,indent=2))
 z,g,b,r=load(6);weights=[.1]*3;trials=[]
 assert np.max(np.abs(predict(g,b,r,weights)-z['pred']))<1e-8
 assert abs(roc_auc_score(z['y'],predict(g,b,r,weights))-roc_auc_score(z['y'],z['pred']))<1e-8
 for group in range(3):
  candidates=[]
  for w in frozen['grid']:
   trial=weights.copy();trial[group]=w
   auc=float(roc_auc_score(z['y'],predict(g,b,r,trial)))
   candidates.append((auc,trial));trials.append(dict(group=group,weights=trial,auc=auc))
  weights=max(candidates,key=lambda v:v[0])[1]
 (ROOT/'v30_gate_selection.json').open('x').write(json.dumps({'weights':weights},indent=2))
 reports=[]
 for day in (6,5):
  z,g,b,r=load(day);p=predict(g,b,r,weights);row=measure(z,p);row['day']=day;reports.append(row)
  assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  np.savez_compressed(ROOT/f'v30_gate_day{day}.npz',y=z['y'],events=z['events'],pred=p)
  print(json.dumps(row),flush=True)
  if row['conditional_interval'][0]<=0 or row['weighted_interval'][0]<=0:break
 for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['inputs'][p.name]
 result=dict(weights=weights,trials=trials,reports=reports,accepted=len(reports)==2 and all(v['conditional_interval'][0]>0 and v['weighted_interval'][0]>0 for v in reports),evaluation_csv_ready=False,note='Reused exploratory validation; conditional intervals exclude selection uncertainty.')
 (ROOT/'v30_gate_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
