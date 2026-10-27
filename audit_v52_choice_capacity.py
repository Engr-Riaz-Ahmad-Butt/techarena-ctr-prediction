"""Independent NumPy softmax replay and selected matched-control metric checks."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from v39_regularized_als32 import rank,weighted_auc
from v24_mass import measure

def replay(model,rows,h):
 history=np.asarray(h[rows[::10,6]]);valid=history>=0;profile=model['U'][rows[::10,0]]+(model['H'][np.maximum(history,0)]*valid[:,:,None]).sum(1)/np.maximum(valid.sum(1),1)[:,None]**.5;items=rows[:,1].reshape(-1,10)
 logits=(-1.8+model['bu'][rows[::10,0],None]+model['bi'][items]+(profile[:,None,:]*model['V'][items]).sum(-1)).clip(-30,30)
 return softmax(np.column_stack([logits,np.zeros(len(logits))]),axis=1)[:,:10].ravel()
def main():
 frozen=json.loads((R/'v52_choice_capacity_frozen.json').read_text())
 for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 result=json.loads((R/'v52_choice_capacity_report.json').read_text());t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');reports=[]
 with threadpool_limits(limits=1):
  for report in result['reports']:
   m=report['metrics'];day=m['day'];epoch=m['epoch'];weight=m['weight'];z=np.load(R/f'v52_choice_capacity_day{day}.npz');base=np.load(R/f'v50_crossfit_experts_day{day}.npz');assert np.array_equal(z['events'],base['events']) and np.array_equal(z['y'],base['y'])
   sample=np.arange(0,len(z['y'])//10,max(1,len(z['y'])//10//128));ix=(sample[:,None]*10+np.arange(10)).ravel();rows=np.asarray(t[(z['events'][ix]//1)*10+np.tile(np.arange(10),len(sample))]);assert np.array_equal(rows[:,6],z['events'][ix])
   errors={}
   for dim,key in [(16,'control_raw'),(32,'raw')]:
    with np.load(R/f'v52_choice_day{day}_d{dim}_ep{epoch}_model.npz') as model:
     assert int(model['cutoff'])==m['cutoff'] and model['U'].shape[1]==dim
     values=replay(model,rows,h)
    error=float(np.max(np.abs(values-z[key][ix])));assert error<2e-6;errors[str(dim)]=error
   assert m['fit_last']<m['cutoff']<=int(rows[:,2].min())
   for rawkey,predkey in [('raw','pred'),('control_raw','control')]:
    expected=rank(rank(base['pred'])+weight*(rank(z[rawkey])-rank(z['old'])));assert np.array_equal(expected,z[predkey])
   assert abs(roc_auc_score(z['y'],z['pred'])-m['auc'])<1e-12 and abs(weighted_auc(z,z['pred'])-m['weighted_auc'])<1e-12
   control={'y':z['y'],'events':z['events'],'pred':z['control']};observed=measure(control,z['pred']);assert abs(observed['weighted_gain']-report['vs_matched_control']['weighted_gain'])<1e-12
   reports.append(dict(day=day,arm_numpy_replay_max_error=errors,cutoff_and_recipes=True,metrics=True))
 assert result['accepted'] is False
 audit=dict(status='PASS',reports=reports,rejection_verified=True,official_auc=None);(R/'v52_choice_capacity_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
