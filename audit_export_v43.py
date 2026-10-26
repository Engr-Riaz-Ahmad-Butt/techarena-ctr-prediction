"""Audit fixed unseen-calendar fallback on fresh fold, then export candidate."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from v36_recent_context import Context,pack,predict,rank,DAY

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4)
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');start=int(t[:,2].min())
 future=np.asarray(t[t[:,2]>=start+4*DAY]);_,first=np.unique(future[::10,0],return_index=True);rows=future[(first[:,None]*10+np.arange(10)).ravel()]
 z=np.load(R/'v42_calendar_holdout.npz');assert np.array_equal(z['events'],rows[:,6]) and np.array_equal(z['y'],rows[:,5])
 with np.load(R/'imp3_model_warm16_cut2.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 model=Context(16);model.load_state_dict(torch.load(R/'v42_calendar_linear.pt',weights_only=True));assert float(model.kind[1].norm().detach())==0
 vd=pack(rows,h,ht,factors);before,_=predict(model,vd,factors);assert np.max(np.abs(before-z['linear']))<1e-7
 with torch.no_grad():model.kind[1].copy_(model.kind[0])
 after,_=predict(model,vd,factors)
 # The seen weekday scores must be bit-for-bit unchanged.
 seen=np.repeat(rows[::10,4]==0,10);assert np.array_equal(before[seen],after[seen])
 ordinary=float(roc_auc_score(z['y'],after)-roc_auc_score(z['y'],before));adjusted=float(roc_auc_score(z['y'],after,sample_weight=z['weights'])-roc_auc_score(z['y'],before,sample_weight=z['weights']))
 audit=dict(fresh_component_ordinary_gain=ordinary,fresh_component_joint_adjusted_gain=adjusted,seen_predictions_unchanged=True,model_replay_max_error=float(np.max(np.abs(before-z['linear']))),scope='Fresh fold component check, not full ensemble improvement estimate')
 (R/'v43_fresh_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
 assert ordinary>0 and adjusted>0, 'Do not export failed fallback'
 basefile=Path('submission_v39.csv');output=Path('submission_v43.csv');assert not output.exists()
 inputs=[basefile,R/'v36_context_final.pt',R/'v36_context_eval.npy',R/'imp2_model_warm1.0_d16_pc1_day5.npz']
 frozen=dict(rule='For unobserved day type only, copy observed context matrix; existing V39 backbone and weight .1 fixed',input_sha256={p.name:sha(p) for p in inputs},source_sha256=sha(Path(__file__)),official_auc=None,qualification='Candidate supported by day5 full-ensemble ablation and fresh component check. Day6 has both levels observed, so rule has no effect. Reused holdouts; no score forecast.')
 (R/'v43_export_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');et=np.load(R/'evaluation_times.npy',mmap_mode='r')
 with np.load(inputs[3]) as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 model=Context(16);model.load_state_dict(torch.load(inputs[1],weights_only=True));assert float(model.kind[0].norm().detach())==0
 ed=pack(np.asarray(e),eh,et,factors);old,_=predict(model,ed,factors);assert np.max(np.abs(old-np.load(inputs[2])))<1e-7
 with torch.no_grad():model.kind[0].copy_(model.kind[1])
 new,_=predict(model,ed,factors);assert np.array_equal(new[np.repeat(e[::10,4]==1,10)],old[np.repeat(e[::10,4]==1,10)])
 with basefile.open(newline='') as f:records=list(csv.DictReader(f))
 baseline=np.array([float(a['predicted_probability']) for a in records]);pred=rank(rank(baseline)+.1*(rank(new)-rank(old)))
 assert len(pred)==len(records)==529390 and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
 ids=[a['rowid'] for a in records];assert len(set(ids))==len(ids)
 with (Path('data')/'evaluation.csv').open(newline='',encoding='utf-8-sig') as f:assert ids==[a['rowid'] for a in csv.DictReader(f)]
 with output.open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['rowid','predicted_probability']);w.writerows(zip(ids,[format(float(p),'.15g') for p in pred]))
 with output.open(newline='') as f:loaded=np.array([float(a['predicted_probability']) for a in csv.DictReader(f)])
 assert np.max(np.abs(loaded-pred))<1e-14
 for p in inputs:assert sha(p)==frozen['input_sha256'][p.name]
 np.savez_compressed(R/'v43_eval.npz',old=old,new=new,pred=pred)
 report=dict(status='PASS',rows=len(pred),csv_sha256=sha(output),output=str(output.resolve()),official_auc=None,qualification=frozen['qualification'],local_day5_auc=.8177810761475498,local_day5_history_adjusted_auc=.8106357223986173,local_day6_history_adjusted_auc=.8166614878850422,day6_rule_is_identity=True,fresh_component_check=audit)
 (R/'v43_submission_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
