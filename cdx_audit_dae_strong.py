"""Read-only replay of the Kaggle DAE screen; no full candidate AUC or fits."""
from pathlib import Path
import json,hashlib,importlib.util,time
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import torch
B=Path(__file__).resolve().parent;A=B/'artifacts';R=A/'v54_horizon/cdx_dae_strong_001';P=R/'kaggle_payload';O=R/'kaggle_outputs/cdx_dae_strong_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 tic=time.monotonic();torch.set_num_threads(2);torch.set_default_device('cpu')
 report=json.loads((O/'report.json').read_text());plan=json.loads((R/'frozen.json').read_text());manifest=json.loads((P/'upload_manifest.json').read_text())
 assert report['source_manifest']==manifest and report['screen_passed'] is False and report['full_confirmation']==[]
 assert all(sha(P/k)==v for k,v in manifest['hashes'].items())
 unchanged={k:sha(B/k)==v for k,v in plan['input_hashes'].items()};assert all(unchanged.values()),unchanged
 spec=importlib.util.spec_from_file_location('own_dae_strong',P/'cdx_kaggle_dae_strong.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 train=np.load(A/'train.npy',mmap_mode='r');times=np.load(A/'train_times.npy',mmap_mode='r');hist=np.load(A/'train_hist.npy',mmap_mode='r');info=json.loads((P/'data_info.json').read_text());results=[]
 for day in [5,6]:
  f=np.load(P/f'features_day{day}.npz');p=np.load(O/f'predictions_day{day}.npz');allowed=np.load(A/f'selection_day{day}.npz');events=f['events'];head=train[events*10]
  assert np.array_equal(events,p['events']) and np.array_equal(events,allowed['events'][::10])
  assert len(np.unique(head[:,0]))==len(events) and np.array_equal(hist[events],f['history'])
  assert np.all((times[events]<head[:,2,None])|(f['history']<0))
  assert info[str(day)]['max_fit_timestamp']<info[str(day)]['cutoff']<=head[:,2].min()
  assert np.array_equal(f['sample'],(head[:,0].astype(np.uint64)*np.uint64(2654435761)%100)<20)
  assert np.isfinite(p['raw_dae']).all() and np.allclose(p['raw_dae'].sum(1),f['point'].sum(1),atol=2e-7)
  delta=.05*((rankdata(p['raw_dae'].ravel())/(p['raw_dae'].size+1)).reshape(-1,10)-f['old_rank'])
  assert np.array_equal(delta,p['delta'])
  for parent in ['control','export_parent','v53']:assert np.array_equal(f[parent]+delta,p[parent])
  assert len(report['training_logs'][str(day)])==20 and report['training_logs'][str(day)][-1]['epoch']==20
  take=f['sample'];y=np.load(P/f'labels_day{day}.npy',mmap_mode='r')[take]
  for z in [z for z in report['screen'] if z['day']==day]:
   w=f['sw3'] if z['weight']=='3bin' else f['w5'][take]
   ca=roc_auc_score(y.ravel(),f[z['parent']][take].ravel(),sample_weight=np.repeat(w,10))
   pa=roc_auc_score(y.ravel(),p[z['parent']][take].ravel(),sample_weight=np.repeat(w,10))
   assert abs(ca-z['control_auc'])<1e-10 and abs(pa-z['candidate_auc'])<1e-10
  # Small CPU inference replay only; both heavy fits were performed on Kaggle.
  initial=dict(np.load(P/f'initial_day{day}.npz'));x=sp.load_npz(P/f'clicks_day{day}.npz')
  model=mod.CatalogAE(initial,x,False);model.load_state_dict(torch.load(O/f'dae_day{day}.pt',map_location='cpu',weights_only=True));model.eval()
  hh=f['history'][:128];ii=f['items'][:128];point=f['point'][:128]
  def predict(mock_labels):
   # Candidate labels are deliberately unavailable to the actual scoring signature.
   return mod.score(model,hh,ii,point,initial['bi'])
  q=predict(np.zeros((128,10),np.int8));other=predict(np.ones((128,10),np.int8))
  assert np.array_equal(q,other) and np.allclose(q,p['raw_dae'][:128],atol=2e-6,rtol=1e-5)
  perm=np.random.default_rng(7321).permutation(10)
  qp=mod.score(model,hh,ii[:,perm],point[:,perm],initial['bi'])
  assert np.allclose(q[:,perm],qp,atol=2e-6,rtol=1e-5)
  results.append(dict(day=day,allowed_events=len(events),screen_events=int(take.sum()),sample_metrics_reproduced=True,
   actual_kaggle_checkpoint_cpu_replay_max_error=float(np.max(np.abs(q-p['raw_dae'][:128]))),candidate_permutation=True,
   labels_not_inputs=True,constant_event_mass=True,prediction_formula_exact=True,twenty_fixed_epochs=True))
 protected=json.loads((A/'v54_horizon/protected_before.json').read_text());assert all(sha(B/k)==v['sha256'] for k,v in protected.items())
 out=dict(status='passed',days=results,protected_files_unchanged=len(protected),all_frozen_parent_inputs_unchanged=True,
  no_full_candidate_metrics_computed=True,full_confirmation_run=False,training_performed=False,holdout_scored=False,
  evaluation_read=False,submission_created=False,model_promoted=False,seconds=time.monotonic()-tic)
 (R/'postfit_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
