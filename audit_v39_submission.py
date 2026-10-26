"""Independent checks for V39 output IDs, recipe and saved final-model replay."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R,EVALUATION_CSV
from v39_regularized_als32 import score,rank
from imp15_als import evaluation,ehist,cutoff_of

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 set_num_threads(2)
 with threadpool_limits(limits=1):
  frozen=json.loads((R/'v39_export_frozen.json').read_text());manifest=json.loads((R/'v39_submission_manifest.json').read_text());output=Path('submission_v39.csv')
  for name,h in frozen['input_sha256'].items():assert sha(Path(name))==h
  for name,h in manifest['artifact_sha256'].items():assert sha(R/name)==h
  assert sha(output)==manifest['csv_sha256']
  base=np.load(R/'v36_final_pred.npy');raw=np.load(R/'v39_final_raw.npy');prior=np.load(R/'imp15_als_final_raw.npy');expected=rank(rank(base)+.05*(rank(raw)-rank(prior)))
  assert np.array_equal(expected,np.load(R/'v39_final_pred.npy')) and np.isfinite(expected).all()
  ih=hashlib.sha256();seen=set();n=0;error=0.
  with output.open(newline='',encoding='utf-8') as f,EVALUATION_CSV.open(newline='',encoding='utf-8-sig') as src:
   reader=csv.DictReader(f);assert reader.fieldnames==['rowid','predicted_probability']
   for row,original,p in zip(reader,csv.DictReader(src),expected,strict=True):
    key=row['rowid'];assert key==original['rowid'] and key not in seen;seen.add(key);ih.update((key+'\n').encode());value=float(row['predicted_probability'])
    assert np.isfinite(value) and 0<=value<=1;error=max(error,abs(value-p));n+=1
  assert n==529390 and ih.hexdigest()==manifest['rowids_sha256'] and error<1e-14
  model=np.load(R/'v39_als32_final.npz');assert model['V'].shape[1]==32 and int(model['iteration'])==6 and float(model['latent_regularization'])==4. and int(model['cutoff'])==cutoff_of(7)
  assert int(evaluation[:,2].min())>=int(model['cutoff'])
  times=np.load(R/'evaluation_times.npy',mmap_mode='r');replay=score(np.asarray(evaluation[:1000]),model['U'],model['V'],model['bu'],model['bi'],float(model['g']),model['exposures'],histories=ehist,history_times=times)
  replay_error=float(np.max(np.abs(replay-raw[:1000])));assert replay_error<1e-10
  report=dict(status='PASS',rows=n,unique_rowids=len(seen),csv_sha256=sha(output),maximum_csv_rounding_error=error,replay_max_error=replay_error,local_adjusted_auc=manifest['local_adjusted_auc'],local_ordinary_auc=manifest['local_ordinary_auc'],official_auc=None,note=manifest['note'])
  (R/'v39_submission_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
