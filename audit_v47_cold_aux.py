"""Replay both saved V47 arms and verify frozen provenance and rejection."""
import hashlib,json
from pathlib import Path
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from config import ARTIFACT_DIR as R
from imp15_als import split,cutoff_of
from v39_regularized_als32 import score,rank,weighted_auc

def main():
 set_num_threads(4);frozen=json.loads((R/'v47_cold_aux_frozen.json').read_text());repair=json.loads((R/'v47_sparse_repair.json').read_text())
 assert hashlib.sha256(Path('v47_cold_aux_als.py').read_bytes()).hexdigest()==repair['repaired_source_sha256']
 assert frozen['source_sha256']==repair['original_source_sha256']==hashlib.sha256((R/'v47_original_source.py').read_bytes()).hexdigest()
 for name,digest in frozen['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 result=json.loads((R/'v47_cold_aux_report.json').read_text());replays=[]
 with threadpool_limits(limits=1):
  for report in result['reports']:
   day=report['day'];_,rows=split(day);z=np.load(R/f'v47_cold_aux_day{day}.npz');base=np.load(R/f'v39_regularized_als32_day{day}.npz');assert np.array_equal(rows[:,6],z['events']) and np.array_equal(rows[:,5],z['y'])
   for arm,rawkey,predkey in [('control','control_raw','control'),('cold_aux','raw','pred')]:
    with np.load(R/f'v47_cold_aux_day{day}_{arm}_checkpoint.npz') as c:
     assert int(c['cutoff'])==cutoff_of(day) and int(c['iteration'])==12
     ev=rows[::10];cold=np.flatnonzero(c['exposures'][ev[:,0]]==0);sample=np.unique(np.r_[np.arange(64),cold[:32]]);ix=(sample[:,None]*10+np.arange(10)).ravel()
     replay=score(rows[ix],*[c[k] for k in ['U','V','bu','bi']],float(c['g']),c['exposures'])
    error=float(np.max(np.abs(replay-z[rawkey][ix])));assert error<1e-10
    predicted=rank(rank(base['pred'])+.05*(rank(z[rawkey])-rank(base['raw'])));assert np.array_equal(predicted,z[predkey])
    assert abs(weighted_auc(z,predicted)-report['arms'][arm]['weighted_auc'])<1e-12
    replays.append(dict(day=day,arm=arm,sample_raw_max_error=error,recipe_and_weighted_auc=True))
   assert report['coverage']['auxiliary_last_timestamp']<cutoff_of(day)
 assert result['accepted'] is False
 audit=dict(status='PASS',reports=replays,rejection_verified=True,scope='Saved-model and metric replay; no official-score or target-achievement claim.')
 (R/'v47_cold_aux_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
