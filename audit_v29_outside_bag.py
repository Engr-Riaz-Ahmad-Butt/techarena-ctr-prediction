"""Audit V29 identity, unchanged inputs, replay and weighted/local score gates."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from config import ARTIFACT_DIR as ROOT
from v29_outside_bag import OutsideDIN, rank, predict
from v21_din import NI, load_v20, pack_events, hist, htimes
from v24_mass import measure

def main():
 torch.set_num_threads(4)
 f=json.loads((ROOT/'v29_outside_bag_frozen.json').read_text())
 for name,h in f['input_sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 for name,h in f['source_sha256'].items(): assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
 report=json.loads((ROOT/'v29_outside_bag_report.json').read_text()); rows=[]
 for result in report['reports']:
  d=result['day'];z=np.load(ROOT/f'v29_outside_bag_day{d}_raw.npz');base=np.load(ROOT/f'v23_outside_day{d}.npz')
  assert np.array_equal(z['y'],base['y']) and np.array_equal(z['events'],base['events'])
  assert np.array_equal(z['base'],base['base'])
  assert np.isfinite(z['raw']).all() and (z['raw']>=0).all() and (z['raw'].reshape(-1,10).sum(1)<=1.000001).all()
  assert np.max(np.abs(z['raw']-(.5*z['new_raw'].astype(float)+.5*base['raw'].astype(float))))<1e-12
  p=.9*rank(z['base'])+.1*rank(z['raw'])
  r=measure(base,p);assert abs(r['auc']-result['auc'])<1e-12
  model=OutsideDIN(NI);model.load_state_dict(torch.load(ROOT/f'v29_outside_bag_day{d}.pt',weights_only=True))
  val,y,events,_=load_v20(d);assert np.array_equal(y,z['y']) and np.array_equal(events,z['events'])
  replay=predict(model,pack_events(val[:1000],hist,htimes));err=float(np.max(np.abs(replay-z['new_raw'][:1000])));assert err<1e-6
  r.update(day=d,replay_max_error=err);rows.append(r)
  np.savez_compressed(ROOT/f'v29_outside_bag_day{d}.npz',y=z['y'],events=z['events'],pred=p)
 accepted=len(rows)==2 and all(r['conditional_interval'][0]>0 and r['weighted_interval'][0]>0 for r in rows)
 output=dict(accepted=accepted,reports=rows,unchanged_hashes=True,evaluation_csv_ready=False,note='Exploratory reused local validation; not official score. Conditional intervals exclude repeated selection.')
 (ROOT/'v29_outside_bag_audit.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2),flush=True)
if __name__=='__main__':main()
