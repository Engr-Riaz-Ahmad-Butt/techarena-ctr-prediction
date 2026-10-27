"""Check actual earliest timestamps; inspect slot signal on fitting labels only."""
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from scipy.stats import chi2
R=Path(__file__).resolve().parent/'artifacts'
def main():
 t=np.load(R/'train.npy',mmap_mode='r');ev=t[::10];start=int(t[:,2].min());cut=start+4*86400;future=ev[ev[:,2]>=cut];nu=int(ev[:,0].max())+1;earliest=np.full(nu,np.iinfo(np.int64).max);np.minimum.at(earliest,future[:,0],future[:,2])
 with np.load(R/'v56_weekday'/'v56_predictions.npz') as z:ids=z['events'][::10];head=t[ids*10]
 assert np.array_equal(head[:,2],earliest[head[:,0]]) and len(head)==len(np.unique(future[:,0]))
 split=dict(status='PASS',events=len(head),actual_minimum_timestamp_per_user=True,all_eligible_users_present=True,scope='Verifies timestamp minima independently of source row/event-index ordering')
 (R/'v56_weekday'/'v56_split_timestamp_audit.json').write_text(json.dumps(split,indent=2))
 y=t[:,5].reshape(-1,10);keep=ev[:,2]<start+5*86400;counts=np.asarray(y[keep].sum(0),dtype=float);expected=counts.mean();stat=float(((counts-expected)**2/expected).sum());rates=counts/keep.sum();by_day=[]
 for day in range(5):
  selected=(ev[:,2]>=start+day*86400)&(ev[:,2]<start+(day+1)*86400);by_day.append(y[selected].sum(0).tolist())
 report=dict(fitting_days=[0,1,2,3,4],validation_days_excluded=[5,6],events=int(keep.sum()),click_counts_by_slot=counts.astype(int).tolist(),click_rate_by_slot=rates.tolist(),max_to_min_rate_ratio=float(rates.max()/rates.min()),chi_square_uniform_slots=stat,degrees_of_freedom=9,p_value=float(chi2.sf(stat,9)),daily_click_counts_by_slot=by_day,scope='Unconditional diagnostic on fitting labels only. Opaque IDs are not interpreted; slot is supplied event row order. This does not establish conditional predictive gain or target achievement.')
 (R/'candidate_slot_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(split=split,slot=report)),flush=True)
if __name__=='__main__':main()
