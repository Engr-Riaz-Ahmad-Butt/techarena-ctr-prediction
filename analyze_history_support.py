"""Label-free support diagnostics for choosing CTR experiments."""
import json
import numpy as np
from config import ARTIFACT_DIR as R

def main():
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');et=np.load(R/'evaluation_times.npy',mmap_mode='r');start=int(t[:,2].min())
 ev=t[::10];order=np.argsort(ev[:,0],kind='stable');same=ev[order[1:],0]==ev[order[:-1],0];before=order[:-1][same];after=order[1:][same]
 changed=np.any(h[before]!=h[after],axis=1);valid=h>=0;evalvalid=eh>=0
 report=dict(training_history_clicks=int(valid.sum()),training_history_clicks_in_training_window=int(((ht>=start)&valid).sum()),training_history_max_timestamp=int(ht[valid].max()),training_start=start,same_user_adjacent_pairs=len(before),pairs_with_changed_history=int(changed.sum()),evaluation_history_clicks=int(evalvalid.sum()),evaluation_history_max_timestamp=int(et[evalvalid].max()),evaluation_history_clicks_after_training_start=int(((et>=start)&evalvalid).sum()),train_history_length_quantiles=np.quantile(valid.sum(1),[0,.1,.5,.9,1]).tolist(),evaluation_history_length_quantiles=np.quantile(evalvalid.sum(1),[0,.1,.5,.9,1]).tolist(),scope='Descriptive support only; no labels or model scores used')
 (R/'history_support_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
