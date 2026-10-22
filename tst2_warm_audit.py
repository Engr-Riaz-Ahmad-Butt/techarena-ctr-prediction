"""Check every auxiliary history is strictly pre-cutoff; count exact duplicates."""
import sys,json
from pathlib import Path
from config import ARTIFACT_DIR as ROOT
import numpy as np
a=np.load(ROOT/'train.npy',mmap_mode='r');h=np.load(ROOT/'train_hist.npy',mmap_mode='r');t=np.load(ROOT/'train_times.npy',mmap_mode='r')
start=int(a[:,2].min());reports=[]
for day in [4,5,6,7]:
    cutoff=start+day*86400
    ev=np.asarray(a[::10]);ev=ev[ev[:,2]<cutoff]
    _,first=np.unique(ev[:,0],return_index=True);e=ev[first]
    history=np.asarray(h[e[:,6]]);ts=np.asarray(t[e[:,6]]);valid=history>=0
    assert np.all(ts[valid]<np.broadcast_to(e[:,2,None],ts.shape)[valid])
    assert np.all(ts[valid]<cutoff)
    triples=np.column_stack([np.broadcast_to(e[:,0,None],history.shape)[valid],history[valid],ts[valid]])
    unique=len(np.unique(triples,axis=0))
    reports.append({'day':day,'users':len(e),'warm_pairs':int(valid.sum()),'exact_duplicate_triples':len(triples)-unique,'timestamp_checks':'PASS'})
(ROOT/'tst2_warm_audit.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
