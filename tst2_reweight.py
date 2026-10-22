"""Diagnostic covariate reweighting; no evaluation labels used."""
import sys,json
from pathlib import Path
from config import ARTIFACT_DIR as ROOT
import numpy as np
from sklearn.metrics import roc_auc_score
h=np.load(ROOT/'train_hist.npy',mmap_mode='r');eh=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
edges=[0,4,9,19];eg=np.digitize((eh>=0).sum(1),edges,right=True)
esh=np.bincount(eg,minlength=5)/len(eg);report=[]
for day in [6,5]:
    d=np.load(ROOT/f'predictions_v11_day{day}.npz')
    vg=np.digitize((h[d['events']]>=0).sum(1),edges,right=True)
    share=np.bincount(vg,minlength=5)/len(vg);weight=(esh/share)[vg]
    report.append({'day':day,'auc':float(roc_auc_score(d['y'],d['pred'])),'auc_history_reweighted':float(roc_auc_score(d['y'],d['pred'],sample_weight=weight)),'evaluation_hist_groups':esh.tolist(),'validation_hist_groups':share.tolist()})
(ROOT/'tst2_reweight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
