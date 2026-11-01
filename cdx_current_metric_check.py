"""Reconcile latest saved candidate scores on allowed selection events only.

No models, no full-day outcome access, no evaluation-file reads or holdout scoring.
This is metric verification, never a model improvement or target-achievement claim.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score

B=Path(__file__).resolve().parent;A=B/'artifacts'

def main():
    out=[];hist=np.load(A/'train_hist.npy',mmap_mode='r')
    for day in [5,6]:
        allowed=np.load(A/f'selection_day{day}.npz')
        saved=np.load(A/f'nxa_h1_day{day}.npz')
        # Never request saved['y']: that full-day file includes reserved day6 outcomes.
        if day==6:
            index=allowed['full_indices']
        else:
            index=np.arange(len(allowed['events']))
        events=saved['events'][index]
        assert np.array_equal(events,allowed['events'])
        hl=(hist[events]>=0).sum(1)
        group=np.where(hl==0,0,np.where(hl<=5,1,2))
        share=np.bincount(group,minlength=3)/len(group)
        w3=(np.array([.0126,.1637,.8237])/share)[group]
        assert np.allclose(np.bincount(group,weights=w3)/w3.sum(),[.0126,.1637,.8237])
        y=allowed['y'];weights=allowed['weights']
        predictions={'V53_parent':allowed['baseline'],'latest_head_V53_parent':saved['pred'][index],
                     'latest_head_V39_parent':saved['pred_v39'][index]}
        for name,p in predictions.items():
            assert p.shape==y.shape and np.isfinite(p).all()
            out.append(dict(day=day,model=name,events=len(y)//10,
                ordinary_auc=float(roc_auc_score(y,p)),
                existing_5bin_auc=float(roc_auc_score(y,p,sample_weight=weights)),
                supplied_3bin_auc=float(roc_auc_score(y,p,sample_weight=w3))))
    result=dict(status='readonly_metric_reconciliation',scores=out,
        note='Different history weighting definitions give different AUCs. Crossing0.82 under another weighting is NOT an improvement or an official score. Rows are allowed selection events; not the full-day cohort used for0.818023.',
        full_day_outcomes_loaded=False,holdout_scored=False,evaluation_read=False,
        model_fitted=False,submission_created=False,official_score_ledger_changed=False)
    path=B/'results/cdx_current_metric_check_20261030.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
