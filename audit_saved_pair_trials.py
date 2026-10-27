"""Read-only recomputation of existing V65/V66/V67 predictions, no fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

from kaggle.runner import metrics, rank

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
TRIALS=["v65_pair_propensity","v66_pair_decay_corrected","v67_event_pairwise_ranker_corrected"]


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def main():
    history=np.load(ART/"train_hist.npy",mmap_mode="r")
    eh=np.load(ART/"evaluation_hist.npy",mmap_mode="r")
    evaluation_groups=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
    target=np.bincount(evaluation_groups,minlength=5)/len(evaluation_groups)
    reports=[]
    for name in TRIALS:
        directory=ART/name
        original=json.loads((directory/"report.json").read_text())
        for day in [5,6]:
            path=directory/("day5_oof.npz" if day==5 else "day6_future.npz")
            with np.load(path) as p,np.load(ART/f"v53_causal_propensity_day{day}.npz") as base:
                assert np.array_equal(p["events"],base["events"])
                assert np.array_equal(p["y"],base["y"])
                assert np.array_equal(p["v53"],base["pred"]), "Comparator differs from final V53"
                events=p["events"]
                assert len(events)%10==0 and np.all(events.reshape(-1,10)==events.reshape(-1,10)[:,:1])
                group=np.digitize((history[events]>=0).sum(1),[0,4,9,19],right=True)
                observed=np.bincount(group,minlength=5)/len(group)
                weights=(target/observed)[group]
                reproduced=rank(.8*rank(base["pred"])+.2*rank(p["raw"]))
                error=float(np.max(np.abs(reproduced-p["pred"])))
                assert error==0.,error
                m=metrics(p["y"],reproduced,base["pred"],weights)
                claimed=original[f"day{day}"]
                assert abs(m["ordinary_auc"]-claimed["ordinary_candidate"])<1e-12
                assert abs(m["history_adjusted_auc"]-claimed["adjusted_candidate"])<1e-12
                m.update(trial=name,day=day,rows=len(events),events_match=True,labels_match=True,
                         exact_final_v53_comparator=True,rank_blend_max_error=error,
                         prediction_sha256=sha(path),baseline_sha256=sha(ART/f"v53_causal_propensity_day{day}.npz"),
                         saved_model_replay="not performed; no fitting or feature generation in this audit",
                         promoted=False,official_auc=None,target_achieved=False)
                reports.append(m)
                print(json.dumps({k:m[k] for k in ["trial","day","history_adjusted_auc","adjusted_gain_interval","positive_gate"]}),flush=True)
    result=dict(status="PASS prediction/metric recomputation",reports=reports,training_performed=False,
                target=.83,target_achieved=False,official_auc=None,csv_ready=False,
                note="Read-only review of existing reused-date trials. Model/feature replay and fresh holdout are not covered. No promotion or new 0.83 claim.")
    (ROOT/"results"/"saved_pair_trials_recheck.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
