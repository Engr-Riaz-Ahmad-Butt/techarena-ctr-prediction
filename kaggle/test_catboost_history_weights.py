"""Verify population weighting and chronological label alignment on the private local bundle."""
import json
from pathlib import Path
import numpy as np
import catboost_gpu_weighted_runner as cb
import runner

def main():
    root=Path(__file__).resolve().parents[1]
    bundle=root/"kaggle_upload/neural_bundle"
    manifest=json.loads((bundle/"neural_manifest.json").read_text())
    with np.load(bundle/"events.npz") as z:
        day=(z["timestamp"]-manifest["start"])//86400
        target=z["target"].copy()
    reports=[]
    for cutoff in [5,6]:
        labels=(target[(day>=cutoff-3)&(day<cutoff),None]==np.arange(10)).astype("float32").ravel()
        weights,recipe=cb.training_weights(bundle,cutoff,labels,runner)
        assert len(weights)==len(labels) and np.isclose(weights.mean(),1)
        try:cb.training_weights(bundle,cutoff,1-labels,runner)
        except AssertionError:pass
        else:raise AssertionError("Incorrect label alignment accepted")
        reports.append(dict(day=cutoff,rows=len(labels),recipe=recipe,incorrect_labels_rejected=True))
    (root/"results/catboost_history_weight_alignment.json").write_text(json.dumps(dict(status="PASS",reports=reports),indent=2)+"\n")
    print("PASS: both temporal slices, population shares, weight normalization and wrong-label rejection")

if __name__=="__main__":main()
