"""Synthetic CPU integration for the GPU driver's data and replay path."""
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import catboost_gpu_runner as cb
import runner


def main():
    parent=Path(__file__).resolve().parents[1]/"artifacts"
    original=dict(cb.PARAMS)
    try:
        cb.PARAMS.update(iterations=4,depth=2,task_type="CPU",thread_count=2,verbose=False)
        cb.PARAMS.pop("devices")
        with tempfile.TemporaryDirectory(dir=parent,prefix="cb-gpu-driver-smoke-") as temp:
            root=Path(temp).resolve();assert root.is_relative_to(parent.resolve())
            rng=np.random.default_rng(6801)
            manifest=dict(schema=1,feature_width=354,train={},validation={},files={})
            def record(path):
                manifest["files"][path.name]=dict(bytes=path.stat().st_size,sha256=runner.digest(path))
                return path.name
            shutil.copy2(Path(__file__).with_name("runner.py"),root/"runner.py")
            record(root/"runner.py")
            for day in [3,4,5]:
                x=root/f"x{day}.npy";y=root/f"y{day}.npy";meta=root/f"m{day}.json"
                np.save(x,rng.normal(size=(100,354)).astype("float32"))
                target=np.zeros(100,dtype="float32");target[::10]=1;np.save(y,target)
                meta.write_text(json.dumps(dict(last_timestamp=day)))
                manifest["train"][str(day)]=dict(features=[record(x)],labels=record(y),metadata=record(meta))
            x=root/"val6.npy";v=root/"val6.npz"
            np.save(x,rng.normal(size=(100,354)).astype("float32"))
            target=np.zeros(100,dtype="float32");target[::10]=1
            np.savez(v,y=target,baseline=.1+.8*target,weights=np.ones(100),events=np.repeat(np.arange(10),10))
            manifest["validation"]["6"]=dict(features=[record(x)],data=record(v),cutoff=6)
            (root/"bundle_manifest.json").write_text(json.dumps(manifest))
            with contextlib.redirect_stdout(io.StringIO()):
                result=cb.run(root,root/"result")
            assert result["reports"][0]["saved_checkpoint_replay"]=="PASS"
            assert result["reports"][0]["saved_model_replay_max_error"]<1e-12
            assert not (root/"result/day6/fit_features.npy").exists()
            assert not result["eligible_for_review"] and not result["csv_ready"]
            assert not (root/"result/day5").exists()
            # Boundary violations must fail before fitting.
            (root/"m5.json").write_text(json.dumps(dict(last_timestamp=6)))
            try:cb.fit(root,root/"bad_cutoff",manifest,6,runner)
            except AssertionError:pass
            else:raise AssertionError("Temporal boundary violation accepted")
    finally:
        cb.PARAMS.clear();cb.PARAMS.update(original)
    report=dict(status="PASS",scope="Synthetic CPU integration only; actual Kaggle GPU run untested here",
                checks=["3-day feature/label alignment","fixed rank blend and metrics",
                        "saved model replay","scratch input cleanup","failed gate skips confirmation",
                        "strict temporal cutoff rejection","production parameters restored"])
    (parent.parent/"results/catboost_gpu_driver_smoke.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":main()
