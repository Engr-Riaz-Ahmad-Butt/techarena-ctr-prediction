"""Fixed-capacity CatBoost GPU expert on the existing causal feature bundle.

No evaluation labels enter fitting or early stopping. Main day6 is followed by
unchanged day5 confirmation only if its conditional gain gate passes.
"""
import gc
import importlib.util
import json
import time
from pathlib import Path

import catboost
import numpy as np
from catboost import CatBoostClassifier, Pool

PARAMS=dict(iterations=1600,depth=8,learning_rate=.04,l2_leaf_reg=10.,
            loss_function="Logloss",task_type="GPU",devices="0",border_count=63,
            bootstrap_type="Bernoulli",subsample=.8,random_seed=6801,
            boosting_type="Plain",thread_count=4,allow_writing_files=False,
            verbose=100)
BLEND=.25


def helpers(root):
    spec=importlib.util.spec_from_file_location("ctr_cb_gpu_helpers",Path(root)/"runner.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(model,features):
    out=[]
    for off in range(0,len(features),25000):
        x=features.block(off,min(off+25000,len(features)))
        out.append(model.predict_proba(x,thread_count=4)[:,1])
    return np.concatenate(out)


def fit(root,output,manifest,day,h):
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    days=list(range(day-3,day))
    validation=manifest["validation"][str(day)]
    specs=[manifest["train"][str(d)] for d in days]
    labels=[]
    for spec in specs:
        meta=json.loads((root/spec["metadata"]).read_text())
        assert meta["last_timestamp"]<validation["cutoff"]
        labels.append(np.load(root/spec["labels"]))
    y=np.concatenate(labels).astype("float32")
    feature=h.FeatureBlocks(root,[s["features"] for s in specs])
    assert len(feature)==len(y)
    recipe=dict(parameters=PARAMS,blend_weight=BLEND,day=day,fit_days=days,
                fit_rows=len(y),columns=354,validation_cutoff=validation["cutoff"],
                input_manifest_sha256=h.digest(root/"bundle_manifest.json"),
                source_sha256=h.digest(Path(__file__)),catboost=catboost.__version__,
                selection="fixed before scoring; no eval_set or early stopping")
    (output/"recipe.json").write_text(json.dumps(recipe,indent=2))
    temporary=output/"fit_features.npy"
    assert temporary.resolve().parent==output.resolve()
    matrix=np.lib.format.open_memmap(temporary,mode="w+",dtype="float32",shape=(len(y),354))
    for off in range(0,len(y),50000):
        end=min(off+50000,len(y));matrix[off:end]=feature.block(off,end)
        if off%500000==0:print(dict(stage="prepare_features",day=day,rows=end,total=len(y)),flush=True)
    matrix.flush()
    pool=Pool(matrix,label=y,thread_count=4)
    model=CatBoostClassifier(**PARAMS)
    print(dict(stage="catboost_gpu_fit",day=day,rows=len(y),iterations=PARAMS["iterations"]),flush=True)
    model.fit(pool)
    model_path=output/"model.cbm"
    model.save_model(str(model_path))
    del pool
    matrix._mmap.close()
    del matrix,y
    gc.collect()
    # Remove only the scratch array authored by this fresh experiment.
    assert temporary.resolve().parent==output.resolve()
    temporary.unlink()
    with np.load(root/validation["data"]) as z:
        val={k:z[k].copy() for k in z.files}
    xv=h.FeatureBlocks(root,[validation["features"]])
    assert len(xv)==len(val["y"]) and len(xv)%10==0
    raw=predict(model,xv)
    assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
    pred=h.rank((1-BLEND)*h.rank(val["baseline"])+BLEND*h.rank(raw))
    m=h.metrics(val["y"],pred,val["baseline"],val["weights"])
    restored=CatBoostClassifier()
    restored.load_model(str(model_path))
    replay=predict(restored,xv)
    replay_error=float(np.max(np.abs(replay-raw)))
    assert replay_error<1e-12,replay_error
    m.update(day=day,raw_auc=float(h.roc_auc_score(val["y"],raw)),rows=len(raw),
             saved_model_replay_max_error=replay_error,saved_checkpoint_replay="PASS",
             model_sha256=h.digest(model_path),elapsed_seconds=round(time.monotonic()-started,1),
             official_auc=None,csv_ready=False)
    np.savez_compressed(output/"predictions.npz",raw=raw,pred=pred,**val)
    (output/"report.json").write_text(json.dumps(m,indent=2))
    print(json.dumps(m,indent=2),flush=True)
    return m


def run(root,output):
    root,output=Path(root),Path(output)
    h=helpers(root)
    manifest=h.load_bundle(root,verify=True)
    output.mkdir(parents=True,exist_ok=False)
    (output/"catboost_gpu_runner.py").write_text(Path(__file__).read_text())
    reports=[fit(root,output/"day6",manifest,6,h)]
    if reports[0]["positive_gate"]:
        reports.append(fit(root,output/"day5",manifest,5,h))
    eligible=len(reports)==2 and all(m["positive_gate"] for m in reports)
    summary=dict(reports=reports,eligible_for_review=eligible,target=.83,
                 target_achieved=bool(eligible and reports[0]["history_adjusted_auc"]>=.83),
                 csv_ready=False,official_auc=None,
                 note="Same reused main dates; fixed CatBoost GPU expert. Local metric, not official forecast.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary
