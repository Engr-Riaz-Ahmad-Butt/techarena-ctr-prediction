"""Correct optimizer gradient units before any validation AUC is computed."""
from pathlib import Path
import json,hashlib,shutil
B=Path(__file__).resolve().parent;R=B/'artifacts/v54_horizon/cdx_graph_001';OLD=R/'upload';NEW=R/'repaired_upload';assert not NEW.exists();assert not (R/'local_run/report.json').exists();assert not (R/'local_run/frozen_predictions.npz').exists();NEW.mkdir()
for p in OLD.iterdir():
    if p.is_file() and p.name not in ['manifest.json','frozen_plan.json','cdx_graph_runner.py']:shutil.copyfile(p,NEW/p.name)
old=(OLD/'cdx_graph_runner.py').read_text();needle='(loss+regularization*reg).backward()';assert old.count(needle)==1;fixed=old.replace(needle,'((loss+regularization*reg)*len(u)).backward()');(NEW/'cdx_graph_runner.py').write_text(fixed)
plan=json.loads((OLD/'frozen_plan.json').read_text());plan['optimizer_gradient_units']='Backprop batch SUM of the same mean objective to match the online-unit Adagrad initial accumulator0.1. Report mean loss. Same correction in both arms.';plan['implementation_correction']='Original attempt stopped after training-only loss revealed mean-gradient/online-accumulator mismatch; no validation AUC or labels used for the correction. Original files retained.';(NEW/'frozen_plan.json').write_text(json.dumps(plan,indent=2))
manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in NEW.iterdir() if p.is_file()};(NEW/'manifest.json').write_text(json.dumps(manifest,indent=2))
audit=dict(reason=plan['implementation_correction'],original_source_sha256=hashlib.sha256(old.encode()).hexdigest(),corrected_source_sha256=hashlib.sha256(fixed.encode()).hexdigest(),old_artifacts_preserved=True,validation_metrics_before_correction=False,datasets_unchanged=True,hyperparameter_search=False,corrected_directory=str(NEW));(R/'implementation_correction.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
