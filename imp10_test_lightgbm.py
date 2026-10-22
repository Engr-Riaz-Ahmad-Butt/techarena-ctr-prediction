"""Tiny native LightGBM smoke, model reload, and determinism check."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
import lightgbm as lgb

tree = ast.parse((ROOT / 'imp10_leafwise.py').read_text())
tree.body = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'SETTINGS' for target in node.targets)]
ns = {}
exec(compile(tree, 'imp10_leafwise.py', 'exec'), ns)
settings = dict(ns['SETTINGS'])
settings['num_threads'] = 1
rng = np.random.default_rng(1042)
x = rng.normal(size=(2048, 12)).astype(np.float32)
y = (x[:, 0] + .4 * x[:, 1] - .2 * x[:, 2] > .3).astype(np.int32)
params = dict(settings)
models = [lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=5) for _ in range(2)]
prediction = models[0].predict(x)
assert np.isfinite(prediction).all() and ((prediction > 0) & (prediction < 1)).all()
repeat_error = float(np.max(np.abs(prediction - models[1].predict(x))))
assert repeat_error < 1e-15, repeat_error
model_path = ARTIFACT_DIR / 'imp10_test_native_model.txt'
models[0].save_model(str(model_path))
reloaded = lgb.Booster(model_file=str(model_path))
reload_error = float(np.max(np.abs(prediction - reloaded.predict(x))))
assert reload_error < 1e-15, reload_error
assert reloaded.num_trees() == 5 and reloaded.num_feature() == 12
assert lgb.__version__ == '4.6.0'
result = dict(status='PASS', version=lgb.__version__, rows=len(x), features=x.shape[1], trees=5,
              tiny_smoke_threads=1, production_plan_threads=ns['SETTINGS']['num_threads'],
              predictions_finite_and_bounded=True, deterministic_repeated_fit_max_error=repeat_error,
              native_reload_max_error=reload_error)
(ARTIFACT_DIR / 'imp10_test_lightgbm_report.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
