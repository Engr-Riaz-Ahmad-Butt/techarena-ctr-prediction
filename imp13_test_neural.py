"""Tiny streaming scaler/optimizer/persistence tests; no competition fitting."""
import ast
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

tree = ast.parse((ROOT / 'imp13_neural.py').read_text())
tree.body = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in ['transform', 'neural_predict']) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ['CHUNK', 'SETTINGS'] for t in n.targets))]
ns = dict(np=np)
exec(compile(tree, 'imp13_neural.py', 'exec'), ns)
rng = np.random.default_rng(1342)
x = rng.normal(size=(768, 290)).astype(np.float32)
labels = (x[:, 0] + .4 * x[:, 1] > 0).astype(int)
validation = rng.normal(100, 5, (64, 290)).astype(np.float32)


def fit_tiny(scaler):
    model = MLPClassifier(**ns['SETTINGS'])
    traces = []
    optimizer = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Got `batch_size` less than 1 or larger than sample size.*')
        for sl in [slice(0, 384), slice(384, 768)]:
            model.partial_fit(ns['transform'](scaler, x[sl]), labels[sl], classes=np.array([0, 1]))
            if optimizer is None:
                optimizer = model._optimizer
            else:
                assert model._optimizer is optimizer
            traces.append(dict(adam_step=int(model._optimizer.t), samples_seen=int(model.t_)))
    assert traces == [dict(adam_step=1, samples_seen=384), dict(adam_step=2, samples_seen=768)]
    assert any(np.any(value) for value in model._optimizer.ms)
    assert any(np.any(value) for value in model._optimizer.vs)
    return model, traces


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        scaler = StandardScaler()
        scaler.partial_fit(x[:384])
        scaler.partial_fit(x[384:])
        reference = StandardScaler().fit(x)
        mean_error = float(np.max(np.abs(scaler.mean_ - reference.mean_)))
        variance_error = float(np.max(np.abs(scaler.var_ - reference.var_)))
        assert mean_error < 1e-14 and variance_error < 1e-13
        original = x.copy()
        transformed = ns['transform'](scaler, x)
        expected = np.clip(reference.transform(x.copy()), -8, 8)
        transform_error = float(np.max(np.abs(transformed - expected)))
        assert transform_error < 1e-6 and np.array_equal(x, original)
        before = (scaler.mean_.copy(), scaler.var_.copy(), int(scaler.n_samples_seen_))
        clipped = ns['transform'](scaler, validation)
        assert clipped.dtype == np.float32 and np.isfinite(clipped).all() and (np.abs(clipped) <= 8).all()
        assert np.array_equal(before[0], scaler.mean_) and np.array_equal(before[1], scaler.var_)
        assert int(scaler.n_samples_seen_) == before[2] == 768
        model, traces = fit_tiny(scaler)
        repeated, _ = fit_tiny(scaler)
        p = ns['neural_predict'](model, scaler, validation)
        deterministic_error = float(np.max(np.abs(p - ns['neural_predict'](repeated, scaler, validation))))
        assert deterministic_error == 0
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
        ns['y'] = labels.copy()
        ns['yv'] = labels.copy()
        before_p = ns['neural_predict'](model, scaler, validation)
        ns['y'] = labels[::-1].copy()
        ns['yv'] = labels[::-1].copy()
        assert np.array_equal(before_p, ns['neural_predict'](model, scaler, validation))
        destination = ARTIFACT_DIR / 'imp13_test_tiny_model.joblib'
        joblib.dump(dict(model=model, scaler=scaler), destination)
        restored = joblib.load(destination)
        reload_error = float(np.max(np.abs(p - ns['neural_predict'](restored['model'], restored['scaler'], validation))))
        assert reload_error == 0
        assert restored['model']._optimizer.t == model._optimizer.t == 2
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Got `batch_size` less than 1 or larger than sample size.*')
            for learner in [model, restored['model']]:
                learner.partial_fit(transformed[:384], labels[:384], classes=np.array([0, 1]))
        assert restored['model']._optimizer.t == model._optimizer.t == 3
        assert all(np.array_equal(a, b) for a, b in zip(model.coefs_, restored['model'].coefs_))
    result = dict(status='PASS', synthetic_training_rows=768, features=290,
                  streaming_scaler_mean_max_error=mean_error, streaming_scaler_variance_max_error=variance_error,
                  transform_max_error=transform_error, validation_did_not_fit_scaler=True,
                  input_array_not_mutated=True, optimizer_state_continuation=traces,
                  deterministic_repeat_max_error=deterministic_error, joblib_reload_max_error=reload_error,
                  restored_optimizer_continues_identically=True, prediction_label_independent=True,
                  note='Tiny chunks384 use sklearn automatic batch-size clipping; production batch4096 unchanged.')
    (ARTIFACT_DIR / 'imp13_test_neural_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
