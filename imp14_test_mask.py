"""Tiny causal constant-feature rule invariance check; no neural fitting."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from sklearn.preprocessing import StandardScaler

ns = dict(np=np, CHUNK=7)
for filename, names in [('imp13_neural.py', {'transform'}), ('imp14_constant_mask.py', {'masked_predict'})]:
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(tree, filename, 'exec'), ns)


class FixedModel:
    def predict_proba(self, x):
        logit = x @ np.linspace(-.1, .1, x.shape[1])
        positive = 1/(1+np.exp(-logit))
        return np.column_stack([1-positive, positive])


if __name__ == '__main__':
    rng = np.random.default_rng(1442)
    train = rng.normal(size=(32, 12)).astype(np.float32)
    train[:, 3] = 0
    train[:, 7] = 2.5
    scaler = StandardScaler().fit(train)
    mask = scaler.var_ <= 1e-12
    assert np.array_equal(np.flatnonzero(mask), [3, 7])
    before = [scaler.mean_.copy(), scaler.var_.copy(), int(scaler.n_samples_seen_)]
    validation = rng.normal(size=(23, 12)).astype(np.float32)
    validation[:, mask] = scaler.mean_[mask]
    changed = validation.copy()
    changed[:, mask] = rng.uniform(-1e6, 1e6, (len(changed), mask.sum()))
    original = validation.copy()
    p = ns['masked_predict'](FixedModel(), scaler, validation)
    altered = ns['masked_predict'](FixedModel(), scaler, changed)
    assert np.array_equal(p, altered)
    assert np.array_equal(validation, original)
    ordinary = FixedModel().predict_proba(ns['transform'](scaler, validation))[:, 1]
    assert np.allclose(p, ordinary, atol=1e-15, rtol=0)
    assert np.array_equal(scaler.mean_, before[0]) and np.array_equal(scaler.var_, before[1])
    assert int(scaler.n_samples_seen_) == before[2]
    varying = StandardScaler().fit(rng.normal(size=(32, 12)).astype(np.float32))
    assert not (varying.var_ <= 1e-12).any()
    empty_mask = ns['masked_predict'](FixedModel(), varying, validation)
    expected = FixedModel().predict_proba(ns['transform'](varying, validation))[:, 1]
    assert np.allclose(empty_mask, expected, atol=1e-15, rtol=0)
    result = dict(status='PASS', synthetic_training_rows=32, validation_rows=23,
                  masked_columns=[3,7], constant_column_perturbation_invariant=True,
                  training_only_scaler_unchanged=True, input_not_mutated=True,
                  zero_mask_original_prediction_identity=True, labels_not_used=True,
                  note='Variance threshold uses training statistics only; experiment follows a diagnosed confirmation failure.')
    (ARTIFACT_DIR / 'imp14_test_mask_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
