"""Tiny compiled-kernel finite-difference checks, without dataset/training imports."""
import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from numba import njit, set_num_threads
set_num_threads(1)

tree = ast.parse((ROOT / 'imp7_sequence_core.py').read_text())
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
namespace = dict(np=np, math=math, njit=njit)
exec(compile(tree, 'imp7_sequence_core.py', 'exec'), namespace)


def initial():
    rng = np.random.default_rng(174)
    return [rng.normal(0, .2, shape) for shape in
            [(2, 3), (5, 3), (5, 3), (2,), (5,), (5, 4), (5, 2)]]


history = np.array([1, 1, 3] + [-1] * 17, np.int32)
event_items = np.array([0, 1, 1, 2, 3, 4, 0, 2, 3, 4], np.int32)
warm_items = np.array([2, 1, 1, 3, 4], np.int32)
pop_bias = np.array([.1, -.1, .2, -.2, .05])
REG = .002
CTXREG = .01
LR = .001


def objective(params, mode, target):
    U, V, H, bu, bi, B, T = params
    items = event_items if mode == 'event' else warm_items
    valid = history[history >= 0]
    z = U[0] + H[valid].sum(0) / np.sqrt(len(valid))
    logits = V[items] @ z + B[items, 2] + T[items, 1]
    if mode == 'event':
        logits += -1.8 + bu[0] + bi[items]
        all_logits = np.r_[logits, 0.]
        chosen = target
        weight = 1.
    else:
        logits += pop_bias[items]
        all_logits = logits
        chosen = 0
        weight = .7
    maximum = all_logits.max()
    loss = weight * (np.log(np.exp(all_logits - maximum).sum()) + maximum - all_logits[chosen])
    penalty = (U[0] ** 2).sum() + (V[items] ** 2).sum() + (H[valid] ** 2).sum()
    if mode == 'event':
        penalty += bu[0] ** 2 + (bi[items] ** 2).sum()
    context_penalty = (B[items, 2] ** 2).sum() + (T[items, 1] ** 2).sum()
    return loss + weight * (.5 * REG * penalty + .5 * CTXREG * context_penalty)


def test_gradient(mode, target):
    before = initial()
    after = [p.copy() for p in before]
    acc = [np.full_like(p, .1) for p in before]
    if mode == 'event':
        namespace['contextual_epoch'](np.array([0]), np.array([0], np.int32), event_items[None],
            history[None], np.array([target], np.int32), np.array([2], np.int32), np.array([1], np.int32),
            *after, *acc, LR, REG, CTXREG)
    else:
        namespace['sequence_epoch'](np.array([0]), np.array([0], np.int32), np.array([warm_items[0]], np.int32),
            history[None], np.array([2], np.int32), np.array([1], np.int32), warm_items[None, 1:],
            *after, *acc, LR, REG, CTXREG, .7, pop_bias)
    errors = {}
    for name, index in zip(['U', 'V', 'H', 'bu', 'bi', 'B', 'T'], range(7)):
        inferred = (before[index] - after[index]) * np.sqrt(acc[index]) / LR
        numerical = np.zeros_like(inferred)
        for position in np.ndindex(numerical.shape):
            original = before[index][position]
            epsilon = 1e-6
            before[index][position] = original + epsilon
            plus = objective(before, mode, target)
            before[index][position] = original - epsilon
            minus = objective(before, mode, target)
            before[index][position] = original
            numerical[position] = (plus - minus) / (2 * epsilon)
        error = float(np.max(np.abs(inferred - numerical)))
        assert error < 2e-8, (mode, target, name, error)
        errors[name] = error
    return dict(mode=mode, target=target, errors=errors)


def test_sequence():
    h = np.full((1, 20), -1, np.int32)
    t = np.zeros((1, 20), np.int64)
    h[0, :5] = [1, 1, 2, 3, 1]
    t[0, :5] = [100, 100, 100, 200, 300]
    users, items, contexts, buckets, kinds = namespace['sequence_examples'](np.array([0], np.int32), h, t)
    assert np.array_equal(items, [1, 2, 3, 1])
    assert (contexts[:2] == -1).all(), contexts
    assert np.array_equal(contexts[2, :2], [1, 2])
    assert np.array_equal(contexts[3, :3], [1, 2, 3])
    assert (contexts[2, 2:] == -1).all() and (contexts[3, 3:] == -1).all()
    return dict(dedup_and_tied_timestamps='PASS')


def test_production_dtype():
    for mode in ['event', 'warm']:
        params = [p.astype(np.float32) for p in initial()]
        acc = [np.full_like(p, .1) for p in params]
        if mode == 'event':
            loss = namespace['contextual_epoch'](np.array([0]), np.array([0], np.int32), event_items[None],
                history[None], np.array([2], np.int32), np.array([2], np.int32), np.array([1], np.int32),
                *params, *acc, LR, REG, CTXREG)
        else:
            loss = namespace['sequence_epoch'](np.array([0]), np.array([0], np.int32),
                np.array([warm_items[0]], np.int32), history[None], np.array([2], np.int32),
                np.array([1], np.int32), warm_items[None, 1:], *params, *acc,
                LR, REG, CTXREG, .7, pop_bias.astype(np.float32))
        assert np.isfinite(loss)
        assert all(np.isfinite(p).all() and p.dtype == np.float32 for p in params + acc)
    return 'PASS'


if __name__ == '__main__':
    results = [test_gradient('event', target) for target in [2, 10]] + [test_gradient('warm', 0)]
    arrays = initial()
    probabilities = namespace['contextual_predict'](np.array([0], np.int32), event_items[None], history[None],
        np.array([2], np.int32), np.array([1], np.int32), *arrays)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert (probabilities.sum(1) < 1).all()
    result = dict(status='PASS', finite_differences=results, sequence=test_sequence(),
                  production_float32_kernel_steps=test_production_dtype(),
                  event_click_probability_mass=float(probabilities.sum()))
    (ARTIFACT_DIR / 'imp7_test_gradients_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
