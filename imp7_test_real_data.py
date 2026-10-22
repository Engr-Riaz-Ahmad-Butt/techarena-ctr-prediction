"""No-fit integration checks for sequence factor kernels on observed data."""
import ast
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from numba import njit, set_num_threads
set_num_threads(1)


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, filename, 'exec'), namespace)


if __name__ == '__main__':
    train = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    hist = np.load(ARTIFACT_DIR / 'train_hist.npy', mmap_mode='r')
    times = np.load(ARTIFACT_DIR / 'train_times.npy', mmap_mode='r')
    start = int(train[:, 2].min())
    def normalize(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    namespace = dict(np=np, math=math, pickle=pickle, ROOT=ARTIFACT_DIR, njit=njit, normalize=normalize)
    load_functions('imp2_warm_choice.py', {'init32'}, namespace)
    load_functions('choice64_core.py', {'predict_events'}, namespace)
    load_functions('imp7_sequence_core.py', {'profile', 'contextual_predict'}, namespace)
    params, _ = namespace['init32'](start + 4 * 86400, 16)
    ni = len(params[1])
    ref = np.load(ARTIFACT_DIR / 'predictions_imp5_aligned_day6.npz')
    events = ref['events'].reshape(-1, 10)[:100, 0]
    rows = train[(events[:, None] * 10 + np.arange(10)).ravel()]
    assert np.array_equal(rows[:, 6], np.repeat(events, 10))
    users = rows[::10, 0].copy()
    items = rows[:, 1].reshape(-1, 10).copy()
    history = np.asarray(hist[events])
    old = namespace['predict_events'](users, items, history, *params)
    extended = [*params, np.zeros((ni, 4), np.float32), np.zeros((ni, 2), np.float32)]
    new = namespace['contextual_predict'](users, items, history, rows[::10, 3].copy(),
                                        rows[::10, 4].copy(), *extended)
    error = float(np.max(np.abs(old - new)))
    assert error <= 5e-7, error
    assert np.isfinite(new).all() and ((new >= 0) & (new <= 1)).all()
    assert (new.sum(1) <= 1).all()
    with np.load(ARTIFACT_DIR / 'imp7_sequence_data_cut6.npz') as source_cache:
        cache = {name: source_cache[name] for name in source_cache.files}
    assert int(cache['cutoff']) == start + 6 * 86400
    source = train[::10]
    source = source[source[:, 2] < int(cache['cutoff'])]
    uid, first = np.unique(source[:, 0], return_index=True)
    chosen = np.unique(cache['users'])
    chosen = chosen[np.linspace(0, len(chosen) - 1, 100, dtype=int)]
    checked = 0
    for user in chosen:
        row = source[first[np.searchsorted(uid, user)]]
        h, t = hist[row[6]], times[row[6]]
        valid = h >= 0
        assert (t[valid] < row[2]).all()
        assert (np.diff(t[valid]) >= 0).all()
        seen = set()
        expected = []
        for item, stamp in zip(h[valid], t[valid]):
            key = (int(item), int(stamp))
            if key in seen:
                continue
            # Preserve original source order for distinct clicks tied in time.
            context = []
            context_seen = set()
            for hi, ht in zip(h[valid], t[valid]):
                pair = (int(hi), int(ht))
                if ht < stamp and pair not in context_seen:
                    context.append(int(hi))
                    context_seen.add(pair)
            padded = context + [-1] * (20 - len(context))
            bucket = stamp % 86400 // 21600
            kind = int(((stamp // 86400 + 3) % 7) >= 5)
            expected.append((int(item), padded, int(bucket), kind))
            seen.add(key)
        indices = np.flatnonzero(cache['users'] == user)
        assert len(indices) == len(expected)
        for index, (item, context, bucket, kind) in zip(indices, expected):
            assert cache['items'][index] == item
            assert np.array_equal(cache['history'][index], context)
            assert cache['buckets'][index] == bucket and cache['types'][index] == kind
            checked += 1
    result = dict(status='PASS', validation_events=100, initialized_prefix_day=4,
                  zero_context_bias_prediction_max_error=error,
                  probabilities_valid=True, maximum_event_mass=float(new.sum(1).max()),
                  sequence_cache_users=len(chosen), sequence_examples_checked=checked,
                  sequence_dedup_and_strict_prior_context='PASS')
    (ARTIFACT_DIR / 'imp7_test_real_data_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
