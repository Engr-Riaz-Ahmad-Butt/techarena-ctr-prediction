"""Independent no-fit replay of the real packed latent smoke data."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from imp11_bilinear_core import loss_gradient

if __name__ == '__main__':
    with np.load(ARTIFACT_DIR / 'imp11_data_smoke_val6.npz') as source:
        packed = {k: source[k] for k in source.files}
    data = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    hist = np.load(ARTIFACT_DIR / 'train_hist.npy', mmap_mode='r')
    times = np.load(ARTIFACT_DIR / 'train_times.npy', mmap_mode='r')
    reference = np.load(ARTIFACT_DIR / 'predictions_imp5_warmcontext_day6.npz')
    events = reference['events'].reshape(-1, 10)[:100, 0]
    assert np.array_equal(packed['events'], events)
    rows = data[(events[:, None] * 10 + np.arange(10)).ravel()]
    assert np.array_equal(rows[:, 5], reference['y'][:1000])
    e = rows[::10]
    start = int(data[:, 2].min())
    assert int(packed['cutoff']) == start + 6 * 86400
    assert ((e[:, 2] >= int(packed['cutoff'])) & (e[:, 2] < int(packed['cutoff']) + 86400)).all()
    histories = hist[events]
    valid = histories >= 0
    assert ((times[events] < e[:, 2, None]) | ~valid).all()
    labels = rows[:, 5].reshape(-1, 10)
    expected_target = np.where(labels.sum(1) > 0, labels.argmax(1), 10)
    assert np.array_equal(packed['target'], expected_target)
    with np.load(ARTIFACT_DIR / 'imp2_model_warm1.0_d16_pc1_day6.npz') as model:
        U, V, H, bu, bi = [model[k].astype(float) for k in ['U', 'V', 'H', 'bu', 'bi']]
    r = packed['rotation']
    assert np.max(np.abs(r.T @ r - np.eye(16))) < 1e-10
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    raw_history = (H[np.maximum(histories, 0)] * valid[:, :, None]).sum(1)
    ids = rows[:, 1].reshape(-1, 10)
    expected_u = unit(U[e[:, 0]] @ r).astype(np.float32)
    expected_h = unit(raw_history @ r).astype(np.float32)
    expected_v = unit(V @ r)[ids].astype(np.float32)
    assert np.array_equal(expected_u, packed['u'])
    assert np.array_equal(expected_h, packed['h'])
    assert np.array_equal(expected_v, packed['v'])
    products = np.concatenate([packed['u'][:, None, :] * packed['v'],
                               packed['h'][:, None, :] * packed['v']], axis=2).reshape(-1, 32)
    old_products = np.load(ARTIFACT_DIR / 'features' / 'imp5_aligned_val6.npy', mmap_mode='r')[:1000]
    product_error = float(np.max(np.abs(products - old_products)))
    assert product_error < 5e-7, product_error
    profile = U[e[:, 0]] + raw_history / np.sqrt(np.maximum(valid.sum(1)[:, None], 1))
    expected_offset = np.clip(-1.8 + bu[e[:, 0], None] + bi[ids] + (profile[:, None, :] * V[ids]).sum(2), -30, 30)
    assert np.allclose(packed['offset'], expected_offset.astype(np.float32), rtol=0, atol=1e-6)
    assert np.array_equal(packed['extra'][:, :, 0], np.ones((100, 10), np.float32))
    assert np.array_equal(packed['extra'][:, :, 1], np.broadcast_to(bu[e[:, 0], None], (100, 10)).astype(np.float32))
    assert np.array_equal(packed['extra'][:, :, 2], bi[ids].astype(np.float32))
    theta = np.zeros(515, np.float32)
    _, _, probability = loss_gradient(theta, *[packed[k] for k in ['u', 'h', 'v', 'offset', 'extra', 'target']])
    old_probability = np.load(ARTIFACT_DIR / 'features' / 'imp3_wf_val6.npy', mmap_mode='r')[:1000, 0]
    probability_error = float(np.max(np.abs(probability.ravel() - old_probability)))
    assert probability_error < 5e-7, probability_error
    tree = ast.parse((ROOT / 'imp11_bilinear.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict']
    ns = dict(np=np, loss_gradient=loss_gradient)
    exec(compile(tree, 'imp11_bilinear.py', 'exec'), ns)
    before = ns['predict'](theta, packed)
    altered = dict(packed)
    altered['target'] = np.arange(100, dtype=np.int32) % 11
    assert np.array_equal(before, ns['predict'](theta, altered))
    result = dict(status='PASS', events=100, rows=1000,
                  source_latent_vectors_exact=True, aligned_product_max_error=product_error,
                  zero_theta_warm_probability_max_error=probability_error,
                  chronology_targets_extra_fields_metadata='PASS', validation_target_permutation_identity=True)
    (ARTIFACT_DIR / 'imp11_test_pack_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
