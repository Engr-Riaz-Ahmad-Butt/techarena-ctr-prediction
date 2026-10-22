"""Full tiny finite differences and real-context smoke without fitting."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from imp11_bilinear_core import loss_gradient
from imp12_context_core import context_loss_gradient

rng = np.random.default_rng(1242)
u = rng.normal(0, .2, (4, 16))
h = rng.normal(0, .2, (4, 16))
v = rng.normal(0, .2, (4, 10, 16))
offset = rng.normal(-1.8, .3, (4, 10))
extra = rng.normal(0, .3, (4, 10, 3))
target = np.array([0, 10, 4, 10])
groups = np.array([0, 0, 3, 7])
theta = rng.normal(0, .02, (9, 515))


def manual(t):
    losses = []
    probs = []
    for e in range(4):
        effective = t[0] + t[groups[e] + 1]
        a, c = effective[:256].reshape(16, 16), effective[256:512].reshape(16, 16)
        logits = np.array([offset[e, j] + u[e] @ a @ v[e, j] + h[e] @ c @ v[e, j] + extra[e, j] @ effective[512:] for j in range(10)])
        all_logits = np.r_[logits, 0.]
        lse = np.logaddexp.reduce(all_logits)
        losses.append(lse - all_logits[target[e]])
        probs.append(np.exp(logits - lse))
    loss = np.mean(losses) + .005 * np.sum(t[0] ** 2) + .001 * np.sum(t[1:] ** 2)
    return float(loss), np.array(probs)


if __name__ == '__main__':
    loss, gradient, p = context_loss_gradient(theta, u, h, v, offset, extra, target, groups)
    reference, manual_p = manual(theta)
    assert abs(loss - reference) < 1e-14
    assert np.max(np.abs(p - manual_p)) < 1e-14
    numerical = np.empty_like(theta)
    for position in np.ndindex(theta.shape):
        value = theta[position]
        theta[position] = value + 1e-6
        plus = manual(theta)[0]
        theta[position] = value - 1e-6
        minus = manual(theta)[0]
        theta[position] = value
        numerical[position] = (plus - minus) / 2e-6
    error = float(np.max(np.abs(numerical - gradient)))
    assert error < 2e-8, error
    unused = [index + 1 for index in range(8) if index not in groups]
    assert np.allclose(gradient[unused], .002 * theta[unused], atol=1e-15, rtol=0)
    shared = theta.copy()
    shared[1:] = 0
    shared_loss, shared_gradient, shared_p = context_loss_gradient(shared, u, h, v, offset, extra, target, groups)
    base_loss, base_gradient, base_p = loss_gradient(shared[0], u, h, v, offset, extra, target, penalty=.01)
    assert abs(shared_loss - base_loss) < 1e-14
    assert np.max(np.abs(shared_p - base_p)) < 1e-14
    assert np.max(np.abs(shared_gradient[0] - base_gradient)) < 1e-14
    zero = np.zeros_like(theta)
    _, _, zero_p = context_loss_gradient(zero, u, h, v, offset, extra, target, groups)
    _, _, old_p = loss_gradient(zero[0], u, h, v, offset, extra, target)
    assert np.max(np.abs(zero_p - old_p)) < 1e-14
    arrays32 = [x.astype(np.float32) for x in [theta, u, h, v, offset, extra]]
    loss32, grad32, p32 = context_loss_gradient(*arrays32, target, groups)
    assert np.isfinite(loss32) and np.isfinite(grad32).all() and np.isfinite(p32).all()
    assert ((p32 > 0) & (p32 < 1)).all() and (p32.sum(1) < 1).all()
    with np.load(ARTIFACT_DIR / 'imp11_data_smoke_val6.npz') as f:
        packed = {k: f[k] for k in f.files}
    data = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    eids = packed['events']
    rows = data[(eids[:, None] * 10 + np.arange(10)).ravel()]
    def cached_pack(name, passed_rows, cut):
        assert cut == 6 and np.array_equal(passed_rows, rows)
        return dict(packed)
    tree = ast.parse((ROOT / 'imp12_context.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['context_pack', 'context_predict']]
    ns = dict(np=np, pack=cached_pack, context_loss_gradient=context_loss_gradient,
              KEYS=['u', 'h', 'v', 'offset', 'extra', 'target', 'groups'])
    exec(compile(tree, 'imp12_context.py', 'exec'), ns)
    real = ns['context_pack']('test', rows, 6)
    expected_groups = ((rows[::10, 2] % 86400 // 21600) +
                       4 * (((rows[::10, 2] // 86400 + 3) % 7) >= 5))
    assert np.array_equal(real['groups'], expected_groups)
    pred_before = ns['context_predict'](theta.astype(np.float32), real)
    altered = dict(real)
    altered['target'] = np.arange(100, dtype=np.int32) % 11
    assert np.array_equal(pred_before, ns['context_predict'](theta.astype(np.float32), altered))
    # Context availability: Jan25 train blocks2/3/4 cannot train weekend deviations.
    start = int(data[:, 2].min())
    train_e = data[::10]
    selected = train_e[(train_e[:, 2] >= start + 2 * 86400) & (train_e[:, 2] < start + 5 * 86400)]
    assert (selected[:, 4] == 0).all()
    result = dict(status='PASS', coefficients_checked=theta.size,
                  manual_loss_max_error=abs(loss-reference), finite_difference_gradient_max_error=error,
                  unused_groups_ridge_only=True, shared_only_identity=True, zero_theta_offset_identity=True,
                  float32_valid=True, real_context_events=len(eids), timestamp_context_mapping=True,
                  prediction_target_permutation_identity=True, jan25_weekend_deviations_global_fallback=True)
    (ARTIFACT_DIR / 'imp12_test_context_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
