"""Finite differences for all 515 dense residual coefficients; tiny data only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from imp11_bilinear_core import loss_gradient, adam_step

rng = np.random.default_rng(1142)
u = rng.normal(0, .3, (3, 16))
h = rng.normal(0, .3, (3, 16))
v = rng.normal(0, .3, (3, 10, 16))
offset = rng.normal(-1.8, .4, (3, 10))
extra = np.stack([np.ones((3, 10)), rng.normal(size=(3, 10)), rng.normal(size=(3, 10))], axis=2)
target = np.array([0, 7, 10])
theta = rng.normal(0, .03, 515)
penalty = .01


def manual(t):
    a = t[:256].reshape(16, 16)
    c = t[256:512].reshape(16, 16)
    b = t[512:]
    losses = []
    probabilities = []
    for e in range(3):
        logits = []
        for j in range(10):
            logits.append(offset[e, j] + u[e] @ a @ v[e, j] + h[e] @ c @ v[e, j] + extra[e, j] @ b)
        all_logits = np.r_[logits, 0.]
        lse = np.logaddexp.reduce(all_logits)
        losses.append(lse - all_logits[target[e]])
        probabilities.append(np.exp(np.array(logits) - lse))
    return float(np.mean(losses) + .5 * penalty * np.sum(t * t)), np.array(probabilities)


if __name__ == '__main__':
    loss, gradient, p = loss_gradient(theta, u, h, v, offset, extra, target, penalty)
    ref_loss, ref_p = manual(theta)
    loss_error = abs(loss - ref_loss)
    prediction_error = float(np.max(np.abs(p - ref_p)))
    assert loss_error < 1e-14 and prediction_error < 1e-14
    numeric = np.empty_like(theta)
    for k in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[k] += 1e-6
        minus[k] -= 1e-6
        numeric[k] = (manual(plus)[0] - manual(minus)[0]) / 2e-6
    gradient_error = float(np.max(np.abs(numeric - gradient)))
    assert gradient_error < 2e-8, gradient_error
    zero = np.zeros_like(theta)
    _, _, identity = loss_gradient(zero, u, h, v, offset, extra, target)
    _, expected_identity = manual(zero)
    identity_error = float(np.max(np.abs(identity - expected_identity)))
    assert identity_error < 1e-14
    _, _, uniform = loss_gradient(zero, u, h, v, np.zeros_like(offset), extra, target)
    assert np.max(np.abs(uniform - 1/11)) < 1e-15
    args32 = [x.astype(np.float32) for x in [theta, u, h, v, offset, extra]]
    loss32, grad32, p32 = loss_gradient(*args32, target, penalty)
    assert np.isfinite(loss32) and np.isfinite(grad32).all() and np.isfinite(p32).all()
    assert ((p32 > 0) & (p32 < 1)).all() and (p32.sum(1) < 1).all()
    # Large gradient forces the documented norm cap; test exact first-step Adam formula.
    big_gradient = np.linspace(-4, 4, 515)
    expected_gradient = big_gradient / max(1., np.linalg.norm(big_gradient) / 5.)
    actual = theta.copy()
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    adam_step(actual, big_gradient, first, second, 1, rate=.01)
    expected_step = theta - .01 * expected_gradient / (np.abs(expected_gradient) + 1e-8)
    adam_error = float(np.max(np.abs(actual - expected_step)))
    assert adam_error < 1e-14
    assert np.allclose(first, .1 * expected_gradient, atol=1e-15, rtol=0)
    assert np.allclose(second, .001 * expected_gradient ** 2, atol=1e-15, rtol=0)
    result = dict(status='PASS', coefficients_checked=len(theta), includes_no_click_target=True,
                  independent_manual_loss_error=loss_error, manual_prediction_max_error=prediction_error,
                  finite_difference_gradient_max_error=gradient_error,
                  zero_coefficients_offset_identity_max_error=identity_error,
                  zero_offset_uniform_probability='1/11', float32_finite_probabilities=True,
                  adam_first_step_max_error=adam_error, gradient_norm_cap_checked=True)
    (ARTIFACT_DIR / 'imp11_test_gradients_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
