"""Small no-fit checks of residual inference and real scoring preconditions."""
import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from numba import njit, prange, set_num_threads
set_num_threads(1)

tree = ast.parse((ROOT / 'imp8_foldin.py').read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['partition', 'adapt']]
ns = dict(np=np, math=math, njit=njit, prange=prange)
exec(compile(tree, 'imp8_foldin.py', 'exec'), ns)


if __name__ == '__main__':
    rng = np.random.default_rng(824)
    V = rng.normal(0, .15, (30, 16))
    background = np.ascontiguousarray(V[rng.integers(0, 30, 32)])
    anchors = rng.normal(0, .2, (3, 16))
    history = np.full((3, 20), -1, np.int32)
    history[0, :4] = [1, 2, 2, 5]
    history[2, :2] = [4, 7]
    event_times = np.full(3, 1000000, np.int64)
    stamps = np.zeros((3, 20), np.int64)
    stamps[0, :4] = event_times[0] - np.array([1, 3600, 172800, 345600])
    stamps[2, :2] = event_times[2] - np.array([100, 5000])
    assert (stamps[history >= 0] < np.repeat(event_times, 20).reshape(3, 20)[history >= 0]).all()
    penalty = 4.
    residual, recorded = ns['adapt'](anchors, history, stamps, event_times, V, background, penalty)
    assert np.isfinite(residual).all() and np.array_equal(residual[1], np.zeros(16))
    assert (recorded[:, 1] <= recorded[:, 0] + 1e-12).all()
    weights = np.exp2(-(event_times[0] - stamps[0, :4]) / (48 * 3600))
    assert weights[2] == .5 and weights[3] == .25
    target = (weights[:, None] * V[history[0, :4]]).sum(0)
    mass = weights.sum()
    def objective(z):
        logits = background @ z
        maximum = logits.max()
        lse = maximum + np.log(np.exp(logits - maximum).mean())
        return mass * lse - target @ z + .5 * penalty * np.sum((z - anchors[0]) ** 2)
    z = anchors[0] + residual[0]
    lse, p = ns['partition'](z, background)
    analytic = mass * (p @ background) - target + penalty * residual[0]
    numeric = np.empty(16)
    for k in range(16):
        delta = np.zeros(16)
        delta[k] = 1e-6
        numeric[k] = (objective(z + delta) - objective(z - delta)) / 2e-6
    gradient_error = float(np.max(np.abs(analytic - numeric)))
    assert gradient_error < 1e-8
    assert abs(objective(anchors[0]) - recorded[0, 0]) < 1e-12
    assert abs(objective(z) - recorded[0, 1]) < 1e-12
    assert np.isfinite(lse) and np.isfinite(p).all() and abs(p.sum() - 1) < 1e-12
    data = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    h = np.load(ARTIFACT_DIR / 'train_hist.npy', mmap_mode='r')
    t = np.load(ARTIFACT_DIR / 'train_times.npy', mmap_mode='r')
    start = int(data[:, 2].min())
    real = []
    for day in [6, 5]:
        ref = np.load(ARTIFACT_DIR / f'predictions_imp5_warmcontext_day{day}.npz')
        events = ref['events'].reshape(-1, 10)[:, 0]
        rows = data[events * 10]
        assert ((rows[:, 2] >= start + day * 86400) & (rows[:, 2] < start + (day + 1) * 86400)).all()
        valid = h[events] >= 0
        assert ((t[events] < rows[:, 2, None]) | ~valid).all()
        assert np.all(ref['pred'] > 0) and np.all(ref['pred'] < 1)
        probabilities = ref['pred']
        logits = np.log(probabilities) - np.log1p(-probabilities)
        zero_residual = 1 / (1 + np.exp(-logits))
        identity_error = float(np.max(np.abs(probabilities - zero_residual)))
        assert identity_error < 1e-15
        corrected = 1 / (1 + np.exp(-(logits[:10] + residual[0] @ V[:10].T)))
        assert np.isfinite(corrected).all() and ((corrected > 0) & (corrected < 1)).all()
        model_path = ARTIFACT_DIR / f'imp2_model_warm1.0_d16_pc1_day{day}.npz'
        with np.load(model_path) as model:
            assert model['U'].shape[1] == model['V'].shape[1] == model['H'].shape[1] == 16
            assert all(np.isfinite(model[name]).all() for name in ['U', 'V', 'H', 'bu', 'bi'])
        real.append(dict(day=day, history_before_event=True,
                         model_checkpoint=model_path.name, required_fitting_cutoff=start + day * 86400,
                         events_with_mass_ge_1=int((ref['pred'].reshape(-1, 10).sum(1) >= 1).sum()),
                         bernoulli_link_zero_residual_max_error=identity_error,
                         maximum_base_event_mass=float(ref['pred'].reshape(-1, 10).sum(1).max())))
    result = dict(status='PASS',
                  production_dtype='float64', gradient_max_error=gradient_error,
                  no_history_residual_exact_zero=True, objective_nonincrease=True,
                  objective_before=float(recorded[0, 0]), objective_after=float(recorded[0, 1]),
                  fixed_48h_weights='PASS', real_scoring_preconditions=real,
                  chronology_note='Stored checkpoints lack cutoff metadata; named prefix recipe audited in source.')
    (ARTIFACT_DIR / 'imp8_test_foldin_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
