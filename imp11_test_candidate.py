"""Independent accepted-bilinear score, model state, and causal pack audit."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from sklearn.metrics import roc_auc_score
from imp11_bilinear_core import loss_gradient
from tst2_candidate import influence, weighted_influence


def replay(theta, packed):
    outputs = []
    for off in range(0, len(packed['u']), 4096):
        sl = slice(off, off + 4096)
        dummy = np.full(len(packed['u'][sl]), 10, np.int32)
        _, _, p = loss_gradient(theta, *[packed[k][sl] for k in ['u', 'h', 'v', 'offset', 'extra']], dummy)
        outputs.append(p)
    return np.concatenate(outputs).ravel()


def audit(day, selected):
    candidate = np.load(ARTIFACT_DIR / f'imp11_bilinear_day{day}.npz')
    baseline = np.load(ARTIFACT_DIR / f'imp10_leafwise_day{day}.npz')
    with np.load(ARTIFACT_DIR / f'imp11_data_val{day}.npz') as f:
        packed = {k: f[k] for k in f.files}
    with np.load(ARTIFACT_DIR / f'imp11_model_day{day}.npz') as f:
        model = {k: f[k] for k in f.files}
    data = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    hist = np.load(ARTIFACT_DIR / 'train_hist.npy', mmap_mode='r')
    stamps = np.load(ARTIFACT_DIR / 'train_times.npy', mmap_mode='r')
    y, events = baseline['y'], baseline['events']
    assert np.array_equal(candidate['y'], y) and np.array_equal(candidate['events'], events)
    groups = events.reshape(-1, 10)
    assert (groups == groups[:, :1]).all() and len(np.unique(groups[:, 0])) == len(groups)
    assert np.array_equal(packed['events'], groups[:, 0])
    labels = y.reshape(-1, 10)
    assert (labels.sum(1) <= 1).all()
    target = np.where(labels.sum(1) > 0, labels.argmax(1), 10)
    assert np.array_equal(packed['target'], target)
    cutoff = int(data[:, 2].min()) + day * 86400
    assert int(model['cutoff']) == int(packed['cutoff']) == cutoff
    assert int(model['epoch']) == selected['epoch'] == 6
    training_events = sum(len(np.load(ARTIFACT_DIR / f'imp11_data_day{d}.npz')['events']) for d in [day-3, day-2, day-1])
    expected_steps = int(selected['epoch']) * ((training_events + 1023) // 1024)
    assert int(model['step']) == expected_steps
    assert model['theta'].shape == model['first'].shape == model['second'].shape == (515,)
    assert all(np.isfinite(model[k]).all() for k in ['theta', 'first', 'second'])
    assert (model['second'] >= 0).all()
    raw = replay(model['theta'], packed)
    replay_error = float(np.max(np.abs(raw - candidate['raw'])))
    assert replay_error < 1e-12, replay_error
    assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
    assert (raw.reshape(-1, 10).sum(1) <= 1 + 1e-6).all()
    weight = selected['weight']
    pred = (1 - weight) * baseline['pred'] + weight * raw
    blend_error = float(np.max(np.abs(pred - candidate['pred'])))
    assert blend_error < 1e-15
    assert np.isfinite(pred).all() and ((pred > 0) & (pred < 1)).all()
    sample = np.linspace(0, len(groups) - 1, 100, dtype=int)
    eids = packed['events'][sample]
    rows = data[(eids[:, None] * 10 + np.arange(10)).ravel()]
    assert np.array_equal(rows[:, 6], events.reshape(-1, 10)[sample].ravel())
    e = rows[::10]
    assert ((e[:, 2] >= cutoff) & (e[:, 2] < cutoff + 86400)).all()
    h = hist[eids]
    valid = h >= 0
    assert ((stamps[eids] < e[:, 2, None]) | ~valid).all()
    with np.load(ARTIFACT_DIR / f'imp2_model_warm1.0_d16_pc1_day{day}.npz') as f:
        U, V, H, bu, bi = [f[k].astype(float) for k in ['U', 'V', 'H', 'bu', 'bi']]
    r = packed['rotation']
    assert np.max(np.abs(r.T @ r - np.eye(16))) < 1e-10
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    hv = (H[np.maximum(h, 0)] * valid[:, :, None]).sum(1)
    items = rows[:, 1].reshape(-1, 10)
    assert np.array_equal(unit(U[e[:, 0]] @ r).astype(np.float32), packed['u'][sample])
    assert np.array_equal(unit(hv @ r).astype(np.float32), packed['h'][sample])
    assert np.array_equal(unit(V @ r)[items].astype(np.float32), packed['v'][sample])
    profile = U[e[:, 0]] + hv / np.sqrt(np.maximum(valid.sum(1)[:, None], 1))
    offset = np.clip(-1.8 + bu[e[:, 0], None] + bi[items] + (profile[:, None] * V[items]).sum(2), -30, 30)
    assert np.allclose(offset.astype(np.float32), packed['offset'][sample], rtol=0, atol=2e-6)
    assert np.array_equal(packed['extra'][sample, :, 0], np.ones((100, 10), np.float32))
    assert np.array_equal(packed['extra'][sample, :, 1], np.broadcast_to(bu[e[:, 0], None], (100, 10)).astype(np.float32))
    assert np.array_equal(packed['extra'][sample, :, 2], bi[items].astype(np.float32))
    vg = np.digitize((hist[events] >= 0).sum(1), [0, 4, 9, 19], right=True)
    eh = np.load(ARTIFACT_DIR / 'evaluation_hist.npy', mmap_mode='r')
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    sw = ((np.bincount(eg, minlength=5)/len(eg)) / (np.bincount(vg, minlength=5)/len(vg)))[vg]
    result = dict(day=day, rows=len(y), epoch=int(model['epoch']), step=int(model['step']),
                  weight=weight, full_raw_model_replay_max_error=replay_error, recipe_max_error=blend_error,
                  independent_causal_pack_events=len(sample), causal_pack_vectors_exact=True)
    for name, weights, fn in [('auc', None, influence), ('weighted_auc', sw, weighted_influence)]:
        old = roc_auc_score(y, baseline['pred'], sample_weight=weights)
        new = roc_auc_score(y, pred, sample_weight=weights)
        args = () if weights is None else (weights,)
        se = np.linalg.norm(fn(y, pred, *args) - fn(y, baseline['pred'], *args))
        result[name] = dict(baseline=float(old), candidate=float(new), delta=float(new-old),
                            conditional_paired_event_95pct_interval=[float(new-old-1.96*se), float(new-old+1.96*se)])
    return result


if __name__ == '__main__':
    selected = json.loads((ARTIFACT_DIR / 'imp11_bilinear_report.json').read_text())
    assert {r['day'] for r in selected} == {5, 6}
    assert len({r['epoch'] for r in selected}) == len({r['weight'] for r in selected}) == 1
    assert all(r['baseline_prefix'] == 'imp10_leafwise' for r in selected)
    trials = json.loads((ARTIFACT_DIR / 'imp11_bilinear_trials_day6.json').read_text())
    eligible = [r for r in trials if r['auc'] > r['baseline_auc'] and r['reweighted'] > r['baseline_reweighted']]
    best = max(eligible, key=lambda r: r['auc'])
    day6 = next(r for r in selected if r['day'] == 6)
    assert best['epoch'] == day6['epoch'] and best['weight'] == day6['weight']
    assert all(r['epoch'] in [2, 4, 6, 8] and r['weight'] in [.1, .25, .5] for r in trials)
    reports = [audit(r['day'], r) for r in selected]
    result = dict(status='PASS', reports=reports,
                  both_dates_improved=all(r[k]['delta'] > 0 for r in reports for k in ['auc', 'weighted_auc']),
                  note='Intervals exclude reused-validation epoch/blend/model selection uncertainty.')
    (ARTIFACT_DIR / 'imp11_test_candidate_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
