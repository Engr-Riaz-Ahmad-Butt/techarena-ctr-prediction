"""Independent count, feature, and V19 ensemble replay; no training imports."""
import json
from pathlib import Path
import imp5_test_aligned as audit_engine
import numpy as np

from config import ARTIFACT_DIR as ROOT


def codes(t):
    return (t % 86400 // 21600).astype(int), (((t // 86400 + 3) % 7) >= 5).astype(int)


def check_features(day, events):
    data = np.load(ROOT / 'train.npy', mmap_mode='r')
    histories = np.load(ROOT / 'train_hist.npy', mmap_mode='r')
    times = np.load(ROOT / 'train_times.npy', mmap_mode='r')
    stats = np.load(ROOT / f'imp6_context_counts_cut{day}.npz')
    cutoff = int(data[:, 2].min()) + day * 86400
    assert int(stats['cutoff']) == cutoff and int(stats['max_click_timestamp']) < cutoff
    before = np.asarray(data[data[:, 2] < cutoff])
    eb, ek = codes(before[::10, 2])
    assert np.array_equal(eb, before[::10, 3]) and np.array_equal(ek, before[::10, 4])
    event_rows = before[::10]
    _, first = np.unique(event_rows[:, 0], return_index=True)
    first_events = event_rows[first]
    h = histories[first_events[:, 6]]
    t = times[first_events[:, 6]]
    valid = h >= 0
    assert ((t < first_events[:, 2, None]) | ~valid).all()
    u = np.broadcast_to(first_events[:, 0, None], h.shape)
    records = np.column_stack([u[valid], h[valid], t[valid]])
    unique = np.unique(records, axis=0)
    clicked = before[before[:, 5] == 1]
    items = np.concatenate([unique[:, 1], clicked[:, 1]])
    stamps = np.concatenate([unique[:, 2], clicked[:, 2]])
    bucket, kind = codes(stamps)
    ni = len(stats['total'])
    joint = np.bincount(items * 8 + bucket * 2 + kind, minlength=ni * 8).reshape(ni, 4, 2)
    assert np.array_equal(joint, stats['joint'])
    assert np.array_equal(joint.sum((1, 2)), stats['total'])
    assert np.array_equal(joint.sum(2), stats['bucket'])
    assert np.array_equal(joint.sum(1), stats['kind'])
    assert len(unique) == int(stats['warm_records'])
    assert len(clicked) == int(stats['labelled_records'])
    gj = (joint.sum(0) + 1) / (len(items) + 8)
    assert np.array_equal(gj, stats['globaljoint'])
    assert np.array_equal(gj.sum(1), stats['globalbucket'])
    assert np.array_equal(gj.sum(0), stats['globalkind'])
    cache = np.load(ROOT / 'features' / f'imp6_context_val{day}.npy', mmap_mode='r')
    errors = []
    for index in np.linspace(0, len(events) // 10 - 1, 100, dtype=int):
        eid = int(events[index * 10])
        a = data[eid * 10:eid * 10 + 10]
        assert (a[:, 6] == eid).all()
        i, b, k = a[:, 1], a[:, 3], a[:, 4]
        total = stats['total'][i]
        cb, ck, cj = stats['bucket'][i, b], stats['kind'][i, k], stats['joint'][i, b, k]
        gb, gk, globalj = stats['globalbucket'][b], stats['globalkind'][k], stats['globaljoint'][b, k]
        cols = [np.log1p(total), np.log1p(cb), np.log1p(ck), np.log1p(cj)]
        for alpha in [5., 20., 100.]:
            for count, prior in [(cb, gb), (ck, gk), (cj, globalj)]:
                cols.append(np.log((count + alpha * prior) / (total + alpha) / prior))
        valid = histories[eid] >= 0
        ht = times[eid][valid]
        assert (ht < a[0, 2]).all()
        hb, hk = codes(ht)
        counts = [(hb == b[0]).sum(), (hk == k[0]).sum(), ((hb == b[0]) & (hk == k[0])).sum()]
        priors = [gb[0], gk[0], globalj[0]]
        cols.extend([np.full(10, np.log((c + 5 * p) / (len(ht) + 5) / p)) for c, p in zip(counts, priors)])
        raw = np.column_stack(cols)
        relative = []
        for c in [0, 1, 3, 4, 5, 6, 7, 9]:
            relative.extend([raw[:, c] - raw[:, c].mean(), raw[:, c] - raw[:, c].max()])
        expected = np.column_stack([raw, *relative]).astype(np.float32)
        errors.append(float(np.max(np.abs(expected - cache[index * 10:index * 10 + 10]))))
    assert max(errors) < 2e-6, max(errors)
    aligned = audit_engine._original_feature_check(day, events)
    return dict(exact_full_prefix_count_replay=True, warm_records=len(unique),
                labelled_records=len(clicked), sampled_events=len(errors),
                sampled_warmcontext_max_error=max(errors), aligned_features=aligned)


if __name__ == '__main__':
    audit_engine.TAG = 'warmcontext'
    audit_engine.BASE_PREFIX = 'imp5_aligned'
    audit_engine._original_feature_check = audit_engine.check_features
    audit_engine.check_features = check_features
    reports = json.loads((ROOT / 'imp5_warmcontext_report.json').read_text())
    result = dict(reports=[audit_engine.audit(r['day'], r['mix']) for r in reports],
                  note='Conditional event intervals exclude reused-validation selection uncertainty.')
    (ROOT / 'imp5_test_warmcontext_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
