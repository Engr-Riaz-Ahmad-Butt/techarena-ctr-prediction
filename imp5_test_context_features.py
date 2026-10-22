"""Independent, event-at-a-time replay of timestamp-context feature cache."""
from pathlib import Path
import numpy as np

from config import ARTIFACT_DIR as ROOT


def check_features(day, events):
    train = np.load(ROOT / 'train.npy', mmap_mode='r')
    histories = np.load(ROOT / 'train_hist.npy', mmap_mode='r')
    times = np.load(ROOT / 'train_times.npy', mmap_mode='r')
    cache = np.load(ROOT / 'features' / f'imp5_timecontext_val{day}.npy', mmap_mode='r')
    m = np.load(ROOT / f'imp2_model_warm1.0_d16_pc1_day{day}.npz')
    errors = []
    for index in np.linspace(0, len(events) // 10 - 1, 100, dtype=int):
        event_id = int(events[index * 10])
        rows = train[event_id * 10:event_id * 10 + 10]
        assert (rows[:, 6] == event_id).all()
        valid = histories[event_id] >= 0
        item_history = histories[event_id][valid]
        stamps = times[event_id][valid]
        now = int(rows[0, 2])
        assert (stamps < now).all()
        age = (now - stamps.astype(float)) / 3600
        n = len(age)
        last = float(age.min()) if n else 1e6
        oldest = float(age.max()) if n else 0.
        mean = float(age.mean()) if n else 0.
        span = max(oldest - last, 0.)
        same = (stamps % 86400 // 21600) == (now % 86400 // 21600)
        weekend = ((stamps // 86400 + 3) % 7) >= 5
        current_weekend = ((now // 86400 + 3) % 7) >= 5
        hour_distance = np.abs((now % 86400) / 3600 - (stamps % 86400) / 3600)
        hour_distance = np.minimum(hour_distance, 24 - hour_distance)
        activity = [np.log1p(last), np.log1p(oldest), np.log1p(mean),
                    np.log1p(span), np.log1p(span / max(n - 1, 1)),
                    same.sum() / max(n, 1), (weekend == current_weekend).sum() / max(n, 1),
                    np.log1p(n / max(oldest, 1))]
        v = m['V'][rows[:, 1]]
        hv = m['H'][item_history]
        raw = [np.tile(activity, (10, 1))]
        for weight in [np.exp2(-age / 6), np.exp2(-age / 24), same.astype(float),
                       np.exp(-.5 * (hour_distance / 3) ** 2)]:
            z = (hv * weight[:, None]).sum(0) / np.sqrt(max(np.dot(weight, weight), 1e-12))
            dot = (v * z).sum(1)
            cosine = dot / np.maximum(np.linalg.norm(z) * np.linalg.norm(v, axis=1), 1e-12)
            repeat = ((rows[:, 1, None] == item_history) * weight).sum(1)
            raw.append(np.column_stack([dot, cosine, np.full(10, np.log1p(weight.sum())), repeat]))
        raw = np.column_stack(raw)
        relative = []
        for col in [8, 12, 16, 20]:
            relative.extend([raw[:, col] - raw[:, col].mean(), raw[:, col] - raw[:, col].max()])
        expected = np.column_stack([raw, *relative]).astype(np.float32)
        errors.append(float(np.max(np.abs(expected - cache[index * 10:index * 10 + 10]))))
    maximum_error = max(errors)
    assert maximum_error < 5e-6, maximum_error
    return dict(sampled_events=len(errors), sampled_feature_max_error=maximum_error,
                history_timestamps_strictly_before_event=True)
