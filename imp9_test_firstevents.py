"""Independent first-event indices, chronology, grouping, and baseline replay."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np

tree = ast.parse((ROOT / 'imp9_firstevents.py').read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'first_indices']
ns = dict(np=np)
exec(compile(tree, 'imp9_firstevents.py', 'exec'), ns)


if __name__ == '__main__':
    for users, expected_events in [(np.array([4, 2, 4, 1, 2, 1]), [0, 1, 3]),
                                   (np.array([9, 9, 9]), [0]), (np.array([], dtype=int), [])]:
        indices = ns['first_indices'](users)
        expected = (np.array(expected_events, dtype=int)[:, None] * 10 + np.arange(10)).ravel()
        assert np.array_equal(indices, expected)
    data = np.load(ARTIFACT_DIR / 'train.npy', mmap_mode='r')
    start = int(data[:, 2].min())
    training = []
    per_day = {}
    for day in [2, 3, 4, 5]:
        a = np.asarray(data[(data[:, 2] >= start + day * 86400) & (data[:, 2] < start + (day + 1) * 86400)])
        e = a[::10]
        indices = ns['first_indices'](e[:, 0])
        assert (np.diff(indices) > 0).all()
        chosen = a[indices]
        assert np.unique(chosen[::10, 0]).size == len(indices) // 10 == np.unique(e[:, 0]).size
        assert (chosen[:, 6].reshape(-1, 10) == chosen[::10, 6, None]).all()
        assert (chosen[:, 5].reshape(-1, 10).sum(1) <= 1).all()
        minimum = np.full(int(data[:, 0].max()) + 1, np.iinfo(np.int32).max, np.int64)
        np.minimum.at(minimum, e[:, 0], e[:, 2])
        assert np.array_equal(chosen[::10, 2], minimum[chosen[::10, 0]])
        assert (chosen[:, 2] < start + (day + 1) * 86400).all()
        per_day[day] = (len(a), indices)
        training.append(dict(day=day, original_rows=len(a), first_event_rows=len(indices),
                             unique_users=len(indices) // 10, timestamp_minimum_verified=True,
                             complete_candidate_groups=True))
    for validation_day in [6, 5]:
        saved = ARTIFACT_DIR / f'imp9_training_indices_day{validation_day}.npy'
        if saved.exists():
            offset = 0
            expected = []
            for day in [validation_day - 3, validation_day - 2, validation_day - 1]:
                rows, indices = per_day[day]
                expected.append(offset + indices)
                offset += rows
            assert np.array_equal(np.load(saved), np.concatenate(expected))
    recipe = []
    for day in [6, 5]:
        ref = np.load(ARTIFACT_DIR / f'predictions_imp5_warmcontext_day{day}.npz')
        aligned = np.load(ARTIFACT_DIR / f'predictions_imp5_aligned_day{day}.npz')
        assert np.array_equal(ref['y'], aligned['y']) and np.array_equal(ref['events'], aligned['events'])
        v6 = np.load(ARTIFACT_DIR / f'predictions_v6_day{day}.npz')['pred']
        v7 = np.load(ARTIFACT_DIR / f'predictions_v7_day{day}.npz')['pred']
        def logit(p):
            p = np.clip(p, 1e-8, 1 - 1e-8)
            return np.log(p) - np.log1p(-p)
        member = .5 * aligned['raw'] + .5 * ref['raw']
        p = 1 / (1 + np.exp(-np.clip(.7 * logit(member) + .2 * logit(v6) + .1 * logit(v7), -35, 35)))
        p = np.clip(p, 1e-8, 1 - 1e-8).reshape(-1, 10)
        odds = p / (1 - p)
        competition = (.5 * p + .5 * odds / (1 + odds.sum(1, keepdims=True))).ravel()
        warm = np.mean([np.load(ARTIFACT_DIR / f'{name}_day{day}.npz')['raw'] for name in
                        ['imp2_warm1.0_d16_pc1', 'imp3_warm16_s7', 'imp3_warm16_s1009']], axis=0)
        error = float(np.max(np.abs(.9 * competition + .1 * warm - ref['pred'])))
        assert error < 1e-12, error
        recipe.append(dict(day=day, mix0_baseline_max_error=error))
    result = dict(status='PASS', repeated_unsorted_empty_users='PASS', training=training,
                  baseline_recipe=recipe, validation_cohort_unchanged=True)
    (ARTIFACT_DIR / 'imp9_test_firstevents_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
