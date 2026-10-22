"""Independently verify final candidate; never train or modify submissions."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from config import ARTIFACT_DIR as ROOT
import numpy as np
from catboost import CatBoostClassifier


def read(name):
    with (ROOT / name).open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        assert next(reader) == ['rowid', 'predicted_probability']
        rows = list(reader)
    assert all(len(r) == 2 for r in rows)
    return [r[0] for r in rows], np.array([float(r[1]) for r in rows])


def logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p) - np.log1p(-p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', choices=['aligned', 'timecontext'], required=True)
    parser.add_argument('--version', type=int, default=19)
    args = parser.parse_args()
    path = ROOT / f'submission_v{args.version}_candidate.csv'
    ids, pred = read(path.name)
    oldids, baseline = read('submission_v18.csv')
    old = np.load(ROOT / 'imp4_decompose_final_raw.npy')
    ids6, v6 = read('submission_v6.csv')
    ids7, v7 = read('submission_v7.csv')
    ids_baseline, _ = read('submission_v18.csv')
    assert ids == oldids == ids6 == ids7 == ids_baseline
    assert len(ids) == len(set(ids)) == 529390
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    reports = json.loads((ROOT / f'imp5_{args.tag}_report.json').read_text())
    assert {r['day'] for r in reports} == {5, 6}
    mix = reports[0]['mix']
    assert all(r['mix'] == mix and r['trees'] == 599 for r in reports)
    raw = np.load(ROOT / f'imp5_{args.tag}_final_raw.npy')
    assert raw.shape == (529390,) and np.isfinite(raw).all()
    warm = np.mean([np.load(ROOT / f'{name}.npy').ravel() for name in
                    ['imp2_final_raw_warm1.0_d16_pc1', 'imp3_warm16_s7_final_raw',
                     'imp3_warm16_s1009_final_raw']], axis=0)
    def combine(member):
        z = .7 * logit(member) + .2 * logit(v6) + .1 * logit(v7)
        p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
        p = np.clip(p, 1e-8, 1 - 1e-8).reshape(-1, 10)
        odds = p / (1 - p)
        adjusted = (.5 * p + .5 * odds / (1 + odds.sum(1, keepdims=True))).ravel()
        return .9 * adjusted + .1 * warm
    expected = combine((1 - mix) * old + mix * raw)
    error = float(np.max(np.abs(pred - expected)))
    assert error < 1e-15, error
    baseline_error = float(np.max(np.abs(baseline - combine(old))))
    assert baseline_error < 1e-9, baseline_error
    model = CatBoostClassifier()
    model.load_model(str(ROOT / f'imp5_{args.tag}_final.cbm'))
    params = model.get_all_params()
    assert model.tree_count_ == 599 and params['depth'] == 7
    assert abs(params['learning_rate'] - .08) < 1e-7
    assert len(model.feature_names_) == 258
    # Replay complete events to preserve cached candidate grouping and row order.
    sampled_events = np.linspace(0, 52939 - 1, 1000, dtype=int)
    sampled_rows = (sampled_events[:, None] * 10 + np.arange(10)).ravel()
    matrices = [np.load(ROOT / 'features' / name, mmap_mode='r') for name in
                ['stack_eval_c7.npy', 'imp3_wf_eval.npy', 'imp4_decompose_eval.npy',
                 f'imp5_{args.tag}_eval.npy']]
    assert [matrix.shape for matrix in matrices] == [(529390, 190), (529390, 4),
                                                     (529390, 32), (529390, 32)]
    sampled_features = np.column_stack([matrix[sampled_rows] for matrix in matrices])
    replay_raw = model.predict_proba(sampled_features)[:, 1]
    raw_replay_error = float(np.max(np.abs(replay_raw - raw[sampled_rows])))
    assert raw_replay_error < 1e-12, raw_replay_error
    result = dict(candidate=path.name, rows=len(ids), unique_ids=len(set(ids)),
                  ids_match_verified_v18=True, probabilities_valid=True,
                  frozen_recipe_max_error=error, v18_replay_max_error=baseline_error,
                  trees=model.tree_count_, depth=params['depth'], learning_rate=params['learning_rate'],
                  features=len(model.feature_names_), variant=args.tag,
                  raw_model_replay_rows=len(sampled_rows), raw_model_replay_max_error=raw_replay_error,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(), official_score=None,
                  status='PASS')
    (ROOT / f'imp5_{args.tag}_final_independent_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
