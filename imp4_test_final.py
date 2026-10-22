"""Independently verify final candidate; never train or modify submissions."""
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
    path = ROOT / 'submission_v18_candidate.csv'
    ids, pred = read(path.name)
    oldids, old = read('submission_imp3_warmfeat_raw.csv')
    ids6, v6 = read('submission_v6.csv')
    ids7, v7 = read('submission_v7.csv')
    ids17, v17 = read('submission_v17.csv')
    assert ids == oldids == ids6 == ids7 == ids17
    assert len(ids) == len(set(ids)) == 529390
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    reports = json.loads((ROOT / 'imp4_decompose_report.json').read_text())
    assert {r['day'] for r in reports} == {5, 6}
    mix = reports[0]['mix']
    assert all(r['mix'] == mix and r['trees'] == 599 for r in reports)
    raw = np.load(ROOT / 'imp4_decompose_final_raw.npy')
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
    baseline_error = float(np.max(np.abs(v17 - combine(old))))
    assert baseline_error < 1e-9, baseline_error
    model = CatBoostClassifier()
    model.load_model(str(ROOT / 'imp4_decompose_final.cbm'))
    params = model.get_all_params()
    assert model.tree_count_ == 599 and params['depth'] == 7
    assert abs(params['learning_rate'] - .08) < 1e-7
    result = dict(candidate=path.name, rows=len(ids), unique_ids=len(set(ids)),
                  ids_match_verified_v17=True, probabilities_valid=True,
                  frozen_recipe_max_error=error, v17_replay_max_error=baseline_error,
                  trees=model.tree_count_, depth=params['depth'], learning_rate=params['learning_rate'],
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(), official_score=None,
                  status='PASS')
    (ROOT / 'imp4_final_independent_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
