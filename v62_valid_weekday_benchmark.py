"""Frozen benchmark audit for V53 and candidate weekday experiments.

This script records the repo's accepted protocol and rejects any candidate whose
results are not supported by an unseen validation cohort. It is intentionally
conservative: it refuses to claim any score improvement unless the benchmark is
valid and the candidate is not a same-cohort replay.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'


def compare(day: int, label: str):
    base_path = ART / f'v53_causal_propensity_day{day}.npz'
    cand_path = ART / f'v61_weekday_causal_day{day}.npz'
    base = np.load(base_path)
    cand = np.load(cand_path)

    same_events = bool(np.array_equal(base['events'], cand['events']))
    same_labels = bool(np.array_equal(base['y'], cand['y']))
    base_auc = float(roc_auc_score(base['y'], base['pred']))
    cand_auc = float(roc_auc_score(cand['y'], cand['pred']))
    gain = cand_auc - base_auc

    return {
        'day': day,
        'label': label,
        'same_events': same_events,
        'same_labels': same_labels,
        'baseline_auc': base_auc,
        'candidate_auc': cand_auc,
        'gain': gain,
        'candidate_valid_under_protocol': False,
        'reason': 'Candidate is scored on the same timestamps/labels as the V53 replay; it is not a valid future unseen holdout.'
    }


def main():
    benchmark = {
        'name': 'V53_accepted_protocol_audit',
        'description': (
            'Use the repo\'s accepted V53 benchmark: day5 full-user crossfit and day6 future-date holdout. '
            'A candidate is valid only if it is evaluated on a genuinely unseen future cohort and not merely on '
            'the same retained label set used to generate the V53 replay.'
        ),
        'metric': 'ROC AUC',
        'target': 0.85,
        'days': {
            5: 'full-day5 user-crossfit; not a second future temporal fold',
            6: 'future-day6 benchmark; valid future event set'
        },
        'status': 'REJECTED',
        'target_achieved': False,
        'notes': [
            'V53 remains the accepted local baseline.',
            'The earlier v61 results are not a valid model improvement because the same labels and cohort are used for scoring.',
            'No official or upload-worthy score is implied by this benchmark.'
        ],
    }

    reports = [compare(5, 'day5_user_crossfit'), compare(6, 'day6_future_date')]
    benchmark['reports'] = reports
    benchmark['accepted_candidate'] = False
    benchmark['score_summary'] = {
        'max_candidate_auc': max(r['candidate_auc'] for r in reports),
        'max_baseline_auc': max(r['baseline_auc'] for r in reports),
        'max_gain': max(r['gain'] for r in reports),
    }

    out = ROOT / 'results' / 'v62_valid_weekday_benchmark.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(benchmark, indent=2))

    md = ROOT / 'results' / 'v62_valid_weekday_benchmark.md'
    md.write_text(
        '# Valid weekday benchmark audit\n\n'
        + '## Benchmark\n'
        + '- Metric: ROC AUC\n'
        + '- Accepted repo protocol: day5 user-crossfit + day6 future-date validation\n'
        + '- Target: 0.85\n\n'
        + '## Result\n'
        + '- Status: REJECTED\n'
        + '- Reason: candidate predictions are scored on the same retained labels/cohort as the V53 replay, so they are not a valid future benchmark.\n'
        + '- Target achieved: no\n\n'
        + '## Comparison\n'
        + '\n| Day | Baseline AUC | Candidate AUC | Gain | Valid under protocol |\n'
        + '|---|---:|---:|---:|---|\n'
        + '\n'.join(
            f"| {r['day']} | {r['baseline_auc']:.6f} | {r['candidate_auc']:.6f} | {r['gain']:+.6f} | {r['candidate_valid_under_protocol']} |"
            for r in reports
        )
        + '\n'
    )

    print(json.dumps(benchmark, indent=2), flush=True)


if __name__ == '__main__':
    main()
