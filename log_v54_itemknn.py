"""Record experiment 1 outcome without modifying baseline artifacts."""
import json
from pathlib import Path

p=Path('artifacts/v54_itemknn_screen.json')
r=json.loads(p.read_text())
best=r['best_joint_configuration']
lines=['\n## 2026-10-29 — Experiment 1: restricted ItemKNN screen\n',
'- **Hypothesis:** popular-item co-click similarity may aggregate enough evidence to improve V53 beyond sparse exact transitions.',
'- **Change:** binary user-item clicks strictly before each day cutoff; symmetric co-click cosine, zero diagonal; top 5k/10k/20k items; shrinkage 1/10/100; last 5/20 history clicks; max/mean/recency aggregation; rank blends 1/2/5/10%. Average ranks preserve ties. Cached sparse matrices. ItemKNN tested; dense EASE inverse not tested.',
f'- **Runtime:** {r["elapsed_seconds"]:.2f} seconds locally on CPU; 216 configurations per day on deterministic 20% user samples (7,017 day-5 events; 4,896 day-6 events).',
'- **Metric:** current user brief specifies global ROC AUC; older history-adjusted gates are superseded. Adjusted scores retained as secondary diagnostics. Sample scores are not full-day scores.']
for x in best:
    lines.append(f'- **Day {x["day"]}:** baseline {x["baseline_auc"]:.9f}; best joint configuration {x["auc"]:.9f}; gain {x["gain"]:+.9f}.')
lines += ['- **Result:** every tested blend worsened sample global AUC. Even top-20k, last-20 similarity is nonzero for only 7.39%/9.71% of day-5/day-6 rows.',
'- **Decision:** reject tested ItemKNN family; no full-data confirmation justified, no candidate promoted. V53 cumulative gain remains zero. Holdout and evaluation files were not opened; no submission written. Pause after experiment 1 as explicitly requested. Next route after user review: repeat-item deep dive.',
'- **Leakage controls:** only training labels strictly before cutoff used for popularity and similarity; no evaluation statistics; no validation labels used to fit matrices; self-similarity excluded. Supplied histories follow the existing data contract (prior clicks).\n']
with Path('results_log.md').open('a',encoding='utf-8') as f:
    f.write('\n'.join(lines))
sfile=Path('agent_state.json'); s=json.loads(sfile.read_text())
s['status']='awaiting_user_review_after_experiment_1'
s['next_action']='User-requested pause; then experiment 2 repeat-item deep dive'
s['completed_directions'].append('Restricted ItemKNN screen rejected: all 216 configurations/day worsened global AUC')
s['experiment_1']={'report':str(p),'decision':'reject tested ItemKNN; EASE inverse untested','best_joint_sample_results':best,'holdout_evaluated':False,'submission_written':False}
s['cumulative_gain_over_v53']={'day5':0.0,'day6':0.0,'holdout':None}
sfile.write_text(json.dumps(s,indent=2))
print('Results log and agent state updated.')
