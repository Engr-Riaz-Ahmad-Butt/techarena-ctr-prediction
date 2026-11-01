"""Verify stored screens and write the stop-condition report and research state."""
import json,hashlib
from pathlib import Path
import numpy as np
from v54_abcd import R,load_samples,metrics

data,cut=load_samples(); stages={k:json.loads((R/f'v54_abcd_{k}.json').read_text()) for k in 'ABCD'}; best={}; checks=[]
for stage in 'BCD':
    rr=stages[stage]['results']; configs={}
    for r in rr:
        path=R/f'v54_abcd_{stage}_{r["label"]}_day{r["day"]}.npy';p=np.load(path);a=data[r['day']]
        assert p.shape==a['baseline'].shape and np.isfinite(p).all()
        m=metrics(a,p,r['day'],r['label']);assert abs(m['auc']-r['auc'])<1e-12
        if stage=='C':assert np.array_equal(p[~a['repeat']],a['baseline'][~a['repeat']])
        configs.setdefault(r['label'],[]).append(r)
        checks.append(dict(file=path.name,metric_recomputed=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    best[stage]=max(configs.values(),key=lambda pair:min(r['gain'] for r in pair))
survivors=[dict(stage=k,results=v) for k,v in best.items() if all(r['gain']>=.0005 for r in v)]
assert not survivors,'A survivor requires full-cohort confirmation before this stop report.'
full=[r for r in stages['A']['reports'] if r['scope']=='full_selection'];ev=stages['A']['evaluation_counts']
lines=['# V54 A–D final research report','',
'No candidate passed the +0.0005 global-AUC gate on both selection samples. Retained cumulative gain is zero, below the requested +0.001 stopping threshold. Stop this research batch and retain V53. No new submission is justified.','',
'## Segment shares','',
'These are full selection cohorts (day 6 excludes the reserved holdout). Evaluation has 529,390 rows; only descriptive item/history fields were used.','',
'| Segment | Day 5 count / share | Day 6 count / share | Evaluation count / share |',
'|---|---:|---:|---:|']
for key,label in [('history_0','History 0'),('history_1_5','History 1–5'),('history_6_20','History 6–20'),('repeat','Repeat'),('non_repeat','Non-repeat')]:
    vals=[r['counts'] for r in full]+[ev]
    lines.append('| '+label+' | '+' | '.join(f'{v[key]:,} / {v[key]/v["rows"]:.2%}' for v in vals)+' |')
lines+=['',f'Perfectly resolving every zero-history positive/negative, including all comparisons against other segments, raises full-selection AUC at most by +{full[0]["maximum_zero_history_gain"]:.6f} / +{full[1]["maximum_zero_history_gain"]:.6f} on days 5/6. This is a label-informed oracle bound, not a usable feature or expected model gain. Even perfection on those rows does not supply +0.005. Evaluation gain cannot be computed without labels.','',
'## Model screens','',
'All models fit the earliest 40% of sampled day-5 events (28,030 rows). Day-5 results below use only the later 42,140 rows; day 6 uses 48,960 sampled selection rows. This earlier-block protocol was explicitly approved by the user. Scores cannot be compared to previous full-day sample headlines without matching these cohorts.','',
'| Best shared configuration in each family | Day 5 baseline → candidate | Day 6 baseline → candidate |',
'|---|---:|---:|']
for k,v in best.items():
    lines.append('| '+k+': '+v[0]['label']+' | '+' | '.join(f'{r["baseline_auc"]:.9f} → {r["auc"]:.9f} ({r["gain"]:+.9f})' for r in v)+' |')
iso={}
for r in stages['B']['results']:
    if r['label'].startswith('segment_isotonic'):iso.setdefault(r['label'],[]).append(r)
ib=max(iso.values(),key=lambda v:min(r['gain'] for r in v))
lines+=['',f'Per-segment isotonic calibration also failed: best shared choice {ib[0]["label"]}, gains '+ ' / '.join(f'{r["gain"]:+.9f}' for r in ib)+'.',
'', 'The dedicated repeat model had only 718 early-block repeat training rows. At its best shared 10% blend, within-repeat AUC fell from '+ ' / '.join(f'{r["repeat_baseline_auc"]:.6f} to {r["repeat_auc"]:.6f}' for r in best['C'])+' (days 5/6). Every non-repeat score was preserved exactly.',
'', 'D reused the three saved neural seeds, a distinct existing tree model, and an ALS model, then also fitted three new shallow stacker seeds. None passed. These saved components are already ingredients of V53, so their marginal diversity is limited. Full V53 base pipelines were not retrained with new seeds; the original logistic meta learner is deterministic. The new three-seed experiment is second-stage diversity, not three full V53 replicas.',
'', '## Leakage and reproducibility','',
f'Fit timestamps are strictly below {cut}; all reported day-5/day-6 evaluation timestamps are later. Repeat features/statistics retain the prior experiment’s strict pre-cutoff construction and audited history timestamps. Saved model source rows were joined by event and candidate position, with exact label alignment assertions. No evaluation labels or evaluation-derived model statistics were used; evaluation access was limited to the authorized segment counts. The reserved holdout was never opened in this batch. No V53, V20, baseline backup, or submission file was written.',
'', 'Inherited limitation: V53 day-5 scores are user-cross-fitted out-of-sample scores, not fully temporal parent predictions; the original V53 meta layer uses other day-5 users. The new models themselves use only the approved earlier time block. Selection days and model components have been heavily reused, so these results are screens rather than untouched performance estimates.',
'', f'All {len(checks)} saved prediction arrays were checked for shape, finiteness, and exact metric reproduction. No gain exceeded +0.01. The full-cohort and holdout stages were not warranted because every sample gate failed.',
'', 'Reproduce from the project folder with `python v54_abcd.py A`, then `python v54_abcd.py B`, then `python v54_abcd.py C`, then `python v54_abcd.py D`, then `python report_v54_abcd.py`. A repeats only the explicitly authorized evaluation segment count. Existing experiment-2 caches and model prediction artifacts are required.',
'', '## Honest ceiling assessment','',
'These tests support a practical plateau for the current features and model family near existing V53 performance. They do not establish a mathematical ceiling or prove the leaderboard target is impossible. The +0.005 target is unsupported by the measured residual signals; no positive leaderboard gain should be claimed from this batch. Global validation AUC and historical history-adjusted AUC are different quantities; the higher global values do not imply an official improvement. Keep the previously validated submission; no new upload file was created.','']
out=Path('results/v54_abcd_final_report.md');out.write_text('\n'.join(lines),encoding='utf-8')
audit=dict(status='PASS_FOR_NEW_SCREEN_OPERATIONS',prediction_checks=checks,fit_cutoff=cut,holdout_accessed=False,evaluation_access='authorized descriptive segment counts only',survivors=survivors,inherited_limitation='V53 day5 parent score is user-crossfit OOS, not fully temporal',source_sha256=hashlib.sha256(Path('v54_abcd.py').read_bytes()).hexdigest())
(R/'v54_abcd_audit.json').write_text(json.dumps(audit,indent=2))
with Path('results_log.md').open('a',encoding='utf-8') as f:
    for stage in 'ABCD':
        f.write(f'\n## 2026-10-29 — Requested screen {stage}\n\n')
        descriptions={'A':'Hypothesis: cold/short histories or repeat population mismatch might explain the missing global AUC. Descriptive segment counts only. Oracle zero-history bound includes cross-segment pairs.','B':'Hypothesis: cross-segment ordering improves with a shallow regularized LightGBM stacker or per-segment isotonic. Fit approved early40% day5; evaluate later60% and future day6.','C':'Hypothesis: a repeat-only expert can improve within-repeat discrimination. Eight specified features; only repeats fit/scored; non-repeat V53 scores unchanged.','D':'Hypothesis: model/seed diversity adds marginal signal. Reused three existing neural seeds and tree/ALS; additionally three new shallow LightGBM stacker seeds on early day5. Full V53 replicas not refitted.'}
        f.write('- '+descriptions[stage]+'\n')
        f.write(f'- Runtime: {stages[stage]["elapsed_seconds"]:.3f} seconds locally on CPU.\n')
        if stage=='A':f.write('- Full-selection zero-history oracle gain: '+ '/'.join(f'+{r["maximum_zero_history_gain"]:.9f}' for r in full)+'. Evaluation shares in artifacts/v54_abcd_A.json; no labels used.\n')
        else:
            f.write('- Best shared configuration: '+best[stage][0]['label']+'. '+ '; '.join(f'day{r["day"]}: {r["baseline_auc"]:.9f} -> {r["auc"]:.9f} ({r["gain"]:+.9f})' for r in best[stage])+'.\n')
            f.write('- Decision: reject; fails +0.0005 on both days. Full cohort/holdout not evaluated.\n')
        f.write(f'- Details: artifacts/v54_abcd_{stage}.json. Final audit/report: results/v54_abcd_final_report.md.\n')
    f.write('\nA–D retained cumulative gain = zero (< +0.001): stop per user instruction. Practical plateau for tested feature/model family; no mathematical ceiling established. Baseline unchanged; no submission generated.\n')
p=Path('agent_state.json');s=json.loads(p.read_text());s['status']='stopped_abcd_gain_below_0.001';s['next_action']='Stop and report per user; await new direction';s['abcd']={'final_report':str(out),'best_sample_configs':best,'accepted':False,'retained_gain':0.0,'holdout_checked':False,'day5_test_protocol':'Later60% after fitting earliest40%; approved by user','full_v53_seed_retraining':False};s['completed_directions']+=['A segment shares and zero-history oracle bound','B calibration/stacker rejected','C dedicated repeat expert rejected','D model/seed diversity screen rejected'];s['current_best_scores_global_selection']={f'day{r["day"]}':r['baseline_auc'] for r in full};p.write_text(json.dumps(s,indent=2))
print(json.dumps(dict(final_report=str(out),audit_checks=len(checks),best=best,retained_gain=0.0)))
