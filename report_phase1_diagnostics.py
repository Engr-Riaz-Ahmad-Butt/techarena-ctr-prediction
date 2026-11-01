"""Complete descriptive horizon checks and produce Phase1 report. No fitting."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import chisquare
from config import ARTIFACT_DIR as R
from phase1_diagnostics import DAY,summary

z=json.loads((R/'v54_phase1_diagnostics.json').read_text());t=np.load(R/'train.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');start=int(t[:,2].min());ni=int(t[:,1].max())+1;nu=int(max(t[:,0].max(),e[:,0].max()))+1
first=np.full(ni,np.iinfo(np.int32).max,dtype=np.int32);np.minimum.at(first,t[:,1],t[:,2]);z['catalog_first_observed']=dict(items=ni,first_observation_hours_since_start=summary((first-start)/3600),share_first24h=float(np.mean(first<start+DAY)))
horizons=[]
for label,rows,cut in [('validation5',None,start+5*DAY),('validation6',None,start+6*DAY),('evaluation',np.asarray(e),start+7*DAY)]:
    if rows is None:
        d=int(label[-1]);a=np.load(R/f'selection_day{d}.npz');ev=a['events'];rows=np.asarray(t[ev*10+np.tile(np.arange(10),len(ev)//10)])
    head=rows[::10];prior=np.asarray(t[t[:,2]<cut]);last=np.zeros(nu,dtype=np.int32);np.maximum.at(last,prior[:,0],prior[:,2]);known=last[head[:,0]]>0;horizon=(head[:,2]-cut)/DAY;gap=(head[known,2]-last[head[known,0]])/DAY
    bins=np.floor(horizon).astype(int);horizons.append(dict(cohort=label,users=len(head),prediction_horizon_days=summary(horizon),events_by_day_after_cutoff={str(k+1):int((bins==k).sum()) for k in np.unique(bins)},time_since_last_training_activity_at_event_days=summary(gap)))
z['prediction_horizons']=horizons
for r in z['validation']:
    pos=r['clicked_list_position']['position_counts'];r['clicked_list_position']['uniform_chisquare_pvalue']=float(chisquare(pos).pvalue)
    r['id_summary']={}
    for name,v in r['id_patterns'].items():
        rates=[x['click_rate'] for x in v['hash_deciles'] if x['click_rate'] is not None];r['id_summary'][name]=dict(raw_hash_label_pearson=v['raw_hash_vs_label']['pearson'],raw_hash_time_pearson=v['raw_hash_vs_timestamp']['pearson'],encoded_index_label_pearson=v['encoded_index_vs_label']['pearson'],encoded_index_time_pearson=v['encoded_index_vs_timestamp']['pearson'],hash_decile_click_rate_min=min(rates),hash_decile_click_rate_max=max(rates))
# The tie-aware mass must form a probability distribution, even for tied counts.
for r in z['validation']:
    for rank in r['clicked_rank_distributions'].values():assert np.isclose(sum(rank['rank_percent']),100)
assert len(e)==529390 and all(r['rows'] in (352300,244220) for r in z['validation'])
(R/'v54_phase1_diagnostics.json').write_text(json.dumps(z,indent=2))

rs=z['validation'];lines=['# Phase 1 diagnostics — no model changes','',
'Scope: all saved selection rows: 352,300 day5 and 244,220 day6. The reserved day6 holdout was not opened or scored. Evaluation was inspected only for authorized candidate/user ID and timestamp counts. No model was trained or changed; no submission was written.','',
'## 1. Drift and recent popularity','',z['drift_limitation'],'',
'| Diagnostic | Day5 | Day6 selection |','|---|---:|---:|',
f'| V53 AUC | {rs[0]["v53_auc"]:.6f} | {rs[1]["v53_auc"]:.6f} |']
for key in ('last1','last3','last7','all'):lines.append('| Impression-popularity AUC '+key+' | '+' | '.join(f'{r["popularity_auc"][key]:.6f}' for r in rs)+' |')
lines+=['','Last7 equals all-time because only 5/6 prefix days exist. Recent impression popularity does not beat all-time. Fixed day5 popularity, without refreshing counts, gets day2 AUC 0.520113 (last1), 0.521675 (last3), 0.521779 (all). That is a real two-date diagnostic for the same simple score, not for one frozen V53.','',
'V53 six-hour slices after each day-specific cutoff:','',
'| Hours | Day5 | Day6 |','|---|---:|---:|']
for k in range(4):lines.append(f'| {k*6}–{(k+1)*6} | '+' | '.join(f'{r["cutoff_hour_slices"][k]["v53_auc"]:.6f}' for r in rs)+' |')
lines+=['','This decline can reflect time-of-day and cohort changes, not necessarily model aging. Evaluation prediction-horizon counts are reported below.','',
'## 2. Item age and lifecycle','',
f'The catalog has {ni:,} items; {z["catalog_first_observed"]["share_first24h"]:.3%} were first observed within24h. First-observation timing in hours: {z["catalog_first_observed"]["first_observation_hours_since_start"]}. This is age within the available window, not actual catalog creation age.','',
'| Quantity | Day5 clicked / not-clicked mean | Day6 clicked / not-clicked mean |','|---|---:|---:|']
for name in ('age_days','recent3_to_all_popularity_share_ratio','alltime_impression_share'):
    lines.append('| '+name+' | '+' | '.join(f'{r["lifecycle"][name]["clicked"]["mean"]:.9g} / {r["lifecycle"][name]["not_clicked"]["mean"]:.9g}' for r in rs)+' |')
lines+=['','Older-first-seen score on ALL users has AUC '+ '/'.join(f'{r["lifecycle"]["age_days"]["auc"]:.6f}' for r in rs)+'. On full zero-history cohorts it is '+ '/'.join(f'{r["zero_history_age_auc"]:.6f}' for r in rs)+'. The earlier promising small-cohort age result does not generalize into a strong all-user signal. Age within each event ranks the clicked candidate only slightly above random (see table). Recent/all popularity-ratio AUC is '+ '/'.join(f'{r["lifecycle"]["recent3_to_all_popularity_share_ratio"]["auc"]:.6f}' for r in rs)+'.','',
'## 3. Candidate generation','',
'Rank1 = highest score/most popular/oldest. Only events with a click enter this distribution. Ties share mass across tied ranks; a flat reference is 10% in each rank. Flat clicked ranks are not implied by uniform negative sampling: clicked items may have different relevance.','',
'| Day / score | Rank1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Mean rank |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rs:
    for name,v in r['clicked_rank_distributions'].items():lines.append(f'| Day{r["day"]} {name} | '+' | '.join(f'{p:.2f}%' for p in v['rank_percent'])+f' | {v["mean_rank"]:.3f} |')
lines+=['', 'Negative candidates concentrate in the highest popularity decile: '+ '/'.join(f'{r["negative_sampling"]["negative_share"][-1]:.2%}' for r in rs)+', versus catalog shares '+ '/'.join(f'{r["negative_sampling"]["uniform_catalog_share"][-1]:.2%}' for r in rs)+'. Bottom-decile negative shares are '+ '/'.join(f'{r["negative_sampling"]["negative_share"][0]:.2%}' for r in rs)+'. Their distribution across popularity deciles is much closer to historical impression-weighted sampling (total variation '+ '/'.join(f'{r["negative_sampling"]["total_variation_negative_vs_impression_weighted"]:.4f}' for r in rs)+') than uniform catalog sampling ('+ '/'.join(f'{r["negative_sampling"]["total_variation_negative_vs_uniform"]:.4f}' for r in rs)+'). This supports popularity-skewed exposure, not a proven negative-sampler algorithm. No useful recent-popularity tilt was detected. Clicked items are mildly more popular than negatives; ALS separates them much more strongly.','',
'## 4. Cold items','',
'Validation day5: 0/352,300 unseen and 0 first-seen-within-two-days. Day6: 0/244,220 for both. Evaluation: 0/529,390 for both. Their V53 AUC is undefined because there are no rows. Cold-item handling cannot explain the observed gap within this catalog/window.','',
'## 5. User-level structure, positions, IDs','',
f'Across {z["user_event_structure"]["events"]:,} training events and {z["user_event_structure"]["users"]:,} users: {z["user_event_structure"]["consecutive_any_overlap_share"]:.2%} of consecutive same-user events share any candidate; mean overlap {z["user_event_structure"]["consecutive_overlap_count"]["mean"]:.5f}/10; zero identical consecutive slates; {z["user_event_structure"]["user_item_repeat_presentation_share"]:.2%} of presentations had appeared in an earlier event for that user. These structure counts inspect candidate IDs/timestamps, not labels.','']
for r in rs:
    v=r['clicked_list_position'];lines.append(f'Day{r["day"]} clicked-position percentages, positions1–10: '+', '.join(f'{p:.2f}' for p in v['position_percent'])+f'. Uniformity chi-square p={v["uniform_chisquare_pvalue"]:.4f}.')
lines+=['','Raw user/item IDs are hash-like strings, not natural numeric ranges. We tested their hash-order deciles, separately from insertion-order encoded IDs. Encoded indices partly encode first appearance and should not be mistaken for meaningful identifier ranges.','']
for r in rs:
    for name,v in r['id_summary'].items():lines.append(f'Day{r["day"]} {name}: raw-hash correlation with click={v["raw_hash_label_pearson"]:+.6f}, timestamp={v["raw_hash_time_pearson"]:+.6f}; encoded-index correlation with click={v["encoded_index_label_pearson"]:+.6f}, timestamp={v["encoded_index_time_pearson"]:+.6f}; raw-hash decile click rates {v["hash_decile_click_rate_min"]:.3%}–{v["hash_decile_click_rate_max"]:.3%}.')
lines+=['', '## 6. Simple-score ceiling and residual correlations','',
'These are observed AUCs of simple scores, not a mathematical ceiling for future models. V53 is an ensemble rank score, not a calibrated probability. Correlation with label minus that rank can reflect score scaling; it does not measure independent predictive gain.','',
'| Signal | Day5 AUC / residual Pearson r | Day6 AUC / residual Pearson r |','|---|---:|---:|']
for name in ('pop_all','pop_last3','ALS','item_age_older_higher','recent3_all_share_ratio'):
    pp=[next(v for v in r['simple_scores'] if v['signal']==name) for r in rs];lines.append('| '+name+' | '+' | '.join(f'{v["auc"]:.6f} / {v["vs_label_minus_V53_rank"]["pearson"]:+.6f}' for v in pp)+' |')
lines+=['','ALS is the single signal most correlated in absolute value with the requested rank residual: r=-0.489916/-0.487990. This mostly reflects shared ranking structure and the mismatch between ranks and probabilities; it is not evidence that another ALS blend would help. Existing model/stacker work remains closed.','',
'## 7. Evaluation user activity and prediction horizon','',
'Training activity means observed impressions, not warm-up clicks. Gap is cutoff minus last training activity among previously seen users; users with no earlier impression are listed separately.','',
'| Cohort | No prior activity | Median cutoff gap, days | 90th percentile | Within1day | Within2days |','|---|---:|---:|---:|---:|---:|']
for label,v in [(f'Day{v["day"]}',v) for v in z['user_activity_gap']['validation']]+[('Evaluation',z['user_activity_gap']['evaluation'])]:lines.append(f'| {label} | {v["no_prior_training_activity"]}/{v["users"]} | {v["gap_days"]["median"]:.4f} | {v["gap_days"]["q90"]:.4f} | {v["within_1_day"]:.2%} | {v["within_2_days"]:.2%} |')
for v in horizons:lines.append(f'\n{v["cohort"]}: events by day1/day2/… after cutoff={v["events_by_day_after_cutoff"]}; median prediction horizon={v["prediction_horizon_days"]["median"]:.4f} days; median time since last training activity at actual event={v["time_since_last_training_activity_at_event_days"]["median"]:.4f} days.')
lines+=['','Evaluation users are less recently active at the cutoff, and evaluation has a longer prediction horizon than the one-day selection windows. This is a real descriptive mismatch that history-length reweighting alone does not remove.','',
'## Ranked follow-up hypotheses — approval required before any modeling','',
'1. Prediction-horizon and user-activity freshness mismatch: evaluation spans seven post-cutoff days and has older user activity; quantify matched fixed-model future-date behavior if later labeled dates exist. This is a diagnostic lead, not permission to reopen short-history, bucket or weighting recipes.','2. Candidate-generator popularity skew: negative candidates are far from uniform over items and match historical exposure weights closely. Obtain the organizer sampling specification before deciding if it leaves any unmodeled information. Existing popularity/stacker models are not reopened.','3. ALS/personal preference structure: standalone ALS AUC is about0.76 and clicked items lead its rank distribution strongly. It is already in V53 and previous diversity tests failed, so incremental value is unproven. It ranks third only as a structural clue; there are not three strong new residual signals in these diagnostics.','',
'Needed: labeled predictions from the SAME frozen V53 across later dates for a genuine multi-day drift curve, and organizer information on how the nine negatives are generated. No new models or submissions should be built without user approval.','']
out=Path('results/v54_phase1_diagnostics_report.md');out.write_text('\n'.join(lines),encoding='utf-8')
with Path('results_log.md').open('a',encoding='utf-8') as f:f.write('\n## 2026-10-29 — Phase1 diagnostics only (previous plan stopped)\n\n- No training/model changes/submissions. Full saved selection cohorts; holdout not opened/scored.\n- Runtime: '+str(z['elapsed_seconds'])+'s diagnostic computation plus report generation.\n- Main results: all-time exposure-popularity AUC0.523103/0.521470; last3 AUC0.522827/0.520886; older-age AUC0.505028/0.505347; ALS AUC0.757775/0.760686. No cold/recent items in validation/evaluation. Negatives match popularity-skewed exposure, not uniform catalog sampling. Evaluation has longer prediction horizon and less recently active users.\n- Same-frozen-V53 multi-day drift is unavailable; no fabricated drift curve. Rank residual correlation is scale-confounded.\n- Decision: diagnostics complete, pause; all closed modeling directions remain closed. Detailed distributions and limitations: results/v54_phase1_diagnostics_report.md.\n')
p=Path('agent_state.json');s=json.loads(p.read_text());s['status']='phase1_diagnostics_complete_awaiting_explicit_approval';s['next_action']='Pause; no modeling without approval';s['phase1']={'report':str(out),'raw':'artifacts/v54_phase1_diagnostics.json','models_changed':False,'holdout_scored':False,'submission_written':False};p.write_text(json.dumps(s,indent=2));print(json.dumps(dict(report=str(out),horizons=horizons,catalog=z['catalog_first_observed'],id_summaries=[r['id_summary'] for r in rs],position_p=[r['clicked_list_position']['uniform_chisquare_pvalue'] for r in rs])))
