"""Resume reporting saved matched scores; never rescore or refit models."""
import json,time
from pathlib import Path
import numpy as np
from horizon_matched_control import R,RUN,BASE,A,sha,auc,AUCPlan
from horizon_foldin import self_test

tic=time.monotonic();frozen=json.loads((RUN/'frozen.json').read_text());pred=dict(np.load(RUN/'predictions.npz'));co=dict(np.load(R/'cohort.npz'));y=np.load(R/'validation_labels.npy').ravel();ledger=json.loads((RUN/'history_ledger_audit.json').read_text())
assert sha(RUN/'scored_source_snapshot.py')==frozen['source_sha256']
assert sha(R/'cohort.npz')==frozen['cohort_sha256'] and sha(R/'score_rows.npy')==frozen['scoring_rows_sha256']
assert sha(A/'imp17_als_cut3.npz')==frozen['model_sha256'] and sha(A/'imp3_model_warm16_cut3.npz')==frozen['warm_expert_sha256']
assert np.array_equal(pred['control_als'],pred['candidate_als']) and np.array_equal(pred['control_blend'],pred['candidate_blend'])
assert ledger['all_users_post_cutoff_history_records']==0 and ledger['warm_history_records_not_represented_in_cutoff_fit']==0
metrics=[]
for scope,take in [('screen20',co['sample']),('full',np.ones(len(co['events']),bool))]:
    ew=co['event_weight'].copy()
    if scope=='screen20':ew[take]=co['sample_event_weight']
    groups=[('overall',take),('arrival1',take&(co['arrival']==0)),('arrival2',take&(co['arrival']==1)),('arrival3plus',take&(co['arrival']>=2)),('history0',take&(co['history_length']==0)),('history1to5',take&(co['history_length']>=1)&(co['history_length']<=5)),('history6to20',take&(co['history_length']>=6))]
    for arm,pk,qk in [('standalone_ALS','control_als','candidate_als'),('simplified_blend','control_blend','candidate_blend')]:
        for name,mask in groups:
            eix=np.flatnonzero(mask);ix=(eix[:,None]*10+np.arange(10)).ravel();w=np.repeat(ew[mask],10);yy=y[ix];p=pred[pk][ix];q=pred[qk][ix]
            control=auc(yy,p,w);candidate=auc(yy,q,w);assert control==candidate
            m=dict(scope=scope,arm=arm,group=name,events=len(eix),rows=len(yy),clicked_events=int(yy.sum()),effective_events=float(ew[mask].sum()**2/np.square(ew[mask]).sum()),weighted_control_auc=control,weighted_candidate_auc=candidate,weighted_delta=0.,unweighted_control_auc=auc(yy,p),unweighted_candidate_auc=auc(yy,q),unweighted_delta=0.,paired_delta_95=[0.,0.],uncertainty_method='Identical per-row score arrays: paired delta is exactly zero in every possible user-bootstrap resample.')
            metrics.append(m)
# Actual whole-user resampling of 20% overall and late cohort verifies degeneracy.
boots=[]
for name,take in [('overall',co['sample']),('arrival3plus',co['sample']&(co['arrival']>=2))]:
    eix=np.flatnonzero(take);ix=(eix[:,None]*10+np.arange(10)).ravel();ew=co['event_weight'].copy();ew[co['sample']]=co['sample_event_weight'];w=np.repeat(ew[take],10);rng=np.random.default_rng(541701);plans={k:AUCPlan(y[ix],pred[k][ix],w) for k in ('control_als','candidate_als','control_blend','candidate_blend')};deltas={k:[] for k in ('ALS','blend')}
    for _ in range(400):
        mult=np.repeat(np.bincount(rng.integers(0,len(eix),len(eix)),minlength=len(eix)),10)
        deltas['ALS'].append(plans['candidate_als'].value(mult)-plans['control_als'].value(mult));deltas['blend'].append(plans['candidate_blend'].value(mult)-plans['control_blend'].value(mult))
    assert all(np.max(np.abs(v))==0 for v in deltas.values())
    boots.append(dict(cohort=name,replicates=400,ALS_delta_95=np.quantile(deltas['ALS'],[.025,.975]).tolist(),blend_delta_95=np.quantile(deltas['blend'],[.025,.975]).tolist()))
protected=json.loads((R/'protected_before.json').read_text())
for name,v in protected.items():assert sha(BASE/name)==v['sha256']
report=dict(status='complete',decision='Stop Idea1: no new history evidence and exact zero effect. Do not schedule full V53 rebuild on this evidence.',metrics=metrics,bootstrap=boots,history_ledger=ledger,advance_to_full_v53_rebuild=False,baseline_is_V53=False,cold_users='Cutoff vectors/biases unchanged in both arms; no new negative evidence generated.',recipe=frozen,unit_tests=self_test(),protected_files_unchanged=len(protected),reporting_seconds=time.monotonic()-tic,technical_note='First reporting pass stopped on an overly strict AUC replay tolerance: full-cohort independent formulas differed by 2.1e-12/2.5e-12. Tolerance corrected to1e-10; saved scores were reused without refitting or rescoring. Original scoring-source snapshot retained.')
(RUN/'report.json').write_text(json.dumps(report,indent=2))
lines=['# Matched control: warm-user anchored ALS refresh','',
'Decision: stop Idea1. No new post-cutoff clicks exist in the frozen first-event cohort, and every warm-user history click was already represented in the cutoff fitting ledger. Candidate and control scores are identical on every row. No full V53 rebuild is scheduled from this result.','',
'## Matched setup','',
'49,420 first-post-cutoff users; 44,627 warm users. 9,878-user screening sample. Cutoff after three training days. Item vectors, user/item biases and global offset fixed. Anchor16, history half-life2days frozen before outcomes. New evidence would include only unique post-cutoff positive clicks strictly before the event; zero such clicks were found. The independent ledger also verified zero missing pre-cutoff history clicks among604,857 warm-user history records. No fake negatives were generated.','',
'ALS control uses the existing causal cutoff3 ALS16 checkpoint. Cold users keep their cutoff vectors identically in both arms; this intentionally avoids invoking the old cold fold-in approximation that introduces uniform background negatives. Thus this is a controlled warm-refresh experiment, not a replay of full V53.','',
'The simplified blend is fixed95% rank(warm16 history-preference expert) +5% rank(ALS). Warm16 is an existing V53 contributor, includes supplied event history, and stays identical in both arms. Ranks computed once across the full four-day horizon, not within arrival day. No new stacker or propensity model was fitted.','',
'## Full-cohort weighted AUC','',
'Weights jointly match four-day arrival proportions67:17:7:4 normalized over supported days, and history proportions1.26%/16.37%/82.37%. Subgroup AUC uses the corresponding frozen weights, retaining the same reference distribution. Evaluation days5–7 remain unsupported.','',
'| Group | Users | ALS control = refresh | Simplified blend control = refresh | Paired delta, 95% interval |','|---|---:|---:|---:|---:|']
for name in ('overall','arrival1','arrival2','arrival3plus','history0','history1to5','history6to20'):
    a=next(m for m in metrics if m['scope']=='full' and m['arm']=='standalone_ALS' and m['group']==name);b=next(m for m in metrics if m['scope']=='full' and m['arm']=='simplified_blend' and m['group']==name);lines.append(f'| {name} | {a["events"]:,} | {a["weighted_control_auc"]:.6f} | {b["weighted_control_auc"]:.6f} | 0.000000 [0,0] |')
lines+=['','20% screen overall: '+ '; '.join(f'{m["arm"]} control={m["weighted_control_auc"]:.6f}, refresh={m["weighted_candidate_auc"]:.6f}' for m in metrics if m['scope']=='screen20' and m['group']=='overall')+'.','',
'Every paired user-bootstrap difference is exactly zero because the two per-row score arrays are identical. Actual400-resample user bootstraps on the screening overall and late-arrival cohorts confirm [0,0]. This is a degenerate paired-difference interval for this experiment, not a claim that baseline AUC itself has no uncertainty.','',
'## Meaning and limits','',
'Longer waiting after the cutoff does not automatically supply more observed clicks: each selected event is the first event for that user after the cutoff. Here, the later arrival histories contain no new clicks to update from. Reusing their pre-cutoff clicks as fresh evidence would double-count information. This explains the exact zero result without needing a full ensemble rebuild. A separate proposal to reweight old evidence would be a different experiment, not a successful refresh.','',
'These AUCs are standalone ALS and a deliberately simplified factor-expert blend, not V53 scores. No positive leaderboard gain is claimed. Full V53 has not been rebuilt; Idea2 and Idea3 have not started.','',
'Model/score inputs and29 protected files passed hashes. History timestamps and independent training-click-ledger coverage checked. Kernel tests passed. No holdout opened/scored, no submissions written, no external data, no copied repository code.','',
'Scoring took8.734 seconds. Actual bootstrap/report timing is in report.json. The reporting-only recovery fixed a2e-12 numerical tolerance assertion; it reused already frozen prediction arrays.','',
'Reproduce: `python horizon_matched_control.py` in a fresh output namespace; for the existing completed run, inspect `artifacts/v54_horizon/matched_refresh_001/report.json` and its frozen scores. Do not rerun a completed experiment to select a favorable outcome.','']
out=BASE/'results/v54_horizon_matched_refresh_report.md';out.write_text('\n'.join(lines),encoding='utf-8')
sfile=BASE/'agent_state.json';s=json.loads(sfile.read_text());s['status']='idea1_matched_control_complete_rejected_awaiting_review';s['next_action']='Pause after matched control; stop Idea1; no full V53 rebuild scheduled; Idea2 requires further user direction';s['horizon']['idea1_tested']=True;s['horizon']['matched_control_report']=str(out);s['horizon']['idea1_gain']=0.;s['horizon']['baseline_fitted']=False;sfile.write_text(json.dumps(s,indent=2))
with (BASE/'results_log.md').open('a',encoding='utf-8') as f:f.write('\n## 2026-10-29 — Matched warm-user ALS refresh\n\n- Hypothesis: new event-time history changes stale warm ALS vectors. Frozen anchor16/half-life2days; shared item factors and biases unchanged. Standalone ALS, then fixed95% existing warm16 expert/5% ALS blend.\n- 49,420 first-event users;44,627 warm;604,857 warm history records all already represented in fitting ledger. Zero post-cutoff history clicks, zero refreshed vectors. Candidate/control scores exactly identical.\n- Full weighted overall ALS0.724784897273; simplified blend0.755280328306; both gains0.000000. All arrival/history subgroup deltas0; paired95% intervals[0,0].400 actual user bootstraps of screen overall/late verify degeneracy.\n- Scoring8.734s. First metric report pass hit2e-12 tolerance mismatch; reporting-only recovery reused saved scores and relaxed replay tolerance to1e-10.\n- Decision: flat by exact identity, stop Idea1; do not schedule full V53 reconstruction from this result. Full V53 remains unbuilt; no holdout/submission. Pause before Idea2/3. Details: results/v54_horizon_matched_refresh_report.md.\n')
print(json.dumps(dict(report=str(out),metrics=[m for m in metrics if m['scope']=='full'],bootstrap=boots,seconds=report['reporting_seconds']),indent=2))
