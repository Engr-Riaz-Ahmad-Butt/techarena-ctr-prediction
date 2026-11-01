"""Audit history-matched screens, quantify cold uncertainty, and record stop."""
import json,hashlib
from pathlib import Path
import numpy as np
from v54_distribution import R,load_samples,evaluate,auc,weights,group,TARGET

data,cut=load_samples();t=np.load(R/'train.npy',mmap_mode='r');start=int(t[:,2].min());reports={s:json.loads((R/f'v54_distribution_step{s}.json').read_text()) for s in (1,2,3,4)};best={};audit=[]
for s in (2,3,4):
    configs={}
    for r in reports[s]['reports']:
        path=R/f'v54_distribution_step{s}_{r["label"]}_day{r["day"]}.npy';p=np.load(path);assert p.shape==data[r['day']]['baseline'].shape and np.isfinite(p).all();m=evaluate(data[r['day']],p,r['day'],r['label']);assert abs(m['weighted_auc']-r['weighted_auc'])<1e-12
        configs.setdefault(r['label'],[]).append(r);audit.append(dict(file=path.name,metric_recomputed=True))
    if s==2:configs={k:v for k,v in configs.items() if not k.endswith('weight0')}
    best[s]=max(configs.values(),key=lambda pair:min(r['weighted_gain'] for r in pair))
assert not any(all(r['weighted_gain']>=.0005 for r in pair) for pair in best.values()),'Survivor: stop report needs confirmation protocol.'
probes=[];rng=np.random.default_rng(5439)
for d,a in data.items():
    z=np.load(R/f'v54_distribution_segment_features_day{d}.npz');zero=a['test']&(a['length']==0);rows=a['rows'][zero];y=a['y'][zero];x=z['x'][zero];prior=np.asarray(t[t[:,2]<start+d*86400]);ni=int(t[:,1].max())+1
    ib=prior[:,1]*4+prior[:,3];n=np.bincount(ib,minlength=ni*4);v=n[rows[:,1]*4+rows[:,3]]
    vals={'item_popularity_by_bucket':v,'item_popularity_by_bucket_daytype':x[:,12],'joint_item_CTR':x[:,13],'first_appearance_age_older_higher':x[:,14],'first_appearance_recency_newer_higher':-x[:,14],'ALS_item_bias':x[:,15]}
    assert len(y)%10==0
    # Whole-event bootstrap keeps the ten correlated candidate rows together.
    for name,p in vals.items():
        estimates=[]
        for _ in range(400):
            ev=rng.integers(0,len(y)//10,len(y)//10);ix=(ev[:,None]*10+np.arange(10)).ravel();score=auc(y[ix],p[ix])
            if score is not None:estimates.append(score)
        probes.append(dict(day=d,feature=name,rows=len(y),positives=int(y.sum()),auc=auc(y,p),event_bootstrap_95_interval=np.quantile(estimates,[.025,.975]).tolist()))
    # Brute-force check a short user's full training-window counts and Bayesian pair feature.
    short=np.flatnonzero(a['length']<=5)[:3]
    for k in short:
        row=a['rows'][k];u,i=map(int,row[:2]);pair=prior[(prior[:,0]==u)&(prior[:,1]==i)&(prior[:,5]==1)];uc=int(((prior[:,0]==u)&(prior[:,5]==1)).sum());assert np.isclose(z['x'][k,4],np.log1p(uc));assert np.isclose(z['x'][k,6],np.log1p(len(pair)))
    for mask in (np.ones(len(a['y']),bool),a['test']):
        w=weights(a['length'][mask]);g=group(a['length'][mask]);assert np.allclose([w[g==k].sum()/w.sum() for k in range(3)],TARGET)
(R/'v54_distribution_zero_probes.json').write_text(json.dumps(probes,indent=2))
full=[r for r in reports[1]['reports'] if r['cohort']=='full_selection'];sample=[r for r in reports[1]['reports'] if r['cohort']=='sample20']
lines=['# History-matched validation and distribution experiments: final report','',
'Stop: no candidate passed +0.0005 history-reweighted AUC on both days. Retained cumulative gain from steps 2–4 is zero, below the requested +0.001 threshold. Keep V53. No holdout evaluation or submission generation.','',
'## Fairer baseline','',
'Weights = target history share / observed cohort history share. Target: 0 clicks 1.26%, 1–5 clicks 16.37%, 6–20 clicks 82.37%. Applied as ROC-AUC sample weights, not an average of separate segment AUCs. These are supplied marginal-history weights; they do not fully replicate evaluation user/item/calendar distribution.','',
'| Cohort | Day 5 unweighted / reweighted AUC | Day 6 unweighted / reweighted AUC |','|---|---:|---:|']
for name,rr in [('Full selection',full),('20% sample',sample)]:lines.append('| '+name+' | '+' | '.join(f'{r["unweighted_auc"]:.9f} / {r["weighted_auc"]:.9f}' for r in rr)+' |')
lines+=['', 'Step2 refits only V53’s original final logistic model: unchanged 190 features, C=.01, scaler, and .75 parent/.25 head rank blend. All upstream experts remain fixed. This is not a new stacker architecture or full upstream pipeline retrain. Weight1 matches the target; weight2/4 multiply short-history training weights further. Weight0 is an unweighted matched retrain control. Only early40% of sampled day5 fits this head.','',
'Steps3/4 train on first-user events from days3/4 (20% hash sample). Each training day’s counts/profile/ALS checkpoint precedes its own cutoff. Step3 uses every earlier training-window click in sparse user-item counts and item-factor profiles, Bayesian priors 5/20, smoothed time_bucket×day_type item popularity/CTR, and ALS features; fits only short histories. Step4 adds user bucket affinity/support/CTR and blends either bucket3 only or all buckets.','',
'All comparisons below use later60% of sampled day5 (42,140 rows) and sampled day6 (48,960 rows), excluding the reserved holdout. The day5 restriction avoids fitting any new predictor on its tested labels. Its baseline differs from the full sample. The early-day5 monotone probability map used for blends was approved in the prior batch and is reused as a protocol, not fitted on test labels.','',
'| Best shared setting | Day5 weighted gain | Day6 weighted gain | Day5 unweighted gain | Day6 unweighted gain |','|---|---:|---:|---:|---:|']
for s,pair in best.items():lines.append(f'| Step{s}: {pair[0]["label"]} | '+ ' | '.join(f'{r[k]:+.9f}' for k in ('weighted_gain','unweighted_gain') for r in pair)+' |')
lines+=['','No configuration passes the gate. No rejected change was promoted or combined into the baseline.','',
'## Zero-history probes','',
'| Feature | Day5 AUC | Day6 AUC |','|---|---:|---:|']
for name in vals:
    pp=[p for p in probes if p['feature']==name];lines.append('| '+name+' | '+' | '.join(f'{p["auc"]:.6f}' for p in pp)+' |')
lines+=['', 'There are only 17 positive rows in the day5 cold cohort and 11 in day6. Several item-level signals score above 0.5 descriptively, but this is weak evidence: the event-bootstrap intervals are stored in artifacts/v54_distribution_zero_probes.json. Neither those signals nor the dedicated short model produce a passing global weighted gain. First-appearance age and recency have opposite score directions; both are explicitly reported.','',
'## Checks and limits','',
f'All {len(audit)} saved prediction arrays were checked for finite values, shape and exact metric reproduction. Weight sums reproduce the requested mix. Full earlier-window click counts spot-checked directly against training rows. Prefix timestamps and ALS cutoffs asserted. No evaluation file was needed in this batch. V53/V20, baseline_backup and submission artifacts were only read where needed, never written. The separate holdout was not opened or scored. No gain > +0.01 occurred.','',
'Inherited limitation: V53 day5 parent scores/features include the original user-cross-fitted meta pipeline, which is out-of-sample by user rather than a fully temporal parent model. All new feature statistics and dedicated model fitting use earlier data. These heavily reused dates do not provide fresh independent generalization evidence.','',
'The sample results do not support the +0.005 official improvement goal. They show a plateau for the tested recipes, not a mathematical ceiling or proof a better model is impossible. Matching history mix lowers the baseline relative to unweighted AUC; that is a fairer descriptive comparison, not a new official leaderboard estimate. No positive leaderboard gain can be claimed from this batch.','',
'Reproduce from the project folder: `python v54_distribution.py 1`, `python v54_distribution.py 2`, `python v54_distribution.py 3`, `python v54_distribution.py 4`, then `python report_v54_distribution.py`. Existing source models, original feature scripts and experiment2 sample caches are required. New caches/predictions use only the v54_distribution prefix.','']
out=Path('results/v54_distribution_final_report.md');out.write_text('\n'.join(lines),encoding='utf-8')
(R/'v54_distribution_audit.json').write_text(json.dumps(dict(status='PASS_NEW_SCREEN_CHECKS',prediction_checks=audit,weight_marginals_verified=True,full_click_count_spot_checks=6,best=best,holdout_evaluated=False,protected_artifacts_written=False,source_sha256=hashlib.sha256(Path('v54_distribution.py').read_bytes()).hexdigest()),indent=2))
with Path('results_log.md').open('a',encoding='utf-8') as f:
    f.write('\n## 2026-10-29 — Test-like history weighting baseline\n\n- Target history mix: 1.26%/16.37%/82.37%; exact target/source weights. Full selection reweighted V53 AUC '+ '/'.join(f'{r["weighted_auc"]:.9f}' for r in full)+'. No model fitted. Step1 runtime not separately captured.\n')
    for s in (2,3,4):
        r=reports[s];f.write(f'\n## 2026-10-29 — Distribution batch step{s}\n\n- Hypothesis/change: '+r.get('recipe',r.get('protocol',''))+'\n');f.write(f'- Runtime: {r["elapsed_seconds"]:.3f}s local CPU.\n');f.write('- Best shared configuration: '+best[s][0]['label']+'. '+ '; '.join(f'day{x["day"]} weighted {x["baseline_weighted_auc"]:.9f}->{x["weighted_auc"]:.9f} ({x["weighted_gain"]:+.9f}); unweighted {x["baseline_unweighted_auc"]:.9f}->{x["unweighted_auc"]:.9f} ({x["unweighted_gain"]:+.9f})' for x in best[s])+'.\n- Decision: reject, fails +0.0005 weighted gain on both days. No holdout or submission.\n')
    f.write('\nRetained cumulative weighted gain = 0 (<+0.001). Stop per user. Detailed features, zero-history probes, limitations and reproduction: results/v54_distribution_final_report.md.\n')
p=Path('agent_state.json');s=json.loads(p.read_text());s['status']='stopped_distribution_batch_weighted_gain_below_0.001';s['active_metric']='Global ROC AUC reweighted to supplied 3-bin evaluation history mix';s['next_action']='Stop; await user direction';s['distribution_batch']={'report':str(out),'weighted_full_selection_baselines':full,'target_mix':TARGET.tolist(),'best_sample_results':best,'retained_gain':0.0,'holdout_evaluated':False,'final_head_only_retrained':True};s['completed_directions']+=['History-matched validation established','Training distribution weights1/2/4 final V53 head rejected','All-window short-history expert rejected','Bucket3/joint-context expert rejected'];p.write_text(json.dumps(s,indent=2));print(json.dumps(dict(final_report=str(out),best=best,probes=probes)))
