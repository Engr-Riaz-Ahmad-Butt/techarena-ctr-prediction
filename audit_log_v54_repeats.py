"""Independent pair checks and durable outcome for experiment 2."""
import json
from pathlib import Path
import numpy as np
from config import ARTIFACT_DIR as R

t=np.load(R/'train.npy',mmap_mode='r'); h=np.load(R/'train_hist.npy',mmap_mode='r'); ht=np.load(R/'train_times.npy',mmap_mode='r')
diag=json.loads((R/'v54_repeat_diagnostics.json').read_text()); report=json.loads((R/'v54_repeat_screen_report.json').read_text()); checks=[]
maxcut=max(r['cutoff'] for r in diag['reports'])
events=np.flatnonzero(t[::10,2]<maxcut)
for off in range(0,len(events),20000):
    ev=events[off:off+20000]; assert np.all((ht[ev]<t[ev*10,2,None])|(h[ev]<0))
for d in (5,6):
    z=np.load(R/f'v54_repeat_screen_day{d}.npz'); x=z['x']; cutoff=next(r['cutoff'] for r in diag['reports'] if r['day']==d)
    for k in np.r_[np.flatnonzero(x[:,15]>0)[:5],np.flatnonzero(x[:,15]==0)[:3]]:
        ev=int(z['events'][k]); row=t[ev*10+int(k%10)]; u,i,stamp=map(int,row[:3])
        pair=np.asarray(t[(t[:,0]==u)&(t[:,1]==i)&(t[:,2]<cutoff)])
        neg=pair[pair[:,5]==0]; pos=pair[pair[:,5]==1]
        assert np.isclose(x[k,14],np.log1p(len(neg)))
        assert np.isclose(x[k,17],np.log1p(len(neg)/(len(pos)+1)))
        expected=np.log1p((stamp-int(neg[:,2].max()))/86400) if len(neg) else 0
        assert np.isclose(x[k,16],expected)
        hist=np.asarray(h[ev]); times=np.asarray(ht[ev]); m=(hist==i)&(hist>=0)
        assert x[k,2]==m.sum()
        if m.any():
            assert x[k,5]==(hist>=0).sum()-1-np.flatnonzero(m)[-1]
            assert np.isclose(x[k,6],np.log1p((stamp-int(times[m].max()))/86400))
        checks.append(dict(day=d,event=ev,negative_count=len(neg),click_count=len(pos),passed=True))
audit=dict(status='PASS',all_prior_history_timestamps_checked=True,pair_spot_checks=checks,holdout_accessed=False,evaluation_accessed=False,notes=['Prefix statistics frozen at cutoff, strictly earlier than each scored event.','All pair exposure counts restricted to earlier observed training events.','OOF day5 split keeps user labels disjoint; day6 fit labels from day5 only.','Existing V53 score is inherited; this is a screen, not a new independent temporal validation.'])
(R/'v54_repeat_screen_audit.json').write_text(json.dumps(audit,indent=2))
lines=['\n## 2026-10-29 — Experiment 2: repeat-item and negative-exposure deep dive\n',
'- **Hypothesis:** repeated click recency and ignored user-item presentations add residual signal beyond V53.',
'- **Existing-feature review:** V53 explicitly includes candidate history click count; core/core_v2 add repeat count and recent-3 matching; core_v5 and V53 contain user/item exposure/CTR summaries; core_v6 contains exposure-adjusted latent user profiles. No explicit same-user same-item ignored-presentation count, recency, or negative/click ratio found in the inspected active feature lineage. New pair features are distinct; repeat count retained as an existing control.',
'- **Change:** strict pre-cutoff user-item unclicked exposure count, last unclicked gap, smoothed negative/click ratio and pair click count; smoothed user seen/new CTR and seen-click share; repeat clicks-ago, last-click gap, same-bucket history fraction, user/item repeat-click rate and user bucket/same-bucket repeat CTR. Warm-up exposures unknown, so seen/new uses observed training presentations only.',
'- **Protocol:** deterministic same 20% user selection; day5 two-fold user OOF, day6 trained on day5 only; 180 fixed CatBoost trees, depth4, seed5402; control/repeat/exposure/combined arms, rank blends 5/10/25%. No early stopping. Global AUC is primary; history-adjusted AUC secondary. Caches contain no evaluation data.']
for r in diag['reports']:
    seg={x['repeat_status']:x for x in r['segments'] if x['history_length']=='all'}
    lines.append(f'- **Day {r["day"]} diagnostics:** repeat AUC {seg["repeat"]["auc"]:.6f}; non-repeat AUC {seg["non_repeat"]["auc"]:.6f}; negative-pair exposure coverage {r["pair_negative_exposure_share"]:.4%}. Full history-length and repeat cross-tab in artifacts/v54_repeat_diagnostics.json.')
for arm in ('control','repeat','exposure','combined'):
    rr=[r for r in report['results'] if r['arm']==arm and r['weight']==.05]
    lines.append(f'- **{arm}, 5% blend:** '+ '; '.join(f'day {r["day"]} {r["baseline_auc"]:.9f} -> {r["auc"]:.9f} ({r["gain"]:+.9f})' for r in rr)+'.')
lines += [f'- **Runtime:** feature preparation {diag["elapsed_seconds"]:.2f}s; model screening {report["elapsed_seconds"]:.2f}s locally, CPU.',
'- **Decision:** reject tested repeat/exposure screen. No configuration gains +0.0005 on both days; best shared exposure-only 5% blend gains -0.000029/+0.000015, incremental to matched control just +0.000006/+0.000009. V53 remains best, cumulative gain zero. Strong repeat click rate does not establish useful residual predictability.',
'- **Audit:** independent brute-force pair counts, ratios and gaps pass; all pre-cutoff supplied-history timestamps checked; no single-step +0.01; holdout/evaluation unopened. No baseline/submission overwritten. Full confirmation is not justified. Pause after experiment 2 as requested. Dense EASE excluded by user.\n']
with Path('results_log.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines))
statefile=Path('agent_state.json'); state=json.loads(statefile.read_text()); state['status']='awaiting_user_review_after_experiment_2'; state['next_action']='User-requested pause; experiment 3 short-history users after review'; state['completed_directions'].append('Repeat-item/negative exposure screen rejected: fails +0.0005 on both days'); state['experiment_2']={'report':'artifacts/v54_repeat_screen_report.json','diagnostics':'artifacts/v54_repeat_diagnostics.json','audit':'artifacts/v54_repeat_screen_audit.json','decision':'reject tested screen','best':report['best']}; state['excluded_directions']=['dense EASE per user instruction']; statefile.write_text(json.dumps(state,indent=2)); print(json.dumps(dict(audit='PASS',spot_checks=len(checks),best=report['best'])))
