"""Finalize two fixed preference diagnostics; replay predictions without new fitting."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
import cdx_collective as c
import cdx_multi_interest as multi

B,H=c.B,c.H
J=H/'cdx_collective_001';M=H/'cdx_multi_interest_001'

def append_once(path,marker,body):
    if marker not in path.read_text(encoding='utf-8'):
        with path.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n\n'+body+'\n')

def main():
    joint=json.loads((J/'report.json').read_text());mixture=json.loads((M/'report.json').read_text())
    assert not joint['screen_passed'] and not mixture['screen_passed']
    assert not joint['full_confirmation'] and not mixture['full_confirmation']
    for root in [J,M]:
        plan=json.loads((root/'frozen.json').read_text())
        assert all(c.metrics.sha(B/p)==v for p,v in plan['hashes'].items())
    reproduction=json.loads((J/'control_reproduction.json').read_text());assert reproduction['exact']
    ns=c.definitions();rows=np.load(H/'score_rows.npy',mmap_mode='r');co=np.load(H/'cohort.npz')
    hist=np.asarray(ns['hist'][rows[::10,6]])
    times=np.load(c.A/'train_times.npy',mmap_mode='r')[rows[::10,6]]
    assert np.all((times<rows[::10,2,None])|(hist<0))
    assert len(np.unique(rows[::10,0]))==49420
    jp=np.load(J/'frozen_predictions.npz');mp=np.load(M/'frozen_predictions.npz')
    assert np.array_equal(jp['control'],mp['control'])
    assert np.array_equal(jp['events'],mp['events']) and np.array_equal(jp['events'],co['events'])
    probe=np.asarray(rows[:1280]).copy();changed=probe.copy();changed[:,5]=np.arange(1280)%2
    for name,stored in [('control','raw_control'),('collective','raw_candidate')]:
        m=np.load(J/f'{name}_model.npz');ps=[m[k] for k in ['U','V','H','bu','bi']]
        def score(rr):
            return ns['predict_events'](rr[::10,0].copy(),rr[:,1].reshape(-1,10).copy(),
                np.asarray(ns['hist'][rr[::10,6]]).copy(),*ps)
        original=score(probe);altered=score(changed)
        assert np.array_equal(original,altered)
        assert np.array_equal(original,jp[stored][:128])
    assert np.allclose(mp['raw'].sum(1),jp['raw_control'].sum(1),atol=1e-12)
    assert np.array_equal(mp['raw'][mp['split']==0],jp['raw_control'][mp['split']==0])
    assert all(np.isfinite(obj[k]).all() for obj in [jp,mp] for k in ['control','candidate'])
    w=co['sample_event_weight'];sample=co['sample']
    assert np.allclose(np.bincount(co['arrival'][sample],weights=w)/w.sum(),np.array([67,17,7,4])/95,atol=1e-9)
    assert np.allclose(np.bincount(co['history_group'][sample],weights=w)/w.sum(),[.0126,.1637,.8237],atol=1e-9)
    protected=json.loads((H/'protected_before.json').read_text());assert all(c.metrics.sha(B/p)==v['sha256'] for p,v in protected.items())
    audit=dict(status='passed',control_model_bit_exact=True,original_control_max_parameter_differences=reproduction['max_differences'],
        shared_control_predictions_identical=True,saved_main_predictions_replayed_on_128_events=True,
        label_mutation_invariance=True,strict_history_timestamps=True,one_event_per_user=True,
        mixture_event_mass_preserved=True,mixture_unsplit_rows_unchanged=True,
        mixture_history_order_invariant=mixture['history_order_invariant'],
        mixture_candidate_order_equivariant=mixture['candidate_order_equivariant'],
        finite_predictions=True,weight_marginals_match=True,protected_files_unchanged=len(protected),
        source_and_input_manifests_unchanged=True,full_cohort_outcome_confirmation=False,
        holdout_scored=False,evaluation_scored=False,submission_created=False)
    (J/'postfit_audit.json').write_text(json.dumps(audit,indent=2))
    names=dict(overall='Overall',arrival1='Arrival 1',arrival2='Arrival 2',arrival3='Arrival 3',arrival4='Arrival 4',
        arrival3plus='Arrival 3+',history0='History 0',history1='History 1–5',history2='History 6–20')
    lines=['# Preference research — 30 October 2026','',
        '**The 0.82 target has not been reached. Both new fixed recipes fail screening and are rejected.**','',
        'The working target is the same local adjusted metric as the reported 0.818023 candidate, a gap of about '
        '0.001977. The last user-confirmed official score in the project ledger remains 0.816623. '
        'This round uses the earlier-cutoff matched horizon to screen new signals. Its approximately 0.759 AUC '
        'is a different cohort, weighting and training-window length; gains cannot be added to 0.818023. '
        'No full V53/nxa_h1 or official-score improvement is claimed.','',
        '## Shared exposure/click item representation','',
        'New question: can supervision from the logging policy improve the underlying click representation '
        'when it updates shared item factors, rather than acting as a separate expert? '
        'The general source principle is parameter sharing across related relations in '
        '[Singh & Gordon (2008), Collective Matrix Factorization]'
        '(https://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf). '
        'The particular exposure/click objective here is an original adaptation, not that paper’s implementation '
        'or evidence of competition gains.','',
        'Both arms use the exact existing warm16 kernels: 16 dimensions, ten epochs, seed 151, learning rate '
        '0.045, regularization 0.002, warm-positive auxiliary weight 1 and popularity correction. '
        'Prefix-2 initial statistics precede the prefix-3 scoring cutoff. Fitting uses all 295,114 earlier events '
        'and 557,429 genuine earliest-history positive pairs. Each arm also learns a private exposure-user head '
        'from the 2,951,140 observed impression triples and independent uniform-catalog statistical noise '
        '(weight 0.1, seed 8531). Only the treatment enables its data gradients into the shared click-item V; '
        'no additional shared-V regularization is added. No missing warm-up impressions are imputed.','',
        'The control **reproduces every archived warm16 parameter bit-for-bit** before validation metrics are '
        'calculated. Thus the original warm-history and event-choice architecture, optimizer and calendar '
        'are retained. Existing project functions are reused as definitions to avoid module side effects; '
        'new auxiliary code was written here and passed independent finite-difference, frozen-V and zero-weight checks.','',
        f"Runtime: {joint['seconds']:.1f} seconds (5.2 minutes), below the conservative 11.6-minute pilot estimate.", '',
        '## Two-interest representation diagnostic','',
        'A second new question: does a single history mean conceal two coherent interests? '
        '[Li et al. (2019), MIND](https://arxiv.org/abs/1904.08030) motivates multiple user representations. '
        'This test is an original **static diagnostic**, not MIND training, capsule routing, learned attention '
        'or a sequence model. Existing warm16 factors remain frozen.','',
        'Split the at-most-20 history H vectors into two deterministic Euclidean clusters: sorted item IDs, '
        'farthest-pair initialization, at most six Lloyd steps. Component weights are history cluster shares. '
        'Their weighted mean exactly conserves the original preference profile. Average their conditional '
        'ten-item softmax probabilities, keeping the original event-click probability fixed. '
        'Histories shorter than two or with degenerate identical vectors are unchanged. '
        'No parameters, temperature, gates or strengths are fitted. History order is irrelevant.','',
        f"It splits {mixture['split_events']:,} of {mixture['total_events']:,} permitted horizon events. "
        f"Runtime {mixture['seconds']:.1f} seconds; the 16.5-second pilot estimate omitted an extra "
        'compiler specialization used by the permutation check.','',
        '## Frozen validation and decision','',
        'Same frozen 9,878-user sample, one first-post-cutoff event per user, ten candidates. '
        'Arrival counts: 6,919 / 1,794 / 856 / 309. IPF weights match arrival 67:17:7:4 normalized over '
        'four supported days and history 1.26% / 16.37% / 82.37%. Evaluation days 5–7 remain unsupported. '
        'Weights and cohorts are unchanged. Supplied history clicks after the cutoff remain usable only '
        'when strictly earlier than their event.','',
        'The unchanged simplified bridge is 95% warm-expert rank plus 5% frozen-ALS rank. '
        'Ranks are calculated jointly over the full permitted horizon without inspecting its labels; '
        'only the 20% sample’s outcomes are assessed. This is not a full V53/nxa_h1 replay. '
        'The tiny control AUC difference from the older 0.759031195 reflects the original compiled scorer '
        'versus the prior NumPy scorer; both new tests share the same control, 0.759031201.','',
        'Require +0.0002 overall and on each of four arrival days, followed only for survivors by '
        '+0.0005 full-cohort confirmation. Neither survives; no full-cohort outcome check or parameter search '
        'was run. Confidence intervals use 200 paired user-bootstrap draws, conditional on fixed models '
        'and weights, and omit repeated-validation and model-selection uncertainty.','']
    for title,obj in [('Shared representation — REJECT',joint),('Two-interest mixture — REJECT',mixture)]:
        lines += ['### '+title,'',
            '| Segment | Events | Control AUC | Candidate AUC | Difference | Paired 95% interval |',
            '|---|---:|---:|---:|---:|---|']
        for z in obj['screen']:
            lo,hi=z['paired95'];lines.append(f"| {names[z['group']]} | {z['events']:,} | {z['control_auc']:.9f} | "
                f"{z['candidate_auc']:.9f} | {z['gain']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
        lines += ['']
    lines += ['## Findings and audit','',
        'Sharing exposure gradients reduces AUC on every arrival day, overall -0.000279634 '
        '(paired 95% [-0.000410215, -0.000149736]). Lower auxiliary training loss did not improve '
        'click generalization. This rejects the fixed shared-representation recipe, without proving '
        'that all forms of multitask recommendation are ineffective.','',
        'The two-interest mixture loses -0.001595433 overall (paired 95% [-0.002749275, -0.000406804]). '
        'History 1–5 and arrival 4 have favorable point estimates, but their intervals include losses; '
        'they do not justify a subgroup specialist or a change to the gate. This rejects the static '
        'diagnostic and is not a full MIND ablation.','',
        'Post-fit audit passes: exact control parameter reproduction; saved probabilities replayed '
        'on 128 events; changing labels leaves predictions unchanged; strict history timestamps; '
        'one event per user; identical controls; mixture event mass and degenerate cases preserved; '
        'history/candidate order checks; finite predictions and correct weight marginals; '
        'all 29 protected files and source/input manifests unchanged. No holdout or evaluation '
        'was scored and no submission was created.','',
        'Reports, frozen plans, models and predictions: `artifacts/v54_horizon/cdx_collective_001/` '
        'and `cdx_multi_interest_001/`. The replay audit is `cdx_collective_001/postfit_audit.json`. '
        'Results are logged. The current candidate is retained; 0.82 remains unverified. '
        'The actual competition/Instructions URL was requested to check official details before choosing '
        'another new signal. No target attainment or general statistical ceiling is claimed.','']
    report=B/'results/cdx_oct30_preference_report.md';report.write_text('\n'.join(lines),encoding='utf-8')
    body=('Matched collective exposure/click item factors: original warm16 parameters reproduced bit-exact in control. '
        'Same16dims/10epochs/seed151/warmaux1/lr.045/reg.002; exposure target actual impressions vs uniform noise, '
        'weight.1/seed8531, only treatment shares V data gradients. Runtime311.719s. '
        'Frozen20%horizon weighted0.759031201->0.758751566 (-0.000279634;paired95%[-0.000410215,-0.000149736]); '
        'all4arrivaldays lose. Separate no-training two-interest diagnostic using frozen H-centroids, '
        'conditional-choice mixture with unchanged event mass:0.757435768 (-0.001595433;paired95%[-0.002749275,-0.000406804]),44.594s. '
        'Both REJECT; no full confirmation or tuning. Same frozen95%warm-rank+5%ALS bridge, not fullV53/nxa_h1. '
        'Sources Singh/Gordon2008 and Li etal2019MIND motivation; own adaptations, no external code copied. '
        'Replay/label invariance/timestamps/order/mass/weight audits pass;29protected unchanged. '
        'No holdout/evaluation/submission.0.82 target unmet; reported best local0.818023 unchanged. '
        'Report:results/cdx_oct30_preference_report.md. Actual competition URL requested for further research.')
    append_once(B/'results_log.md','### Codex 30 October — collective and multi-interest screens',body)
    append_once(B/'results/cdx_research_journal.md','## 2026-10-30 — both new screens rejected',body)
    state=dict(date='2026-10-30',namespace='cdx',status='oct30_screens_complete_no_survivors',target_auc=.82,
        target_working_metric='Same reported local adjusted metric as0.818023; official score separate',
        target_achieved=False,last_reported_local_candidate_auc=.818023,
        collective='rejected_all_arrival_days_worse',multi_interest='rejected_overall_worse',
        full_confirmation_run=False,protected_files_unchanged=29,report=str(report),
        holdout_scored=False,evaluation_scored=False,submission_created=False,no_background_job_remaining=True,
        pending_optional_input='Actual TechArena competition/Instructions URL',shared_agent_state_not_overwritten=True)
    (B/'cdx_agent_state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(dict(audit=audit,report=str(report)),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
