"""Replay and document the two fixed exposure-policy tests without further fitting."""
from pathlib import Path
import json,pickle
import numpy as np
from threadpoolctl import threadpool_limits
import cdx_policy_hpf as common
import cdx_policy_contrast as contrast

B,H=common.B,common.H
P=H/'cdx_policy_hpf_001'
C=H/'cdx_policy_contrast_001'

def append_once(path, marker, body):
    text=path.read_text(encoding='utf-8')
    if marker not in text:
        with path.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+'\n\n'+body+'\n')

def main():
    hpf=json.loads((P/'report.json').read_text())
    nce=json.loads((C/'report.json').read_text())
    assert not hpf['screen_passed'] and not nce['screen_passed']
    for root in [P,C]:
        plan=json.loads((root/'frozen.json').read_text())
        assert all(common.sha(B/k)==v for k,v in plan['hashes'].items())
    protected=json.loads((H/'protected_before.json').read_text())
    assert all(common.sha(B/k)==v['sha256'] for k,v in protected.items())
    ps=np.load(P/'screen_predictions.npz')
    cs=np.load(C/'screen_predictions.npz')
    co=np.load(H/'cohort.npz');sample=co['sample']
    assert np.array_equal(ps['events'],cs['events'])
    assert np.array_equal(ps['control'],cs['control'])
    assert np.array_equal(cs['events'],co['events'][sample])
    assert all(cs[k].shape==(9878,10) and np.isfinite(cs[k]).all()
               for k in ['baseline','control','candidate','signal'])
    hp=pickle.loads((P/'prefix3_models.pkl').read_bytes())
    nc=pickle.loads((C/'model.pkl').read_bytes())
    rows=np.asarray(np.load(H/'score_rows.npy',mmap_mode='r').reshape(-1,10,7)[sample][:128]).copy()
    assert len(np.unique(rows[:,0,0]))==128 and np.all(rows[:,:,5]==-1)
    def scores(rr):
        u=rr[:,0,0];i=rr[:,:,1]
        svd,poisson=common.score_models(hp['svd'],hp['hpf'],hp['cdfs'],u,i)
        policy=common.centered(common.cdf(nc['reference'],contrast.raw(nc['state'],u,i)))
        return svd,poisson,policy
    original=scores(rows)
    changed=rows.copy();changed[:,:,5]=np.arange(1280).reshape(128,10)%2
    altered=scores(changed)
    assert all(np.array_equal(x,y) for x,y in zip(original,altered))
    permutation=np.array([8,0,3,9,1,7,2,5,4,6]);undo=np.argsort(permutation)
    reordered=scores(rows[:,permutation,:])
    assert all(np.allclose(x,z[:,undo],atol=1e-10) for x,z in zip(original,reordered))
    assert np.allclose(original[0],ps['svd_signal'][:128],atol=1e-10)
    assert np.allclose(original[1],ps['hpf_signal'][:128],atol=1e-10)
    assert np.allclose(original[2],cs['signal'][:128],atol=1e-10)
    assert np.array_equal(ps['candidate'],ps['baseline']+.015*ps['hpf_signal'])
    assert np.array_equal(cs['candidate'],cs['baseline']+.015*cs['signal'])
    assert all(d['max_source_timestamp']<d['cutoff'] for obj in [hpf,nce] for d in obj['fit_audits'])
    for logs in hpf['training_logs'].values():
        assert np.all(np.diff([z['elbo'] for z in logs])>=-1e-5)
    w=co['sample_event_weight']
    assert np.allclose(np.bincount(co['arrival'][sample],weights=w)/w.sum(),np.array([67,17,7,4])/95,atol=1e-9)
    assert np.allclose(np.bincount(co['history_group'][sample],weights=w)/w.sum(),[.0126,.1637,.8237],atol=1e-9)
    audit=dict(status='passed',protected_files_unchanged=len(protected),
        source_and_input_manifests_unchanged=True,matched_SVD_control_identical=True,
        predictions_replayed_on_128_events=True,label_mutation_invariant=True,
        candidate_order_equivariant=True,finite_aligned_predictions=True,
        cutoff_checks=True,poisson_elbo_monotone=True,contrastive_finite_difference_checks=True,
        IPF_marginals_match=True,effective_events=float(w.sum()**2/(w@w)),
        full_confirmation_run=False,holdout_scored=False,evaluation_scored=False,submission_created=False)
    (C/'postfit_audit.json').write_text(json.dumps(audit,indent=2))
    names=dict(overall='Overall',arrival1='Arrival 1',arrival2='Arrival 2',arrival3='Arrival 3',
        arrival4='Arrival 4',arrival3plus='Arrival 3+',history0='History 0',history1='History 1–5',history2='History 6–20')
    lines=['# Further exposure-policy tests — 2026-10-29','',
        '**Neither fixed recipe passes the arrival-day screening rule. No model is promoted.**', '',
        '## What changed', '',
        'The current reported 0.818023 local candidate uses a spectral exposure-policy signal. Its tree/richer-feature '
        'follow-ups were already rejected by the parallel session. This round tested two different ways of learning '
        'the exposure policy itself, using only actual shown-item counts/triples from before the cutoff. '
        'No click outcomes entered either fit. Existing nx/nxa/tst work was read only.', '',
        '1. A 32-factor hierarchical Gamma-Poisson model with 80 preset variational iterations, '
        'shape priors 0.3 and hierarchical means 1. The feature is personal exposure lift, with population '
        'and activity mass removed. Source: [Gopalan, Hofman & Blei (2015)]'
        '(https://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf). '
        'The paper supports the count-model approach; it provides no evidence of gains on this competition.', '',
        '2. A 32-factor signed model that jointly learns user/item logging factors for three preset passes. '
        'It distinguishes observed shown items from independent uniform-catalog noise, with a fixed marginal exposure offset '
        'and learned user normalizer. The scored feature is the personal factor dot product. '
        'Parameters: seed 8231, Adagrad learning rate 0.03/initial accumulator 0.1, anchor 0.01 to cutoff-matched '
        'spectral vectors. Source principle: [Gutmann & Hyvärinen (2010)]'
        '(https://proceedings.mlr.press/v9/gutmann10a.html). '
        'This applies contrastive estimation to exposure density. The earlier closed NCE experiments predicted clicks; '
        'the previous cheap policy proxy adjusted users in a fixed item space. Here the user and item logging factors both learn. '
        'Noise examples are statistical density-class examples, not claims of unclicked ads or missing warm-up exposures.', '',
        'Both implementations were written here from the mathematical ideas; no repository code or external data was used. '
        'No additional calibration head, blend-strength search, seed search, or short-history specialist was fitted.', '',
        '## Matched validation', '',
        'The frozen 9,878-user sample contains one first-post-cutoff event per user and ten aligned candidates. '
        'Arrival counts are 6,919 / 1,794 / 856 / 309. Models freeze after the first three training days; '
        'the same model scores all four arrival days. No holdout events are in this cohort. '
        'Joint IPF event weights match arrival 67:17:7:4 normalized over four supported days and history '
        '1.26% / 16.37% / 82.37%. Evaluation arrival days 5–7 remain unsupported.', '',
        'The original reference is the frozen simplified V53-component horizon blend, AUC **0.759031195**. '
        'It is not a full V53 rebuild or nxa_h1 replay. The SVD exposure control uses the fixed '
        'typ16/typ32/cos32 recipe with IDF-log normalized counts, 64 spectral components and seed 4403. '
        'Historical CDFs come from 7,047 first-user day-2 events, scored by prefix-2 models. '
        'The current validation file does not supply percentiles or fitted normalization.', '',
        'Control = original reference + 0.015 × standardized-within-event SVD exposure signal. '
        'Candidate = the same original reference + 0.015 × the corresponding new exposure signal. '
        'The score bridge, cutoff, cohort, weights and historical normalization block are fixed. '
        'This tests replacing the exposure model, without fitting a new head. '
        'The same SVD control scores are used in both tests.', '',
        'Screening requires +0.0002 overall **and each of four arrival days**, against both the SVD control '
        'and original reference. Full-cohort +0.0005 confirmation is allowed only for survivors. '
        'Both recipes failed screening, so confirmation was not run. '
        'Intervals are 200-draw paired user bootstraps with fixed models and weights; '
        'they omit recipe-selection, factor-refitting and repeated-validation uncertainty.', '']
    for title,obj in [('Gamma-Poisson count model — reject',hpf),('Learned contrastive policy — reject under arrival-day gate',nce)]:
        lines += ['## '+title,'',
            '| Segment | Events | SVD control AUC | New model AUC | Difference | Paired 95% interval |',
            '|---|---:|---:|---:|---:|---|']
        for m in obj['screen']:
            lo,hi=m['paired95']
            lines.append(f"| {names[m['group']]} | {m['events']:,} | {m['control_auc']:.9f} | {m['candidate_auc']:.9f} | {m['gain']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
        lines += ['',f"Runtime {obj['seconds']:.1f} seconds locally; final fixed checkpoint only.",'']
    lines += ['## Interpretation', '',
        'The Gamma-Poisson model gains +0.000261 over the original reference but loses -0.000245 against '
        'the stronger SVD exposure control. It supplies no accepted improvement.', '',
        'The learned policy reaches **0.759759776** versus **0.759536768** for SVD, a conditional '
        '**+0.000223008** overall (paired 95% [+0.000025950, +0.000410513]). '
        'It gains +0.000728581 over the original reference, which includes the existing exposure signal’s contribution. '
        'This is a small positive observation, not an accepted model: arrival 1 gains +0.000317588, '
        'arrival 2 +0.000002686, arrival 3 +0.000079195, and arrival 4 -0.000071857. '
        'Arrivals 3+ combined gain only +0.000004984. History 1–5 also loses -0.000211430 versus SVD. '
        'No gate is relaxed and the failed recipe is not tuned or expanded.', '',
        'Later-arrival estimates are noisy, especially the 309-event fourth day. This does not prove that '
        'the method is intrinsically worse later or establish a general performance ceiling. It does show '
        'that this test has not demonstrated the required gains across the agreed horizon.', '',
        'The earlier-cutoff horizon AUC cannot be added to or compared numerically with the reported '
        '0.818023 local result. The full V53/nxa_h1 chain was not reproduced here. '
        'The last confirmed official score in the ledger is still 0.816623; no official improvement or target attainment is claimed.', '',
        '## Audit and artifacts', '',
        'Saved models reproduce the new signal arrays on 128 events. Changing candidate labels leaves all '
        'signals identical; permuting candidate order permutes scores correspondingly. All predictions '
        'are finite/aligned; both tests share identical controls. Fitting uses user/item/time only, '
        'with maximum timestamps strictly earlier than prefix-2/prefix-3 cutoffs. '
        'Variational count allocations conserve observed counts, its ELBO rises at every logged checkpoint, '
        'and contrastive updates pass finite-difference checks including a noise/data item collision. '
        'Input/source manifests and all 29 protected artifacts are unchanged. '
        'No holdout or evaluation data was scored, no submission was created, and no upload was attempted.', '',
        'Full numerical reports, frozen plans, predictions and checkpoints are in '
        '`artifacts/v54_horizon/cdx_policy_hpf_001/` and `cdx_policy_contrast_001/`. '
        'Post-fit replay is recorded in the latter’s `postfit_audit.json`.', '']
    path=B/'results/cdx_policy_followup_report.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    body=('Learned logging-policy factors, genuine shown triples vs independent uniform noise; no click-label fitting. '
        '32dims,3fixed epochs,seed8231,Adagrad0.03/acc0.1,anchor0.01; same historical-CDF0.015 bridge/SVD control as HPF. '
        'Runtime38.672s. SVD0.759536768->0.759759776 (+0.000223008;paired95%[+0.000025950,+0.000410513]), '
        'original baseline0.759031195. Arrival gains1–4:+0.000317588/+0.000002686/+0.000079195/-0.000071857; '
        '3+only+.000004984; history1–5-.000211430. REJECT under all-arrival gate; no full confirmation, tuning or promotion. '
        'Protected29 unchanged; manifests/replay/label and order invariance pass. No holdout/evaluation/submission. '
        'Not a fullV53/nxa_h1 or official-score gain. Report:results/cdx_policy_followup_report.md.')
    append_once(B/'results_log.md','### Codex learned logging-policy screen — 2026-10-29',body)
    append_once(B/'results/cdx_research_journal.md','## Explicit policy screen completed — no survivor',body)
    state=dict(date='2026-10-29',namespace='cdx',status='policy_followups_complete_no_survivors',target_achieved=False,
        hpf='rejected',contrastive_policy='overall_positive_but_fails_arrival_day_gate',
        last_screen_control_auc=.759536767641717,last_screen_candidate_auc=.7597597758390581,
        last_screen_gain=.00022300819734111244,full_confirmation_run=False,
        protected_files_unchanged=29,report=str(path),holdout_scored=False,evaluation_scored=False,
        submission_created=False,no_background_job_remaining=True,shared_agent_state_not_overwritten=True)
    (B/'cdx_agent_state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(dict(audit=audit,report=str(path)),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
