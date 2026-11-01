"""Post-fit checks and honest report for the fixed independent preference screens."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import json, hashlib, importlib.util
import numpy as np
import torch

B = Path(__file__).resolve().parent
H = B / 'artifacts/v54_horizon'
R = H / 'cdx_graph_001'
S = R / 'repaired_upload'
O = R / 'local_run_corrected'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    result = json.loads((O / 'report.json').read_text())
    assert not result['screen_passed'], 'A survivor needs full-cohort confirmation first.'
    protected = json.loads((H / 'protected_before.json').read_text())
    changes = [k for k, v in protected.items()
               if not (B / k).exists() or digest(B / k) != v['sha256']]
    assert not changes, changes
    manifest = json.loads((S / 'manifest.json').read_text())
    assert all(digest(S / k) == v['sha256'] for k, v in manifest.items())
    old = json.loads((R / 'upload/manifest.json').read_text())
    data_unchanged = all(digest(S / k) == v['sha256']
                         for k, v in old.items() if k.endswith('.npz'))
    assert data_unchanged
    v = dict(np.load(S / 'screen.npz'))
    p = np.load(O / 'frozen_predictions.npz')
    assert len(np.unique(v['users'])) == len(v['users']) == 9878
    assert np.array_equal(p['events'], v['events'])
    assert np.array_equal(p['control'], v['baseline'])
    assert np.array_equal(p['candidate'], p['control'] + .05 * (p['graph'] - p['bag']))
    assert all(np.isfinite(p[k]).all() for k in ['control','candidate','graph','bag'])
    assert v['y'].shape == (9878, 10) and np.all(v['y'].sum(1) <= 1)
    a_target = np.array([67,17,7,4], float) / 95
    h_target = np.array([.0126,.1637,.8237])
    w = v['weight']
    am = np.bincount(v['arrival'], weights=w, minlength=4) / w.sum()
    hm = np.bincount(v['history_group'], weights=w, minlength=3) / w.sum()
    assert np.allclose(am, a_target, atol=1e-9)
    assert np.allclose(hm, h_target, atol=1e-9)
    spec = importlib.util.spec_from_file_location('cdx_frozen_graph', S / 'cdx_graph_runner.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.set_num_threads(2)
    initial = dict(np.load(S / 'initial.npz'))
    device = torch.device('cpu')
    adj = mod.graph(S / 'graph3.npz', len(initial['U']) + len(initial['V']), device)
    original = {k: x[:128].copy() for k,x in v.items()}
    altered = {k: x.copy() for k,x in original.items()}
    altered['y'] = 1 - altered['y']  # deliberately invalid labels must have no scoring effect
    invariance = {}
    for arm, enabled, stored in [('bag_control',False,'bag'),('graph_candidate',True,'graph')]:
        model = mod.Preference(initial, enabled)
        model.load_state_dict(torch.load(O / (arm + '.pt'), weights_only=True, map_location='cpu'))
        clean = mod.predict(model, original, adj, device)
        changed = mod.predict(model, altered, adj, device)
        assert np.array_equal(clean, changed)
        assert np.allclose(clean, p[stored][:128], rtol=2e-6, atol=2e-7)
        invariance[arm] = True
    audit = dict(status='passed', protected_files=len(protected), protected_changes=changes,
        manifest_unchanged=True, corrected_arrays_identical_to_original=data_unchanged,
        one_event_per_user=True, candidates_per_event=10, predictions_finite=True,
        label_mutation_invariant_on_128_events=invariance, saved_predictions_reproduced=True,
        arrival_weight_marginals=am.tolist(), history_weight_marginals=hm.tolist(),
        effective_events=float(w.sum()**2/(w@w)), weight_range=[float(w.min()),float(w.max())],
        validation_labels_used_for_metrics_and_screen_gate=True,
        validation_labels_used_to_fit_models_or_select_epochs_or_tune_recipe=False,
        full_confirmation_run=False, holdout_scored=False, evaluation_scored=False,
        submission_created=False, full_V53_rebuild=False)
    (R / 'postfit_audit.json').write_text(json.dumps(audit, indent=2))
    taste = json.loads((H / 'cdx_taste_groups_001/report.json').read_text())
    names = {'overall':'Overall','arrival1':'Arrival 1','arrival2':'Arrival 2',
             'arrival3':'Arrival 3','arrival4':'Arrival 4',
             'history0':'History 0','history1':'History 1–5','history2':'History 6–20'}
    lines = ['# Independent preference screens — 2026-10-29', '',
        'Neither fixed recipe passes screening. V53 remains unchanged; no full-cohort confirmation was run.', '',
        '## Reference and decision rule', '',
        'Frozen 9,878-user first-event horizon screen; one event per user, ten candidates per event. '
        'Arrival and history marginals use the previously frozen IPF evaluation-mix weights. '
        'The reference is the **simplified causal V53-component horizon blend**, not a full V53 rebuild. '
        'Its weighted AUC is 0.759031195. These earlier-cutoff scores cannot be compared directly with the official 0.816623.', '',
        'Screening requires +0.0002 overall AND on each of four arrival days; confirmation would require '
        '+0.0005 on the full permitted cohort. Uncertainty below is a 200-draw paired bootstrap by user '
        'with fixed event weights; it does not include refitting or recipe-selection uncertainty.', '']
    for title, obj, description in [
        ('Latent taste-group residual', taste,
         'Fixed 16 ALS user groups, shrunk group/item response interaction, with all scored-user counts removed. '
         'Historical-only reference transform and fixed 0.02 increment. '
         'The overall gain +0.000195975 is strictly below the gate; arrival 1 also falls short and arrival 4 is negative. '
         'Do not round this into a pass or tune the rejected recipe.'),
        ('Matched graph preference model', result,
         'Own two-hop LightGCN-inspired gated user/item graph propagation against the identical warm bag model. '
         'Both use the same 16-dimensional initialization, data order, true ten-candidate/no-click loss, '
         'regularization and four fixed epochs. Each training day uses a graph from strictly before that day; '
         'validation uses the graph frozen at the three-day cutoff. Same event-time histories in both arms. '
         'Candidate = frozen reference + 0.05 × (graph probability − matched bag probability). '
         'No epoch selection or validation-fitted blend scale. The overall effect is effectively zero; reject.')]:
        lines += ['## ' + title, '', description, '',
            '| Segment | Events | Control AUC | Candidate AUC | Difference | Paired 95% interval |',
            '|---|---:|---:|---:|---:|---|']
        for m in obj['screen']:
            lo, hi = m['paired95']
            lines.append(f"| {names[m['group']]} | {m['events']:,} | {m['control_auc']:.9f} | "
                         f"{m['candidate_auc']:.9f} | {m['gain']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
        lines += ['']
    lines += ['## Scope and checks', '',
        f"Graph standalone weighted AUC: bag {result['standalone_bag_auc']:.9f}, graph "
        f"{result['standalone_graph_auc']:.9f}. Both are weaker than the reference; this is evidence against "
        'this fixed graph diagnostic, not a general disproof of graph recommenders. '
        'The small positive short-history effect is below the gate and does not justify a specialist model.', '',
        f"Corrected graph matched run: {result['seconds']:.1f} seconds locally, two CPU threads "
        '(11.5 minutes, versus 10.7-minute timing-pilot estimate). '
        'An earlier attempt was stopped before validation metrics after training loss exposed an Adagrad '
        'gradient-unit mismatch. The same objective was changed from mean to batch-sum backpropagation in BOTH arms; '
        'a training-only sanity check confirmed learning. Original files were preserved. '
        'No validation metrics informed that correction. See implementation_correction.json.', '',
        'Automatic approval review rejected the proposed Kaggle data upload. No upload or workaround was used; '
        'the completed experiment ran locally. No repository code was copied. '
        'Research inspiration: [LightGCN, He et al. (2020)](https://arxiv.org/abs/2002.02126); '
        'the gated history-bag architecture here is an original adaptation, not a paper reproduction. '
        'Taste grouping was inspired by [Dhillon et al. (2003)]'
        '(https://bigdata.oden.utexas.edu/publication/information-theoretic-co-clustering/), '
        'not an implementation of their co-clustering algorithm.', '',
        f"Post-fit audit: all {len(protected)} protected artifacts unchanged; input manifests unchanged; "
        'finite/aligned predictions; one event per user; at most one positive per event; target weight marginals '
        'matched; mutating labels leaves recomputed predictions identical on 128 events in both arms. '
        'Training/graph timestamps precede their permitted cutoffs. '
        'Labels are used only for the fixed validation metrics and screening gate, not fitting or tuning. '
        'No holdout or evaluation data were scored, and no submission was created.', '',
        'The official target has not been reached or demonstrated. These tests add no accepted model. '
        'Other researchers’ nx/nxa/tst sources and models were treated as read-only; their results are '
        'not promoted or claimed as independently confirmed here.', '']
    report = B / 'results/cdx_independent_preference_report.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    journal = ('\n\n## Graph screen completed — REJECT\n\n'
        f"Corrected matched run {result['seconds']:.3f}s, weighted AUC0.759031195->0.759027097 "
        '(-0.000004099), paired95%[-0.000010819,+0.000003880]. '
        'Arrival gains1–4:-0.000007692/+0.000007754/+0.000012331/-0.000002581. '
        'Standalone bag0.711418694/graph0.711085878. No survivor, no full confirmation or sweep. '
        'Protected29 unchanged; saved predictions reproduced; label-mutation check passes in both arms. '
        'Complete report results/cdx_independent_preference_report.md.\n')
    with (B / 'results/cdx_research_journal.md').open('a', encoding='utf-8') as f:
        f.write(journal)
    with (B / 'results_log.md').open('a', encoding='utf-8') as f:
        f.write('\n\n### Codex independent graph preference screen — 2026-10-29\n'
            + journal.split('## Graph screen completed — REJECT\n\n',1)[1])
    state = dict(date='2026-10-29', namespace='cdx', status='screening_complete_no_survivors',
        target_achieved=False, taste_groups='rejected', graph='rejected',
        full_confirmation_run=False, protected_files_unchanged=len(protected),
        report=str(report), holdout_scored=False, evaluation_scored=False, submission_created=False,
        no_background_job_remaining=True, shared_agent_state_not_overwritten=True)
    (B / 'cdx_agent_state.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(dict(audit=audit, report=str(report)), indent=2))

if __name__ == '__main__':
    main()
