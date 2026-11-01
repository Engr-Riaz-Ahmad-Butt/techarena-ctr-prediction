"""nxa H1 final outputs (owner: Idea-A improver session). Applies the recipe frozen in artifacts/nxa_h1_frozen.json (feature set F1
of nxa_h1.py: [logit(rank(parent)), E3z], LogisticRegression C=0.01 with evaluation-mix weights) without any change:
  day 6       <- fit on the full day-5 cohort;   day 5 <- 2-fold user cross-fit (seed 9901);   evaluation <- fit on the full day-6
  cohort with the V39 parent, applied to the evaluation rows (V39 = v39_final_pred.npy, features nxa_exposure2_day7_features.npz).
Outputs: artifacts/nxa_h1_day{5,6}.npz (y, events, pred = head on V53, pred_v39 = head on V39, raw = E3 expert, baseline = V53),
artifacts/nxa_h1_eval_pred_v39.npy (head on V39, rank-transformed to [0,1]) and artifacts/nxa_h1_report.json.
"""
import json, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import rank, weighted_auc
import nxa_h1 as H

FS = 1
frozen = json.loads((R / 'nxa_h1_frozen.json').read_text()); assert frozen['featureset'] == FS
assert hashlib.sha256(open('nxa_h1.py', 'rb').read()).hexdigest() == frozen['source_sha256'], 'nxa_h1.py changed after freezing'
assert hashlib.sha256(open('nxa_exposure2.py', 'rb').read()).hexdigest() == frozen['feature_source_sha256']
rep = dict(frozen=frozen)
store = {}
for parent_name in ('v53', 'v39'):
    p6, v6, par6, m6 = H.day6_pred(FS, parent_name); p5, v5, par5 = H.day5_pred(FS, parent_name); store[parent_name] = (p5, p6)
    for day, pred, v, par in ((6, p6, v6, par6), (5, p5, v5, par5)):
        r = dict(parent=weighted_auc(v['events'], v['y'], par), head=weighted_auc(v['events'], v['y'], pred), parent_plain=float(roc_auc_score(v['y'], par)), head_plain=float(roc_auc_score(v['y'], pred)))
        r['gain'] = r['head'] - r['parent']
        if day == 6:
            sel = np.load(R / 'selection_day6.npz'); FI = sel['full_indices']; r['selection_gain'] = float(roc_auc_score(sel['y'], pred[FI], sample_weight=sel['weights']) - roc_auc_score(sel['y'], par[FI], sample_weight=sel['weights']))
        hl = (np.load(R / 'train_hist.npy', mmap_mode='r')[v['events']] >= 0).sum(1)
        for nm, m in (('bin0', hl == 0), ('bin1-4', (hl >= 1) & (hl <= 4)), ('bin5-9', (hl >= 5) & (hl <= 9)), ('bin10-19', (hl >= 10) & (hl <= 19)), ('bin20', hl == 20)):
            r[nm] = dict(parent=float(roc_auc_score(v['y'][m], par[m])), head=float(roc_auc_score(v['y'][m], pred[m])))
        rep[f'{parent_name}_day{day}'] = r; print(parent_name, day, json.dumps({k: (round(x, 6) if isinstance(x, float) else x) for k, x in r.items() if not k.startswith('bin')}), flush=True)
    if parent_name == 'v53': rep['coef_day6_head_v53'] = m6.coef_.ravel().tolist()
for day in (5, 6):
    v = np.load(R / f'v53_causal_propensity_day{day}.npz'); f = np.load(R / f'nxa_exposure2_day{day}_features.npz')
    raw = np.mean([rank(f[c]) for c in H.COLS], axis=0); i = 0 if day == 5 else 1
    np.savez_compressed(R / f'nxa_h1_day{day}.npz', y=v['y'], events=v['events'], pred=store['v53'][i], pred_v39=store['v39'][i], raw=raw, baseline=v['pred'])
# evaluation: fit on full day-6 cohort with the V39 parent, predict evaluation rows
f6, v6, par6, h6, _ = H.day_inputs(6, 'v39'); fe = np.load(R / 'nxa_exposure2_day7_features.npz'); pe = np.load(R / 'v39_final_pred.npy')
he = np.repeat((np.asarray(np.load(R / 'evaluation_hist.npy', mmap_mode='r')) >= 0).sum(1), 10); assert len(he) == len(pe) == len(fe['typ16_all']) == 529390
pred_e, me = H.fit_predict(H.design(par6, f6, h6, FS), v6['y'], H.weights(h6), H.design(pe, fe, he, FS)); out = rank(pred_e)
assert np.isfinite(out).all() and out.min() > 0 and out.max() < 1; np.save(R / 'nxa_h1_eval_pred_v39.npy', out)
rep['eval'] = dict(rows=int(len(out)), coef=me.coef_.ravel().tolist(), spearman_vs_v39=float(np.corrcoef(out, rank(pe))[0, 1]), fit='full day-6 cohort, V39 parent')
(R / 'nxa_h1_report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep['eval']))
