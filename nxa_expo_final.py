"""nxa idea C final: exposure-policy expert + fixed within-event blend recipe (owner: Idea-A improver session).

Features: nxa_exposure2.py --day 5 / 6 / 7 (one code path; impressions with time < cutoff 1737763200 / 1737849600 / 1737936000,
no labels anywhere in the features). Expert (raw) = mean of the average ranks of typ16_all, typ32_all and cos32 over the scored set.
Recipe (chosen on day 6 only among 5 rank-average experts x {global rank blend w in .03-.1, within-event centred rank w .03-.07,
within-event centred z beta .01-.05}; applied unchanged to day 5 and to evaluation):
    zc   = raw - mean(raw over the event's 10 candidates);  zc /= std(zc over the scored set)   (label-free normalisation)
    pred = rank(parent) + 0.015 * zc        parent = V53 'pred' (days 5/6) or V39 (days 5/6 'pred', evaluation v39_final_pred.npy)
Outputs: artifacts/nxa_expo_day{5,6}.npz (y, events, raw, pred [= recipe on V53], pred_v39 [= recipe on V39], baseline [= V53]),
artifacts/nxa_expo_eval.npy (raw expert, evaluation order), artifacts/nxa_expo_eval_pred_v39.npy (recipe applied to V39 eval).
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc, rank

COLS = ('typ16_all', 'typ32_all', 'cos32'); BETA = .015


def expert(f): return np.mean([rank(f[c]) for c in COLS], axis=0)


def recipe(parent, raw):
    zc = raw - np.repeat(raw.reshape(-1, 10).mean(1), 10); zc = zc / zc.std(); return rank(parent) + BETA * zc


if __name__ == '__main__':
    rep = {}
    for day in (6, 5):
        f = np.load(R / f'nxa_exposure2_day{day}_features.npz'); v = np.load(R / f'v53_causal_propensity_day{day}.npz'); p39 = np.load(R / f'v39_regularized_als32_day{day}.npz')
        assert np.array_equal(f['events'], v['events']) and np.array_equal(f['y'], v['y']) and np.array_equal(p39['events'], v['events'])
        y, e = v['y'], v['events']; raw = expert(f); pred = recipe(v['pred'], raw); pred39 = recipe(p39['pred'], raw)
        np.savez_compressed(R / f'nxa_expo_day{day}.npz', y=y, events=e, raw=raw, pred=pred, pred_v39=pred39, baseline=v['pred'])
        H = np.load(R / 'train_hist.npy', mmap_mode='r'); hl = (H[e] >= 0).sum(1)
        r = dict(v53=weighted_auc(e, y, v['pred']), candidate=weighted_auc(e, y, pred), v39=weighted_auc(e, y, p39['pred']), candidate_on_v39=weighted_auc(e, y, pred39), raw_expert=weighted_auc(e, y, raw),
                 ordinary_v53=float(roc_auc_score(y, v['pred'])), ordinary_candidate=float(roc_auc_score(y, pred)))
        r['gain_vs_v53'] = r['candidate'] - r['v53']; r['gain_vs_v39'] = r['candidate_on_v39'] - r['v39']
        for nm, m in (('bin0', hl == 0), ('bin1-4', (hl >= 1) & (hl <= 4)), ('bin5-9', (hl >= 5) & (hl <= 9)), ('bin10-19', (hl >= 10) & (hl <= 19)), ('bin20', hl == 20)):
            r[nm] = dict(events=int(m.sum() // 10), v53=float(roc_auc_score(y[m], v['pred'][m])), candidate=float(roc_auc_score(y[m], pred[m])))
        rep[f'day{day}'] = r; print(day, json.dumps({k: (round(x, 6) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
    f = np.load(R / 'nxa_exposure2_day7_features.npz'); raw = expert(f); base = np.load(R / 'v39_final_pred.npy'); assert len(raw) == len(base) == 529390
    ev = np.load(R / 'evaluation.npy', mmap_mode='r'); assert np.array_equal(np.asarray(ev[:, 6]).reshape(-1, 10), np.repeat(np.asarray(ev[::10, 6])[:, None], 10, 1))
    pe = recipe(base, raw); np.save(R / 'nxa_expo_eval.npy', raw); np.save(R / 'nxa_expo_eval_pred_v39.npy', pe)
    rep['eval'] = dict(rows=int(len(raw)), raw_mean=float(raw.mean()), pred_v39_min=float(pe.min()), pred_v39_max=float(pe.max()), finite=bool(np.isfinite(pe).all()),
                       spearman_vs_v39=float(np.corrcoef(rank(pe), rank(base))[0, 1]))
    rep['recipe'] = dict(columns=COLS, beta=BETA, form='rank(parent) + beta * within-event-centred raw / std', selection='day 6 only; day 5 unchanged')
    print(json.dumps(rep['eval'])); (R / 'nxa_expo_report.json').write_text(json.dumps(rep, indent=1))
