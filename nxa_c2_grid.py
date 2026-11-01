"""nxa C2 selection grid (owner: Idea-A improver session). Day-6 SELECTION SUBSET only (selection_day6.npz full_indices,
the evaluate_candidate.py gate cohort; the 10,716 fresh-holdout events are never scored here). Saves every setting and its score.

Expert family (label-free features, nxa_exposure2.py / nxa_c2_features.py at cutoff day 6) and recipe family:
  pred = rank(parent) + beta(h) * zc,  zc = expert - event mean(expert), zc /= std(zc over the scored set),
  beta(h) = beta_short for supplied-history length h <= 4, beta_long otherwise (h known at prediction time, no labels).
Usage: python nxa_c2_grid.py OUT_JSON EXPERT_NAME=npz:col+col+... [more experts]   ('-col' = use the negated column)
"""
import sys, json, itertools
import numpy as np
from config import ARTIFACT_DIR as R
from nx_seq_core import rank
from sklearn.metrics import roc_auc_score

SEL = np.load(R / 'selection_day6.npz'); FI = SEL['full_indices']; W = SEL['weights']


def zc(x):
    c = x - np.repeat(x.reshape(-1, 10).mean(1), 10); return c / c.std()


def score(p): return float(roc_auc_score(SEL['y'], p[FI], sample_weight=W))


if __name__ == '__main__':
    out_path = sys.argv[1]; v = np.load(R / 'v53_causal_propensity_day6.npz'); assert np.array_equal(v['y'][FI], SEL['y']) and np.array_equal(v['pred'][FI], SEL['baseline'])
    H = np.load(R / 'train_hist.npy', mmap_mode='r'); short = (H[v['events']] >= 0).sum(1) <= 4; rv = rank(v['pred']); base = score(v['pred']); grid = []
    for spec in sys.argv[2:]:
        name, rest = spec.split('='); path, cols = rest.split(':'); f = np.load(R / path); assert np.array_equal(f['events'], v['events'])
        z = zc(np.mean([rank(-f[c[1:]]) if c.startswith('-') else rank(f[c]) for c in cols.split('+')], axis=0))
        for bs, bl in itertools.product((.01, .015, .02, .03, .04, .06), (.01, .015, .02)):
            g = score(rv + np.where(short, bs, bl) * z) - base; grid.append(dict(expert=name, columns=cols, beta_short=bs, beta_long=bl, selection_gain=g))
            print(name.ljust(12), f'bs={bs:.3f} bl={bl:.3f} gain={g:+.6f}', flush=True)
    best = max(grid, key=lambda r: r['selection_gain'])
    json.dump(dict(cohort='day-6 selection subset (24,422 events), selection_day6.npz weights', v53=base, grid=grid, best=best), open(R / out_path, 'w'), indent=1)
    print('BEST', json.dumps(best))
