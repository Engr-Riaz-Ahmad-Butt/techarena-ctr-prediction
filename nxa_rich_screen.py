"""nxa idea 2 screen (owner: Idea-A improver session): do the richer exposure features add to nxa_h1?
Head exactly as nxa_h1 (LogisticRegression C=0.01, StandardScaler, evaluation-mix weights), fit on the full day-5 cohort, scored on
the day-6 SELECTION subset only. Base design = nxa_h1 F1 [logit rank(V53), E3z]; each candidate adds columns from
nxa_expo_rich_day{5,6}_features.npz as within-event centred z (zc) and/or raw values. Grid saved to artifacts/nxa_rich_screen.json.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import rank
import nxa_h1 as H

NEW = ('aud32', 'xcon32', 'hcl32', 'pct32', 'rec32', 'nrec')
sel = np.load(R / 'selection_day6.npz'); FI = sel['full_indices']


def design(day, extra_cols, mode):
    f, v, parent, hl, _ = H.day_inputs(day, 'v53'); X = H.design(parent, f, hl, 1); g = np.load(R / f'nxa_expo_rich_day{day}_features.npz'); assert np.array_equal(g['events'], v['events'])
    cols = [X]
    for c in extra_cols:
        if mode in ('zc', 'both') and c != 'nrec': cols.append(H.zc(g[c])[:, None])
        if mode in ('raw', 'both'): cols.append(g[c][:, None])
    return np.column_stack(cols), v, hl


if __name__ == '__main__':
    h1 = np.load(R / 'nxa_h1_day6.npz'); base = float(roc_auc_score(sel['y'], h1['pred'][FI], sample_weight=sel['weights'])); grid = []
    specs = [((), 'zc')] + [((c,), m) for c in NEW for m in (('raw',) if c == 'nrec' else ('zc', 'raw'))] + [(NEW, 'zc'), (NEW, 'both')]
    for cols, mode in specs:
        X5, v5, h5 = design(5, cols, mode); X6, v6, h6 = design(6, cols, mode); pred, m = H.fit_predict(X5, v5['y'], H.weights(h5), X6)
        g = float(roc_auc_score(sel['y'], pred[FI], sample_weight=sel['weights'])) - base; grid.append(dict(columns=list(cols), mode=mode, gain_vs_h1=g, coef=m.coef_.ravel().round(4).tolist()))
        print(('+'.join(cols) or 'F1 (= nxa_h1)').ljust(40), mode.ljust(5), 'gain vs nxa_h1 %+.6f' % g, m.coef_.ravel().round(3).tolist(), flush=True)
    (R / 'nxa_rich_screen.json').write_text(json.dumps(dict(cohort='day-6 selection subset; heads fit on full day 5', nxa_h1_selection=base, grid=grid), indent=1))
