"""nxa: day-wise evaluation of fixed exposure-expert recipes (rank averages) blended into V53 and V39 (owner: Idea-A improver)."""
import sys, json
import numpy as np
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc, rank
RECIPES = {'E1_typ32': ['typ32_all'], 'E2_typ32_cos32': ['typ32_all', 'cos32'], 'E3_typ16_typ32_cos32': ['typ16_all', 'typ32_all', 'cos32'],
           'E4_typ16_typ32': ['typ16_all', 'typ32_all'], 'E5_typ32_raw32_cos32': ['typ32_all', 'raw32', 'cos32']}
def expert(f, cols): return np.mean([rank(f[c]) for c in cols], axis=0)
if __name__ == '__main__':
    day = int(sys.argv[1]); f = np.load(R / f'nxa_exposure2_day{day}_features.npz'); v = np.load(R / f'v53_causal_propensity_day{day}.npz'); p39 = np.load(R / f'v39_regularized_als32_day{day}.npz')
    assert np.array_equal(f['events'], v['events']) and np.array_equal(p39['events'], v['events']); y, e = v['y'], v['events']
    b53 = weighted_auc(e, y, v['pred']); b39 = weighted_auc(e, y, p39['pred']); H = np.load(R / 'train_hist.npy', mmap_mode='r'); hl = (H[e] >= 0).sum(1); s = hl <= 4
    out = dict(day=day, v53=b53, v39=b39)
    for name, cols in RECIPES.items():
        x = expert(f, cols); r = dict(raw_adjusted=weighted_auc(e, y, x))
        for w in (.03, .05, .07, .1):
            p = (1 - w) * rank(v['pred']) + w * rank(x); r[f'v53_{w}'] = weighted_auc(e, y, p) - b53; r[f'v53_{w}_short_auc'] = float(roc_auc_score(y[s], p[s]))
            r[f'v39_{w}'] = weighted_auc(e, y, (1 - w) * rank(p39['pred']) + w * rank(x)) - b39
        out[name] = r; print(name.ljust(22), ' '.join(f'{k}={vv:+.6f}' if 'short' not in k and k != 'raw_adjusted' else f'{k}={vv:.4f}' for k, vv in r.items()), flush=True)
    (R / f'nxa_combo_screen_day{day}.json').write_text(json.dumps(out, indent=1))
