"""nxb Track A: one LightGBM on the clean-room features (nxb_a_features.py). Self-contained.

--target 6: train blocks 2..5, score the day-6 cohort (first event per user on day 6); trees chosen on the
            day-6 SELECTION subset only (grid saved).
--target 5: train blocks 1..4 with the frozen tree count, score the day-5 cohort.
--target 7: train blocks 3..6 with the frozen tree count, score the evaluation rows.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent / 'artifacts'
NAMES = json.loads((R / 'nxb_a_names.json').read_text())
FAMILY = {'item': ('i_',), 'user': ('u_',), 'ui': ('ui_',), 'hist': ('h_', 'c_hist'), 'cc': ('cc_', 'ev_cc'),
          'mf': ('mf_',), 'expo': ('expo_',), 'event': ('ev_ictr', 'ev_in_hist', 'tb')}
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12, seed=6201,
              deterministic=True, force_row_wise=True, verbose=-1)


def family_of(n):
    for f, pre in FAMILY.items():
        if any(n.startswith(p) for p in pre):
            return f
    raise KeyError(n)


def eval_hist_share():
    eh = np.load(R / 'evaluation_hist.npy')
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    return np.bincount(eg, minlength=5) / len(eg)


def adj_weights(hbin_rows):
    vg = hbin_rows[::10]
    ratio = eval_hist_share() / (np.bincount(vg, minlength=5) / len(vg))
    return ratio[hbin_rows]


def cohort_rows(k, cohort_events):
    """Map cohort rows (event e, slot j in train order) to row indices of block k."""
    meta = np.load(R / f'nxb_a_meta_k{k}.npz')
    ev = meta['events']
    e0 = ev[0]
    assert np.all(ev[::10] - e0 == np.arange(len(ev) // 10))
    ce = cohort_events.reshape(-1, 10)
    assert np.all(ce == ce[:, :1])
    idx = ((ce[:, :1] - e0) * 10 + np.arange(10)[None, :]).ravel()
    assert np.array_equal(ev[idx], cohort_events)
    assert meta['first'][idx].all()
    return idx, meta


SHIFT_LOG = ('i_imp', 'i_clk', 'u_imp', 'u_ev')   # log1p(counts over the k days before the cutoff)
FIX_SHIFT = False


def load_rows(k, rows, cols):
    X = np.load(R / f'nxb_a_feat_k{k}.npy', mmap_mode='r')
    out = np.asarray(X[rows][:, cols], dtype=np.float32)
    if FIX_SHIFT:  # per-day normalisation of block-length dependent counts (block k has k days of past rows)
        for j, c in enumerate(cols):
            if NAMES[c] in SHIFT_LOG:
                out[:, j] -= np.float32(np.log(k))
    return out


def train(blocks, cols, rows_mode, trees, log):
    Xs, ys = [], []
    for k in blocks:
        meta = np.load(R / f'nxb_a_meta_k{k}.npz')
        sel = np.where(meta['first'])[0] if rows_mode == 'first' else np.arange(len(meta['y']))
        Xs.append(load_rows(k, sel, cols)); ys.append(meta['y'][sel])
    X = np.concatenate(Xs); y = np.concatenate(ys); del Xs
    log(f'train blocks {blocks} rows {len(y)} ctr {y.mean():.4f} cols {len(cols)}')
    ds = lgb.Dataset(X, y, feature_name=[NAMES[c] for c in cols], free_raw_data=True)
    t = time.time()
    bst = lgb.train(PARAMS, ds, num_boost_round=trees)
    log(f'fit {trees} trees {time.time()-t:.1f}s')
    return bst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, required=True)
    ap.add_argument('--name', default='a1')
    ap.add_argument('--rows', default='first', choices=['first', 'all'])
    ap.add_argument('--drop', default='')            # comma list of families to drop
    ap.add_argument('--trees', type=int, default=0)  # 0 -> frozen value (targets 5/7) or 1500 grid (target 6)
    ap.add_argument('--nblocks', type=int, default=4)
    ap.add_argument('--fix-shift', action='store_true')
    ap.add_argument('--drop-cols', default='')
    a = ap.parse_args()
    t0 = time.time()
    logf = open(R / f'nxb_a_{a.name}_t{a.target}.log', 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    global FIX_SHIFT
    FIX_SHIFT = a.fix_shift
    drop = [d for d in a.drop.split(',') if d]
    dcols = [d for d in a.drop_cols.split(',') if d]
    cols = [j for j, n in enumerate(NAMES) if family_of(n) not in drop and n not in dcols]
    blocks = list(range(a.target - a.nblocks, a.target))
    frozen_path = R / f'nxb_a_{a.name}_frozen.json'
    if a.target == 6:
        trees = a.trees or 1500
    else:
        trees = a.trees or json.loads(frozen_path.read_text())['trees']
    log(json.dumps(dict(args=vars(a), blocks=blocks, trees=trees, n_cols=len(cols), params=PARAMS)))
    bst = train(blocks, cols, a.rows, trees, log)
    if a.target in (5, 6):
        ref = np.load(R / f'nxa_h1_day{a.target}.npz')
        idx, meta = cohort_rows(a.target, ref['events'])
        X = load_rows(a.target, idx, cols)
        y = ref['y']
        assert np.array_equal(meta['y'][idx], y)
        w_full = adj_weights(meta['hbin'][idx])
        if a.target == 6:
            sel = np.load(R / 'selection_day6.npz')
            fi = sel['full_indices']
            assert np.array_equal(sel['y'], y[fi])
            grid = {}
            for it in list(range(100, trees + 1, 100)):
                p = bst.predict(X[fi], num_iteration=it)
                grid[it] = float(roc_auc_score(sel['y'], p, sample_weight=sel['weights']))
                log(f'  trees {it} selection adjusted {grid[it]:.6f}')
            best = max(grid, key=grid.get)
            (R / f'nxb_a_{a.name}_grid_day6.json').write_text(json.dumps(dict(grid=grid, best=best), indent=1))
            if not a.trees:
                frozen_path.write_text(json.dumps(dict(trees=best, rows=a.rows, drop=drop, drop_cols=dcols, fix_shift=a.fix_shift, params=PARAMS,
                                                       time=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
            use = best
        else:
            use = trees
        raw = bst.predict(X, num_iteration=use)
        res = dict(target=a.target, trees=use, adj_full=float(roc_auc_score(y, raw, sample_weight=w_full)),
                   plain=float(roc_auc_score(y, raw)),
                   nxa_h1_adj_full=float(roc_auc_score(y, ref['pred'], sample_weight=w_full)))
        if a.target == 6:
            res['adj_sel'] = float(roc_auc_score(sel['y'], raw[fi], sample_weight=sel['weights']))
            res['nxa_h1_adj_sel'] = float(roc_auc_score(sel['y'], ref['pred'][fi], sample_weight=sel['weights']))
        hb = meta['hbin'][idx]
        res['bins'] = {}
        for b in range(5):
            m = hb == b
            res['bins'][b] = dict(n_events=int(m.sum() // 10), clean=float(roc_auc_score(y[m], raw[m])),
                                  nxa_h1=float(roc_auc_score(y[m], ref['pred'][m])),
                                  v53=float(roc_auc_score(y[m], ref['baseline'][m])))
        log(json.dumps(res))
        np.savez(R / f'nxb_a_{a.name}_day{a.target}.npz', y=y, events=ref['events'], raw=raw, pred=raw)
        imp = dict(zip(bst.feature_name(), bst.feature_importance('gain').tolist()))
        (R / f'nxb_a_{a.name}_importance_t{a.target}.json').write_text(json.dumps(
            dict(sorted(imp.items(), key=lambda kv: -kv[1])), indent=1))
    else:
        meta = np.load(R / 'nxb_a_meta_k7.npz')
        X = load_rows(7, np.arange(len(meta['y'])), cols)
        raw = bst.predict(X, num_iteration=trees)
        np.save(R / f'nxb_a_{a.name}_eval.npy', raw)
        log(f'eval rows {len(raw)} mean {raw.mean():.5f}')
    log(f'total {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
