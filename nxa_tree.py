"""nxa T1 (owner: Idea-A improver session): the main tree expert (imp19 LightGBM: same 354 cached columns, same SETTINGS, seed 42,
735 trees, rolling 3-day fit) with the 8 label-free exposure columns of nxa_tree_feats.py appended (362 columns).

Blocks/cutoffs exactly as imp19: fit days [d-3, d-2, d-1] (each block's features cut at its own start), predict val{d} (cut d) for
d = 5, 6 and the evaluation rows (fit days 4, 5, 6; cut 7). Column order: stack 190 | imp3_wf 4 | imp4_decompose 32 | imp5_aligned 32 |
imp6_context 32 | imp17_als 32 | imp19_als_aligned 32 | nxa_expo 8.
Modes:
  check            rebuild the val6 354-column matrix and replay imp19_tree_d6.txt against imp19_tree_raw_day6.npz (layout test)
  screen FRAC      matched pair on a fixed FRAC of training events (seed 7401): control (354) and treatment (362), day 6 only
  full TARGET      full fit with 362 columns for TARGET in {6, 5, eval}; writes artifacts/nxa_tree_raw_{day6,day5,eval}.np[yz]
"""
import sys, json, time, gc
import numpy as np
import lightgbm as lgb
from config import ARTIFACT_DIR as R
from nx_seq_core import cutoff_of

FEAT = R / 'features'; T = np.load(R / 'train.npy', mmap_mode='r'); HEAD_T = np.asarray(T[::10, 2])
SETTINGS = dict(objective='binary', metric='auc', num_leaves=31, learning_rate=.04, min_data_in_leaf=200, feature_fraction=.9, bagging_fraction=.8,
                bagging_freq=1, lambda_l2=2., max_bin=127, force_col_wise=True, deterministic=True, num_threads=8, seed=42, verbosity=-1)  # = imp10_leafwise.SETTINGS
TREES = 735


def parts(name):
    stack = FEAT / (f'stack_{name}_c{name[-1]}.npy' if name != 'eval' else 'stack_eval_c7.npy')
    return [stack] + [FEAT / f'{p}_{name}.npy' for p in ('imp3_wf', 'imp4_decompose', 'imp5_aligned', 'imp6_context', 'imp17_als', 'imp19_als_aligned')]


def matrix(name, extra, rows=None):
    files = parts(name) + ([FEAT / f'nxa_expo_{name}.npy'] if extra else []); arrs = [np.load(f, mmap_mode='r') for f in files]
    widths = [a.shape[1] for a in arrs]; assert widths[:7] == [190, 4, 32, 32, 32, 32, 32] and len({len(a) for a in arrs}) == 1, (name, widths)
    sel = slice(None) if rows is None else rows
    return np.column_stack([np.asarray(a[sel], np.float32) for a in arrs])


def day_block(d):
    lo, hi = np.searchsorted(HEAD_T, [cutoff_of(d), cutoff_of(d + 1)], side='left'); return np.asarray(T[lo * 10:hi * 10])


def assemble(days, label, extra, frac=None, seed=7401):
    rng = np.random.default_rng(seed); chunks = []; ys = []; width = 354 + (8 if extra else 0)
    idx = {}
    for d in days:
        a = day_block(d); n_ev = len(a) // 10
        keep = np.sort(rng.choice(n_ev, int(round(frac * n_ev)), replace=False)) if frac else np.arange(n_ev)
        idx[d] = (keep[:, None] * 10 + np.arange(10)).ravel(); ys.append(a[idx[d], 5].astype(np.float32))
    total = sum(len(v) for v in idx.values()); path = FEAT / f'nxa_train_{label}.npy'
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(total, width)); off = 0
    for d in days:
        base = np.load(parts(f'day{d}')[0], mmap_mode='r'); a = day_block(d); assert np.array_equal(base[:, 4], a[:, 3]) and np.array_equal(base[:, 5], a[:, 4])
        for s in range(0, len(idx[d]), 200000):
            r = idx[d][s:s + 200000]; x[off:off + len(r)] = matrix(f'day{d}', extra, r); off += len(r)
        gc.collect()
    x.flush(); print(json.dumps(dict(stage='assembled', label=label, shape=list(x.shape))), flush=True); return x, np.concatenate(ys), path


def fit_predict(days, label, extra, targets, frac=None):
    tic = time.monotonic(); x, y, path = assemble(days, label, extra, frac)
    data = lgb.Dataset(x, label=y, free_raw_data=True); model = lgb.train(SETTINGS, data, num_boost_round=TREES)
    del data, x; gc.collect(); path.unlink()
    out = {t: model.predict(matrix(t, extra)) for t in targets}; model.save_model(str(R / f'nxa_tree_{label}.txt'))
    print(json.dumps(dict(stage='fit', label=label, extra=extra, frac=frac, trees=TREES, elapsed_s=round(time.monotonic() - tic, 1))), flush=True); return out


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'check':
        xv = matrix('val6', False); m = lgb.Booster(model_file=str(R / 'imp19_tree_d6.txt')); raw = m.predict(xv); ref = np.load(R / 'imp19_tree_raw_day6.npz')['raw']
        print('imp19 day-6 replay max abs diff', float(np.max(np.abs(raw - ref))))
    elif mode == 'screen':
        frac = float(sys.argv[2]); res = {}
        for extra in (False, True):
            res['treatment' if extra else 'control'] = fit_predict([3, 4, 5], f'screen_{"t" if extra else "c"}', extra, ['val6'], frac)['val6']
        v = np.load(R / 'v53_causal_propensity_day6.npz'); np.savez_compressed(R / 'nxa_tree_screen_day6.npz', y=v['y'], events=v['events'], control=res['control'], treatment=res['treatment'], frac=frac)
    elif mode == 'full':
        tgt = sys.argv[2]
        if tgt == 'eval':
            raw = fit_predict([4, 5, 6], 'final', True, ['eval'])['eval']; np.save(R / 'nxa_tree_raw_eval.npy', raw)
        else:
            d = int(tgt); raw = fit_predict([d - 3, d - 2, d - 1], f'd{d}', True, [f'val{d}'])[f'val{d}']; v = np.load(R / f'v53_causal_propensity_day{d}.npz')
            np.savez_compressed(R / f'nxa_tree_raw_day{d}.npz', y=v['y'], events=v['events'], raw=raw)
