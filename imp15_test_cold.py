"""No-fit synthetic checks of cold-history scoring and exact fold-in."""
import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from scipy.sparse import coo_matrix
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from imp15_als_core import solve_rows
set_num_threads(4)


if __name__ == '__main__':
    rng = np.random.default_rng(1551)
    V = rng.normal(0, .2, (12, 3))
    U = rng.normal(0, .2, (3, 3))
    bu = np.array([.02, .5, -.4])
    bi = rng.normal(0, .03, 12)
    g = .06
    exposures = np.array([5, 0, 0])
    h = np.full((3, 20), -1, np.int32)
    h[1, :4] = [2, 2, 3, 2]
    stamps = np.zeros((3, 20), np.int64)
    stamps[1, :4] = [100, 100, 200, 300]
    rows = np.zeros((30, 7), np.int32)
    rows[:, 0] = np.repeat([0, 1, 2], 10)
    rows[:, 1] = np.tile(np.arange(10), 3)
    rows[:, 2] = 1000
    rows[:, 6] = np.repeat([0, 1, 2], 10)
    tree = ast.parse((ROOT / 'imp15_als.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['paired_matrices', 'augmented', 'score']]
    with tempfile.TemporaryDirectory(prefix='imp15_test_', dir=ARTIFACT_DIR) as directory:
        temp = Path(directory)
        assert temp.resolve().is_relative_to(ARTIFACT_DIR.resolve())
        np.save(temp / 'train_times.npy', stamps)
        ns = dict(np=np, ROOT=temp, hist=h, ni=12, solve_rows=solve_rows, coo_matrix=coo_matrix)
        exec(compile(tree, 'imp15_als.py', 'exec'), ns)
        original_U, original_bu = U.copy(), bu.copy()
        with threadpool_limits(limits=1):
            result = ns['score'](rows, U, V, bu, bi, g, exposures).reshape(3, 10)
            assert np.isfinite(result).all()
            assert np.allclose(result[0], g+bu[0]+bi[:10]+V[:10]@U[0], rtol=0, atol=1e-15)
            assert np.allclose(result[2], g+bi[:10], rtol=0, atol=1e-15)
            x = np.column_stack([np.ones(12), V])
            ix, weights = np.array([2, 3]), np.array([2., 1.])
            confidence = 9*3/12
            design = np.vstack([np.sqrt(weights)[:, None]*x[ix], np.sqrt(confidence)*x,
                                np.diag(np.sqrt([20., 1., 1., 1.]))])
            targets = np.r_[np.sqrt(weights)*(1-g-bi[ix]), -np.sqrt(confidence)*(g+bi), np.zeros(4)]
            exact = np.linalg.lstsq(design, targets, rcond=None)[0]
            expected = g+exact[0]+bi[:10]+V[:10]@exact[1:]
            error = float(np.max(np.abs(result[1]-expected)))
            assert error < 1e-12
            ns['hist'] = h.copy()
            ns['hist'][1, 1] = -1
            without_duplicate = ns['score'](rows, U, V, bu, bi, g, exposures).reshape(3, 10)
            duplicate_error = float(np.max(np.abs(result-without_duplicate)))
            assert duplicate_error == 0
            # A valid future click must be rejected before any cold solve.
            future = stamps.copy()
            future[1, 0] = 1000
            np.save(temp / 'train_times.npy', future)
            rejected = False
            try:
                ns['score'](rows, U, V, bu, bi, g, exposures)
            except AssertionError:
                rejected = True
            assert rejected
        assert np.array_equal(U, original_U) and np.array_equal(bu, original_bu)
    report = dict(status='PASS', cold_dense_foldin_max_error=error,
                  duplicate_removal_prediction_max_error=duplicate_error,
                  known_users_not_refitted=True, empty_history_global_item_fallback=True,
                  future_history_rejected=True, source_model_arrays_unmodified=True)
    (ARTIFACT_DIR / 'imp15_test_cold_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
