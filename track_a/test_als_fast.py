import time
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def solve_rows(indptr, indices, weights, targets, factors, other_bias,
               gram, rhs, background, global_rate, prior_bias,
               latent_reg=4.0, bias_reg=20.0):
    n = len(indptr) - 1
    d = factors.shape[1]
    bias = np.empty(n, np.float64)
    latent = np.empty((n, d), np.float64)
    for row in prange(n):
        a = background[row] * gram.copy()
        b = background[row] * rhs.copy()
        a[0, 0] += bias_reg
        b[0] += bias_reg * prior_bias[row]
        for j in range(1, d + 1):
            a[j, j] += latent_reg
        vector = np.empty(d + 1, np.float64)
        vector[0] = 1.0
        for q in range(indptr[row], indptr[row + 1]):
            item = indices[q]
            w = weights[q]
            t = targets[q]
            for j in range(d):
                vector[j + 1] = factors[item, j]
            residual = t - w * (global_rate + other_bias[item])
            for j in range(d + 1):
                b[j] += residual * vector[j]
                for k in range(d + 1):
                    a[j, k] += w * vector[j] * vector[k]
        value = np.linalg.solve(a, b)
        bias[row] = value[0]
        latent[row] = value[1:]
    return bias, latent

def main():
    print("Testing solve_rows compilation and speed...")
    from scipy.sparse import csr_matrix
    np.random.seed(42)
    n_users = 1000
    n_items = 500
    dim = 64
    
    # Random sparse interactions
    nnz = 10000
    u_idx = np.random.randint(0, n_users, size=nnz)
    i_idx = np.random.randint(0, n_items, size=nnz)
    w_data = np.random.uniform(0.5, 2.0, size=nnz).astype(np.float64)
    t_data = (np.random.uniform(0, 1, size=nnz) > 0.8).astype(np.float64) * w_data
    
    mat_w = csr_matrix((w_data, (u_idx, i_idx)), shape=(n_users, n_items))
    mat_t = csr_matrix((t_data, (u_idx, i_idx)), shape=(n_users, n_items))
    
    V = np.random.randn(n_items, dim) * 0.1
    bi = np.random.randn(n_items) * 0.05
    x = np.column_stack([np.ones(n_items), V])
    gram = x.T @ x
    rhs = -x.T @ (0.03 + bi)
    background = np.full(n_users, 0.001)
    prior_bias = np.zeros(n_users)
    
    t0 = time.time()
    # First call triggers jit compilation
    bu, U = solve_rows(mat_w.indptr, mat_w.indices, mat_w.data, mat_t.data,
                       V, bi, gram, rhs, background, 0.03, prior_bias,
                       latent_reg=4.0, bias_reg=20.0)
    t1 = time.time()
    print(f"JIT compile + solve 1000 users dim {dim}: {t1 - t0:.3f} s")
    
    # Second call (pure execution)
    t0 = time.time()
    bu, U = solve_rows(mat_w.indptr, mat_w.indices, mat_w.data, mat_t.data,
                       V, bi, gram, rhs, background, 0.03, prior_bias,
                       latent_reg=4.0, bias_reg=20.0)
    t1 = time.time()
    print(f"Pure execution 1000 users dim {dim}: {t1 - t0:.4f} s")
    assert np.isfinite(bu).all()
    assert np.isfinite(U).all()
    print("Verification passed successfully!")

if __name__ == '__main__':
    main()
