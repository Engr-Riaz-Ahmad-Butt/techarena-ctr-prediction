"""Collaborative Factorization Library for Track A.

Strict temporal causality:
Day 5 models only use data < Day 5 cutoff (start + 5 * 86400).
Day 6 models only use data < Day 6 cutoff (start + 6 * 86400).

Features:
- Exposure-aware weighted alternating least squares (ALS)
- Exponential click recency time-decay (e.g. 3-day half-life)
- Pre-window warm history inclusion with timestamps
- Exact cold-user fold-in via pre-event history
- Multi-seed factor averaging
"""
import math
import time
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix, csr_matrix

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DAY = 86400
N_USERS = 53003
N_ITEMS = 50000


def get_cutoff(day, start_time=1737331200):
    return start_time + day * DAY


@njit(parallel=True)
def solve_rows(indptr, indices, weights, targets, factors, other_bias,
               gram, rhs, background, global_rate, prior_bias,
               latent_reg=4.0, bias_reg=20.0):
    """
    Exact weighted least squares solve for one side of ALS.
    residual = targets[q] - weights[q] * (global_rate + other_bias[item])
    where targets[q] = weights[q] * label_q.
    """
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


def prepare_als_data(day, half_life_days=3.0, confidence_scale=5.0,
                     exposure_weight=0.5):
    """
    Extracts strictly causal interaction matrices for the given day cutoff.
    """
    train = np.load(ARTIFACTS / "train.npy", mmap_mode="r")
    train_hist = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    train_times = np.load(ARTIFACTS / "train_times.npy", mmap_mode="r")
    start = int(train[:, 2].min())
    cutoff = get_cutoff(day, start)

    # 1. In-window prior interactions
    mask = train[:, 2] < cutoff
    prior = np.asarray(train[mask])
    assert int(prior[:, 2].max()) < cutoff, "Prior timestamp must be strictly < cutoff"

    global_rate = float(prior[:, 5].mean())
    prior_users = prior[:, 0]
    prior_items = prior[:, 1]
    prior_times = prior[:, 2]
    prior_labels = prior[:, 5].astype(np.float64)

    # Calculate exponential click decay
    ages_sec = (cutoff - prior_times).astype(np.float64)
    tau_sec = (half_life_days * DAY) / math.log(2.0)
    click_decay = np.exp(-ages_sec / tau_sec)

    # In-window weights and targets
    # For non-clicks: weight = exposure_weight, target = 0.0
    # For clicks: weight = exposure_weight + confidence_scale * click_decay, target = weight * 1.0
    in_weights = np.full(len(prior), exposure_weight, dtype=np.float64)
    in_weights += (prior_labels * confidence_scale * click_decay)
    in_targets = prior_labels * in_weights

    # 2. Pre-window warm history clicks
    # Take each user's earliest event in prior
    ev = prior[::10]
    _, first_idx = np.unique(ev[:, 0], return_index=True)
    first_events = ev[first_idx]
    
    h_items = np.asarray(train_hist[first_events[:, 6]])
    h_times = np.asarray(train_times[first_events[:, 6]])
    h_users = np.repeat(first_events[:, 0], 20)
    h_items_flat = h_items.ravel()
    h_times_flat = h_times.ravel()

    valid = (h_items_flat >= 0) & (h_times_flat < cutoff)
    u_warm = h_users[valid]
    i_warm = h_items_flat[valid]
    t_warm = h_times_flat[valid]

    # Deduplicate (user, item, timestamp)
    keys = np.rec.fromarrays([u_warm, i_warm, t_warm], names="u,i,t")
    _, unique_warm = np.unique(keys, return_index=True)
    u_warm = u_warm[unique_warm]
    i_warm = i_warm[unique_warm]
    t_warm = t_warm[unique_warm]

    warm_ages = (cutoff - t_warm).astype(np.float64)
    warm_decay = np.exp(-warm_ages / tau_sec)
    warm_weights = (confidence_scale * warm_decay).astype(np.float64)
    warm_targets = warm_weights.copy()  # label is 1.0

    # Combine all interactions
    all_users = np.concatenate([prior_users, u_warm])
    all_items = np.concatenate([prior_items, i_warm])
    all_weights = np.concatenate([in_weights, warm_weights])
    all_targets = np.concatenate([in_targets, warm_targets])

    # Build CSR matrices
    w_mat = coo_matrix((all_weights, (all_users, all_items)),
                       shape=(N_USERS, N_ITEMS)).tocsr()
    t_mat = coo_matrix((all_targets, (all_users, all_items)),
                       shape=(N_USERS, N_ITEMS)).tocsr()
    assert np.array_equal(w_mat.indptr, t_mat.indptr)
    assert np.array_equal(w_mat.indices, t_mat.indices)

    # Item-transposed CSR matrices
    wt_mat = w_mat.T.tocsr()
    tt_mat = t_mat.T.tocsr()

    # Prior CTR estimates for bias regularization
    user_exp = np.bincount(prior_users, minlength=N_USERS)
    item_exp = np.bincount(prior_items, minlength=N_ITEMS)
    user_clicks = np.bincount(prior_users, weights=prior_labels, minlength=N_USERS)
    item_clicks = np.bincount(prior_items, weights=prior_labels, minlength=N_ITEMS)

    up = (user_clicks + 50.0 * global_rate) / (user_exp + 50.0) - global_rate
    ip = (item_clicks + 50.0 * global_rate) / (item_exp + 50.0) - global_rate

    warm_counts = np.bincount(u_warm, minlength=N_USERS)
    # Background confidence: unobserved items pull towards 0 with small weight
    background = (exposure_weight * 9.0 * warm_counts / N_ITEMS).astype(np.float64)

    return {
        "day": day,
        "cutoff": cutoff,
        "w_mat": w_mat,
        "t_mat": t_mat,
        "wt_mat": wt_mat,
        "tt_mat": tt_mat,
        "global_rate": global_rate,
        "up": up,
        "ip": ip,
        "background": background,
        "user_exp": user_exp,
        "item_exp": item_exp,
    }


def fit_single_seed(data, dim=64, seed=101, iterations=6,
                     latent_reg=4.0, bias_reg=20.0):
    """
    Fits ALS model for a single seed.
    """
    w = data["w_mat"]
    t = data["t_mat"]
    wt = data["wt_mat"]
    tt = data["tt_mat"]
    g = data["global_rate"]
    up = data["up"]
    ip = data["ip"]
    background = data["background"]
    
    rng = np.random.default_rng(seed)
    V = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(N_ITEMS, dim)).astype(np.float64)
    U = np.zeros((N_USERS, dim), dtype=np.float64)
    bu = up.copy()
    bi = ip.copy()

    for it in range(1, iterations + 1):
        # 1. Update user factors U, bu
        x_v = np.column_stack([np.ones(N_ITEMS), V])
        gram_v = x_v.T @ x_v
        rhs_v = -x_v.T @ (g + bi)
        bu, U = solve_rows(w.indptr, w.indices, w.data, t.data,
                           V, bi, gram_v, rhs_v, background, g, up,
                           latent_reg=latent_reg, bias_reg=bias_reg)
        
        # 2. Update item factors V, bi
        x_u = np.column_stack([np.ones(N_USERS), U])
        gram_u = x_u.T @ (background[:, None] * x_u)
        rhs_u = -x_u.T @ (background * (g + bu))
        bi, V = solve_rows(wt.indptr, wt.indices, wt.data, tt.data,
                           U, bu, gram_u, rhs_u, np.ones(N_ITEMS), g, ip,
                           latent_reg=latent_reg, bias_reg=bias_reg)

    return {"U": U, "V": V, "bu": bu, "bi": bi, "g": g, "dim": dim, "seed": seed}


def score_events(event_ids, model, data=None):
    """
    Computes candidate predictions for given event_ids.
    Returns:
      full_score: g + bu + bi + u @ v
      dot_score: u @ v
      cos_score: (u @ v) / (||u|| * ||v||)
      slate_diff: full_score - mean(full_score, axis=1)
    """
    train = np.load(ARTIFACTS / "train.npy", mmap_mode="r")
    train_hist = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    train_times = np.load(ARTIFACTS / "train_times.npy", mmap_mode="r")

    row_idx = (event_ids[:, None] * 10 + np.arange(10)[None, :]).ravel()
    rows = np.asarray(train[row_idx])
    
    users = rows[::10, 0]
    items = rows[:, 1].reshape(-1, 10)

    U = model["U"]
    V = model["V"]
    bu = model["bu"]
    bi = model["bi"]
    g = model["g"]

    # Cold user fold-in
    user_vecs = U[users].copy()
    user_biases = bu[users].copy()
    
    if data is not None:
        cold = data["user_exp"][users] == 0
        if cold.any():
            # Fold in cold users from their history before the event
            cold_ev = event_ids[cold]
            cold_users = users[cold]
            h = np.asarray(train_hist[cold_ev])
            stamps = np.asarray(train_times[cold_ev])
            ev_times = train[cold_ev * 10, 2]
            
            valid = (h >= 0) & (stamps < ev_times[:, None])
            local_idx = np.repeat(np.arange(len(cold_ev)), 20)[valid.ravel()]
            target_items = h.ravel()[valid.ravel()]
            
            if len(local_idx) > 0:
                vals = np.ones(len(local_idx), dtype=np.float64)
                w_cold = coo_matrix((vals, (local_idx, target_items)),
                                    shape=(len(cold_ev), N_ITEMS)).tocsr()
                t_cold = w_cold.copy()
                x_v = np.column_stack([np.ones(N_ITEMS), V])
                gram_v = x_v.T @ x_v
                rhs_v = -x_v.T @ (g + bi)
                cnt = np.bincount(local_idx, minlength=len(cold_ev))
                bg_cold = (9.0 * cnt / N_ITEMS).astype(np.float64)
                
                c_bu, c_U = solve_rows(w_cold.indptr, w_cold.indices, w_cold.data, t_cold.data,
                                       V, bi, gram_v, rhs_v, bg_cold, g, np.zeros(len(cold_ev)),
                                       latent_reg=4.0, bias_reg=20.0)
                user_vecs[cold] = c_U
                user_biases[cold] = c_bu

    # Dot products
    # user_vecs: (n_events, dim), V[items]: (n_events, 10, dim)
    dot = np.einsum("ed,ecd->ec", user_vecs, V[items])
    
    # Norms for cosine
    u_norm = np.linalg.norm(user_vecs, axis=1, keepdims=True)
    v_norm = np.linalg.norm(V[items], axis=2)
    cos = dot / np.maximum(u_norm * v_norm, 1e-9)

    # Full linear score
    full = g + user_biases[:, None] + bi[items] + dot
    
    # Within-slate centered difference
    slate_diff = full - full.mean(axis=1, keepdims=True)
    slate_dot_diff = dot - dot.mean(axis=1, keepdims=True)

    return {
        "full": full.ravel(),
        "dot": dot.ravel(),
        "cos": cos.ravel(),
        "slate_diff": slate_diff.ravel(),
        "slate_dot_diff": slate_dot_diff.ravel(),
    }
