"""Tiny exact ALS versus explicit augmented least squares; no dataset fitting."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from imp15_als_core import solve_rows
set_num_threads(4)


def csr(weight, target):
    indptr = [0]
    indices, weights, targets = [], [], []
    for row in range(len(weight)):
        ix = np.flatnonzero(weight[row])
        indices.extend(ix)
        weights.extend(weight[row, ix])
        targets.extend(target[row, ix])
        indptr.append(len(indices))
    return (np.array(indptr, np.int64), np.array(indices, np.int64),
            np.array(weights, float), np.array(targets, float))


def explicit_rows(weight, target, factors, other_bias, confidence, global_rate, prior, latent_reg, bias_reg):
    x = np.column_stack([np.ones(len(factors)), factors])
    answers = []
    regularization = np.r_[bias_reg, np.full(factors.shape[1], latent_reg)]
    for row in range(len(weight)):
        ix = np.flatnonzero(weight[row])
        design = [np.sqrt(weight[row, ix])[:, None] * x[ix],
                  np.sqrt(confidence[row])[:, None] * x,
                  np.diag(np.sqrt(regularization))]
        observed_y = np.sqrt(weight[row, ix]) * (target[row, ix]/weight[row, ix] - global_rate - other_bias[ix])
        background_y = -np.sqrt(confidence[row]) * (global_rate + other_bias)
        values = [observed_y, background_y, np.r_[np.sqrt(bias_reg)*prior[row], np.zeros(factors.shape[1])]]
        answers.append(np.linalg.lstsq(np.vstack(design), np.concatenate(values), rcond=None)[0])
    return np.array(answers)


if __name__ == '__main__':
    rng = np.random.default_rng(1542)
    n, m, dim = 4, 5, 3
    U = rng.normal(0, .2, (n, dim))
    V = rng.normal(0, .2, (m, dim))
    bu = np.array([.03, -.04, .02, .1])
    bi = np.array([-.02, .06, -.05, .07, -.03])
    prior_u = np.array([.01, -.02, .04, .15])
    prior_i = np.array([-.03, .01, -.02, .05, .02])
    confidence = np.array([.1, .2, .3, 0.])
    weight = np.array([[2.,0,3.,0,0], [0,1.5,2.,0,0], [0,0,0,0,0], [0,0,0,0,0]])
    target = np.array([[1.,0,2.,0,0], [0,.5,.2,0,0], [0,0,0,0,0], [0,0,0,0,0]])
    global_rate, latent_reg, bias_reg = .17, 1., 20.
    def objective(u,v,user_bias,item_bias):
        scores = global_rate + user_bias[:,None] + item_bias + u@v.T
        known = 0.
        active = weight>0
        known = np.sum(weight[active]*(scores[active]-target[active]/weight[active])**2)
        background = np.sum(confidence[:,None]*scores**2)
        ridge = latent_reg*(np.sum(u*u)+np.sum(v*v))
        ridge += bias_reg*(np.sum((user_bias-prior_u)**2)+np.sum((item_bias-prior_i)**2))
        return float(known+background+ridge)
    before = objective(U,V,bu,bi)
    x = np.column_stack([np.ones(m),V])
    gram, rhs = x.T@x, -x.T@(global_rate+bi)
    gram_before, rhs_before = gram.copy(), rhs.copy()
    with threadpool_limits(limits=1):
        new_bu,new_U = solve_rows(*csr(weight,target),V,bi,gram,rhs,confidence,global_rate,prior_u,latent_reg,bias_reg)
        reference_u = explicit_rows(weight,target,V,bi,np.tile(confidence[:,None],(1,m)),
                                    global_rate,prior_u,latent_reg,bias_reg)
        actual_u = np.column_stack([new_bu,new_U])
        user_error = float(np.max(np.abs(actual_u-reference_u)))
        assert user_error<1e-12,user_error
        assert np.array_equal(gram,gram_before) and np.array_equal(rhs,rhs_before)
        assert np.allclose(new_U[3],0,atol=1e-15,rtol=0) and abs(new_bu[3]-prior_u[3])<1e-15
        after_user = objective(new_U,V,new_bu,bi)
        assert after_user<=before+1e-12
        xu=np.column_stack([np.ones(n),new_U])
        gram_i=xu.T@(confidence[:,None]*xu)
        rhs_i=-xu.T@(confidence*(global_rate+new_bu))
        new_bi,new_V=solve_rows(*csr(weight.T,target.T),new_U,new_bu,gram_i,rhs_i,np.ones(m),
                               global_rate,prior_i,latent_reg,bias_reg)
        reference_i=explicit_rows(weight.T,target.T,new_U,new_bu,np.tile(confidence,(m,1)),
                                  global_rate,prior_i,latent_reg,bias_reg)
        item_error=float(np.max(np.abs(np.column_stack([new_bi,new_V])-reference_i)))
        assert item_error<1e-12,item_error
        after_item=objective(new_U,new_V,new_bu,new_bi)
        assert after_item<=after_user+1e-12
    result=dict(status='PASS',user_dense_lstsq_max_error=user_error,item_dense_lstsq_max_error=item_error,
                empty_zero_background_row_prior_and_zero_latent=True,
                empty_nonzero_background_row_dense_match=True,nonzero_global_other_bias_prior=True,
                uniform_user_and_item_terms_verified=True,input_gram_rhs_not_mutated=True,
                objective_before=before,objective_after_user=after_user,objective_after_item=after_item,
                numba_threads=4,blas_threads=1)
    (ARTIFACT_DIR/'imp15_test_als_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
