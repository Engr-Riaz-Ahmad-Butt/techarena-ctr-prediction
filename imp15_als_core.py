"""Exact alternating weighted ridge solves for implicit preference factors."""
import numpy as np
from numba import njit,prange


@njit(parallel=True)
def solve_rows(indptr,indices,weights,targets,factors,other_bias,gram,rhs,background,
               global_rate,prior_bias,latent_reg=1.,bias_reg=20.):
    n=len(indptr)-1;d=factors.shape[1]
    bias=np.empty(n,np.float64);latent=np.empty((n,d),np.float64)
    for row in prange(n):
        a=background[row]*gram.copy();b=background[row]*rhs.copy()
        a[0,0]+=bias_reg;b[0]+=bias_reg*prior_bias[row]
        for j in range(1,d+1):a[j,j]+=latent_reg
        vector=np.empty(d+1,np.float64);vector[0]=1.
        for q in range(indptr[row],indptr[row+1]):
            item=indices[q];weight=weights[q]
            for j in range(d):vector[j+1]=factors[item,j]
            residual=targets[q]-weight*(global_rate+other_bias[item])
            for j in range(d+1):
                b[j]+=residual*vector[j]
                for k in range(d+1):a[j,k]+=weight*vector[j]*vector[k]
        value=np.linalg.solve(a,b)
        bias[row]=value[0];latent[row]=value[1:]
    return bias,latent
