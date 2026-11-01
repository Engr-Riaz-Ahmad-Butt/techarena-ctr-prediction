"""Anchored query-only warm-user refresh. No fitting or evaluation on import.

The new evidence is positive clicks only. Unknown accompanying warm-up slates
are not modeled as zero targets. Shared item factors and biases never change.
"""
import numpy as np

def refresh(u0,item_factors,item_bias,global_bias,user_bias,items,times,
            cutoff,event_time,anchor_precision,half_life_days=None):
    u0=np.asarray(u0,dtype=np.float64)
    v=np.asarray(item_factors,dtype=np.float64)
    precision=np.asarray(anchor_precision,dtype=np.float64)
    if precision.ndim==0:precision=np.eye(len(u0))*precision
    if precision.shape!=(len(u0),len(u0)):raise ValueError('Anchor precision shape')
    np.linalg.cholesky(precision)
    items=np.asarray(items);times=np.asarray(times)
    valid=items>=0
    if np.any(times[valid]>=event_time):raise ValueError('History not strictly earlier than event')
    if event_time<cutoff:raise ValueError('Event precedes cutoff')
    valid &= times>=cutoff
    evidence=np.rec.fromarrays([items[valid],times[valid]],names='item,time')
    if not len(evidence):return u0.copy(),0
    _,take=np.unique(evidence,return_index=True)
    ids=items[valid][take];stamps=times[valid][take]
    w=np.ones(len(ids)) if half_life_days is None else np.exp2(-(event_time-stamps)/(86400*half_life_days))
    vv=v[ids];target=1-float(global_bias)-float(user_bias)-np.asarray(item_bias)[ids]
    # Anchor the cutoff posterior, then add observed positive evidence only.
    hessian=precision+vv.T@(w[:,None]*vv)
    rhs=precision@u0+vv.T@(w*target)
    return np.linalg.solve(hessian,rhs),len(ids)

def self_test():
    v=np.array([[1.,0.],[0.,2.],[1.,1.]]);bi=np.array([.1,.2,.3]);u=np.array([.2,-.1]);p=np.diag([2.,3.]);original=v.copy()
    unchanged,n=refresh(u,v,bi,.05,.1,[0],[9],10,20,p);assert n==0 and np.array_equal(u,unchanged)
    # One duplicated record is removed; a second real click of the same item survives.
    actual,n=refresh(u,v,bi,.05,.1,[0,0,0,1],[11,11,12,9],10,20,p)
    expected=np.linalg.solve(p+2*np.outer(v[0],v[0]),p@u+2*v[0]*(1-.05-.1-bi[0]))
    assert n==2 and np.allclose(actual,expected) and np.array_equal(v,original)
    residual=p@(actual-u)+2*v[0]*(v[0]@actual-(1-.05-.1-bi[0]));assert np.linalg.norm(residual)<1e-12
    try:refresh(u,v,bi,.05,.1,[0],[20],10,20,p)
    except ValueError:pass
    else:raise AssertionError('Future/same-time click accepted')
    return dict(status='PASS',tests=['no-new-evidence identity','pre-cutoff clicks not counted twice','duplicate-record removal and distinct repeated clicks retained','independent quadratic optimum and zero gradient','future/same-time click rejected','item factors unchanged'])

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
