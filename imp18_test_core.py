"""Tiny complete-objective, duplicate-gradient and snapshot-update tests."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
SOURCE_HASHES={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in ['imp18_nce_core.py','imp18_nce.py']}
import imp18_nce_core as core
set_num_threads(1)

REG=.002
LR=.02
ITEMS=np.array([0,1,1,2,3,4,0,2,3,4],np.int32)
WARM_ITEMS=np.array([1,1,2,2,3,4,5,0,1,2],np.int32)
HISTORY=np.array([1,1,3]+[-1]*17,np.int32)
OFFSET=np.array([.2,-.1,.3,-.2,.05,-.15])


def initial(dtype):
    rng=np.random.default_rng(1842)
    p={k:rng.normal(0,.2,shape).astype(dtype) for k,shape in
       [('U',(2,3)),('V',(6,3)),('H',(6,3)),('bu',(2,)),('bi',(6,))]}
    p['g']=np.array([-1.7],dtype)
    p['aw']=np.array([-np.log(9)+.1,-np.log(9)-.1],dtype)
    return p


def manual(p,mode,y):
    values={k:v.astype(float) for k,v in p.items()}
    U,V,H,bu,bi,g,aw=[values[k] for k in ['U','V','H','bu','bi','g','aw']]
    gradient={k:np.zeros_like(v) for k,v in values.items()}
    if mode=='real':
        valid=HISTORY[HISTORY>=0]
        scale=1/np.sqrt(len(valid))
        profile=U[0]+scale*H[valid].sum(0)
        logits=g[0]+bu[0]+bi[ITEMS]+V[ITEMS]@profile
        probability=1/(1+np.exp(-logits));error=probability-y
        loss=np.sum(np.logaddexp(0,logits)-y*logits)
        z_gradient=error@V[ITEMS]
        gradient['U'][0]=z_gradient+REG*U[0]
        for j,item in enumerate(ITEMS):
            gradient['V'][item]+=error[j]*profile+REG*V[item]
            gradient['bi'][item]+=error[j]+REG*bi[item]
        for item in valid:gradient['H'][item]+=scale*z_gradient+REG*H[item]
        gradient['bu'][0]=error.sum()+REG*bu[0]
        gradient['g'][0]=error.sum()+REG*(g[0]+1.8)
        loss+=.5*REG*(np.sum(U[0]**2)+np.sum(V[ITEMS]**2)+np.sum(H[valid]**2)+
                        bu[0]**2+np.sum(bi[ITEMS]**2)+(g[0]+1.8)**2)
    else:
        logits=aw[0]+OFFSET[WARM_ITEMS]+V[WARM_ITEMS]@U[0]
        targets=np.zeros(len(WARM_ITEMS));targets[0]=1
        error=1/(1+np.exp(-logits))-targets
        loss=np.sum(np.logaddexp(0,logits)-targets*logits)
        gradient['U'][0]=error@V[WARM_ITEMS]+REG*U[0]
        for j,item in enumerate(WARM_ITEMS):gradient['V'][item]+=error[j]*U[0]+REG*V[item]
        gradient['aw'][0]=error.sum()+REG*(aw[0]+np.log(9))
        loss+=.5*REG*(np.sum(U[0]**2)+np.sum(V[WARM_ITEMS]**2)+(aw[0]+np.log(9))**2)
    return float(loss),gradient


def analytic(p,mode,y):
    out={k:np.zeros(v.shape,float) for k,v in p.items()}
    if mode=='real':
        loss,gu,gv,gh,gub,gb,gg=core.real_grad(0,ITEMS,HISTORY,y,p['U'],p['V'],p['H'],p['bu'],p['bi'],float(p['g'][0]),REG)
        out['U'][0]=gu
        np.add.at(out['V'],ITEMS,gv)
        valid=HISTORY>=0
        np.add.at(out['H'],HISTORY[valid],gh[valid])
        out['bu'][0]=gub
        np.add.at(out['bi'],ITEMS,gb)
        out['g'][0]=gg
    else:
        loss,gu,gv,ga=core.warm_grad(0,WARM_ITEMS,p['U'],p['V'],p['aw'],OFFSET,REG)
        out['U'][0]=gu
        np.add.at(out['V'],WARM_ITEMS,gv)
        out['aw'][0]=ga
    return loss,out


def finite_difference(mode,click):
    p=initial(np.float64)
    y=np.zeros(10)
    if click:y[2]=1
    loss,gradient=analytic(p,mode,y)
    reference,independent=manual(p,mode,y)
    assert abs(loss-reference)<1e-12
    numeric={k:np.zeros_like(v) for k,v in p.items()}
    for name in p:
        for index in np.ndindex(p[name].shape):
            old=p[name][index]
            p[name][index]=old+1e-6;plus=manual(p,mode,y)[0]
            p[name][index]=old-1e-6;minus=manual(p,mode,y)[0]
            p[name][index]=old
            numeric[name][index]=(plus-minus)/2e-6
    error=max(float(np.max(np.abs(gradient[k]-numeric[k]))) for k in gradient)
    closed_error=max(float(np.max(np.abs(gradient[k]-independent[k]))) for k in gradient)
    assert error<2e-8 and closed_error<1e-12,(mode,error,closed_error)
    return dict(mode=mode,clicked=click,finite_difference_max_error=error,
                independent_aggregated_gradient_max_error=closed_error,loss_error=abs(loss-reference))


def production_update(mode):
    p=initial(np.float32)
    before={k:v.copy() for k,v in p.items()}
    accum={k:np.full_like(v,.1) for k,v in p.items()}
    y=np.zeros(10,np.float32);y[2]=1
    _,gradient=manual(before,mode,y)
    active=['U','V','H','bu','bi','g'] if mode=='real' else ['U','V','aw']
    expected={k:v.copy() for k,v in before.items()}
    expected_acc={k:v.copy() for k,v in accum.items()}
    for key in active:
        # Float32 accumulator is rounded BEFORE it is used in the denominator.
        expected_acc[key]=(.1+gradient[key]**2).astype(np.float32)
        expected[key]=(before[key].astype(float)-LR*gradient[key]/np.sqrt(expected_acc[key].astype(float))).astype(np.float32)
    if mode=='real':
        loss=core.real_epoch(np.array([0]),np.array([0],np.int32),ITEMS[None],HISTORY[None],y[None],
            *[p[k] for k in ['U','V','H','bu','bi','g']],*[accum[k] for k in ['U','V','H','bu','bi','g']],LR,REG)
    else:
        loss=core.warm_epoch(np.array([0]),np.array([0],np.int32),WARM_ITEMS[None],
            p['U'],p['V'],p['aw'],accum['U'],accum['V'],accum['aw'],OFFSET.astype(np.float32),LR,REG)
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in p.values())
    error=max(float(np.max(np.abs(p[k]-expected[k]))) for k in p)
    acc_error=max(float(np.max(np.abs(accum[k]-expected_acc[k]))) for k in p)
    assert error<2e-7 and acc_error<2e-6,(mode,error,acc_error)
    for key in set(p)-set(active):
        assert np.array_equal(p[key],before[key]) and np.array_equal(accum[key],np.full_like(accum[key],.1))
    return dict(mode=mode,snapshot_adagrad_parameter_max_error=error,
                snapshot_adagrad_accumulator_max_error=acc_error,parameter_isolation=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        gradients=[finite_difference('real',True),finite_difference('real',False),finite_difference('warm',True)]
        updates=[production_update('real'),production_update('warm')]
        for z in [-10000.,-1000.,0.,1000.,10000.]:
            for y in [0.,1.]:
                loss,error=core.bernoulli(z,y)
                expected=np.logaddexp(0,z)-y*z
                assert np.isfinite(loss) and np.isfinite(error) and abs(loss-expected)<1e-12
        p=initial(np.float32)
        prediction=core.predict(np.array([0],np.int32),ITEMS[None],HISTORY[None],
                               *[p[k] for k in ['U','V','H','bu','bi','g']])
        z=p['U'][0].astype(float)+p['H'][HISTORY[HISTORY>=0]].astype(float).sum(0)/np.sqrt(3)
        expected=p['g'][0]+p['bu'][0]+p['bi'][ITEMS]+p['V'][ITEMS].astype(float)@z
        prediction_error=float(np.max(np.abs(prediction-expected)))
        assert prediction_error<2e-7
    assert SOURCE_HASHES=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    result=dict(status='PASS',core_sha256=SOURCE_HASHES['imp18_nce_core.py'],source_sha256=SOURCE_HASHES,finite_difference_tests=gradients,float32_snapshot_updates=updates,
                repeated_candidate_history_and_positive_noise_collisions=True,
                stable_extreme_bernoulli=True,predict_logit_max_error=prediction_error,
                real_warm_parameter_isolation=True,no_heavy_fit=True)
    (ARTIFACT_DIR/'imp18_test_core_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
