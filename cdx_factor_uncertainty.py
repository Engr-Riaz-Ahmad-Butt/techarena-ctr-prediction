"""Fixed directional uncertainty diagnostic for the causal warm16 model.

Population factor covariance + genuine past-slate Fisher information; positive
spherical cubature for conditional item choice. No new mean-vector fit or head.
This is a local empirical-Bayes approximation, not full BPMF posterior inference.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,time,argparse,math
import numpy as np
from numba import njit,set_num_threads
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import cdx_collective as original
import cdx_policy_hpf as metric
B,A,H=metric.B,metric.A,metric.H
R=H/'cdx_uncertainty_001'

@njit(cache=True)
def fisher(users,items,p,V,nu):
    d=V.shape[1];out=np.zeros((nu,d,d))
    for e in range(len(users)):
        u=users[e];mean=np.zeros(d)
        for j in range(10):
            i=items[e,j]
            for a in range(d):mean[a]+=p[e,j]*V[i,a]
        for a in range(d):
            for b in range(a+1):
                value=-mean[a]*mean[b]
                for j in range(10):
                    i=items[e,j];value+=p[e,j]*V[i,a]*V[i,b]
                out[u,a,b]+=value
                if b!=a:out[u,b,a]+=value
    return out

@njit(cache=True)
def average_choice(users,items,p,V,chol):
    out=np.zeros(p.shape);d=V.shape[1];scale=math.sqrt(d)
    for e in range(len(users)):
        mass=0.
        for j in range(10):mass+=p[e,j]
        for k in range(d):
            for sign in [-1.,1.]:
                logits=np.empty(10);maximum=-1e300
                for j in range(10):
                    shift=0.
                    for a in range(d):shift+=V[items[e,j],a]*chol[users[e],a,k]
                    logits[j]=math.log(max(p[e,j],1e-300))+sign*scale*shift
                    maximum=max(maximum,logits[j])
                denom=0.
                for j in range(10):logits[j]=math.exp(logits[j]-maximum);denom+=logits[j]
                for j in range(10):out[e,j]+=mass*logits[j]/denom/(2*d)
    return out

def load():
    t=np.load(A/'train.npy',mmap_mode='r');start=int(t[0,2]);cut=start+3*86400
    end=int(np.searchsorted(t[::10,2],cut));past=t[:end*10]
    model=np.load(H/'cdx_collective_001/control_model.npz');ps=[model[k] for k in ['U','V','H','bu','bi']]
    ns=original.definitions();users=past[::10,0].copy();items=past[:,1].reshape(-1,10).copy()
    hh=np.asarray(ns['hist'][past[::10,6]]).copy()
    timestamps=np.load(A/'train_times.npy',mmap_mode='r')[past[::10,6]]
    assert np.all((timestamps<past[::10,2,None])|(hh<0))
    count=np.bincount(users,minlength=len(ps[0]));active=count>0
    population=np.cov(ps[0][active].astype(np.float64),rowvar=False)
    population+=np.eye(len(population))*np.trace(population)/len(population)*1e-6
    assert np.linalg.eigvalsh(population).min()>0
    return ps,ns,users,items,hh,population,dict(cutoff=cut,max_fit_timestamp=int(past[-1,2]),
        fit_events=len(users),active_users=int(active.sum()),fit_features='Earlier actual user/item/history plus frozen point-model probabilities; outcomes not used to fit covariance')

def checks():
    # Fisher Hessian agrees with an independent finite difference of 11-way CE.
    rng=np.random.default_rng(8);v=rng.normal(0,.2,(10,3));u=rng.normal(0,.2,3)
    def probability(z):
        logits=v@z;p=np.exp(logits);return p/(1+p.sum())
    p=probability(u);f=fisher(np.array([0]),np.arange(10).reshape(1,10),p[None,:],v,1)[0]
    expected=np.empty((3,3));eps=1e-5
    def grad(z):return v.T@probability(z)-v[2]
    for k in range(3):
        delta=np.zeros(3);delta[k]=eps;expected[:,k]=(grad(u+delta)-grad(u-delta))/(2*eps)
    assert np.allclose(f,expected,atol=1e-10)
    assert np.linalg.eigvalsh(f).min()>-1e-10
    pred=average_choice(np.array([0]),np.arange(10).reshape(1,10),p[None,:],v,np.zeros((1,3,3)))
    assert np.allclose(pred,p[None,:],atol=1e-14)
    # Sigma points reproduce requested first two moments exactly.
    cov=np.diag([.02,.03,.04]);chol=np.linalg.cholesky(cov)
    points=np.concatenate([math.sqrt(3)*chol.T,-math.sqrt(3)*chol.T])
    assert np.allclose(points.mean(0),0) and np.allclose(points.T@points/6,cov)
    return True

def freeze():
    R.mkdir(exist_ok=True);assert not (R/'frozen.json').exists()
    paths=[original.H/'cdx_collective_001/control_model.npz',original.H/'cdx_collective_001/frozen_predictions.npz',H/'cohort.npz',B/'cdx_factor_uncertainty.py']
    plan=dict(date='2026-10-30',hypothesis='Anisotropic uncertainty in latent preference can improve choice beyond a single fitted user vector',
        sources=['https://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf','https://www.cs.ucc.ie/~dgb/papers/Coscrato-Bridge-2022.pdf'],
        source_scope='Posterior prediction/implicit-feedback uncertainty motivation; original local conditional approximation, not BPMF/MCMC reproduction or exact calibrated posterior',
        prior='Empirical covariance of cutoff-model U among prior-impression users, plus trace-scaled1e-6 numerical jitter',
        likelihood_information='11-way softmax expected user Hessian from ONLY actual pre-cutoff slates, no warm-up exposures invented',
        covariance='inverse(population_precision + summed past-slate Fisher)',
        prediction='2d positive spherical cubature points around unchanged point U; item and history factors fixed; average conditional10-way probabilities; preserve original event-click mass',
        novelty='Directional evidence and distributional integration; no scalar segment calibration, no updated mean vectors, no short-history expert, no seed averaging, no new head',
        hyperparameter_search=False,mean_user_factor_update=False,point_model_frozen=True,
        limitation='Item/H/parameter coupling and warm-positive auxiliary curvature omitted; population prior inferred from fitted means; diagnostic is not a full Bayesian fit',
        bridge='Original95percent joint-horizon warm rank plus5percent frozen ALS rank',
        screening='20%9878users:+.0002 overall/all4arrivaldays; full+.0005 only for survivor',
        no_holdout=True,no_evaluation_statistics=True,no_submission=True,
        hashes={str(p.relative_to(B)):metric.sha(p) for p in paths})
    (R/'frozen.json').open('x').write(json.dumps(plan,indent=2));return plan

def pilot():
    freeze();assert checks();ps,ns,u,i,h,pop,info=load();n=min(10000,len(u))
    tic=time.monotonic();p=ns['predict_events'](u[:n],i[:n],h[:n],*ps);f=fisher(u[:n],i[:n],p,ps[1],len(ps[0]));fit_seconds=time.monotonic()-tic
    tic=time.monotonic();cov=np.linalg.inv(np.linalg.inv(pop)[None,:,:]+f);chol=np.linalg.cholesky(cov);inverse_seconds=time.monotonic()-tic
    average_choice(u[:2],i[:2],p[:2],ps[1],chol)
    tic=time.monotonic();q=average_choice(u[:n],i[:n],p,ps[1],chol);score_seconds=time.monotonic()-tic
    assert np.allclose(q.sum(1),p.sum(1),atol=1e-12)
    out=dict(stage='training_only_timing_pilot',events=n,fit_seconds=fit_seconds,inverse_seconds=inverse_seconds,
        score_seconds=score_seconds,estimated_seconds=(fit_seconds*len(u)/n+inverse_seconds+score_seconds*49420/n)*1.5+30,
        numerical_checks=True,validation_scored=False,info=info)
    (R/'pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

def run():
    assert not (R/'report.json').exists();plan=json.loads((R/'frozen.json').read_text())
    assert all(metric.sha(B/p)==v for p,v in plan['hashes'].items())
    assert json.loads((R/'pilot.json').read_text())['estimated_seconds']<1800
    tic=time.monotonic();ps,ns,u,i,h,pop,info=load()
    p=ns['predict_events'](u,i,h,*ps);information=fisher(u,i,p,ps[1],len(ps[0]))
    assert np.isfinite(information).all()
    # Independent observed-slate Hessian and symmetry checks on a small prefix.
    n=128;reference=fisher(u[:n],i[:n],p[:n],ps[1],len(ps[0]))
    assert np.linalg.eigvalsh(reference[np.unique(u[:n])]).min()>-1e-8
    assert np.allclose(information,information.transpose(0,2,1),atol=1e-12)
    covariance=np.linalg.inv(np.linalg.inv(pop)[None,:,:]+information)
    chol=np.linalg.cholesky(covariance)
    assert np.linalg.eigvalsh(pop[None,:,:]-covariance[np.unique(u[:128])]).min()>-1e-8
    np.savez_compressed(R/'covariance.npz',population=pop,covariance=covariance,**info)
    rows=np.load(H/'score_rows.npy',mmap_mode='r');co=np.load(H/'cohort.npz');vu=rows[::10,0].copy();vi=rows[:,1].reshape(-1,10).copy()
    assert rows[::10,2].min()>=info['cutoff']>info['max_fit_timestamp'] and np.all(rows[:,5]==-1)
    legacy=np.load(H/'cdx_collective_001/frozen_predictions.npz');point=legacy['raw_control'];control=legacy['control']
    q=average_choice(vu,vi,point,ps[1],chol)
    assert np.isfinite(q).all() and (q>=0).all() and np.allclose(q.sum(1),point.sum(1),atol=1e-12)
    perm=np.array([9,1,4,6,2,0,3,7,8,5]);check=average_choice(vu[:128],vi[:128,perm],point[:128,perm],ps[1],chol)
    assert np.allclose(check[:,np.argsort(perm)],q[:128],atol=1e-12)
    n=len(rows);als=np.load(H/'matched_refresh_001/predictions.npz')['control_als']
    candidate=(.95*rankdata(q.ravel())/(n+1)+.05*rankdata(als)/(n+1)).reshape(-1,10)
    np.savez_compressed(R/'frozen_predictions.npz',control=control,candidate=candidate,raw=q,events=co['events'])
    # Outcomes enter after covariance, quadrature and scores are frozen.
    sample=co['sample'];y=np.load(H/'validation_labels.npy')[sample];w=co['sample_event_weight'];ar=co['arrival'][sample];hg=co['history_group'][sample]
    masks=[('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',ar==k) for k in range(4)]+[('arrival3plus',ar>=2)]+[(f'history{k}',hg==k) for k in range(3)]
    results=[]
    with threadpool_limits(limits=1):
        for name,mask in masks:
            z=metric.compare(y,control[sample],candidate[sample],w,mask);z['group']=name;results.append(z)
    passed=all(z['gain']>=.0002 for z in results[:5]);full=[]
    if passed:
        fy=np.load(H/'validation_labels.npy')
        for name,mask in [('overall',np.ones(len(fy),bool))]+[(f'arrival{k+1}',co['arrival']==k) for k in range(4)]:
            z=metric.compare(fy,control,candidate,co['event_weight'],mask);z['group']=name;full.append(z)
    protected=json.loads((H/'protected_before.json').read_text());assert all(metric.sha(B/k)==v['sha256'] for k,v in protected.items())
    out=dict(status='complete',screen=results,screen_passed=passed,full_confirmation=full,
        confirmation_passed=bool(full) and all(z['gain']>=.0005 for z in full),seconds=time.monotonic()-tic,
        population_eigenvalues=np.linalg.eigvalsh(pop).tolist(),median_posterior_trace=float(np.median(np.trace(covariance,axis1=1,axis2=2))),
        fit_info=info,event_click_mass_preserved=True,user_mean_factors_unchanged=True,
        fixed_point_model=True,no_validation_label_fitting=True,protected_files_unchanged=len(protected),
        holdout_scored=False,evaluation_scored=False,submission_created=False,
        scope='Conditional local uncertainty diagnostic on simplified horizon control, not fullV53/nxa_h1 or full Bayesian training')
    (R/'report.json').write_text(json.dumps(out,indent=2));print('UNCERTAINTY_RESULT_BEGIN',json.dumps(out),'UNCERTAINTY_RESULT_END',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();set_num_threads(2)
    with threadpool_limits(limits=2):pilot() if a.pilot else run()
