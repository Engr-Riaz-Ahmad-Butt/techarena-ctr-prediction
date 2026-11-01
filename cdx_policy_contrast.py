"""Own contrastive logging-policy fit: shown items versus known catalog noise.

Targets exposure density, never clicks. No imputed warm-up exposures. The novel
target differs from the closed click-label NCE experiments and one-step fixed-space fit.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,pickle,time
import numpy as np
from numba import njit,set_num_threads
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import cdx_policy_hpf as common

B,A,H=common.B,common.A,common.H
R=H/'cdx_policy_contrast_001'
K,EPOCHS,SEED,LR,ANCHOR=32,3,8231,.03,.01

@njit(cache=True)
def sigmoid(x):
    if x>=0:return 1/(1+np.exp(-x))
    z=np.exp(x);return z/(1+z)

@njit(cache=True)
def softplus(x):
    if x>0:return x+np.log1p(np.exp(-x))
    return np.log1p(np.exp(x))

@njit(cache=True)
def epoch(u,i,order,noise,w,v,bu,w0,v0,bias,aw,av,ab,lr,anchor):
    loss=0.
    for step in range(len(order)):
        e=order[step];user=u[e];pos=i[e];neg=noise[step]
        sp=bu[user]+bias[pos];sn=bu[user]+bias[neg]
        for k in range(w.shape[1]):
            sp+=w[user,k]*v[pos,k];sn+=w[user,k]*v[neg,k]
        gp=sigmoid(sp)-1;gn=sigmoid(sn)
        loss+=softplus(-sp)+softplus(sn)
        gb=gp+gn+anchor*bu[user]
        ab[user]+=gb*gb;bu[user]-=lr*gb/np.sqrt(ab[user])
        for k in range(w.shape[1]):
            old=w[user,k];vp=v[pos,k];vn=v[neg,k]
            gu=gp*vp+gn*vn+anchor*(old-w0[user,k])
            aw[user,k]+=gu*gu;w[user,k]-=lr*gu/np.sqrt(aw[user,k])
            gvp=gp*old+anchor*(vp-v0[pos,k])
            gvn=gn*old+anchor*(vn-v0[neg,k])
            if pos==neg:
                g=gvp+gvn;av[pos,k]+=g*g;v[pos,k]-=lr*g/np.sqrt(av[pos,k])
            else:
                av[pos,k]+=gvp*gvp;v[pos,k]-=lr*gvp/np.sqrt(av[pos,k])
                av[neg,k]+=gvn*gvn;v[neg,k]-=lr*gvn/np.sqrt(av[neg,k])
    return loss/(2*len(order))

def initialize(svd,x):
    w=svd['U'][:,:K].copy();v=svd['V'][:,:K].copy()
    w/=np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-8)
    v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
    count=np.asarray(x.sum(0)).ravel()
    bias=np.log((count+1)/(count.sum()+len(count))*len(count))
    return dict(W=w,V=v,bu=np.zeros(len(w)),W0=w.copy(),V0=v.copy(),bias=bias,
        aw=np.full_like(w,.1),av=np.full_like(v,.1),ab=np.full(len(w),.1),
        activity=np.asarray(x.sum(1)).ravel())

def update(state,u,i,order,noise):
    return epoch(u,i,order,noise,*[state[k] for k in
        ['W','V','bu','W0','V0','bias','aw','av','ab']],LR,ANCHOR)

def gradient_check():
    # Independent finite-difference objective and optimizer check, including a noise/data collision.
    rng=np.random.default_rng(9)
    for collision in [False,True]:
        w=rng.normal(0,.2,(1,3));v=rng.normal(0,.2,(2,3));bu=np.array([.1])
        w0=w-.03;v0=v+.02;bias=np.array([.3,-.4]);j=0 if collision else 1
        def objective(ww,vv,bb):
            p=bb[0]+bias[0]+ww[0]@vv[0];n=bb[0]+bias[j]+ww[0]@vv[j]
            reg=((ww[0]-w0[0])**2).sum()+((vv[0]-v0[0])**2).sum()+((vv[j]-v0[j])**2).sum()+bb[0]**2
            return np.logaddexp(0,-p)+np.logaddexp(0,n)+ANCHOR*.5*reg
        grads=[]
        for arr in [w,v,bu]:
            g=np.zeros_like(arr)
            for ix in np.ndindex(arr.shape):
                old=arr[ix];arr[ix]=old+1e-6;p=objective(w,v,bu)
                arr[ix]=old-1e-6;n=objective(w,v,bu);arr[ix]=old;g[ix]=(p-n)/2e-6
            grads.append(g)
        expected=[z-LR*g/np.sqrt(.1+g*g) for z,g in zip([w,v,bu],grads)]
        aw=np.full_like(w,.1);av=np.full_like(v,.1);ab=np.full_like(bu,.1)
        epoch(np.array([0]),np.array([0]),np.array([0]),np.array([j]),
            w,v,bu,w0,v0,bias,aw,av,ab,LR,ANCHOR)
        assert all(np.allclose(z,e,atol=1e-10) for z,e in zip([w,v,bu],expected))
    return True

def positives(cut):
    t=np.load(A/'train.npy',mmap_mode='r')
    n=int(np.searchsorted(t[::10,2],cut))*10
    triples=np.asarray(t[:n,:3])
    assert triples[:,2].max()<cut
    return triples[:,0].copy(),triples[:,1].copy()

def fit(svd,x,cut,tag):
    s=initialize(svd,x);u,i=positives(cut);rng=np.random.default_rng(SEED);logs=[]
    tic=time.monotonic()
    for ep in range(EPOCHS):
        order=rng.permutation(len(u));noise=rng.integers(0,len(s['V']),len(u),dtype=np.int64)
        loss=update(s,u,i,order,noise)
        assert np.isfinite(loss) and np.isfinite(s['W']).all() and np.isfinite(s['V']).all()
        row=dict(stage='policy_contrast_training',block=tag,epoch=ep+1,loss=loss,seconds=time.monotonic()-tic)
        logs.append(row);print(json.dumps(row),flush=True)
    return s,logs

def raw(s,u,i):
    # Separate the learned personal term from item popularity and user normalization.
    z=np.einsum('ed,ejd->ej',s['W'][u],s['V'][i])
    z[s['activity'][u]==0]=0
    return z

def earlier_events(t,start,cut):
    first=int(np.searchsorted(t[::10,2],start+2*86400));last=int(np.searchsorted(t[::10,2],cut))
    head=np.asarray(t[first*10:last*10:10,:3]);_,ix=np.unique(head[:,0],return_index=True)
    ix=ix[(head[ix,0].astype(np.int64)*2654435761)%5==0]
    ev=first+ix
    return head[ix,0],np.asarray(t[ev[:,None]*10+np.arange(10),1])

def pilot():
    R.mkdir(exist_ok=True);assert not (R/'frozen.json').exists()
    plan=dict(hypothesis='Jointly learned signed logging-policy factors may recover targeting preferences missed by spectral or positive-mixture models',
        source='https://proceedings.mlr.press/v9/gutmann10a.html',
        source_scope='NCE density-ratio principle; own conditional exposure factor model and optimizer, not source code or proof of gains',
        target='Actual shown items versus independently sampled uniform catalog noise; these are density-class labels, never click or inferred warm-up exposure labels',
        novelty='Logging-policy target, jointly retrained user AND item factors, rather than closed click-NCE objectives or previous fixed-item one-step user fit',
        factors=K,epochs=EPOCHS,seed=SEED,lr=LR,anchor=ANCHOR,initial_accumulator=.1,
        initialization='Cutoff-matched normalized 32D SVD user/item vectors',noise_per_observed_impression=1,
        bias='Fixed smoothed marginal exposure/noise log ratio; learned user normalizer; infer personal dot term only',
        matched_control='Exactly the previously frozen HPF screen SVD control, unchanged; historical day2 CDF, same.015 score bridge',
        candidate='Replace only exposure expert by standardized historical-CDF contrastive policy personal score',
        screen='9878 frozen users; +.0002 overall/all4arrivals vs both SVD and original baseline',
        confirmation='Only survivor receives full permitted cohort+.0005 overall/all4arrivals',
        no_new_head=True,no_click_labels_in_fit=True,no_evaluation_statistics=True,no_holdout=True,no_submission=True,
        hashes={str(p.relative_to(B)):common.sha(p) for p in [Path(__file__),Path(common.__file__),H/'cohort.npz',
            H/'cdx_policy_hpf_001/screen_predictions.npz',H/'cdx_policy_hpf_001/prefix3_models.pkl']})
    (R/'frozen.json').open('x').write(json.dumps(plan,indent=2))
    assert gradient_check()
    t,start,cut,nu,ni=common.setup()
    model=pickle.loads((H/'cdx_policy_hpf_001/prefix3_models.pkl').read_bytes())
    x,_=common.exposures(cut,nu,ni);s=initialize(model['svd'],x);u,i=positives(cut)
    rng=np.random.default_rng(SEED);n=min(100000,len(u));order=rng.choice(len(u),n,replace=False)
    noise=rng.integers(0,ni,n,dtype=np.int64);tic=time.monotonic();loss=update(s,u,i,order,noise)
    seconds=time.monotonic()-tic
    estimate=seconds*(len(u)*2*EPOCHS/n)*1.5+60
    out=dict(stage='training_only_pilot',gradient_check_passed=True,examples=n,seconds=seconds,
        training_density_loss=loss,estimated_total_seconds=estimate,validation_scored=False)
    (R/'pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

def run():
    assert not (R/'report.json').exists()
    plan=json.loads((R/'frozen.json').read_text())
    assert all(common.sha(B/p)==v for p,v in plan['hashes'].items())
    assert json.loads((R/'pilot.json').read_text())['estimated_total_seconds']<1800
    tic=time.monotonic();t,start,cut,nu,ni=common.setup()
    x2,info2=common.exposures(start+2*86400,nu,ni);svd2=common.train_svd(x2)
    s2,logs2=fit(svd2,x2,start+2*86400,'prefix2')
    ru,ri=earlier_events(t,start,cut);reference=np.sort(raw(s2,ru,ri).ravel())
    del s2,svd2,x2
    stored=pickle.loads((H/'cdx_policy_hpf_001/prefix3_models.pkl').read_bytes())
    x3,info3=common.exposures(cut,nu,ni);s3,logs3=fit(stored['svd'],x3,cut,'prefix3')
    co=np.load(H/'cohort.npz');sample=co['sample'];rows=np.load(H/'score_rows.npy',mmap_mode='r')
    u=np.asarray(rows[::10,0][sample]);i=np.asarray(rows[:,1].reshape(-1,10)[sample])
    assert len(np.unique(u))==len(u)==9878 and np.all(rows[:,5]==-1)
    signal=common.centered(common.cdf(reference,raw(s3,u,i)))
    previous=np.load(H/'cdx_policy_hpf_001/screen_predictions.npz')
    base=previous['baseline'];control=previous['control'];candidate=base+.015*signal
    assert np.isfinite(candidate).all()
    np.savez_compressed(R/'screen_predictions.npz',events=co['events'][sample],baseline=base,
        control=control,candidate=candidate,signal=signal)
    (R/'model.pkl').write_bytes(pickle.dumps(dict(state=s3,reference=reference,plan=plan)))
    # Fixed scores precede validation outcome access.
    y=np.load(H/'validation_labels.npy')[sample];w=co['sample_event_weight']
    arrival=co['arrival'][sample];history=co['history_group'][sample]
    masks=[('overall',np.ones(len(u),bool))]+[(f'arrival{k+1}',arrival==k) for k in range(4)]
    masks+=[('arrival3plus',arrival>=2)]+[(f'history{k}',history==k) for k in range(3)]
    metrics=[]
    with threadpool_limits(limits=1):
        for name,mask in masks:
            m=common.compare(y,control,candidate,w,mask);m['group']=name
            m['original_baseline_auc']=float(roc_auc_score(y[mask].ravel(),base[mask].ravel(),sample_weight=np.repeat(w[mask],10)))
            m['gain_vs_original_baseline']=m['candidate_auc']-m['original_baseline_auc'];metrics.append(m)
    passed=all(m['gain']>=.0002 and m['gain_vs_original_baseline']>=.0002 for m in metrics[:5])
    protected=json.loads((H/'protected_before.json').read_text())
    assert all(common.sha(B/p)==v['sha256'] for p,v in protected.items())
    out=dict(stage='screen_complete',screen=metrics,screen_passed=passed,seconds=time.monotonic()-tic,
        fit_audits=[info2,info3],training_logs={'prefix2':logs2,'prefix3':logs3},gradient_check_passed=True,
        protected_files_unchanged=len(protected),no_validation_label_fit=True,no_epoch_or_scale_selection=True,
        holdout_scored=False,evaluation_scored=False,submission_created=False,full_confirmation_run=False,
        baseline_scope='Same simplified horizon/SVD exposure bridge, not full V53/nxa_h1 or official scoring')
    (R/'report.json').write_text(json.dumps(out,indent=2));print('CONTRAST_SCREEN_JSON_BEGIN',json.dumps(out),'CONTRAST_SCREEN_JSON_END',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):pilot() if args.pilot else run()
