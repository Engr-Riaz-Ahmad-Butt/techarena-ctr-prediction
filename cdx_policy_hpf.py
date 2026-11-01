"""Label-free hierarchical Poisson exposure model versus frozen-form SVD control.

Own implementation from the Gamma-Poisson variational equations, not repository code.
Only user/item/time triples enter either fit. No click labels or warm-up negatives.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import argparse, hashlib, json, pickle, time
import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from numba import njit, prange, set_num_threads
from threadpoolctl import threadpool_limits

B = Path(__file__).resolve().parent
A = B / 'artifacts'
H = A / 'v54_horizon'
R = H / 'cdx_policy_hpf_001'
K, ITERATIONS, SEED, SCALE = 32, 80, 8221, .015

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

@njit(parallel=True, cache=True)
def edge_dot(u, i, w, h):
    out = np.empty(len(u), np.float64)
    for e in prange(len(u)):
        z = 0.
        for k in range(w.shape[1]):
            z += w[u[e],k] * h[i[e],k]
        out[e] = z
    return out

def exposures(cut, nu, ni):
    t = np.load(A/'train.npy', mmap_mode='r')
    n = int(np.searchsorted(t[::10,2], cut)) * 10
    triples = np.asarray(t[:n,:3])  # no outcome column is available to the model
    assert triples[:,2].max() < cut
    x = sp.csr_matrix((np.ones(n), (triples[:,0], triples[:,1])), shape=(nu,ni))
    x.sum_duplicates()
    assert x.sum() == n
    return x, dict(cutoff=int(cut), max_source_timestamp=int(triples[:,2].max()),
                   observed_impressions=n, nonzero_user_item_pairs=int(x.nnz), fit_columns=[0,1,2])

class PoissonExposure:
    def __init__(self, x):
        self.x = x
        self.u = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        self.i = x.indices.astype(np.int64)
        self.ratio = x.copy()
        nu, ni = x.shape
        rng = np.random.default_rng(SEED)
        scale = np.sqrt(x.sum() / (nu * ni * K))
        self.gs = .3 + rng.uniform(.5,1.5,(nu,K))
        self.ls = .3 + rng.uniform(.5,1.5,(ni,K))
        self.gr = self.gs / (scale * rng.uniform(.5,1.5,(nu,K)))
        self.lr = self.ls / (scale * rng.uniform(.5,1.5,(ni,K)))
        self.ks = .3 + K*.3
        self.ts = .3 + K*.3
        self.kr = .3 + (self.gs/self.gr).sum(1)
        self.tr = .3 + (self.ls/self.lr).sum(1)

    def step(self):
        f = np.exp(digamma(self.gs)-np.log(self.gr))
        g = np.exp(digamma(self.ls)-np.log(self.lr))
        normalizer = edge_dot(self.u, self.i, f, g)
        assert np.all(normalizer > 0) and np.isfinite(normalizer).all()
        self.ratio.data = self.x.data / normalizer
        new_gs = .3 + f * (self.ratio @ g)
        new_ls = .3 + g * (self.ratio.T @ f)
        # Latent topic responsibilities must allocate each observed count exactly once.
        mass = self.x.sum()
        assert abs((new_gs-.3).sum()/mass-1) < 1e-8
        assert abs((new_ls-.3).sum()/mass-1) < 1e-8
        self.gr = self.ks/self.kr[:,None] + (self.ls/self.lr).sum(0)[None,:]
        self.gs = new_gs
        self.kr = .3 + (self.gs/self.gr).sum(1)
        self.lr = self.ts/self.tr[:,None] + (self.gs/self.gr).sum(0)[None,:]
        self.ls = new_ls
        self.tr = .3 + (self.ls/self.lr).sum(1)

    @staticmethod
    def entropy(s,r):
        return np.sum(s-np.log(r)+gammaln(s)+(1-s)*digamma(s))

    def elbo(self):
        w, h = self.gs/self.gr, self.ls/self.lr
        lw, lh = digamma(self.gs)-np.log(self.gr), digamma(self.ls)-np.log(self.lr)
        ek, et = self.ks/self.kr, self.ts/self.tr
        lk, lt = digamma(self.ks)-np.log(self.kr), digamma(self.ts)-np.log(self.tr)
        z = edge_dot(self.u,self.i,np.exp(lw),np.exp(lh))
        observation = self.x.data @ np.log(z) - w.sum(0) @ h.sum(0) - gammaln(self.x.data+1).sum()
        def prior(s,r,k,lk,ek):
            n = len(s)
            lp = K*.3*lk.sum() - n*K*gammaln(.3) - .7*(digamma(s)-np.log(r)).sum()
            lp -= ek @ (s/r).sum(1)
            hierarchy = n*(.3*np.log(.3)-gammaln(.3)) - .7*lk.sum() - .3*ek.sum()
            return lp + hierarchy
        return float(observation + prior(self.gs,self.gr,self.ks,lk,ek)
            + prior(self.ls,self.lr,self.ts,lt,et) + self.entropy(self.gs,self.gr)
            + self.entropy(self.ls,self.lr) + self.entropy(self.ks,self.kr)
            + self.entropy(self.ts,self.tr))

    def state(self):
        return dict(W=self.gs/self.gr, V=self.ls/self.lr,
                    activity=np.asarray(self.x.sum(1)).ravel())

def train_hpf(x, iterations, tag):
    tic = time.monotonic()
    model = PoissonExposure(x)
    previous = model.elbo()
    logs = [dict(iteration=0, elbo=previous)]
    for iteration in range(1,iterations+1):
        model.step()
        if iteration in [1,2,5] or iteration % 10 == 0 or iteration == iterations:
            value = model.elbo()
            assert value >= previous - 1e-5 * max(1,abs(previous)), (previous,value)
            previous = value
            log = dict(stage='hpf_training', block=tag, iteration=iteration, elbo=value,
                       seconds=time.monotonic()-tic)
            logs.append(log)
            print(json.dumps(log),flush=True)
    return model.state(), logs

def train_svd(x):
    # Rebuild the established exposure control: same fixed 64-D SVD and 16/32-D views.
    nu = x.shape[0]
    idf = np.log((1+nu)/(1+np.asarray((x>0).sum(0)).ravel()))
    z = x.multiply(idf).tocsr()
    z.data = np.log1p(z.data)
    z = sp.diags(1/np.maximum(np.sqrt(z.multiply(z).sum(1).A.ravel()),1e-8)) @ z
    model = TruncatedSVD(n_components=64,n_iter=5,random_state=4403)
    us = model.fit_transform(z)
    v = model.components_.T
    state = dict(U=us,V=v,activity=np.asarray(x.sum(1)).ravel(),profiles={})
    for k in [16,32]:
        vv = v[:,:k] / np.maximum(np.linalg.norm(v[:,:k],axis=1,keepdims=True),1e-8)
        profile = x @ vv
        profile /= np.maximum(np.linalg.norm(profile,axis=1,keepdims=True),1e-8)
        state['profiles'][k] = profile
    return state

def raw_svd(state,u,i):
    values=[]
    for k in [16,32]:
        v=state['V'][:,:k]
        v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
        values.append(np.einsum('ed,ejd->ej',state['profiles'][k][u],v[i]))
    v=state['V'][:,:32]
    uu=state['U'][u,:32]
    values.append(np.einsum('ed,ejd->ej',uu,v[i]) / np.maximum(
        np.linalg.norm(uu,axis=1)[:,None]*np.linalg.norm(v[i],axis=2),1e-8))
    return np.asarray(values)

def raw_hpf(state,u,i):
    w,h=state['W'],state['V']
    rate=np.einsum('ed,ejd->ej',w[u],h[i])
    row=w[u]@h.sum(0)
    col=h[i]@w.sum(0)
    total=w.sum(0)@h.sum(0)
    score=np.log(np.maximum(rate,1e-30))+np.log(total)-np.log(row[:,None])-np.log(col)
    score[state['activity'][u]==0]=0
    assert np.isfinite(score).all()
    return score

def cdf(reference,values):
    return (np.searchsorted(reference,values,'left')+np.searchsorted(reference,values,'right'))/(2*len(reference))

def centered(z):
    return (z-z.mean(1,keepdims=True))/np.maximum(z.std(1,keepdims=True),1e-6)

def compare(y,b,p,w,mask):
    yy=y[mask].ravel(); ww=np.repeat(w[mask],10)
    def plan(z):
        order=np.argsort(z[mask].ravel(),kind='stable')
        zz=z[mask].ravel()[order]
        groups=np.r_[0,np.flatnonzero(zz[1:]!=zz[:-1])+1]
        def auc(m):
            tw=ww[order]*m[order]
            pos=np.add.reduceat(tw*yy[order],groups)
            neg=np.add.reduceat(tw*(1-yy[order]),groups)
            return float(pos@(np.cumsum(neg)-.5*neg)/(pos.sum()*neg.sum()))
        return auc
    c,t=plan(b),plan(p)
    n=int(mask.sum()); ones=np.ones(n*10); ca,ta=c(ones),t(ones)
    assert abs(ca-roc_auc_score(yy,b[mask].ravel(),sample_weight=ww))<1e-10
    rng=np.random.default_rng(SEED)
    deltas=[]
    for _ in range(200):
        mult=np.repeat(np.bincount(rng.integers(0,n,n),minlength=n),10)
        deltas.append(t(mult)-c(mult))
    return dict(events=n,control_auc=ca,candidate_auc=ta,gain=ta-ca,
                paired95=np.quantile(deltas,[.025,.975]).tolist())

def setup():
    t=np.load(A/'train.npy',mmap_mode='r')
    start=int(t[0,2]);cut=start+3*86400
    nu=len(np.load(H/'imp3_model_warm16_cut1.npz')['U'])
    return t,start,cut,nu,50000

def freeze():
    R.mkdir(exist_ok=True)
    assert not (R/'frozen.json').exists(),'Recipe already frozen; use the recorded pilot/run, not a new search.'
    plan=dict(hypothesis='A Gamma-Poisson likelihood can recover exposure preferences beyond IDF-log SVD',
        source='https://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf',
        authors='Gopalan, Hofman, Blei (2015)', source_scope='Own variational Gamma-Poisson implementation applied to past impressions, not click consumption; not a competition-result claim',
        factors=K,iterations=ITERATIONS,seed=SEED,shape_hyperparameters=[.3,.3,.3,.3],
        hierarchy_means=[1.,1.],svd_components=64,svd_iterations=5,svd_seed=4403,
        raw_hpf_signal='log posterior-mean exposure intensity / model row mass / model item mass, plus log total mass',
        reference='Earlier day2 first-user events; both exposure models fitted strictly before day2; no labels used for historical CDF',
        control='Frozen simplified horizon blend + .015*within-event standardized SVD exposure expert (historical CDFs of typ16/typ32/cos32 averaged)',
        candidate='Identical frozen base and .015 scale, replacing SVD expert by historical-CDF HPF exposure-lift expert',
        matched='Same users, item triples, cutoff, historical normalization block, score bridge, weights and bootstrap; only exposure model changes',
        baseline_scope='Causal shortened simplified horizon blend; not full V53, not the nxa_h1 head and not an official-score comparison',
        novelty='Probability model of actual shown-item counts and sparse population/user factors; no ignored-click-count feature, click-label MF objective, short-history specialist, new stacker or tuning',
        screening='20% user sample; gain>=.0002 overall and all four arrivals versus BOTH SVD control and original frozen baseline',
        confirmation='Only survivor: frozen models on full permitted cohort, +.0005 overall/all arrivals; no holdout',
        no_evaluation_statistics=True,no_holdout=True,no_submission=True,
        hashes={str(p.relative_to(B)):sha(p) for p in [Path(__file__),H/'cohort.npz',H/'score_rows.npy',A/'train.npy',H/'matched_refresh_001/predictions.npz']})
    (R/'frozen.json').open('x',encoding='utf-8').write(json.dumps(plan,indent=2))
    return plan

def pilot():
    plan=freeze()
    t,start,cut,nu,ni=setup()
    x,info=exposures(cut,nu,ni)
    sp.save_npz(R/'prefix3_exposures.npz',x)
    (R/'prefix3_info.json').write_text(json.dumps(info,indent=2))
    tic=time.monotonic()
    _,logs=train_hpf(x,3,'training_only_pilot')
    elapsed=time.monotonic()-tic
    step=(logs[-1]['seconds']-logs[2]['seconds']) # iteration3 minus iteration2
    estimate=2*ITERATIONS*step*1.3+120
    result=dict(stage='training_only_pilot',seconds=elapsed,last_step_seconds=step,
                estimated_total_seconds=estimate,validation_scored=False,labels_loaded=False)
    (R/'pilot.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

def score_models(svd,hpf,cdfs,u,i):
    raw=raw_svd(svd,u,i)
    svd_score=np.mean([cdf(cdfs['svd'][k],raw[k]) for k in range(3)],axis=0)
    hpf_score=cdf(cdfs['hpf'],raw_hpf(hpf,u,i))
    return centered(svd_score),centered(hpf_score)

def run():
    assert not (R/'report.json').exists(),'Completed experiment must not be repeated.'
    plan=json.loads((R/'frozen.json').read_text())
    assert all(sha(B/p)==v for p,v in plan['hashes'].items())
    assert json.loads((R/'pilot.json').read_text())['estimated_total_seconds']<1800
    tic=time.monotonic()
    t,start,cut,nu,ni=setup()
    # Freeze normalization on earlier first-user events before inspecting validation scores.
    first=int(np.searchsorted(t[::10,2],start+2*86400))
    last=int(np.searchsorted(t[::10,2],cut))
    head=np.asarray(t[first*10:last*10:10,:3])
    _,ix=np.unique(head[:,0],return_index=True)
    ix=ix[(head[ix,0].astype(np.int64)*2654435761)%5==0]
    ref_events=first+ix
    ref_u=head[ix,0]
    ref_i=np.asarray(t[ref_events[:,None]*10+np.arange(10),1])
    x2,info2=exposures(start+2*86400,nu,ni)
    hpf2,log2=train_hpf(x2,ITERATIONS,'prefix2')
    svd2=train_svd(x2)
    r=raw_svd(svd2,ref_u,ref_i)
    cdfs=dict(svd=[np.sort(z.ravel()) for z in r],hpf=np.sort(raw_hpf(hpf2,ref_u,ref_i).ravel()))
    (R/'historical_cdfs.pkl').write_bytes(pickle.dumps(cdfs))
    del x2,hpf2,svd2,r
    x3=sp.load_npz(R/'prefix3_exposures.npz')
    info3=json.loads((R/'prefix3_info.json').read_text())
    hpf3,log3=train_hpf(x3,ITERATIONS,'prefix3')
    svd3=train_svd(x3)
    co=np.load(H/'cohort.npz')
    sample=co['sample']
    rows=np.load(H/'score_rows.npy',mmap_mode='r')
    assert np.all(rows[:,5]==-1)
    u=np.asarray(rows[::10,0][sample])
    i=np.asarray(rows[:,1].reshape(-1,10)[sample])
    assert len(np.unique(u))==len(u)==9878
    zc,zt=score_models(svd3,hpf3,cdfs,u,i)
    base=np.load(H/'matched_refresh_001/predictions.npz')['control_blend'].reshape(-1,10)[sample]
    control=base+SCALE*zc
    candidate=base+SCALE*zt
    assert np.isfinite(candidate).all() and np.isfinite(control).all()
    np.savez_compressed(R/'screen_predictions.npz',events=co['events'][sample],baseline=base,
        control=control,candidate=candidate,svd_signal=zc,hpf_signal=zt)
    (R/'prefix3_models.pkl').write_bytes(pickle.dumps(dict(svd=svd3,hpf=hpf3,cdfs=cdfs,plan=plan)))
    # Only after freezing predictions are validation outcomes used for assessment.
    y=np.load(H/'validation_labels.npy')[sample]
    w=co['sample_event_weight'];arrival=co['arrival'][sample];history=co['history_group'][sample]
    masks=[('overall',np.ones(len(u),bool))]+[(f'arrival{k+1}',arrival==k) for k in range(4)]
    masks += [('arrival3plus',arrival>=2)]+[(f'history{k}',history==k) for k in range(3)]
    metrics=[]
    with threadpool_limits(limits=1):
        for name,mask in masks:
            m=compare(y,control,candidate,w,mask)
            m['group']=name
            m['original_baseline_auc']=float(roc_auc_score(y[mask].ravel(),base[mask].ravel(),sample_weight=np.repeat(w[mask],10)))
            m['gain_vs_original_baseline']=m['candidate_auc']-m['original_baseline_auc']
            metrics.append(m)
    survives=all(m['gain']>=.0002 and m['gain_vs_original_baseline']>=.0002 for m in metrics[:5])
    protected=json.loads((H/'protected_before.json').read_text())
    assert all(sha(B/p)==m['sha256'] for p,m in protected.items())
    # Scorer accepts no outcomes; adding/mutating a label column cannot enter its inputs.
    zc2,zt2=score_models(svd3,hpf3,cdfs,u[:128],i[:128])
    assert np.allclose(zc2,zc[:128],atol=1e-12) and np.allclose(zt2,zt[:128],atol=1e-12)
    result=dict(stage='screen_complete',screen=metrics,screen_passed=survives,
        seconds=time.monotonic()-tic,training_logs={'prefix2':log2,'prefix3':log3},
        fit_audits=[info2,info3],historical_reference_events=len(ref_events),
        protected_files_unchanged=len(protected),validation_labels_used_for_fitting=False,
        validation_epoch_or_scale_selection=False,holdout_scored=False,evaluation_scored=False,
        submission_created=False,full_confirmation_run=False,
        limitations=['Simplified frozen horizon blend with a fixed exposure bridge; not full V53/nxa_h1.',
                     'Poisson model treats finite-window shown counts as its target, not click labels or causal propensities.',
                     'No validation signal informs the preset80iterations or initialization.'])
    (R/'report.json').write_text(json.dumps(result,indent=2))
    print('HPF_SCREEN_JSON_BEGIN',json.dumps(result),'HPF_SCREEN_JSON_END',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        pilot() if args.pilot else run()
