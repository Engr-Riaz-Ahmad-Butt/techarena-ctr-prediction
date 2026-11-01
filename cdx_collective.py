"""Matched original warm16 versus shared-item exposure/click multitask learning.

Existing project kernels are loaded as definitions only, avoiding import side effects
and evaluation-file reads. No searches, original-file changes or submissions.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import ast,math,json,pickle,time,argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from numba import njit,set_num_threads
from threadpoolctl import threadpool_limits
import cdx_policy_hpf as metrics

B=Path(__file__).resolve().parent;A=B/'artifacts';H=A/'v54_horizon';R=H/'cdx_collective_001'
POLICY_WEIGHT=.1;EPOCHS=10;SEED=151;POLICY_SEED=8531

def definitions():
    ns=dict(np=np,math=math,njit=njit,ROOT=A,pickle=pickle,normalize=normalize,
            hist=np.load(A/'train_hist.npy',mmap_mode='r'),__name__=__name__)
    for source,names in [('choice64_core.py',['train_epoch','predict_events','event_data']),
                         ('imp2_warm_choice.py',['warm_epoch','warm_data','init32'])]:
        tree=ast.parse((B/source).read_text())
        nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
        assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(B/source),'exec'),ns)
    return ns

@njit(cache=True)
def sigmoid(x):
    if x>=0:return 1/(1+np.exp(-x))
    z=np.exp(x);return z/(1+z)

@njit(cache=True)
def softplus(x):
    return x+np.log1p(np.exp(-x)) if x>0 else np.log1p(np.exp(x))

@njit(cache=True)
def exposure_epoch(order,users,items,noise,E,V,bu,bias,ae,av,ab,share,weight):
    loss=0.
    for k0 in range(len(order)):
        e=order[k0];u=users[e];i=items[e];j=noise[k0]
        zp=bu[u]+bias[i];zn=bu[u]+bias[j]
        for k in range(V.shape[1]):
            zp+=E[u,k]*V[i,k];zn+=E[u,k]*V[j,k]
        gp=weight*(sigmoid(zp)-1);gn=weight*sigmoid(zn)
        loss+=softplus(-zp)+softplus(zn)
        gb=gp+gn+weight*.002*bu[u]
        ab[u]+=gb*gb;bu[u]-=.045*gb/math.sqrt(ab[u])
        for k in range(V.shape[1]):
            eu=E[u,k];vi=V[i,k];vj=V[j,k]
            ge=gp*vi+gn*vj+weight*.002*eu
            ae[u,k]+=ge*ge;E[u,k]-=.045*ge/math.sqrt(ae[u,k])
            if share:
                gi=gp*eu;gj=gn*eu
                if i==j:
                    g=gi+gj;av[i,k]+=g*g;V[i,k]-=.045*g/math.sqrt(av[i,k])
                else:
                    av[i,k]+=gi*gi;V[i,k]-=.045*gi/math.sqrt(av[i,k])
                    av[j,k]+=gj*gj;V[j,k]-=.045*gj/math.sqrt(av[j,k])
    return loss/max(2*len(order),1)

def kernel_checks():
    rng=np.random.default_rng(4)
    for collision in [False,True]:
        e=rng.normal(0,.2,(1,3));v=rng.normal(0,.2,(2,3));bu=np.array([.1]);bias=np.array([.3,-.2]);j=0 if collision else 1
        def objective():
            return POLICY_WEIGHT*(np.logaddexp(0,-(bu[0]+bias[0]+e[0]@v[0]))
                +np.logaddexp(0,bu[0]+bias[j]+e[0]@v[j])+.001*((e[0]**2).sum()+bu[0]**2))
        gradients=[]
        for z in [e,v,bu]:
            g=np.zeros_like(z)
            for ix in np.ndindex(z.shape):
                old=z[ix];z[ix]=old+1e-6;plus=objective();z[ix]=old-1e-6;minus=objective();z[ix]=old
                g[ix]=(plus-minus)/2e-6
            gradients.append(g)
        expected=[z-.045*g/np.sqrt(.1+g*g) for z,g in zip([e,v,bu],gradients)]
        ef,vf,bf=e.copy(),v.copy(),bu.copy();vf0=vf.copy()
        args=[np.array([0]),np.array([0]),np.array([0]),np.array([j])]
        exposure_epoch(*args,e,v,bu,bias,np.full_like(e,.1),np.full_like(v,.1),np.full_like(bu,.1),True,POLICY_WEIGHT)
        assert all(np.allclose(z,x,atol=1e-10) for z,x in zip([e,v,bu],expected))
        exposure_epoch(*args,ef,vf,bf,bias,np.full_like(ef,.1),np.full_like(vf,.1),np.full_like(bf,.1),False,POLICY_WEIGHT)
        assert np.array_equal(vf,vf0)
        assert np.allclose(ef,e,atol=1e-12) and np.allclose(bf,bu,atol=1e-12)
    # Zero auxiliary weight must leave all parameters and accumulators unchanged.
    es=np.ones((1,2));vs=np.ones((2,2));bs=np.ones(1)
    ae=np.full_like(es,.1);av=np.full_like(vs,.1);ab=np.full_like(bs,.1)
    before=[x.copy() for x in [es,vs,bs,ae,av,ab]]
    exposure_epoch(np.array([0]),np.array([0]),np.array([0]),np.array([1]),es,vs,bs,np.zeros(2),ae,av,ab,True,0.)
    assert all(np.array_equal(x,y) for x,y in zip(before,[es,vs,bs,ae,av,ab]))
    return True

def data(ns):
    t=np.load(A/'train.npy',mmap_mode='r');start=int(t[0,2]);cut=start+3*86400
    n=int(np.searchsorted(t[::10,2],cut))*10;past=np.asarray(t[:n])
    u,i,h,y=ns['event_data'](past,ns['hist']);wu,wi=ns['warm_data'](past)
    stamps=np.load(A/'train_times.npy',mmap_mode='r')[past[::10,6]]
    assert np.all((stamps<past[::10,2,None])|(h<0))
    assert np.array_equal(h>=0,np.arange(20)[None,:]<(h>=0).sum(1)[:,None])
    ps,acc=ns['init32'](start+2*86400,16)
    ni=len(ps[1]);nu=len(ps[0]);x=sp.csr_matrix((np.ones(n),(past[:,0],past[:,1])),shape=(nu,ni))
    profile=x@ps[1];ee=(.25*normalize(profile)).astype(np.float32)
    counts=np.asarray(x.sum(0)).ravel();bias=np.log((counts+1)/(counts.sum()+ni)*ni).astype(np.float32)
    pop=np.bincount(wi,minlength=ni)+1;pop=np.log(pop/pop.mean()).astype(np.float32)
    return dict(u=u,i=i,h=h,y=y,wu=wu,wi=wi,pop=pop,ps=ps,acc=acc,
        eu=past[:,0].copy(),ei=past[:,1].copy(),E=ee,eb=np.zeros(nu,np.float32),bias=bias,
        cut=cut,start=start,max_timestamp=int(past[:,2].max()))

def extra(d):
    return [d['E'].copy(),d['eb'].copy(),np.full_like(d['E'],.1),np.full_like(d['eb'],.1)]

def warm_batch(rng,d,ns,ps,acc,order_limit=None):
    neg=rng.integers(0,len(ps[1]),size=(len(d['wu']),11),dtype=np.int32)
    coll=neg==d['wi'][:,None]
    while coll.any():
        neg[coll]=rng.integers(0,len(ps[1]),int(coll.sum()),dtype=np.int32);coll=neg==d['wi'][:,None]
    order=rng.permutation(len(d['wu']))
    if order_limit:order=order[:order_limit]
    return ns['warm_epoch'](order,d['wu'],d['wi'],neg,ps[0],ps[1],acc[0],acc[1],.045,.002,1.,d['pop'])

def freeze():
    R.mkdir(exist_ok=True);assert not (R/'frozen.json').exists()
    files=[Path(__file__),B/'choice64_core.py',B/'imp2_warm_choice.py',A/'preference_stats_1737504000.pkl',
           A/'imp3_model_warm16_cut3.npz',H/'cohort.npz',H/'matched_refresh_001/predictions.npz']
    plan=dict(date='2026-10-30',hypothesis='Shared item representation transfers logging-policy supervision into click prediction',
        source='https://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf',
        scope='Collective-representation principle; original adaptation, not paper reproduction or evidence of competition gains',
        novelty='Exposure gradients update the CLICK item factors while original warm-positive and actual10+no-click losses remain unchanged; prior standalone experts did not transfer gradients',
        main_model='Exact existing warm16 functions,16factors,10epochs,seed151,lr.045,reg.002,warmweight1,pop-corrected',
        initialization='Prefix2 original statistics strictly before cutoff3; full prefix0..2 fits both arms',
        auxiliary='Observed pre-cutoff impression versus independent uniform catalog noise; weight.1; seed8531',
        matched_control='Identical auxiliary-user head trained with shared item updates DISABLED; main parameters exactly original recipe',
        candidate='Enable only auxiliary data gradients to shared click-item V; no extra shared-V regularizer',
        score='Same95%rank(warm expert)+5%rank(frozen ALS), joint rank over permitted complete horizon',
        scoring='All horizon FEATURES scored for joint ranks; only9878 screen outcomes assessed; no holdout in cohort',
        gate='+.0002 overall AND each4arrivaldays; only survivor full+.0005 all; no holdout',
        no_new_head=True,no_seed_or_weight_search=True,no_evaluation_statistics=True,no_submission=True,
        hashes={str(p.relative_to(B)):metrics.sha(p) for p in files})
    (R/'frozen.json').open('x').write(json.dumps(plan,indent=2));return plan

def pilot():
    freeze();assert kernel_checks();ns=definitions();d=data(ns);ps=d['ps'];acc=d['acc'];ex=extra(d)
    rng=np.random.default_rng(SEED);prng=np.random.default_rng(POLICY_SEED)
    # Compile the existing project kernels separately from timing.
    ns['train_epoch'](np.array([0]),d['u'],d['i'],d['h'],d['y'],*ps,*acc,.045,.002)
    ns['warm_epoch'](np.array([0]),d['wu'],d['wi'],np.zeros((len(d['wu']),11),np.int32),ps[0],ps[1],acc[0],acc[1],.045,.002,1.,d['pop'])
    exposure_epoch(np.array([0]),d['eu'],d['ei'],np.array([1],np.int32),ex[0],ps[1],ex[1],d['bias'],ex[2],acc[1],ex[3],True,POLICY_WEIGHT)
    tic=time.monotonic();warm_batch(rng,d,ns,ps,acc,20000);wt=time.monotonic()-tic
    tic=time.monotonic();ns['train_epoch'](rng.permutation(len(d['u']))[:10000],d['u'],d['i'],d['h'],d['y'],*ps,*acc,.045,.002);ct=time.monotonic()-tic
    order=prng.permutation(len(d['eu']))[:100000];noise=prng.integers(0,len(ps[1]),len(order),dtype=np.int32)
    tic=time.monotonic();exposure_epoch(order,d['eu'],d['ei'],noise,ex[0],ps[1],ex[1],d['bias'],ex[2],acc[1],ex[3],True,POLICY_WEIGHT);et=time.monotonic()-tic
    ep=wt*len(d['wu'])/20000+ct*len(d['u'])/10000+et*len(d['eu'])/100000
    out=dict(stage='training_only_pilot',warm_seconds=wt,event_seconds=ct,exposure_seconds=et,
        estimate_seconds=ep*EPOCHS*2*1.4+30,training_events=len(d['u']),warm_pairs=len(d['wu']),
        actual_exposure_pairs=len(d['eu']),kernel_checks=True,validation_scored=False)
    (R/'pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

def run():
    assert not (R/'report.json').exists();plan=json.loads((R/'frozen.json').read_text())
    assert all(metrics.sha(B/p)==v for p,v in plan['hashes'].items())
    assert json.loads((R/'pilot.json').read_text())['estimate_seconds']<1800
    ns=definitions();d=data(ns);tic=time.monotonic();raws={};logs={};control_exact=False
    rows=np.load(H/'score_rows.npy',mmap_mode='r');co=np.load(H/'cohort.npz')
    assert np.all(rows[:,5]==-1) and rows[::10,2].min()>=d['cut']>d['max_timestamp']
    vu=rows[::10,0].copy();vi=rows[:,1].reshape(-1,10).copy();vh=np.asarray(ns['hist'][rows[::10,6]]).copy()
    for name,share in [('control',False),('collective',True)]:
        ps=[p.copy() for p in d['ps']];acc=[p.copy() for p in d['acc']];ex=extra(d)
        rng=np.random.default_rng(SEED);prng=np.random.default_rng(POLICY_SEED);logs[name]=[]
        for epoch in range(EPOCHS):
            st=time.monotonic();wl=warm_batch(rng,d,ns,ps,acc)
            cl=ns['train_epoch'](rng.permutation(len(d['u'])),d['u'],d['i'],d['h'],d['y'],*ps,*acc,.045,.002)
            order=prng.permutation(len(d['eu']));noise=prng.integers(0,len(ps[1]),len(order),dtype=np.int32)
            el=exposure_epoch(order,d['eu'],d['ei'],noise,ex[0],ps[1],ex[1],d['bias'],ex[2],acc[1],ex[3],share,POLICY_WEIGHT)
            entry=dict(arm=name,epoch=epoch+1,warm_loss=wl,event_loss=cl,exposure_loss=el,seconds=time.monotonic()-st)
            logs[name].append(entry);print(json.dumps(entry),flush=True)
        saved=dict(zip(['U','V','H','bu','bi'],ps));np.savez_compressed(R/f'{name}_model.npz',**saved)
        if name=='control':
            archived=np.load(A/'imp3_model_warm16_cut3.npz')
            differences={k:float(np.max(np.abs(saved[k]-archived[k]))) for k in saved}
            control_exact=all(v==0 for v in differences.values())
            (R/'control_reproduction.json').write_text(json.dumps(dict(exact=control_exact,max_differences=differences),indent=2))
            assert control_exact,'Control does not reproduce original warm16; stop before validation metrics.'
        raws[name]=ns['predict_events'](vu,vi,vh,*ps)
    legacy=np.load(H/'matched_refresh_001/predictions.npz');als_rank=rankdata(legacy['control_als'])/(len(rows)+1)
    control=.95*rankdata(raws['control'].ravel())/(len(rows)+1)+.05*als_rank
    candidate=.95*rankdata(raws['collective'].ravel())/(len(rows)+1)+.05*als_rank
    assert np.isfinite(candidate).all() and np.array_equal(rows[::10,6],co['events'])
    np.savez_compressed(R/'frozen_predictions.npz',control=control.reshape(-1,10),candidate=candidate.reshape(-1,10),
        raw_control=raws['control'],raw_candidate=raws['collective'],events=co['events'])
    sample=co['sample'];y=np.load(H/'validation_labels.npy')[sample];w=co['sample_event_weight']
    cb=control.reshape(-1,10)[sample];cp=candidate.reshape(-1,10)[sample]
    ar=co['arrival'][sample];hg=co['history_group'][sample]
    masks=[('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',ar==k) for k in range(4)]
    masks += [('arrival3plus',ar>=2)]+[(f'history{k}',hg==k) for k in range(3)]
    results=[]
    with threadpool_limits(limits=1):
        for name,mask in masks:
            result=metrics.compare(y,cb,cp,w,mask);result['group']=name;results.append(result)
    passed=all(z['gain']>=.0002 for z in results[:5]);full=[]
    if passed:
        fy=np.load(H/'validation_labels.npy')
        for name,mask in [('overall',np.ones(len(fy),bool))]+[(f'arrival{k+1}',co['arrival']==k) for k in range(4)]:
            m=metrics.compare(fy,control.reshape(-1,10),candidate.reshape(-1,10),co['event_weight'],mask);m['group']=name;full.append(m)
    protected=json.loads((H/'protected_before.json').read_text());assert all(metrics.sha(B/p)==v['sha256'] for p,v in protected.items())
    result=dict(status='complete',screen=results,screen_passed=passed,full_confirmation=full,
        confirmation_passed=bool(full) and all(z['gain']>=.0005 for z in full),training_logs=logs,
        control_reproduced_original_bit_exact=control_exact,seconds=time.monotonic()-tic,
        cutoff=d['cut'],max_fit_timestamp=d['max_timestamp'],protected_files_unchanged=len(protected),
        heldout_metrics_used_for_epoch_or_weight_selection=False,holdout_scored=False,evaluation_scored=False,submission_created=False,
        scope='Matched original warm16/simplified horizon control; not full V53/nxa_h1 or an official gain')
    (R/'report.json').write_text(json.dumps(result,indent=2));print('COLLECTIVE_RESULT_BEGIN',json.dumps(result),'COLLECTIVE_RESULT_END',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();set_num_threads(2)
    with threadpool_limits(limits=2):pilot() if a.pilot else run()
