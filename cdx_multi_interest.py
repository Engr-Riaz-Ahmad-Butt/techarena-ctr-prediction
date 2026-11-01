"""Fixed two-interest diagnostic using existing warm16 factors, no model fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,time,argparse,math
from pathlib import Path
import numpy as np
from numba import njit,set_num_threads
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import cdx_policy_hpf as metrics
B,A,H=metrics.B,metrics.A,metrics.H
R=H/'cdx_multi_interest_001'

@njit(cache=True)
def mixture(users,items,history,U,V,Hh,bu,bi,control):
    out=control.copy();split=np.zeros(len(users),np.int8)
    for e in range(len(users)):
        n=0
        for q in range(20):
            if history[e,q]>=0:n+=1
        if n<2:continue
        ids=np.sort(history[e,:n]);d=U.shape[1];x=np.empty((n,d))
        for j in range(n):
            for k in range(d):x[j,k]=Hh[ids[j],k]
        # Farthest-pair initialization, independent of original history ordering.
        far=0.;left=0;right=0
        for a in range(n):
            for b in range(a):
                dist=0.
                for k in range(d):dist+=(x[a,k]-x[b,k])**2
                if dist>far:far=dist;left=a;right=b
        if far<1e-16:continue
        centers=np.empty((2,d));centers[0]=x[left];centers[1]=x[right]
        assignment=np.full(n,-1,np.int64);counts=np.zeros(2,np.int64)
        for iteration in range(6):
            new=np.empty(n,np.int64);sums=np.zeros((2,d));counts[:]=0
            for j in range(n):
                dist0=0.;dist1=0.
                for k in range(d):
                    dist0+=(x[j,k]-centers[0,k])**2;dist1+=(x[j,k]-centers[1,k])**2
                c=0 if dist0<=dist1 else 1;new[j]=c;counts[c]+=1
                for k in range(d):sums[c,k]+=x[j,k]
            if counts[0]==0 or counts[1]==0:break
            changed=False
            for j in range(n):
                if new[j]!=assignment[j]:changed=True
                assignment[j]=new[j]
            for c in range(2):
                for k in range(d):centers[c,k]=sums[c,k]/counts[c]
            if not changed:break
        if counts[0]==0 or counts[1]==0:continue
        # Weighted component profiles conserve the original mean preference vector.
        for k in range(d):
            mean=0.
            for j in range(n):mean+=x[j,k]/n
            assert abs((counts[0]*centers[0,k]+counts[1]*centers[1,k])/n-mean)<1e-9
        conditional=np.zeros(10)
        for c in range(2):
            logits=np.empty(10);maximum=-1e100
            for j in range(10):
                item=items[e,j];value=-1.8+bu[users[e]]+bi[item]
                for k in range(d):value+=(U[users[e],k]+math.sqrt(n)*centers[c,k])*V[item,k]
                value=max(-30.,min(30.,value));logits[j]=value;maximum=max(maximum,value)
            denom=0.
            for j in range(10):logits[j]=math.exp(logits[j]-maximum);denom+=logits[j]
            for j in range(10):conditional[j]+=(counts[c]/n)*logits[j]/denom
        mass=0.
        for j in range(10):mass+=control[e,j]
        for j in range(10):out[e,j]=mass*conditional[j]
        split[e]=1
    return out,split

def arrays():
    p=np.load(H/'cdx_collective_001/frozen_predictions.npz');raw=p['raw_control']
    model=dict(np.load(H/'cdx_collective_001/control_model.npz'))
    rows=np.load(H/'score_rows.npy',mmap_mode='r');hist=np.load(A/'train_hist.npy',mmap_mode='r')
    u=rows[::10,0].copy();i=rows[:,1].reshape(-1,10).copy();hh=np.asarray(hist[rows[::10,6]]).copy()
    times=np.load(A/'train_times.npy',mmap_mode='r')[rows[::10,6]]
    assert np.all((times<rows[::10,2,None])|(hh<0))
    assert np.all(rows[:,5]==-1)
    return u,i,hh,[model[k] for k in ['U','V','H','bu','bi']],raw

def pilot():
    R.mkdir(exist_ok=True);assert not (R/'frozen.json').exists()
    plan=dict(date='2026-10-30',hypothesis='A user can have two coherent interests hidden by the single history mean',
        source='https://arxiv.org/abs/1904.08030',source_scope='Multiple user-representation motivation only; own static diagnostic, NOT MIND/capsule/attention reproduction',
        algorithm='Two deterministic Euclidean history-H centroids; sorted item IDs, farthest-pair initialization, six Lloyd steps, count-based weights',
        inference='Mix conditional10-way item probabilities across component profiles; multiply by UNCHANGED original event-click mass',
        conservation='Weighted component profile equals original user+sqrt(n)*history_mean; n<2 or identical vectors unchanged',
        model='Same archived bit-exact warm16 cutoff3 model; all parameters frozen; no training or global/new user-vector update',
        bridge='Original95%joint-horizon warm-score rank+5%frozen ALS rank',
        gate='20%9878users:+.0002 overall/all4arrivals; only survivor full+.0005 overall/all4',
        no_new_head=True,no_order_attention=True,no_search=True,no_holdout=True,no_evaluation_statistics=True,no_submission=True,
        hashes={str(p.relative_to(B)):metrics.sha(p) for p in [Path(__file__),H/'cdx_collective_001/control_model.npz',
          H/'cdx_collective_001/frozen_predictions.npz',H/'cohort.npz']})
    (R/'frozen.json').open('x').write(json.dumps(plan,indent=2))
    u,i,h,ps,p=arrays();mixture(u[:2],i[:2],h[:2],*ps,p[:2])
    tic=time.monotonic();q,split=mixture(u[:1024],i[:1024],h[:1024],*ps,p[:1024]);elapsed=time.monotonic()-tic
    assert np.allclose(q.sum(1),p[:1024].sum(1),atol=1e-12)
    result=dict(stage='label_free_timing_pilot',seconds=elapsed,estimate_seconds=elapsed*len(u)/1024*2+15,events=1024,labels_loaded=False)
    (R/'pilot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def run():
    assert not (R/'report.json').exists();plan=json.loads((R/'frozen.json').read_text())
    assert all(metrics.sha(B/p)==v for p,v in plan['hashes'].items())
    tic=time.monotonic();u,i,h,ps,p=arrays();q,split=mixture(u,i,h,*ps,p)
    assert np.isfinite(q).all() and (q>=0).all() and np.allclose(q.sum(1),p.sum(1),atol=1e-12)
    assert np.array_equal(p[split==0],q[split==0])
    # Sequence order is irrelevant; candidate order merely permutes scores.
    rev=h[:128].copy()
    for k in range(len(rev)):
        n=int((rev[k]>=0).sum());rev[k,:n]=rev[k,:n][::-1]
    test,_=mixture(u[:128],i[:128],rev,*ps,p[:128]);assert np.allclose(test,q[:128],atol=1e-12)
    perm=np.array([9,1,7,2,5,8,0,4,6,3]);test,_=mixture(u[:128],i[:128,perm],h[:128],*ps,p[:128,perm])
    assert np.allclose(test[:,np.argsort(perm)],q[:128],atol=1e-12)
    previous=np.load(H/'cdx_collective_001/frozen_predictions.npz');control=previous['control']
    legacy=np.load(H/'matched_refresh_001/predictions.npz');n=len(p)*10
    candidate=(.95*rankdata(q.ravel())/(n+1)+.05*rankdata(legacy['control_als'])/(n+1)).reshape(-1,10)
    co=np.load(H/'cohort.npz');np.savez_compressed(R/'frozen_predictions.npz',control=control,candidate=candidate,raw=q,split=split,events=co['events'])
    sample=co['sample'];y=np.load(H/'validation_labels.npy')[sample];w=co['sample_event_weight'];ar=co['arrival'][sample];hg=co['history_group'][sample]
    masks=[('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',ar==k) for k in range(4)]
    masks += [('arrival3plus',ar>=2)]+[(f'history{k}',hg==k) for k in range(3)]
    results=[]
    with threadpool_limits(limits=1):
        for name,mask in masks:
            z=metrics.compare(y,control[sample],candidate[sample],w,mask);z['group']=name;results.append(z)
    passed=all(x['gain']>=.0002 for x in results[:5]);full=[]
    if passed:
        fy=np.load(H/'validation_labels.npy')
        for name,mask in [('overall',np.ones(len(fy),bool))]+[(f'arrival{k+1}',co['arrival']==k) for k in range(4)]:
            z=metrics.compare(fy,control,candidate,co['event_weight'],mask);z['group']=name;full.append(z)
    protected=json.loads((H/'protected_before.json').read_text());assert all(metrics.sha(B/k)==v['sha256'] for k,v in protected.items())
    result=dict(status='complete',screen=results,screen_passed=passed,full_confirmation=full,
        confirmation_passed=bool(full) and all(z['gain']>=.0005 for z in full),seconds=time.monotonic()-tic,
        split_events=int(split.sum()),total_events=len(p),event_click_mass_preserved=True,
        short_or_degenerate_unchanged=True,history_order_invariant=True,candidate_order_equivariant=True,
        protected_files_unchanged=len(protected),holdout_scored=False,evaluation_scored=False,submission_created=False,
        scope='No-training two-centroid representation diagnostic; not fullV53/nxa_h1 or full MIND training')
    (R/'report.json').write_text(json.dumps(result,indent=2));print('MULTI_RESULT_BEGIN',json.dumps(result),'MULTI_RESULT_END',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();set_num_threads(2)
    with threadpool_limits(limits=2):pilot() if a.pilot else run()
