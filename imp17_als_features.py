"""Causal exposure-ALS scalar features for the tree learner."""
from imp15_als import *


def ensure_model(cut):
    existing=ROOT/f'imp15_model_day{cut}.npz'
    destination=existing if cut in [5,6] else ROOT/f'imp17_als_cut{cut}.npz'
    if destination.exists():
        with np.load(destination) as model:
            assert int(model['cutoff'])==cutoff_of(cut) and int(model['iteration'])==6
        return destination
    w,t,wt,tt,g,up,ip,confidence,exposures=prepare(cut)
    with np.load(model_path(cut)) as initial:V=initial['V'].astype(np.float64)
    V/=np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-12)
    U=np.zeros((nu,16));bu=up.copy();bi=ip.copy()
    with threadpool_limits(limits=1):
        for iteration in range(6):
            x=augmented(V)
            bu,U=solve_rows(w.indptr,w.indices,w.data,t.data,V,bi,x.T@x,-x.T@(g+bi),confidence,g,up)
            x=augmented(U)
            bi,V=solve_rows(wt.indptr,wt.indices,wt.data,tt.data,U,bu,x.T@(confidence[:,None]*x),-x.T@(confidence*(g+bu)),np.ones(ni),g,ip)
    np.savez_compressed(destination,U=U,V=V,bu=bu,bi=bi,g=g,exposures=exposures,cutoff=cutoff_of(cut),iteration=6)
    print('Causal ALS checkpoint',cut,flush=True)
    return destination


def als_features(name,rows,histories,cut):
    path=FEAT/f'imp17_als_{name}.npy'
    if path.exists():
        x=np.load(path,mmap_mode='r');assert x.shape==(len(rows),32);return x
    with np.load(ensure_model(cut)) as model:
        U,V,bu,bi,exposures=[model[k] for k in ['U','V','bu','bi','exposures']];g=float(model['g'])
    item_exposures=np.bincount(train[train[:,2]<cutoff_of(cut),1],minlength=ni)
    history_times=np.load(ROOT/('evaluation_times.npy' if name=='eval' else 'train_times.npy'),mmap_mode='r')
    vn_all=np.linalg.norm(V,axis=1);unit=V/np.maximum(vn_all[:,None],1e-12)
    result=np.empty((len(rows),32),np.float32)
    for off in range(0,len(rows),20000):
        a=rows[off:off+20000];e=a[::10];items=a[:,1].reshape(-1,10)
        assert np.all(a[:,6].reshape(-1,10)==e[:,6,None])
        h=np.asarray(histories[e[:,6]]);valid=h>=0;n=valid.sum(1);safe=np.maximum(h,0)
        times=np.asarray(history_times[e[:,6]])
        assert np.all((times<e[:,2,None])|~valid)
        u,bias=score(a,U,V,bu,bi,g,exposures,return_profile=True,histories=histories,history_times=history_times)
        v=V[items];un=np.linalg.norm(u,axis=1);vn=vn_all[items]
        direct=np.einsum('ed,ecd->ec',u,v)
        profile=(V[safe]*valid[:,:,None]).sum(1)/np.maximum(n[:,None],1)
        hn=np.linalg.norm(profile,axis=1);history_dot=np.einsum('ed,ecd->ec',profile,v)
        sim=np.einsum('ehd,ecd->ech',unit[safe],unit[items])*valid[:,None,:]
        mean=sim.sum(2)/np.maximum(n[:,None],1)
        maximum=np.where(valid[:,None,:],sim,-1e6).max(2);maximum[n==0]=0
        std=np.sqrt(np.maximum((sim*sim).sum(2)/np.maximum(n[:,None],1)-mean*mean,0))
        age=n[:,None]-1-np.arange(20)[None,:];recent=[]
        for width in [3,5]:
            mask=valid&(age<width)
            recent.append((sim*mask[:,None,:]).sum(2)/np.maximum(mask.sum(1)[:,None],1))
        def event(x):return np.broadcast_to(x[:,None],direct.shape)
        values=[event(bias),bi[items],direct,event(un),vn,direct/np.maximum(un[:,None]*vn,1e-12),
                g+event(bias)+bi[items]+direct,event(np.log1p(exposures[e[:,0]])),history_dot,event(hn),
                history_dot/np.maximum(hn[:,None]*vn,1e-12),maximum,std,*recent,
                (h[:,None,:]==items[:,:,None]).sum(2),event(np.log1p(n)),np.log1p(item_exposures[items])]
        scalar=np.stack(values,axis=2).reshape(-1,18);relative=[]
        for col in [1,2,5,6,8,10,11]:
            value=scalar[:,col].reshape(-1,10)
            relative.extend([(value-value.mean(1,keepdims=True)).ravel(),(value-value.max(1,keepdims=True)).ravel()])
        result[off:off+len(a)]=np.column_stack([scalar,*relative])
    assert np.isfinite(result).all()
    np.save(path,result);print('ALS features',name,result.shape,flush=True)
    return result
