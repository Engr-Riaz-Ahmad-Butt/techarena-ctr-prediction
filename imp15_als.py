"""Exposure-aware alternating ridge factors, selected on unchanged full cohorts."""
from imp4_decompose import *
from imp2_warm_choice import warm_data,nu
from imp15_als_core import solve_rows
from scipy.sparse import coo_matrix
from scipy.stats import rankdata
from numba import set_num_threads
set_num_threads(4)


def paired_matrices(users,items,weights,positives,shape):
    w=coo_matrix((weights,(users,items)),shape=shape).tocsr()
    t=coo_matrix((positives,(users,items)),shape=shape).tocsr()
    assert np.array_equal(w.indptr,t.indptr) and np.array_equal(w.indices,t.indices)
    return w,t


def prepare(day):
    rows=np.asarray(train[train[:,2]<cutoff_of(day)])
    wu,wi=warm_data(rows);g=float(rows[:,5].mean())
    ue=np.bincount(rows[:,0],minlength=nu);ie=np.bincount(rows[:,1],minlength=ni)
    uc=np.bincount(rows[:,0],weights=rows[:,5],minlength=nu)
    ic=np.bincount(rows[:,1],weights=rows[:,5],minlength=ni)
    up=(uc+50*g)/(ue+50)-g;ip=(ic+50*g)/(ie+50)-g
    warm=np.bincount(wu,minlength=nu)
    users=np.concatenate([rows[:,0],wu]);items=np.concatenate([rows[:,1],wi])
    weights=np.ones(len(users),np.float32);positives=np.concatenate([rows[:,5].astype(np.float32),np.ones(len(wu),np.float32)])
    w,t=paired_matrices(users,items,weights,positives,(nu,ni))
    wt=w.T.tocsr();tt=t.T.tocsr()
    assert np.array_equal(wt.indptr,tt.indptr) and np.array_equal(wt.indices,tt.indices)
    print('ALS prefix',day,'exposure rows',len(rows),'warm positives',len(wu),'unique pairs',w.nnz,flush=True)
    return w,t,wt,tt,g,up,ip,9.*warm/ni,ue


def augmented(factors):return np.column_stack([np.ones(len(factors)),factors])


def score(rows,U,V,bu,bi,g,exposures,return_profile=False,histories=None,history_times=None):
    e=rows[::10];users=e[:,0];items=rows[:,1].reshape(-1,10)
    u=U[users].copy();bias=bu[users].copy();cold=exposures[users]==0
    if cold.any():
        if histories is None:histories=hist
        if history_times is None:history_times=np.load(ROOT/'train_times.npy',mmap_mode='r')
        selected=e[cold];h=np.asarray(histories[selected[:,6]])
        stamps=np.asarray(history_times[selected[:,6]])
        assert np.all((stamps<selected[:,2,None])|(h<0))
        valid=h>=0;local=np.repeat(np.arange(len(h)),20)[valid.ravel()]
        target_items=h.ravel()[valid.ravel()]
        keys=np.rec.fromarrays([local,target_items,stamps.ravel()[valid.ravel()]],names='row,item,timestamp')
        _,keep=np.unique(keys,return_index=True)
        local=local[keep];target_items=target_items[keep];values=np.ones(len(local),np.float32)
        w,t=paired_matrices(local,target_items,values,values,(len(h),ni))
        x=augmented(V);gram=x.T@x;rhs=-x.T@(g+bi)
        count=np.bincount(local,minlength=len(h))
        cb,cu=solve_rows(w.indptr,w.indices,w.data,t.data,V,bi,gram,rhs,9.*count/ni,g,np.zeros(len(h)))
        u[cold]=cu;bias[cold]=cb
    if return_profile:return u,bias
    return (g+bias[:,None]+bi[items]+np.einsum('ed,ecd->ec',u,V[items])).ravel()


def fit(day,chosen=None):
    w,t,wt,tt,g,up,ip,confidence,exposures=prepare(day)
    with np.load(model_path(day)) as initial:
        V=initial['V'].astype(np.float64)
    assert V.shape==(ni,16)
    V/=np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-12)
    U=np.zeros((nu,16));bu=up.copy();bi=ip.copy()
    _,val=split(day);ref=np.load(ROOT/f'imp11_bilinear_day{day}.npz')
    assert np.array_equal(ref['events'],val[:,6]) and np.array_equal(ref['y'],val[:,5])
    y=ref['y'];base=ref['pred'];sw,_=history_weights(ref['events'])
    baseline_rank=rankdata(base,method='average')/(len(base)+1)
    old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    best=None;trials=[]
    for iteration in range(1,(chosen['iteration'] if chosen else 6)+1):
        started=time.time();x=augmented(V)
        bu,U=solve_rows(w.indptr,w.indices,w.data,t.data,V,bi,x.T@x,-x.T@(g+bi),confidence,g,up)
        x=augmented(U)
        bi,V=solve_rows(wt.indptr,wt.indices,wt.data,tt.data,U,bu,x.T@(confidence[:,None]*x),-x.T@(confidence*(g+bu)),np.ones(ni),g,ip)
        assert all(np.isfinite(p).all() for p in [U,V,bu,bi])
        print('ALS iteration',day,iteration,'seconds',time.time()-started,flush=True)
        if (chosen is None and iteration not in [2,4,6]) or (chosen is not None and iteration!=chosen['iteration']):continue
        raw=score(val,U,V,bu,bi,g,exposures)
        raw_rank=rankdata(raw,method='average')/(len(raw)+1)
        for weight in ([chosen['weight']] if chosen else [.05,.1,.2]):
            pred=(1-weight)*baseline_rank+weight*raw_rank
            auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
            result=dict(day=day,iteration=iteration,weight=weight,auc=auc,reweighted=weighted,baseline_auc=old,baseline_reweighted=oldw,raw_auc=float(roc_auc_score(y,raw)))
            trials.append(result);print(json.dumps(result),flush=True)
            if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
                best=result.copy();se=np.linalg.norm(influence(y,pred)-influence(y,base))
                sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
                best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
                np.savez_compressed(ROOT/f'imp15_als_day{day}.npz',y=y,events=ref['events'],raw=raw,pred=pred)
                np.savez_compressed(ROOT/f'imp15_model_day{day}.npz',U=U,V=V,bu=bu,bi=bi,g=g,exposures=exposures,cutoff=cutoff_of(day),iteration=iteration)
        (ROOT/f'imp15_als_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    return best


if __name__=='__main__':
    output=ROOT/'imp15_als_report.json';assert not output.exists(),'Do not restart a completed experiment.'
    with threadpool_limits(limits=1):
        selected=fit(6);reports=[selected];output.write_text(json.dumps(reports,indent=2))
        if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
            reports.append(fit(5,selected));output.write_text(json.dumps(reports,indent=2))
    print('ALS experiment complete; independent audit required for any gain.',flush=True)
