"""Causal dense factor calibration on rolling training days."""
from imp4_decompose import *
from imp11_bilinear_core import loss_gradient,adam_step
import argparse


def pack(name,rows,cut):
    destination=ROOT/f'imp11_data_{name}.npz'
    if destination.exists():
        with np.load(destination) as saved:
            assert int(saved['cutoff'])==cutoff_of(cut)
            assert np.array_equal(saved['events'],rows[::10,6])
            return {k:saved[k] for k in ['u','h','v','offset','extra','target','events']}
    with np.load(model_path(cut)) as model:
        U,V,H,bu,bi=[model[k].astype(np.float64) for k in ['U','V','H','bu','bi']]
    assert U.shape[1]==V.shape[1]==H.shape[1]==16
    def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    with np.load(model_path(2)) as model:anchor=unit(model['V'].astype(np.float64))
    l,s,rt=np.linalg.svd(unit(V).T@anchor,full_matrices=False);rotation=l@rt
    assert np.max(np.abs(rotation.T@rotation-np.eye(16)))<1e-10
    e=rows[::10];ids=rows[:,1].reshape(-1,10)
    assert np.all(rows[:,6].reshape(-1,10)==e[:,6,None])
    histories=np.asarray(hist[e[:,6]]);valid=histories>=0
    stamps=np.asarray(np.load(ROOT/'train_times.npy',mmap_mode='r')[e[:,6]])
    assert np.all((stamps<e[:,2,None])|~valid)
    raw_h=(H[np.maximum(histories,0)]*valid[:,:,None]).sum(1)
    profile=U[e[:,0]]+raw_h/np.sqrt(np.maximum(valid.sum(1)[:,None],1))
    offset=np.clip(-1.8+bu[e[:,0],None]+bi[ids]+np.einsum('ed,ecd->ec',profile,V[ids]),-30,30)
    extra=np.stack([np.ones_like(offset),np.broadcast_to(bu[e[:,0],None],offset.shape),bi[ids]],axis=2)
    labels=rows[:,5].reshape(-1,10);assert np.all(labels.sum(1)<=1)
    target=np.where(labels.sum(1)>0,labels.argmax(1),10).astype(np.int32)
    result=dict(u=unit(U[e[:,0]]@rotation).astype(np.float32),h=unit(raw_h@rotation).astype(np.float32),
                v=unit(V@rotation)[ids].astype(np.float32),offset=offset.astype(np.float32),
                extra=extra.astype(np.float32),target=target,events=e[:,6].copy())
    assert all(np.isfinite(value).all() for value in result.values())
    np.savez(destination,**result,cutoff=cutoff_of(cut),rotation=rotation)
    print('Packed causal bilinear data',name,len(e),flush=True)
    return result


def predict(theta,data):
    out=[]
    for off in range(0,len(data['u']),4096):
        sl=slice(off,off+4096)
        dummy=np.full(len(data['u'][sl]),10,np.int32)
        _,_,p=loss_gradient(theta,*[data[k][sl] for k in ['u','h','v','offset','extra']],dummy)
        out.append(p)
    return np.concatenate(out).ravel()


def fit(day,baseline_prefix,chosen=None):
    training=[]
    for d in [day-3,day-2,day-1]:
        rows=np.asarray(train[(train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1))])
        training.append(pack(f'day{d}',rows,d))
    data={k:np.concatenate([part[k] for part in training]) for k in training[0]};del training;gc.collect()
    _,rows=split(day);validation=pack(f'val{day}',rows,day)
    refpath=ROOT/f'predictions_{baseline_prefix}_day{day}.npz'
    if baseline_prefix=='imp10_leafwise':refpath=ROOT/f'imp10_leafwise_day{day}.npz'
    ref=np.load(refpath)
    assert np.array_equal(ref['events'],rows[:,6]) and np.array_equal(ref['y'],rows[:,5])
    y=ref['y'];base=ref['pred'];sw,_=history_weights(ref['events'])
    old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    theta=np.zeros(515,np.float32);first=theta.copy();second=theta.copy();step=0
    rng=np.random.default_rng(42);best=None;trials=[]
    for epoch in range(1,(chosen['epoch'] if chosen else 8)+1):
        order=rng.permutation(len(data['u']));losses=[];t=time.time()
        for off in range(0,len(order),1024):
            ix=order[off:off+1024]
            loss,gradient,_=loss_gradient(theta,*[data[k][ix] for k in ['u','h','v','offset','extra','target']])
            step+=1;adam_step(theta,gradient,first,second,step);losses.append(loss)
        print('Bilinear epoch',day,epoch,'loss',float(np.mean(losses)),'seconds',time.time()-t,flush=True)
        if (chosen is None and epoch not in [2,4,6,8]) or (chosen is not None and epoch!=chosen['epoch']):continue
        raw=predict(theta,validation)
        assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
        assert np.all(raw.reshape(-1,10).sum(1)<=1+1e-6)
        for weight in ([chosen['weight']] if chosen else [.1,.25,.5]):
            pred=(1-weight)*base+weight*raw
            auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
            result=dict(day=day,epoch=epoch,weight=weight,auc=auc,reweighted=weighted,
                        baseline_auc=old,baseline_reweighted=oldw,raw_auc=float(roc_auc_score(y,raw)),baseline_prefix=baseline_prefix)
            trials.append(result);print(json.dumps(result),flush=True)
            if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
                best=result.copy();se=np.linalg.norm(influence(y,pred)-influence(y,base))
                sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
                best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
                np.savez_compressed(ROOT/f'imp11_bilinear_day{day}.npz',y=y,events=ref['events'],raw=raw,pred=pred)
                np.savez_compressed(ROOT/f'imp11_model_day{day}.npz',theta=theta,first=first,second=second,epoch=epoch,step=step,cutoff=cutoff_of(day))
        (ROOT/f'imp11_bilinear_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    return best


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--baseline-prefix',required=True);args=ap.parse_args()
    output=ROOT/'imp11_bilinear_report.json';assert not output.exists(),'Do not restart a completed experiment.'
    with threadpool_limits(limits=4):
        selected=fit(6,args.baseline_prefix);reports=[selected];output.write_text(json.dumps(reports,indent=2))
        if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
            reports.append(fit(5,args.baseline_prefix,selected));output.write_text(json.dumps(reports,indent=2))
    print('Bilinear experiment complete; independent audit required for any gain.',flush=True)
