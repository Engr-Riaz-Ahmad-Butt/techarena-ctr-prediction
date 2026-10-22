"""Context-dependent dense latent corrections on the unchanged validation cohort."""
from imp11_bilinear import *
from imp12_context_core import context_loss_gradient

KEYS=['u','h','v','offset','extra','target','groups']


def context_pack(name,rows,cut):
    data=pack(name,rows,cut)
    e=rows[::10]
    assert np.array_equal(e[:,3],e[:,2]%86400//21600)
    assert np.array_equal(e[:,4],((e[:,2]//86400+3)%7>=5).astype(np.int32))
    data['groups']=(e[:,3]+4*e[:,4]).astype(np.int32)
    return data


def context_predict(theta,data):
    out=[]
    for off in range(0,len(data['u']),4096):
        sl=slice(off,off+4096);dummy=np.full(len(data['u'][sl]),10,np.int32)
        _,_,p=context_loss_gradient(theta,*[data[k][sl] for k in KEYS[:5]],dummy,data['groups'][sl])
        out.append(p)
    return np.concatenate(out).ravel()


def fit_context(day,chosen=None):
    parts=[]
    for d in [day-3,day-2,day-1]:
        rows=np.asarray(train[(train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1))])
        parts.append(context_pack(f'day{d}',rows,d))
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};del parts;gc.collect()
    _,rows=split(day);validation=context_pack(f'val{day}',rows,day)
    ref=np.load(ROOT/f'imp11_bilinear_day{day}.npz')
    assert np.array_equal(ref['events'],rows[:,6]) and np.array_equal(ref['y'],rows[:,5])
    y=ref['y'];base=ref['pred'];sw,_=history_weights(ref['events'])
    old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    theta=np.zeros((9,515),np.float32);first=theta.copy();second=theta.copy();step=0
    rng=np.random.default_rng(42);best=None;trials=[]
    for epoch in range(1,(chosen['epoch'] if chosen else 8)+1):
        order=rng.permutation(len(data['u']));losses=[];t=time.time()
        for off in range(0,len(order),1024):
            ix=order[off:off+1024]
            value,gradient,_=context_loss_gradient(theta,*[data[k][ix] for k in KEYS])
            step+=1;adam_step(theta,gradient,first,second,step);losses.append(value)
        print('Context bilinear epoch',day,epoch,'loss',float(np.mean(losses)),'seconds',time.time()-t,flush=True)
        if (chosen is None and epoch not in [2,4,6,8]) or (chosen is not None and epoch!=chosen['epoch']):continue
        raw=context_predict(theta,validation)
        assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
        assert np.all(raw.reshape(-1,10).sum(1)<=1+1e-6)
        for weight in ([chosen['weight']] if chosen else [.1,.25,.5]):
            pred=(1-weight)*base+weight*raw
            auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
            result=dict(day=day,epoch=epoch,weight=weight,auc=auc,reweighted=weighted,
                        baseline_auc=old,baseline_reweighted=oldw,raw_auc=float(roc_auc_score(y,raw)))
            trials.append(result);print(json.dumps(result),flush=True)
            if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
                best=result.copy();se=np.linalg.norm(influence(y,pred)-influence(y,base))
                sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
                best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
                np.savez_compressed(ROOT/f'imp12_context_day{day}.npz',y=y,events=ref['events'],raw=raw,pred=pred)
                np.savez_compressed(ROOT/f'imp12_model_day{day}.npz',theta=theta,first=first,second=second,epoch=epoch,step=step,cutoff=cutoff_of(day))
        (ROOT/f'imp12_context_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    return best


if __name__=='__main__':
    output=ROOT/'imp12_context_report.json';assert not output.exists(),'Do not restart a completed experiment.'
    with threadpool_limits(limits=4):
        selected=fit_context(6);reports=[selected];output.write_text(json.dumps(reports,indent=2))
        if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
            reports.append(fit_context(5,selected));output.write_text(json.dumps(reports,indent=2))
    print('Context experiment complete; independent audit required for any gain.',flush=True)
