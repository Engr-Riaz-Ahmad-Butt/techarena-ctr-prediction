"""Fallback: causal elapsed-time activity and contextual warm-history profiles."""
import imp5_aligned as experiment
from imp4_decompose import *


def context_features(name,rows,histories,cut):
    dest=FEAT/f'imp5_timecontext_{name}.npy'
    if dest.exists():
        x=np.load(dest,mmap_mode='r');assert x.shape==(len(rows),32);return x
    times=np.load(ROOT/('evaluation_times.npy' if name=='eval' else 'train_times.npy'),mmap_mode='r')
    with np.load(model_path(cut)) as source:
        V,H=[source[k] for k in ['V','H']]
    x=np.empty((len(rows),32),dtype=np.float32)
    for off in range(0,len(rows),20000):
        a=rows[off:off+20000];event=a[::10]
        h=np.asarray(histories[event[:,6]]);ts=np.asarray(times[event[:,6]])
        valid=h>=0;n=valid.sum(1);safe=np.maximum(h,0)
        assert np.all((ts<event[:,2,None])|~valid)
        hours=np.maximum(event[:,2,None].astype(float)-ts,0)/3600
        last=np.where(valid,hours,1e6).min(1);oldest=np.where(valid,hours,0).max(1)
        mean=(hours*valid).sum(1)/np.maximum(n,1);span=np.maximum(oldest-last,0)
        gap=span/np.maximum(n-1,1)
        current_hour=(event[:,2]%86400)/3600;history_hour=(ts%86400)/3600
        distance=np.abs(current_hour[:,None]-history_hour);distance=np.minimum(distance,24-distance)
        samebucket=valid&((ts%86400//21600)==(event[:,2,None]%86400//21600))
        # Unix epoch was Thursday, weekday indices Monday=0.
        weekend=((ts//86400+3)%7)>=5
        current_weekend=((event[:,2]//86400+3)%7)>=5
        activity=np.column_stack([np.log1p(last),np.log1p(oldest),np.log1p(mean),
            np.log1p(span),np.log1p(gap),samebucket.sum(1)/np.maximum(n,1),
            ((weekend==current_weekend[:,None])&valid).sum(1)/np.maximum(n,1),
            np.log1p(n/np.maximum(oldest,1))])
        hv=H[safe]*valid[:,:,None];v=V[a[:,1].reshape(-1,10)]
        vn=np.linalg.norm(v,axis=2);parts=[np.repeat(activity,10,axis=0)]
        for weight in [np.exp2(-hours/6)*valid,np.exp2(-hours/24)*valid,
                       samebucket.astype(float),np.exp(-.5*(distance/3)**2)*valid]:
            z=(hv*weight[:,:,None]).sum(1)/np.sqrt(np.maximum((weight*weight).sum(1,keepdims=True),1e-12))
            dot=np.einsum('ed,ecd->ec',z,v)
            cosine=dot/np.maximum(np.linalg.norm(z,axis=1)[:,None]*vn,1e-12)
            mass=np.broadcast_to(np.log1p(weight.sum(1))[:,None],dot.shape)
            repeat=((h[:,None,:]==a[:,1].reshape(-1,10,1))*weight[:,None,:]).sum(2)
            parts.append(np.stack([dot,cosine,mass,repeat],axis=2).reshape(-1,4))
        raw=np.column_stack(parts);relative=[]
        for col in [8,12,16,20]:
            values=raw[:,col].reshape(-1,10)
            relative.extend([(values-values.mean(1,keepdims=True)).ravel(),
                             (values-values.max(1,keepdims=True)).ravel()])
        x[off:off+len(a)]=np.column_stack([raw,*relative])
    assert np.isfinite(x).all()
    np.save(dest,x);print('Time-context features',name,x.shape,flush=True)
    return x


if __name__=='__main__':
    experiment.TAG='timecontext'
    experiment.aligned_features=context_features
    report=ROOT/'imp5_aligned_report.json'
    audit=ROOT/'imp5_test_aligned_report.json'
    if report.exists() and audit.exists():
        prior=json.loads(report.read_text());checked=json.loads(audit.read_text())['reports']
        if ({r['day'] for r in prior}=={5,6} and {r['day'] for r in checked}=={5,6}
            and all(r['delta']>0 and r['delta_reweighted']>0 for r in prior)
            and all(r[k]['delta']>0 for r in checked for k in ['auc','weighted_auc'])
            and next(r for r in prior if r['day']==6)['weighted_paired_interval'][0]>0):
            experiment.SELECTION_FLOOR=next(r for r in prior if r['day']==6)['reweighted']
    print('Context selection floor',experiment.SELECTION_FLOOR,flush=True)
    experiment.run_aligned()
