"""Warm clicked-context priors on top of all V19 features; target local0.83."""
import imp5_aligned as engine
from imp4_decompose import *

_aligned_features=engine.aligned_features
_validation=engine.aligned_validation


def context_codes(stamps):
    return stamps%86400//21600,(((stamps//86400+3)%7)>=5).astype(np.int32)


def counts(cut):
    dest=ROOT/f'imp6_context_counts_cut{cut}.npz'
    if dest.exists():
        with np.load(dest) as f:return {k:f[k] for k in f.files}
    cutoff=cutoff_of(cut)
    a=np.asarray(train[train[:,2]<cutoff]);events=a[::10]
    bucket,kind=context_codes(events[:,2])
    assert np.array_equal(bucket,events[:,3]) and np.array_equal(kind,events[:,4])
    _,first=np.unique(events[:,0],return_index=True);e=events[first]
    h=np.asarray(hist[e[:,6]])
    ts=np.asarray(np.load(ROOT/'train_times.npy',mmap_mode='r')[e[:,6]])
    valid=h>=0;users=np.broadcast_to(e[:,0,None],h.shape)
    assert np.all((ts<e[:,2,None])|~valid)
    assert np.all((ts<cutoff)|~valid)
    records=np.rec.fromarrays([users[valid],h[valid],ts[valid]],names='user,item,timestamp')
    _,unique=np.unique(records,return_index=True)
    wi=h[valid][unique];wt=ts[valid][unique]
    clicked=a[a[:,5]==1]
    # Warm clicks precede each user's first observed event. Labelled events are
    # separate real events and are retained even if timestamp resolution ties.
    ids=np.concatenate([wi,clicked[:,1]]).astype(np.int64)
    stamps=np.concatenate([wt,clicked[:,2]]).astype(np.int64)
    bucket,kind=context_codes(stamps);joint=bucket*2+kind
    total=np.bincount(ids,minlength=ni)
    byjoint=np.bincount(ids*8+joint,minlength=ni*8).reshape(ni,4,2)
    bybucket=byjoint.sum(2);bykind=byjoint.sum(1)
    assert np.array_equal(total,byjoint.sum((1,2)))
    globaljoint=(byjoint.sum(0)+1)/(len(ids)+8)
    result=dict(total=total,bucket=bybucket,kind=bykind,joint=byjoint,
                globaljoint=globaljoint,globalbucket=globaljoint.sum(1),globalkind=globaljoint.sum(0),
                cutoff=np.array(cutoff),warm_records=np.array(len(wi)),labelled_records=np.array(len(clicked)),
                warm_duplicates_removed=np.array(int(valid.sum())-len(unique)),
                max_click_timestamp=np.array(stamps.max()))
    assert int(result['max_click_timestamp'])<cutoff
    np.savez_compressed(dest,**result)
    print('Context counts',cut,'warm',len(wi),'labelled',len(clicked),'duplicates',int(result['warm_duplicates_removed']),flush=True)
    return result


def context_priors(name,rows,histories,cut):
    dest=FEAT/f'imp6_context_{name}.npy'
    if dest.exists():
        x=np.load(dest,mmap_mode='r');assert x.shape==(len(rows),32);return x
    stats=counts(cut)
    times=np.load(ROOT/('evaluation_times.npy' if name=='eval' else 'train_times.npy'),mmap_mode='r')
    x=np.empty((len(rows),32),np.float32)
    for off in range(0,len(rows),20000):
        a=rows[off:off+20000];e=a[::10]
        assert np.all(a[:,6].reshape(-1,10)==e[:,6,None])
        h=np.asarray(histories[e[:,6]]);ts=np.asarray(times[e[:,6]])
        valid=h>=0;assert np.all((ts<e[:,2,None])|~valid)
        hb,hk=context_codes(ts)
        b=a[:,3];k=a[:,4];i=a[:,1]
        gb=stats['globalbucket'][b];gk=stats['globalkind'][k];gj=stats['globaljoint'][b,k]
        total=stats['total'][i];cb=stats['bucket'][i,b];ck=stats['kind'][i,k];cj=stats['joint'][i,b,k]
        columns=[np.log1p(total),np.log1p(cb),np.log1p(ck),np.log1p(cj)]
        for alpha in [5.,20.,100.]:
            columns.extend([np.log((cb+alpha*gb)/(total+alpha)/gb),
                            np.log((ck+alpha*gk)/(total+alpha)/gk),
                            np.log((cj+alpha*gj)/(total+alpha)/gj)])
        n=valid.sum(1);eb=e[:,3];ek=e[:,4]
        usercounts=[((hb==eb[:,None])&valid).sum(1),((hk==ek[:,None])&valid).sum(1),
                    ((hb==eb[:,None])&(hk==ek[:,None])&valid).sum(1)]
        priors=[stats['globalbucket'][eb],stats['globalkind'][ek],stats['globaljoint'][eb,ek]]
        columns.extend([np.repeat(np.log((c+5*p)/(n+5)/p),10) for c,p in zip(usercounts,priors)])
        raw=np.column_stack(columns);assert raw.shape==(len(a),16)
        relative=[]
        for col in [0,1,3,4,5,6,7,9]:
            v=raw[:,col].reshape(-1,10)
            relative.extend([(v-v.mean(1,keepdims=True)).ravel(),(v-v.max(1,keepdims=True)).ravel()])
        x[off:off+len(a)]=np.column_stack([raw,*relative])
    assert np.isfinite(x).all()
    np.save(dest,x);print('Warm-context features',name,x.shape,flush=True)
    return x


def combined_features(name,rows,histories,cut):
    return np.column_stack([_aligned_features(name,rows,histories,cut),context_priors(name,rows,histories,cut)])


def validation_v19(day):
    x,old,sw=_validation(day)
    ref=np.load(ROOT/f'predictions_imp5_aligned_day{day}.npz')
    assert np.array_equal(old['events'],ref['events']) and np.array_equal(old['y'],ref['y'])
    return x,ref,sw


def candidate_v19(day,component,mix):
    old=np.load(ROOT/f'predictions_imp5_aligned_day{day}.npz')['raw']
    return candidate(day,(1-mix)*old+mix*component,1.)


def configure():
    engine.TAG='warmcontext';engine.WIDTH=290;engine.BASELINE_NAME='V19';engine.SELECTION_FLOOR=None
    engine.aligned_features=combined_features
    engine.aligned_validation=validation_v19
    engine.aligned_candidate=candidate_v19


if __name__=='__main__':
    configure();engine.run_aligned()
