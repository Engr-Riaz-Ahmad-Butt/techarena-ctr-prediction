"""V6: richer item factors and exposure-adjusted user preference profiles."""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v5 import *
from threadpoolctl import threadpool_limits

threadpool_limits(limits=6)

def preference_stats(cutoff):
    cache=ROOT/('warm_stats_'+str(cutoff)+'.pkl')
    if cache.exists():
        with cache.open('rb') as f:base=pickle.load(f)
    else:
        base=build_warm_stats(cutoff)
        with cache.open('wb') as f:pickle.dump(base,f,protocol=5)
    relations,context=base
    _,keys,counts,hc,cc,_,_=relations
    row=keys//ni;col=keys%ni
    matrix=coo_matrix((counts/np.sqrt((hc[row]+10)*(cc[col]+10)),(row,col)),shape=(ni,ni)).tocsr()
    svd=TruncatedSVD(n_components=128,n_iter=5,random_state=46)
    us=svd.fit_transform(matrix);s=np.maximum(svd.singular_values_,1e-8)
    # Remove the strongest common direction when constructing taste profiles.
    left=(us/np.sqrt(s)).astype('float32');right=(svd.components_.T*np.sqrt(s)).astype('float32')
    taste=normalize(right[:,1:]).astype('float32')
    del matrix,us,svd,row,col
    a=np.asarray(train[train[:,2]<cutoff]);events=a[::10]
    _,first=np.unique(events[:,0],return_index=True)
    e=events[first];h=np.asarray(hist[e[:,6]])
    hu=np.repeat(e[:,0],20);hi=h.ravel();valid=hi>=0
    clicked=a[a[:,5]==1]
    users=np.concatenate([hu[valid],clicked[:,0]]);items=np.concatenate([hi[valid],clicked[:,1]])
    positive=coo_matrix((np.ones(len(users),dtype='float32'),(users,items)),shape=(nu,ni)).tocsr()
    exposure=coo_matrix((np.ones(len(a),dtype='float32'),(a[:,0],a[:,1])),shape=(nu,ni)).tocsr()
    pc=np.asarray(positive.sum(axis=1)).ravel();ec=np.asarray(exposure.sum(axis=1)).ravel()
    pm=(positive@taste)/np.maximum(pc[:,None],1)
    em=(exposure@taste)/np.maximum(ec[:,None],1)
    profile=normalize(pm).astype('float32')
    contrast=normalize(pm-em).astype('float32')
    # Approximate variance adjustment accounts for globally common directions.
    variance=np.maximum(np.mean(taste*taste,axis=0),1e-6)
    adjusted=normalize((pm-em)/np.sqrt(variance)).astype('float32')
    del a,events,e,h,hu,hi,clicked,positive,exposure,pm,em
    gc.collect()
    print('128-factor item model and user preference profiles ready',flush=True)
    return base,left,right,taste,profile,contrast,adjusted,pc

def preference_features(a,histories,stats):
    base,left,right,taste,profile,contrast,adjusted,pc=stats
    x=context_features(a,histories,base);parts=[]
    for off in range(0,len(a),10000):
        b=a[off:off+10000];h=np.asarray(histories[b[:,6]])
        valid=h>=0;safe=np.maximum(h,0);n=valid.sum(axis=1)
        age=n[:,None]-1-np.arange(20)[None,:]
        l=left[safe];r=right[b[:,1]];vals=[]
        for rank in [8,16,32,64,128]:
            # Shared popularity direction is excluded, retaining taste factors.
            ls=l[:,:,1:rank];rs=r[:,1:rank]
            raw=(ls*rs[:,None,:]).sum(axis=2)*valid
            sim=raw/np.maximum(np.linalg.norm(ls,axis=2)*np.linalg.norm(rs,axis=1)[:,None],1e-8)
            vals.extend([raw.sum(axis=1)/np.maximum(n,1),
                         sim.sum(axis=1)/np.maximum(n,1),sim.max(axis=1)])
            for recent in [3,5]:
                mask=valid&(age<recent)
                vals.append((sim*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1))
        t=taste[b[:,1]];u=b[:,0]
        p=profile[u];c=contrast[u];ad=adjusted[u]
        vals.extend([(p*t).sum(axis=1),(c*t).sum(axis=1),(ad*t).sum(axis=1),np.log1p(pc[u])])
        ht=(taste[safe]*valid[:,:,None]).sum(axis=1)/np.maximum(n[:,None],1)
        ht=normalize(ht)
        vals.extend([(ht*t).sum(axis=1),(ht*p).sum(axis=1)])
        # Component-wise interactions let trees combine preference dimensions.
        parts.append(np.column_stack([*vals,(p*t)[:,:16],(ht*t)[:,:16]]).astype('float32'))
    extra=np.concatenate(parts);relative=[]
    for col in [1,6,11,16,21,25,26,27,29]:
        v=extra[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),(v-v.max(axis=1,keepdims=True)).ravel()])
    return np.column_stack([x,extra,*relative]).astype('float32')
