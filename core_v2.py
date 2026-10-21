"""Compare history and event features on two temporal validation windows."""
from config import TRAIN_CSV, EVALUATION_CSV
from core import *
from sklearn.preprocessing import normalize

def extra_statistics(cutoff):
    base=statistics(cutoff)
    a=np.asarray(train[train[:,2]<cutoff]); events=a[::10]
    _,first=np.unique(events[:,0],return_index=True)
    e=events[first]; h=np.asarray(hist[e[:,6]]); hu=np.repeat(e[:,0],20); hi=h.ravel(); valid=hi>=0
    clicked=a[a[:,5]==1]
    rows=np.concatenate([hu[valid],clicked[:,0]]); cols=np.concatenate([hi[valid],clicked[:,1]])
    matrix=coo_matrix((np.ones(len(rows),dtype='float32'),(rows,cols)),shape=(nu,ni)).tocsr()
    # Downweight ubiquitous items so the factors express personal taste.
    matrix.data=np.log1p(matrix.data)
    idf=np.log1p(nu/(1+np.asarray((matrix>0).sum(axis=0)).ravel()))
    weighted=normalize(matrix.multiply(idf).tocsr(),norm='l2')
    svd=TruncatedSVD(n_components=64,n_iter=5,random_state=43)
    svd.fit(weighted)
    emb=normalize(svd.components_.T.copy()).astype('float32')
    profiles=normalize(matrix@emb).astype('float32')
    print('Personalized factors ready',flush=True)
    return base,emb,profiles

def extra_features(a,histories,stats):
    base,emb,profiles=stats
    original=features(a,histories,base)
    extra=np.empty((len(a),12),dtype='float32')
    for off in range(0,len(a),20000):
        b=a[off:off+20000]; h=np.asarray(histories[b[:,6]])
        valid=h>=0; safe=np.maximum(h,0); length=valid.sum(axis=1)
        position=np.arange(20)[None,:]; age=length[:,None]-1-position
        sim=(emb[safe]*emb[b[:,1],None,:]).sum(axis=2)*valid
        vals=[]
        for n in [1,3,5,10]:
            mask=valid&(age<n)
            vals.append((sim*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1))
        for decay in [.5,.85]:
            w=np.power(decay,np.maximum(age,0))*valid
            vals.append((sim*w).sum(axis=1)/np.maximum(w.sum(axis=1),1e-9))
        vals.extend([sim.sum(axis=1)/np.maximum(length,1),sim.max(axis=1),
            np.sqrt((sim**2).sum(axis=1)/np.maximum(length,1)),
            (profiles[b[:,0]]*emb[b[:,1]]).sum(axis=1),
            ((h==b[:,1,None])&(age<3)).sum(axis=1),
            np.where(valid,sim,-2).argmax(axis=1)/np.maximum(length,1)])
        extra[off:off+len(b)]=np.column_stack(vals)
    x=np.column_stack([original,extra])
    # Event groups are preserved when selecting rows for training/validation.
    assert len(a)%10==0 and np.all(a[:,6].reshape(-1,10)==a[::10,6,None])
    relative=[]
    for col in [1,8,9,13,14,15,19,20,22]:
        v=x[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),
                         (v-v.max(axis=1,keepdims=True)).ravel(),
                         (np.argsort(np.argsort(v,axis=1),axis=1)/9).ravel()])
    return np.column_stack([x,*relative]).astype('float32')

def split(day):
    fit=np.asarray(train[(train[:,2]>=start+(day-2)*DAY)&(train[:,2]<start+day*DAY)])
    later=np.asarray(train[(train[:,2]>=start+day*DAY)&(train[:,2]<start+(day+1)*DAY)])
    _,first=np.unique(later[::10,0],return_index=True)
    val=later[(first[:,None]*10+np.arange(10)).ravel()]
    return fit,val
