"""V4: learn directed history-item -> clicked-item relationships."""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v2 import *
import pickle

def build_stats(cutoff):
    path=ROOT/('stats_'+str(cutoff)+'.pkl')
    if path.exists():
        with path.open('rb') as f:base=pickle.load(f)
    else:
        base=extra_statistics(cutoff)
        with path.open('wb') as f:pickle.dump(base,f,protocol=5)
    a=np.asarray(train[(train[:,2]<cutoff)&(train[:,5]==1)])
    h=np.asarray(hist[a[:,6]]);valid=h>=0
    left=h.ravel();right=np.repeat(a[:,1],20);mask=left>=0
    matrix=coo_matrix((np.ones(mask.sum(),dtype='float32'),(left[mask],right[mask])),shape=(ni,ni)).tocsr()
    hc=np.asarray(matrix.sum(axis=1)).ravel();cc=np.asarray(matrix.sum(axis=0)).ravel()
    co=matrix.tocoo();keys=co.row.astype('int64')*ni+co.col
    order=np.argsort(keys);keys=keys[order];counts=co.data[order]
    # Normalize frequent source/target items before reducing the relation table.
    co.data /= np.sqrt((hc[co.row]+10)*(cc[co.col]+10))
    svd=TruncatedSVD(n_components=64,n_iter=5,random_state=44)
    us=svd.fit_transform(co.tocsr());s=np.maximum(svd.singular_values_,1e-8)
    src=(us/np.sqrt(s)).astype('float32');dst=(svd.components_.T*np.sqrt(s)).astype('float32')
    print('Directed relationships:',len(keys),'distinct item pairs',flush=True)
    return base,keys,counts,hc,cc,src,dst

def relation_features(a,histories,stats):
    base,keys,counts,hc,cc,src,dst=stats
    x=extra_features(a,histories,base);parts=[]
    for off in range(0,len(a),20000):
        b=a[off:off+20000];h=np.asarray(histories[b[:,6]])
        valid=h>=0;safe=np.maximum(h,0);n=valid.sum(axis=1)
        query=safe.astype('int64')*ni+b[:,1,None]
        pos=np.searchsorted(keys,query);inside=pos<len(keys);pos=np.minimum(pos,len(keys)-1)
        c=np.where(inside&(keys[pos]==query)&valid,counts[pos],0)
        cosine=c/np.sqrt((hc[safe]+10)*(cc[b[:,1,None]]+10))
        lift=c/(c+3)*np.log1p(c*counts.sum()/((hc[safe]+10)*(cc[b[:,1,None]]+10)))
        latent=(src[safe]*dst[b[:,1],None,:]).sum(axis=2)*valid
        age=n[:,None]-1-np.arange(20)[None,:];vals=[]
        for values in [cosine,lift,latent]:
            vals.extend([values.sum(axis=1)/np.maximum(n,1),values.max(axis=1)])
            for recent in [1,3,5]:
                m=valid&(age<recent)
                vals.append((values*m).sum(axis=1)/np.maximum(m.sum(axis=1),1))
        vals.extend([(c>0).sum(axis=1),np.log1p(c.sum(axis=1)),np.log1p(c.max(axis=1))])
        parts.append(np.column_stack(vals))
    extra=np.concatenate(parts).astype('float32');relative=[]
    for col in [0,1,5,6,10,11]:
        v=extra[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),(v-v.max(axis=1,keepdims=True)).ravel()])
    return np.column_stack([x,extra,*relative]).astype('float32')
