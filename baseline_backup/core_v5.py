"""V5: warm-up click relationships and smoothed contextual click rates."""
from config import TRAIN_CSV, EVALUATION_CSV
from core_v4 import *
from scipy.sparse import csr_matrix
import gc

def build_warm_stats(cutoff):
    path=ROOT/('stats_'+str(cutoff)+'.pkl')
    with path.open('rb') as f:base=pickle.load(f)
    a=np.asarray(train[train[:,2]<cutoff])
    clicked=a[a[:,5]==1]
    h=np.asarray(hist[clicked[:,6]])
    left=h.ravel();right=np.repeat(clicked[:,1],20);mask=left>=0
    matrix=coo_matrix((np.ones(mask.sum(),dtype='float32'),(left[mask],right[mask])),shape=(ni,ni)).tocsr()
    del h,left,right,mask,clicked
    events=a[::10]
    _,first=np.unique(events[:,0],return_index=True)
    histories=np.asarray(hist[events[first,6]])
    # Only the first available pre-cutoff history per user is used, avoiding
    # repeated counting of the same warm-up clicks across overlapping histories.
    # Every pair is directed from an older click to a newer click.
    warm_pairs=0
    for j in range(1,20):
        previous=histories[:,:j];following=histories[:,j]
        source=previous.ravel();target=np.repeat(following,j)
        mask=(source>=0)&(target>=0)
        matrix += coo_matrix((np.ones(mask.sum(),dtype='float32'),(source[mask],target[mask])),shape=(ni,ni)).tocsr()
        warm_pairs+=int(mask.sum())
    del histories,events,previous,following,source,target,mask
    hc=np.asarray(matrix.sum(axis=1)).ravel();cc=np.asarray(matrix.sum(axis=0)).ravel()
    co=matrix.tocoo();del matrix
    keys=co.row.astype('int64')*ni+co.col
    order=np.argsort(keys);keys=keys[order];counts=co.data[order];del order
    co.data /= np.sqrt((hc[co.row]+10)*(cc[co.col]+10))
    svd=TruncatedSVD(n_components=64,n_iter=5,random_state=44)
    us=svd.fit_transform(co.tocsr());s=np.maximum(svd.singular_values_,1e-8)
    src=(us/np.sqrt(s)).astype('float32');dst=(svd.components_.T*np.sqrt(s)).astype('float32')
    print('Warm-up pairs added:',warm_pairs,'Distinct combined pairs:',len(keys),flush=True)
    del co,us,svd
    context={}
    y=a[:,5];g=float(y.mean())
    bucket_impr=np.bincount(a[:,3],minlength=4)
    bucket_click=np.bincount(a[:,3],weights=y,minlength=4)
    context['global']=g
    context['bucket_ctr']=(bucket_click+100*g)/(bucket_impr+100)
    for name,col,size in [('user',0,nu),('item',1,ni)]:
        ids=a[:,col];key=ids*4+a[:,3]
        context[name+'_impr']=np.bincount(ids,minlength=size)
        context[name+'_click']=np.bincount(ids,weights=y,minlength=size)
        context[name+'_bucket_impr']=np.bincount(key,minlength=size*4)
        context[name+'_bucket_click']=np.bincount(key,weights=y,minlength=size*4)
        recent=a[:,2]>=cutoff-DAY
        context[name+'_recent_impr']=np.bincount(ids[recent],minlength=size)
        context[name+'_recent_click']=np.bincount(ids[recent],weights=y[recent],minlength=size)
    del a
    gc.collect()
    return (base,keys,counts,hc,cc,src,dst),context

def context_features(a,histories,stats):
    relations,context=stats
    x=relation_features(a,histories,relations)
    bucket=a[:,3];g=context['global'];columns=[]
    for name,col in [('user',0),('item',1)]:
        ids=a[:,col];key=ids*4+bucket
        exposure=context[name+'_impr'][ids];clicks=context[name+'_click'][ids]
        prior=(clicks+50*g)/(exposure+50)
        # Conditional prior accounts for each bucket's overall click propensity.
        conditional=np.clip(prior*context['bucket_ctr'][bucket]/g,1e-5,.999)
        ni_=context[name+'_bucket_impr'][key];nc=context[name+'_bucket_click'][key]
        ri=context[name+'_recent_impr'][ids];rc=context[name+'_recent_click'][ids]
        columns.extend([(clicks+10*g)/(exposure+10),(clicks+150*g)/(exposure+150),
                        (nc+20*conditional)/(ni_+20),(nc+80*conditional)/(ni_+80),
                        np.log1p(ni_),(rc+30*prior)/(ri+30),
                        (rc+30*prior)/(ri+30)-prior,ni_/np.maximum(exposure,1)])
    columns.append(context['bucket_ctr'][bucket])
    extra=np.column_stack(columns).astype('float32');relative=[]
    for col in [8,10,11,13,14]:
        v=extra[:,col].reshape(-1,10)
        relative.extend([(v-v.mean(axis=1,keepdims=True)).ravel(),(v-v.max(axis=1,keepdims=True)).ravel()])
    return np.column_stack([x,extra,*relative]).astype('float32')
