"""Full-prefix user/item co-occurrence; fixed matched residual-tree pilot."""
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import tied_history_gpu_bag as bag

PARAMS = dict(max_iter=200, learning_rate=.05, max_leaf_nodes=7,
              min_samples_leaf=200, l2_regularization=10., max_bins=63,
              early_stopping=False, random_state=8201)
BLEND = .1


def prefix_ledger(data, stamps, cutoff):
    ledger = defaultdict(set)
    for q in np.flatnonzero(data['timestamp'] < cutoff):
        user=int(data['users'][q])
        for item,stamp in zip(data['hist'][q],stamps[q]):
            if item >= 0:
                assert stamp < data['timestamp'][q] < cutoff
                ledger[user].add((int(stamp),int(item)))
        target=int(data['target'][q])
        if target < 10:
            ledger[user].add((int(data['timestamp'][q]),int(data['items'][q,target])))
    return dict(ledger)


def graph(ledger, n_items):
    profiles=[sorted({item for _,item in clicks}) for clicks in ledger.values() if clicks]
    lengths=np.array([len(p) for p in profiles],dtype='int64')
    rows=np.repeat(np.arange(len(profiles)),lengths)
    cols=np.concatenate(profiles)
    x=sparse.csr_matrix((np.ones(len(rows),dtype='float32'),(rows,cols)),shape=(len(profiles),n_items))
    frequency=np.asarray(x.sum(0)).ravel()
    support=(x.T@x).tocsr();support.setdiag(0);support.eliminate_zeros();support.sort_indices()
    # Direct set-intersection oracle, independent of sparse multiplication.
    take=np.linspace(0,len(support.data)-1,min(12,len(support.data)),dtype=int)
    pairrows=np.searchsorted(support.indptr,take,side='right')-1
    pairs=list(zip(pairrows,support.indices[take]))+[(0,n_items-1)]
    memberships=defaultdict(set)
    for user,p in enumerate(profiles):
        for item in p: memberships[item].add(user)
    checks=[]
    for a,b in pairs:
        expected=len(memberships[int(a)] & memberships[int(b)]) if a!=b else 0
        actual=float(support[a,b]);assert actual==expected
        checks.append(dict(a=int(a),b=int(b),expected=expected,actual=actual))
    del x,rows,cols,profiles,pairrows,memberships;gc.collect()
    return support,frequency,len(lengths),checks


def contexts(ledger,data,stamps,ids):
    result=np.full((len(ids),100),-1,dtype='int32')
    for n,q in enumerate(ids):
        clicks=set(ledger.get(int(data['users'][q]),()))
        for item,stamp in zip(data['hist'][q],stamps[q]):
            if item>=0:
                assert stamp<data['timestamp'][q]
                clicks.add((int(stamp),int(item)))
        chosen=sorted(clicks,reverse=True)[:100]
        result[n,:len(chosen)]=[i for _,i in chosen]
    return result


def features(support,frequency,n_users,history,candidates):
    out=np.zeros((len(candidates)*10,12),dtype='float32')
    for off in range(0,len(candidates),256):
        h=history[off:off+256];c=candidates[off:off+256]
        shape=(len(c),10,h.shape[1]);valid=np.broadcast_to(h[:,None,:]>=0,shape)
        row=np.broadcast_to(np.arange(len(c)*10).reshape(len(c),10,1),shape)[valid]
        a=np.broadcast_to(c[:,:,None],shape)[valid]
        b=np.broadcast_to(h[:,None,:],shape)[valid]
        values=np.asarray(support[a,b]).ravel()
        cosine=values/np.sqrt(np.maximum(frequency[a]*frequency[b],1))
        lift=values*n_users/np.maximum(frequency[a]*frequency[b],1)
        ppmi=np.log(np.maximum(lift,1))*values/(values+5)
        size=len(c)*10;den=np.repeat(np.maximum((h>=0).sum(1),1),10)
        columns=[]
        for v in [cosine,ppmi,np.log1p(values)]:
            mean=np.bincount(row,weights=v,minlength=size)/den
            maximum=np.zeros(size);np.maximum.at(maximum,row,v)
            columns.extend([mean,maximum])
        columns.extend([np.bincount(row,weights=values>0,minlength=size)/den,
                        np.log1p(frequency[c.ravel()]),
                        np.repeat(np.log1p((h>=0).sum(1)),10),
                        ((c[:,:,None]==h[:,None,:])&(h[:,None,:]>=0)).any(2).ravel()])
        for v in [columns[0],columns[2]]:
            slate=v.reshape(-1,10);columns.append((slate-slate.mean(1,keepdims=True)).ravel())
        block=np.column_stack(columns)
        assert block.shape==(size,12) and np.isfinite(block).all()
        out[off*10:(off+len(c))*10]=block
    return out


def run():
    from sklearn.ensemble import HistGradientBoostingClassifier
    start=time.monotonic()
    roots=list(Path('/kaggle/input').rglob('neural_manifest.json'));assert len(roots)==1
    root=roots[0].parent;out=Path('/kaggle/working/full-cooccurrence-001');out.mkdir(exist_ok=False)
    frozen=dict(params=PARAMS,blend=BLEND,source_sha256=bag.digest(__file__),
                construction='binary user-item presence from all unique prefix clicks; diagonal removed',
                ppmi_shrinkage=5,fit_meta_day=5,score_meta_day=6,minimum_gain=.0005,
                fresh_holdout=False,promoted=False,official_auc=None)
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2));print(json.dumps(frozen),flush=True)
    data,stamps=bag.load(root);n_items=max(int(data['items'].max()),int(data['hist'].max()))+1
    arrays={};reports=[]
    for day in [5,6]:
        with np.load(root/f'val{day}.npz') as z: val={k:z[k].copy() for k in z.files}
        cutoff=int(data['timestamp'][0])+day*86400
        ledger=prefix_ledger(data,stamps,cutoff)
        matrix,freq,users,oracle=graph(ledger,n_items)
        h=contexts(ledger,data,stamps,val['indices'])
        x=features(matrix,freq,users,h,data['items'][val['indices']])
        np.save(out/f'features{day}.npy',x)
        baseline=logit(np.clip(val['baseline'],1e-6,1-1e-6))
        count=np.repeat((data['hist'][val['indices']]>=0).sum(1),10)
        control=np.column_stack([baseline,np.log1p(count),baseline*np.log1p(count)])
        arrays[day]=dict(val=val,control=control,treatment=np.column_stack([control,x]))
        r=dict(day=day,cutoff=cutoff,graph_users=users,graph_edges=int(matrix.nnz),
               events=len(h),feature_sha256=bag.digest(out/f'features{day}.npy'),pair_oracle=oracle)
        reports.append(r);print(json.dumps(r),flush=True)
        del matrix,ledger,h,x;gc.collect()
    predictions={}
    for arm in ['control','treatment']:
        model=HistGradientBoostingClassifier(**PARAMS)
        with threadpool_limits(limits=2):
            model.fit(arrays[5][arm],arrays[5]['val']['y'],sample_weight=arrays[5]['val']['weights'])
            raw=model.predict_proba(arrays[6][arm])[:,1]
            joblib.dump(model,out/f'{arm}.joblib')
            replay=joblib.load(out/f'{arm}.joblib')
            assert np.array_equal(raw,replay.predict_proba(arrays[6][arm])[:,1])
        rank=lambda p:rankdata(np.asarray(p,dtype='float64'),method='average')/(len(p)+1)
        predictions[arm+'_raw']=raw
        predictions[arm]=(1-BLEND)*rank(arrays[6]['val']['baseline'])+BLEND*rank(raw)
        print(json.dumps(dict(stage='model_complete',arm=arm,checkpoint_replay='exact_all_rows')),flush=True)
    v=arrays[6]['val']
    def metric(p):
        return dict(ordinary_auc=float(roc_auc_score(v['y'],p)),
                    adjusted_auc=float(roc_auc_score(v['y'],p,sample_weight=v['weights'])))
    result=dict(baseline=metric(v['baseline']),**{k:metric(p) for k,p in predictions.items()},
                source_reports=reports,promoted=False,csv_ready=False,fresh_holdout=False,
                official_auc=None,target_achieved=False,elapsed_seconds=time.monotonic()-start)
    result['adjusted_gain']=result['treatment']['adjusted_auc']-result['baseline']['adjusted_auc']
    result['gain_vs_control']=result['treatment']['adjusted_auc']-result['control']['adjusted_auc']
    result['development_gate']=(result['adjusted_gain']>=.0005 and result['gain_vs_control']>0
                                and result['treatment']['ordinary_auc']>=result['baseline']['ordinary_auc'])
    np.savez_compressed(out/'predictions.npz',y=v['y'],events=np.repeat(v['indices'],10),
                        weights=v['weights'],baseline=v['baseline'],**predictions)
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result
