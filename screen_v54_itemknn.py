"""Restricted causal ItemKNN; selection samples only, never opens holdout/test."""
import json
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R

def rank(x):
    return rankdata(x, method='average') / (len(x) + 1)

def main():
    started = time.monotonic()
    train = np.load(R/'train.npy', mmap_mode='r')
    hist = np.load(R/'train_hist.npy', mmap_mode='r')
    start = int(train[:,2].min())
    ni = int(train[:,1].max())+1
    records = []
    for day in (5,6):
        cutoff = start+day*86400
        z = np.load(R/f'selection_day{day}.npz')
        events = z['events'][::10]
        users = np.asarray(train[events*10,0])
        take = (users.astype(np.uint64)*np.uint64(2654435761)%100)<20
        selected = events[take]
        assert np.all(train[selected*10,2]>=cutoff)
        y = z['y'].reshape(-1,10)[take].ravel()
        base = z['baseline'].reshape(-1,10)[take].ravel()
        weights = z['weights'].reshape(-1,10)[take].ravel()
        candidates = np.asarray(train[(selected[:,None]*10+np.arange(10)).ravel(),1])
        h = np.asarray(hist[selected])
        labels = np.asarray(train[:,5])
        clickrows = np.flatnonzero((train[:,2]<cutoff)&(labels==1))
        u = np.asarray(train[clickrows,0]); items = np.asarray(train[clickrows,1])
        pop = np.bincount(items, minlength=ni)
        order = np.argsort(-pop,kind='stable')
        baseline = float(roc_auc_score(y,base))
        br = rank(base)
        for n in (5000,10000,20000):
            mapping = np.full(ni,-1,dtype=np.int32); mapping[order[:n]]=np.arange(n)
            ci=mapping[items]; valid=ci>=0
            cache=R/f'v54_itemknn_day{day}_n{n}.npz'
            # Cache only the causal co-click matrix, not validation labels.
            if cache.exists():
                from scipy.sparse import load_npz
                gram=load_npz(cache)
            else:
                x=coo_matrix((np.ones(valid.sum(),dtype=np.float32),(u[valid],ci[valid])),shape=(int(u.max())+1,n)).tocsr()
                x.data[:]=1
                gram=(x.T@x).tocsr()
                from scipy.sparse import save_npz
                save_npz(cache,gram)
            freq=gram.diagonal().copy(); gram.setdiag(0); gram.eliminate_zeros()
            c=mapping[candidates]; hh=mapping[np.maximum(h,0)]; hh[h<0]=-1
            lengths=(h>=0).sum(1)
            pair=np.zeros((len(c),20),dtype=np.float32)
            for j in range(20):
                hj=np.repeat(hh[:,j],10); ok=(c>=0)&(hj>=0)
                pair[ok,j]=np.asarray(gram[c[ok],hj[ok]]).ravel()
            for reg in (1,10,100):
                denom=np.sqrt((freq[np.maximum(c,0),None]+reg)*(freq[np.maximum(np.repeat(hh,10,axis=0),0)]+reg))
                sim=pair/denom
                for window in (5,20):
                    age=lengths[:,None]-1-np.arange(20)
                    mask=np.repeat((age>=0)&(age<window),10,axis=0)
                    v=sim*mask
                    rw=mask*np.exp(-np.maximum(np.repeat(age,10,axis=0),0)/5)
                    features={'max':v.max(1),'mean':v.sum(1)/np.maximum(mask.sum(1),1),'recency':(sim*rw).sum(1)/np.maximum(rw.sum(1),1)}
                    for agg,score in features.items():
                        sr=rank(score)
                        for alpha in (.01,.02,.05,.1):
                            pred=(1-alpha)*br+alpha*sr
                            auc=float(roc_auc_score(y,pred))
                            records.append(dict(day=day,n=n,regularization=reg,window=window,aggregation=agg,alpha=alpha,baseline_auc=baseline,auc=auc,gain=auc-baseline,adjusted_auc=float(roc_auc_score(y,pred,sample_weight=weights)),nonzero_share=float((score>0).mean())))
            print(json.dumps(dict(day=day,n=n,sample_events=int(take.sum()),click_rows=len(clickrows),item_click_coverage=float(valid.mean()),candidate_coverage=float((c>=0).mean()),gram_nonzero=gram.nnz,best_gain=max(r['gain'] for r in records if r['day']==day and r['n']==n))),flush=True)
    configs={}
    for r in records:
        key=tuple(r[k] for k in ('n','regularization','window','aggregation','alpha'))
        configs.setdefault(key,[]).append(r)
    best=max(configs.values(),key=lambda rr:min(r['gain'] for r in rr))
    out=dict(experiment='restricted_itemknn',metric='global ROC AUC',scope='20% deterministic users from frozen selection cohorts; holdout unopened',method='Binary user-item pre-cutoff clicks; symmetric co-click cosine with diagonal removed and frequency shrinkage. ItemKNN arm; no dense EASE inverse.',best_joint_configuration=best,sample_gate_passed=all(r['gain']>=.0005 for r in best),trials=records,elapsed_seconds=time.monotonic()-started,leakage_controls=['Only labeled training clicks strictly before each cutoff build popularity and similarities.','No supplied history is used to fit the similarity matrix.','History supplied at scoring is prior to its event; self-similarity diagonal is zero.','No evaluation or fresh-holdout file opened.','Average ranks preserve ties; deterministic user hash; no random fit.'])
    (R/'v54_itemknn_screen.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='trials'},indent=2),flush=True)

if __name__=='__main__':
    main()
