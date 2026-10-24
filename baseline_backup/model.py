from config import TRAIN_CSV, EVALUATION_CSV
import sys, json, csv, time
from pathlib import Path
from config import ARTIFACT_DIR as ROOT

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

train = np.load(ROOT/'train.npy', mmap_mode='r')
hist = np.load(ROOT/'train_hist.npy', mmap_mode='r')
evaluation = np.load(ROOT/'evaluation.npy', mmap_mode='r')
ehist = np.load(ROOT/'evaluation_hist.npy', mmap_mode='r')
ids = json.loads((ROOT/'ids.json').read_text())
nu, ni = len(ids['users']), len(ids['items'])
del ids
start = int(train[:,2].min())
DAY = 86400

def statistics(cutoff):
    a = np.asarray(train[train[:,2] < cutoff])
    y = a[:,5]
    g = float(y.mean())
    ui = np.bincount(a[:,0], minlength=nu); uc = np.bincount(a[:,0], weights=y, minlength=nu)
    ii = np.bincount(a[:,1], minlength=ni); ic = np.bincount(a[:,1], weights=y, minlength=ni)
    # Only histories and labels available before the statistics cutoff.
    events = a[::10]
    _, first = np.unique(events[:,0], return_index=True)
    firstevents = events[first]
    h = np.asarray(hist[firstevents[:,6]])
    hu = np.repeat(firstevents[:,0],20); hi = h.ravel(); valid = hi >= 0
    clicked = a[y == 1]
    matrix = coo_matrix((np.ones(int(valid.sum())+len(clicked),dtype='float32'),
        (np.concatenate([hu[valid],clicked[:,0]]),np.concatenate([hi[valid],clicked[:,1]]))),shape=(nu,ni)).tocsr()
    matrix.data = np.log1p(matrix.data)
    svd = TruncatedSVD(n_components=32,n_iter=4,random_state=42)
    svd.fit(matrix)
    emb = svd.components_.T.copy()
    emb /= np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-8)
    print('Statistics:', len(a), 'rows; CTR:',g,'embedding ready',flush=True)
    return g,ui,uc,ii,ic,emb

def features(a, histories, stats):
    g,ui,uc,ii,ic,emb = stats
    x = np.empty((len(a),13), dtype='float32')
    for offset in range(0,len(a),100000):
        b=a[offset:offset+100000]; u,i=b[:,0],b[:,1]
        h=np.asarray(histories[b[:,6]]); valid=h>=0; safe=np.maximum(h,0)
        lengths=valid.sum(axis=1)
        v=emb[safe]*valid[:,:,None]
        avg=v.sum(axis=1)/np.maximum(lengths[:,None],1)
        similarity=(v*emb[i,None,:]).sum(axis=2)
        x[offset:offset+len(b)] = np.column_stack([
            (uc[u]+50*g)/(ui[u]+50), (ic[i]+50*g)/(ii[i]+50),
            np.log1p(ui[u]),np.log1p(ii[i]),b[:,3],b[:,4],lengths,
            (h==i[:,None]).sum(axis=1), (avg*emb[i]).sum(axis=1),
            similarity.max(axis=1), (similarity*np.arange(1,21)).sum(axis=1)/np.maximum((valid*np.arange(1,21)).sum(axis=1),1),
            (((ic[safe]+50*g)/(ii[safe]+50))*valid).sum(axis=1)/np.maximum(lengths,1),
            (b[:,2]%86400)/86400])
    return x

stats=statistics(start+4*DAY)
fit=np.asarray(train[(train[:,2]>=start+4*DAY)&(train[:,2]<start+6*DAY)])
later=np.asarray(train[train[:,2]>=start+6*DAY])
events=later[::10]
_,first=np.unique(events[:,0],return_index=True)
val=later[(first[:,None]*10+np.arange(10)).ravel()]
print('Model fit rows:',len(fit),'Validation rows:',len(val),flush=True)
xf=features(fit,hist,stats); xv=features(val,hist,stats)
model=CatBoostClassifier(iterations=350,depth=7,learning_rate=.08,loss_function='Logloss',eval_metric='AUC',thread_count=6,random_seed=42,verbose=50,allow_writing_files=False)
model.fit(xf,fit[:,5],eval_set=(xv,val[:,5]),early_stopping_rounds=50)
pred=model.predict_proba(xv)[:,1]
baseline=xv[:,0]*xv[:,1]
report={'validation_rows':len(val),'baseline_auc':float(roc_auc_score(val[:,5],baseline)),'model_auc':float(roc_auc_score(val[:,5],pred)),'iterations':int(model.tree_count_)}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
del xf,xv,fit,val,later,stats
stats=statistics(start+6*DAY)
fit=np.asarray(train[train[:,2]>=start+6*DAY])
xf=features(fit,hist,stats)
final=CatBoostClassifier(iterations=report['iterations'],depth=7,learning_rate=.08,loss_function='Logloss',thread_count=6,random_seed=42,verbose=50,allow_writing_files=False)
final.fit(xf,fit[:,5]); final.save_model(str(ROOT/'model.cbm'))
del xf,fit
xe=features(evaluation,ehist,stats)
prob=final.predict_proba(xe)[:,1]
assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
source=EVALUATION_CSV
seen=set()
with source.open(newline='',encoding='utf-8') as src, (ROOT/'submission.csv').open('w',newline='',encoding='utf-8') as dest:
    writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
    count=0
    for row,p in zip(csv.DictReader(src),prob,strict=True):
        assert row['rowid'] not in seen
        seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')]);count+=1
assert count==529390
print('Verified submission:',count,'unique rows',flush=True)
