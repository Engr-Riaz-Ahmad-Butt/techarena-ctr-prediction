"""Independent nonnegative user-item topics; temporally isolated train/test."""
from imp2_warm_choice import *
from sklearn.decomposition import NMF

def stats(cutoff):
    path=ROOT/f'imp2_nmf_stats_{cutoff}.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    a=np.asarray(train[train[:,2]<cutoff]);wu,wi=warm_data(a);click=a[a[:,5]==1]
    u=np.concatenate([wu,click[:,0]]);i=np.concatenate([wi,click[:,1]])
    matrix=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(nu,ni)).tocsr()
    matrix.data=np.log1p(matrix.data)
    count=np.asarray(matrix.sum(axis=0)).ravel();idf=np.maximum(count,1)**(-.25)
    matrix=matrix.multiply(idf[None,:]).tocsr()
    nm=NMF(n_components=16,init='nndsvda',solver='cd',max_iter=100,tol=.002,random_state=151)
    t=time.time();W=nm.fit_transform(matrix).astype('float32');H=nm.components_.T.astype('float32')
    print('NMF fitted',nm.n_iter_,'iterations seconds',round(time.time()-t,1),flush=True)
    g=float(a[:,5].mean());ui=np.bincount(a[:,0],minlength=nu);uc=np.bincount(a[:,0],weights=a[:,5],minlength=nu)
    ii=np.bincount(a[:,1],minlength=ni);ic=np.bincount(a[:,1],weights=a[:,5],minlength=ni)
    out=W,H,g,ui,uc,ii,ic
    with path.open('wb') as f:pickle.dump(out,f,protocol=5)
    return out

def feats(a,hh,s):
    assert len(a)%10==0 and np.all(a[:,6].reshape(-1,10)==a[::10,6,None])
    W,H,g,ui,uc,ii,ic=s;E=normalize(H).astype('float32');U=normalize(W).astype('float32');out=[]
    for off in range(0,len(a),20000):
        b=a[off:off+20000];u,i=b[:,0],b[:,1];h=np.asarray(hh[b[:,6]])
        valid=h>=0;safe=np.maximum(h,0);n=valid.sum(1);v=E[safe]*valid[:,:,None]
        mean=v.sum(1)/np.maximum(n[:,None],1);q=E[i];sim=(v*q[:,None,:]).sum(2)
        age=n[:,None]-1-np.arange(20);last3=valid&(age<3)
        values=[(uc[u]+50*g)/(ui[u]+50),(ic[i]+50*g)/(ii[i]+50),np.log1p(ui[u]),np.log1p(ii[i]),
                b[:,3],b[:,4],n,(h==i[:,None]).sum(1),(mean*q).sum(1),sim.max(1),
                (sim*last3).sum(1)/np.maximum(last3.sum(1),1),(U[u]*q).sum(1),(W[u]*H[i]).sum(1)]
        out.append(np.column_stack([*values,mean*q,U[u]*q,H[i]]).astype('float32'))
    x=np.concatenate(out);relative=[]
    for j in [1,8,9,10,11,12]:
        v=x[:,j].reshape(-1,10)
        relative.extend([(v-v.mean(1,keepdims=True)).ravel(),(v-v.max(1,keepdims=True)).ravel()])
    return np.column_stack([x,*relative]).astype('float32')

def experiment():
    reports=[];trees=None;blendweight=None
    for day in [6,5]:
        fit,val=split(day);s=stats(start+(day-2)*DAY)
        xf=feats(fit,hist,s);xv=feats(val,hist,s);del s;gc.collect()
        m=CatBoostClassifier(iterations=1200 if trees is None else trees,depth=7,learning_rate=.08,
            thread_count=6,random_seed=151,loss_function='Logloss',eval_metric='AUC',verbose=200,allow_writing_files=False)
        if trees is None:m.fit(xf,fit[:,5],eval_set=(xv,val[:,5]),early_stopping_rounds=60);trees=int(m.tree_count_)
        else:m.fit(xf,fit[:,5])
        raw=m.predict_proba(xv)[:,1];prior=np.load(ROOT/f'predictions_weighted_day{day}.npz')
        reference=np.load(ROOT/f'predictions_v6_day{day}.npz')
        assert np.array_equal(prior['y'],val[:,5]) and np.array_equal(reference['events'],val[:,6])
        old=prior['pred']
        eg=np.digitize((ehist>=0).sum(1),[0,4,9,19],right=True);vg=np.digitize((hist[val[:,6]]>=0).sum(1),[0,4,9,19],right=True)
        sw=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
        if blendweight is None:
            choices={w:roc_auc_score(val[:,5],(1-w)*old+w*raw,sample_weight=sw) for w in [0,.05,.1,.2,.3]}
            blendweight=max(choices,key=choices.get)
        pred=(1-blendweight)*old+blendweight*raw
        r=dict(day=day,trees=trees,weight=blendweight,raw_auc=float(roc_auc_score(val[:,5],raw)),
            auc=float(roc_auc_score(val[:,5],pred)),history_weighted_auc=float(roc_auc_score(val[:,5],pred,sample_weight=sw)))
        print(r,flush=True);reports.append(r)
        np.savez_compressed(ROOT/f'imp2_nmf_day{day}.npz',events=val[:,6],y=val[:,5],raw=raw,pred=pred)
        (ROOT/'imp2_nmf_report.json').write_text(json.dumps(reports,indent=2))
        del fit,val,xf,xv,m;gc.collect()
        if day==6 and blendweight==0:
            print('Rejected on selection date: NMF adds no weighted AUC gain.',flush=True)
            break
if __name__=='__main__':experiment()
