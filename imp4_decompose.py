"""Causal decomposition of warm16 preferences; fixed-tree comparison against V17.

All factor checkpoints use labels strictly before their scored day. Additional
features are invariant scalar products, norms, biases and history similarities.
Choose only the new/old member mix on day 6; freeze it for day 5. No upload.
"""
from imp3_warmfeat import *
from tst2_candidate import influence, weighted_influence

TAG='decompose'
TREES=599
NEXTRA=32

def extras(rows,histories,cut):
    checkpoint=model_path(cut)
    with np.load(checkpoint) as m:
        U,V,H,bu,bi=[m[k] for k in ['U','V','H','bu','bi']]
    assert U.shape[1]==V.shape[1]==H.shape[1]==16
    result=np.empty((len(rows),NEXTRA),np.float32)
    probabilities=np.empty(len(rows),np.float32)
    assert len(rows)%10==0
    for off in range(0,len(rows),20000):
        a=rows[off:off+20000]
        assert np.all(a[:,6].reshape(-1,10)==a[::10,6,None])
        users=a[::10,0];items=a[:,1].reshape(-1,10)
        history=np.asarray(histories[a[::10,6]])
        valid=history>=0;n=valid.sum(1)
        hv=H[np.maximum(history,0)]*valid[:,:,None]
        z=hv.sum(1)/np.sqrt(np.maximum(n[:,None],1))
        u=U[users];v=V[items]
        direct=np.einsum('ed,ecd->ec',u,v)
        dynamic=np.einsum('ed,ecd->ec',z,v)
        un=np.linalg.norm(u,axis=1);zn=np.linalg.norm(z,axis=1);vn=np.linalg.norm(v,axis=2)
        sim=np.einsum('ehd,ecd->ech',hv,v)
        sim*=valid[:,None,:]
        mean=sim.sum(2)/np.maximum(n[:,None],1)
        maximum=np.where(valid[:,None,:],sim,-1e6).max(2)
        maximum[n==0]=0
        variance=np.maximum((sim*sim).sum(2)/np.maximum(n[:,None],1)-mean*mean,0)
        age=n[:,None]-1-np.arange(20)[None,:]
        recents=[]
        for width in [3,5]:
            mask=valid&(age<width)
            recents.append((sim*mask[:,None,:]).sum(2)/np.maximum(mask.sum(1)[:,None],1))
        def event_col(v):return np.broadcast_to(v[:,None],direct.shape)
        agreement=(u*z).sum(1)/np.maximum(un*zn,1e-8)
        user_bias=event_col(bu[users]);item_bias=bi[items]
        logits=np.clip(-1.8+user_bias+item_bias+direct+dynamic,-30,30)
        maximum_logit=np.maximum(logits.max(1,keepdims=True),0)
        exp=np.exp(logits-maximum_logit)
        raw=exp/(np.exp(-maximum_logit)+exp.sum(1,keepdims=True))
        no_click=1-raw.sum(1)
        values=[user_bias,item_bias,direct,dynamic,event_col(un),event_col(zn),vn,
                direct/np.maximum(un[:,None]*vn,1e-8),
                dynamic/np.maximum(zn[:,None]*vn,1e-8),mean,maximum,
                np.sqrt(variance),*recents,event_col(agreement),event_col(no_click),
                direct-dynamic,-1.8+user_bias+item_bias]
        # 18 scalar features plus 14 candidate-relative summaries.
        x=np.stack(values,axis=2).reshape(-1,18)
        relative=[]
        for col in [1,2,3,7,8,10,12]:
            s=x[:,col].reshape(-1,10)
            relative.extend([(s-s.mean(1,keepdims=True)).ravel(),
                             (s-s.max(1,keepdims=True)).ravel()])
        result[off:off+len(a)]=np.column_stack([x,*relative])
        probabilities[off:off+len(a)]=raw.ravel()
    assert np.isfinite(result).all()
    return result,probabilities

def feature_file(name):return FEAT/f'imp4_decompose_{name}.npy'

def build(name,rows,histories,cut):
    p=feature_file(name)
    if p.exists():
        cached=np.load(p,mmap_mode='r')
        assert cached.shape==(len(rows),NEXTRA)
        return cached
    x,probabilities=extras(rows,histories,cut)
    warm_name=name if name.startswith('val') or name=='eval' else f'day{cut}'
    old=np.load(FEAT/f'imp3_wf_{warm_name}.npy',mmap_mode='r')
    # Prove checkpoint, row order and the existing warm feature cache agree.
    difference=float(np.max(np.abs(probabilities-old[:,0])))
    assert np.allclose(probabilities,old[:,0],rtol=5e-5,atol=2e-7),difference
    np.save(p,x);print('Features',name,x.shape,'warm replay max error',difference,flush=True)
    return x

def assemble_new(days,label):
    total=sum(len(np.load(FEAT/f'stack_day{d}_c{d}.npy',mmap_mode='r')) for d in days)
    p=FEAT/f'imp4_train_{label}.npy'
    output=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(total,194+NEXTRA))
    targets=[];offset=0
    for day in days:
        a=np.asarray(train[(train[:,2]>=cutoff_of(day))&(train[:,2]<cutoff_of(day+1))])
        base=np.load(FEAT/f'stack_day{day}_c{day}.npy',mmap_mode='r')
        warm=np.load(FEAT/f'imp3_wf_day{day}.npy',mmap_mode='r')
        new=build(f'day{day}',a,hist,day)
        assert len(a)==len(base)==len(warm)==len(new)
        assert np.array_equal(base[:,4],a[:,3]) and np.array_equal(base[:,5],a[:,4])
        for start_row in range(0,len(a),100000):
            stop=min(start_row+100000,len(a));dest=slice(offset+start_row,offset+stop)
            output[dest,:190]=base[start_row:stop]
            output[dest,190:194]=warm[start_row:stop]
            output[dest,194:]=new[start_row:stop]
        targets.append(a[:,5].copy());offset+=len(a)
        del a,base,warm,new;gc.collect()
    output.flush();print('Training matrix',label,output.shape,flush=True)
    return output,np.concatenate(targets),p

def validation(day):
    _,a=split(day)
    ref=np.load(ROOT/f'predictions_imp3_wf3_day{day}.npz')
    assert np.array_equal(a[:,6],ref['events']) and np.array_equal(a[:,5],ref['y'])
    new=build(f'val{day}',a,hist,day)
    base=np.load(FEAT/f'stack_val{day}_c{day}.npy',mmap_mode='r')
    warm=np.load(FEAT/f'imp3_wf_val{day}.npy',mmap_mode='r')
    x=np.column_stack([base,warm,new]);sw,_=history_weights(ref['events'])
    return x,ref,sw

def candidate(day,component,mix):
    old=np.load(ROOT/f'predictions_imp3_warmfeat_day{day}.npz')['pred']
    v6=np.load(ROOT/f'predictions_v6_day{day}.npz')['pred']
    v7=np.load(ROOT/f'predictions_v7_day{day}.npz')['pred']
    warm=np.mean([np.load(ROOT/f'{prefix}_day{day}.npz')['raw'] for prefix in
                  ['imp2_warm1.0_d16_pc1','imp3_warm16_s7','imp3_warm16_s1009']],axis=0)
    member=(1-mix)*old+mix*component
    return .9*competitive(sigmoid(.7*logit(member)+.2*logit(v6)+.1*logit(v7)))+.1*warm

def run():
    reports=[];chosen_mix=None
    for day in [6,5]:
        print('DECOMPOSITION TEST DAY',day,flush=True)
        x,y,path=assemble_new([day-3,day-2,day-1],f'd{day}')
        xv,ref,sw=validation(day)
        model=CatBoostClassifier(iterations=TREES,**PARAMS,verbose=100)
        model.fit(x,y,eval_set=(xv,ref['y']),use_best_model=False)
        component=model.predict_proba(xv)[:,1]
        model.save_model(str(ROOT/f'imp4_decompose_validation_d{day}.cbm'))
        if chosen_mix is None:
            trials=[(roc_auc_score(ref['y'],candidate(day,component,m),sample_weight=sw),m) for m in [0,.5,1.]]
            chosen_mix=max(trials)[1]
            print('Selection grid',trials,'chosen mix',chosen_mix,flush=True)
        pred=candidate(day,component,chosen_mix)
        base=ref['pred'];old=float(roc_auc_score(ref['y'],base));new=float(roc_auc_score(ref['y'],pred))
        oldw=float(roc_auc_score(ref['y'],base,sample_weight=sw));neww=float(roc_auc_score(ref['y'],pred,sample_weight=sw))
        se=float(np.sqrt(np.sum((influence(ref['y'],pred)-influence(ref['y'],base))**2)))
        sew=float(np.sqrt(np.sum((weighted_influence(ref['y'],pred,sw)-weighted_influence(ref['y'],base,sw))**2)))
        r=dict(day=day,baseline='V17',auc=new,base_auc=old,delta=new-old,
               reweighted=neww,base_reweighted=oldw,delta_reweighted=neww-oldw,
               raw_auc=float(roc_auc_score(ref['y'],component)),trees=TREES,mix=chosen_mix,
               paired_interval=[new-old-1.96*se,new-old+1.96*se],
               weighted_paired_interval=[neww-oldw-1.96*sew,neww-oldw+1.96*sew])
        reports.append(r);print(json.dumps(r),flush=True)
        np.savez_compressed(ROOT/f'predictions_imp4_decompose_day{day}.npz',y=ref['y'],events=ref['events'],pred=pred,raw=component)
        (ROOT/'imp4_decompose_report.json').write_text(json.dumps(reports,indent=2))
        del x,y,xv,model;gc.collect();path.unlink()
        if new<=old or neww<=oldw:
            print('Rejected; V17 unchanged.',flush=True);return
    if reports[0]['weighted_paired_interval'][0]<=0:
        print('Gain inconclusive; V17 unchanged.',flush=True);return
    print('Both dates improved. Candidate ready for independent review before final refit.',flush=True)

if __name__=='__main__':run()
