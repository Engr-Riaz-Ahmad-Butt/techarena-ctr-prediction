"""Causal aligned warm-factor component interactions, compared with V18."""
from imp4_decompose import *

TAG='aligned'
WIDTH=258
SELECTION_FLOOR=None
BASELINE_NAME='V18'


def norm(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def aligned_features(name,rows,histories,cut):
    dest=FEAT/f'imp5_aligned_{name}.npy'
    if dest.exists():
        x=np.load(dest,mmap_mode='r');assert x.shape==(len(rows),32);return x
    with np.load(model_path(2)) as ref:
        anchor=norm(ref['V'].astype(np.float64))
    with np.load(model_path(cut)) as source:
        U,V,H=[source[k] for k in ['U','V','H']]
    current=norm(V.astype(np.float64))
    l,s,rt=np.linalg.svd(current.T@anchor,full_matrices=False);rotation=l@rt
    orth_error=float(np.max(np.abs(rotation.T@rotation-np.eye(16))))
    assert orth_error<1e-10
    sample=np.arange(min(len(U),len(V),1000))
    dot=np.einsum('ij,ij->i',U[sample].astype(float),V[sample].astype(float))
    ur=U.astype(float)@rotation;vr=V.astype(float)@rotation;hr=H.astype(float)@rotation
    dot_error=float(np.max(np.abs(dot-np.einsum('ij,ij->i',ur[sample],vr[sample]))))
    assert dot_error<1e-9
    np.savez_compressed(ROOT/f'imp5_rotation_cut{cut}.npz',rotation=rotation,singular_values=s,
                        orthogonality_error=orth_error,dot_error=dot_error)
    ur=norm(ur).astype(np.float32);vr=norm(vr).astype(np.float32);hr=hr.astype(np.float32)
    x=np.empty((len(rows),32),dtype=np.float32)
    for off in range(0,len(rows),20000):
        a=rows[off:off+20000]
        assert np.all(a[:,6].reshape(-1,10)==a[::10,6,None])
        users=a[::10,0];items=a[:,1].reshape(-1,10)
        history=np.asarray(histories[a[::10,6]]);valid=history>=0
        z=(hr[np.maximum(history,0)]*valid[:,:,None]).sum(1)
        z=norm(z);v=vr[items];u=ur[users]
        direct=u[:,None,:]*v;dynamic=z[:,None,:]*v
        x[off:off+len(a)]=np.concatenate([direct,dynamic],axis=2).reshape(-1,32)
    assert np.isfinite(x).all()
    np.save(dest,x)
    print('Aligned features',name,x.shape,'orthogonality',orth_error,'dot error',dot_error,flush=True)
    return x


def assemble_aligned(days,label):
    total=sum(len(np.load(FEAT/f'stack_day{day}_c{day}.npy',mmap_mode='r')) for day in days)
    path=FEAT/f'imp5_train_{TAG}_{label}.npy'
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(total,WIDTH))
    targets=[];offset=0
    for day in days:
        a=np.asarray(train[(train[:,2]>=cutoff_of(day))&(train[:,2]<cutoff_of(day+1))])
        base=np.load(FEAT/f'stack_day{day}_c{day}.npy',mmap_mode='r')
        warm=np.load(FEAT/f'imp3_wf_day{day}.npy',mmap_mode='r')
        decomp=build(f'day{day}',a,hist,day)
        aligned=aligned_features(f'day{day}',a,hist,day)
        assert len(a)==len(base)==len(warm)==len(decomp)==len(aligned)
        assert np.array_equal(base[:,4],a[:,3]) and np.array_equal(base[:,5],a[:,4])
        for off in range(0,len(a),100000):
            stop=min(off+100000,len(a));sl=slice(offset+off,offset+stop)
            x[sl,:190]=base[off:stop];x[sl,190:194]=warm[off:stop]
            x[sl,194:226]=decomp[off:stop];x[sl,226:]=aligned[off:stop]
        targets.append(a[:,5].copy());offset+=len(a)
        del a,base,warm,decomp,aligned;gc.collect()
    x.flush();print('Aligned training matrix',label,x.shape,flush=True)
    return x,np.concatenate(targets),path


def aligned_validation(day):
    x,oldref,sw=validation(day);_,a=split(day)
    ref=np.load(ROOT/f'predictions_imp4_decompose_day{day}.npz')
    assert np.array_equal(ref['events'],oldref['events']) and np.array_equal(ref['y'],oldref['y'])
    return np.column_stack([x,aligned_features(f'val{day}',a,hist,day)]),ref,sw


def aligned_candidate(day,raw,mix):
    old=np.load(ROOT/f'predictions_imp4_decompose_day{day}.npz')['raw']
    return candidate(day,(1-mix)*old+mix*raw,1.)


def run_aligned():
    reports=[];mix=None
    for day in [6,5]:
        print('ALIGNED TEST DAY',day,flush=True)
        x,y,path=assemble_aligned([day-3,day-2,day-1],f'd{day}')
        xv,ref,sw=aligned_validation(day)
        model=CatBoostClassifier(iterations=TREES,**PARAMS,verbose=100)
        model.fit(x,y,eval_set=(xv,ref['y']),use_best_model=False)
        raw=model.predict_proba(xv)[:,1]
        model.save_model(str(ROOT/f'imp5_{TAG}_d{day}.cbm'))
        if mix is None:
            grid=[(roc_auc_score(ref['y'],aligned_candidate(day,raw,m),sample_weight=sw),m) for m in [0.,.5,1.]]
            mix=max(grid)[1];print('Frozen selection',grid,'mix',mix,flush=True)
        pred=aligned_candidate(day,raw,mix);base=ref['pred'];y=ref['y']
        old=float(roc_auc_score(y,base));new=float(roc_auc_score(y,pred))
        oldw=float(roc_auc_score(y,base,sample_weight=sw));neww=float(roc_auc_score(y,pred,sample_weight=sw))
        se=float(np.linalg.norm(influence(y,pred)-influence(y,base)))
        sew=float(np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw)))
        r=dict(day=day,baseline=BASELINE_NAME,auc=new,base_auc=old,delta=new-old,
               reweighted=neww,base_reweighted=oldw,delta_reweighted=neww-oldw,
               raw_auc=float(roc_auc_score(y,raw)),trees=TREES,mix=mix,
               paired_interval=[new-old-1.96*se,new-old+1.96*se],
               weighted_paired_interval=[neww-oldw-1.96*sew,neww-oldw+1.96*sew])
        reports.append(r);print(json.dumps(r),flush=True)
        np.savez_compressed(ROOT/f'predictions_imp5_{TAG}_day{day}.npz',y=y,events=ref['events'],pred=pred,raw=raw)
        (ROOT/f'imp5_{TAG}_report.json').write_text(json.dumps(reports,indent=2))
        del x,y,xv,model;gc.collect();path.unlink()
        if new<=old or neww<=oldw:
            print('Rejected; V18 preserved.',flush=True);return
        if day==6 and SELECTION_FLOOR is not None and neww<=SELECTION_FLOOR:
            print('Below independently confirmed selection incumbent; no confirmation refit.',flush=True);return
    print('Both dates improved; independent audit required.',flush=True)


if __name__=='__main__':run_aligned()
