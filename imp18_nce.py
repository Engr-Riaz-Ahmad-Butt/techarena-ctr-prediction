"""Joint real Bernoulli / warm conditional NCE; frozen first-event cohorts."""
from imp15_als import *
import imp18_nce_core as core
import hashlib


def fit(day,baseline,chosen=None):
    cutoff=cutoff_of(day);rows=np.asarray(train[train[:,2]<cutoff])
    users,items,history,_=event_data(rows,hist)
    target=rows[:,5].reshape(-1,10).astype(np.float32).copy()
    stamps=np.load(ROOT/'train_times.npy',mmap_mode='r')[rows[::10,6]]
    assert np.all((stamps<rows[::10,2,None])|(history<0))
    wu,wi=warm_data(rows);assert rows[:,2].max()<cutoff
    counts=np.bincount(wi,minlength=ni)+1
    offset=np.log(counts/counts.mean()).astype(np.float32)
    del rows,stamps;gc.collect()
    _,val=split(day);vu,vi,vh,_=event_data(val,hist)
    ref=np.load(ROOT/f'{baseline}_day{day}.npz')
    assert np.array_equal(ref['events'],val[:,6]) and np.array_equal(ref['y'],val[:,5])
    y=ref['y'];base=ref['pred'];sw,_=history_weights(ref['events'])
    br=rankdata(base,method='average')/(len(y)+1)
    old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    initial_path=model_path(day)
    initial_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest()
    with np.load(initial_path) as m:
        ps=[m[k].astype(np.float32).copy() for k in ['U','V','H','bu','bi']]
    assert ps[0].shape==(nu,16) and ps[1].shape==(ni,16)
    acc=[np.full_like(p,.1) for p in ps]
    g=np.array([-1.8],np.float32);ag=np.array([.1],np.float32)
    aw=np.full(nu,-np.log(9),np.float32);aaw=np.full(nu,.1,np.float32)
    rng=np.random.default_rng(42);trials=[];best=None
    for epoch in range(1,(chosen['epoch'] if chosen else 6)+1):
        started=time.time()
        noise=rng.integers(0,ni,size=(len(wu),9),dtype=np.int32)
        contrast=np.column_stack([wi,noise])
        wl=core.warm_epoch(rng.permutation(len(wu)),wu,contrast,ps[0],ps[1],aw,acc[0],acc[1],aaw,offset,.02,.002)
        rl=core.real_epoch(rng.permutation(len(users)),users,items,history,target,*ps,g,*acc,ag,.02,.002)
        assert all(np.isfinite(p).all() for p in [*ps,*acc,g,ag,aw,aaw])
        print('NCE epoch',day,epoch,'warm_loss',wl,'real_loss',rl,'seconds',time.time()-started,flush=True)
        if (chosen is None and epoch not in [2,4,6]) or (chosen and epoch!=chosen['epoch']):continue
        raw=core.predict(vu,vi,vh,*ps,g);assert np.isfinite(raw).all()
        rr=rankdata(raw,method='average')/(len(y)+1)
        for weight in ([chosen['weight']] if chosen else [.05,.1,.2]):
            pred=(1-weight)*br+weight*rr
            auc=float(roc_auc_score(y,pred));wa=float(roc_auc_score(y,pred,sample_weight=sw))
            record=dict(day=day,epoch=epoch,weight=weight,auc=auc,reweighted=wa,baseline_auc=old,baseline_reweighted=oldw,raw_auc=float(roc_auc_score(y,raw)),baseline=baseline,cutoff=cutoff,warm_records=len(wu),real_events=len(users))
            trials.append(record);print(json.dumps(record),flush=True)
            if chosen or (auc>old and wa>oldw and (best is None or auc>best['auc'])):
                se=np.linalg.norm(influence(y,pred)-influence(y,base))
                sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
                best=dict(record,delta=auc-old,delta_reweighted=wa-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[wa-oldw-1.96*sew,wa-oldw+1.96*sew])
                np.savez_compressed(ROOT/f'imp18_nce_day{day}.npz',y=y,events=ref['events'],pred=pred,raw=raw)
                state={k:v for k,v in zip(['U','V','H','bu','bi'],ps)}
                state.update({f'a{k}':v for k,v in zip(['U','V','H','bu','bi'],acc)})
                state.update(g=g,ag=ag,aw=aw,aaw=aaw,offset=offset,cutoff=cutoff,epoch=epoch,rng_state=json.dumps(rng.bit_generator.state),baseline=baseline,seed=42,
                             initial_sha256=initial_sha256,initial_checkpoint=initial_path.name,
                             core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest())
                np.savez_compressed(ROOT/f'imp18_nce_model_day{day}.npz',**state)
        (ROOT/f'imp18_nce_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    return best


if __name__=='__main__':
    output=ROOT/'imp18_nce_report.json';assert not output.exists(),'Completed experiment must not be restarted'
    # Independent core audit and explicit incumbent freeze must precede fitting.
    check=json.loads((ROOT/'imp18_test_core_report.json').read_text())
    assert check['status']=='PASS'
    source=Path(core.__file__).resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==check['core_sha256'],'Core audit is stale'
    frozen=json.loads((ROOT/'imp18_baseline.json').read_text());baseline=frozen['baseline']
    for day in [6,5]:
        path=ROOT/f'{baseline}_day{day}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['sha256'][str(day)]
    selected=fit(6,baseline);reports=[selected];output.write_text(json.dumps(reports,indent=2))
    if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
        reports.append(fit(5,baseline,selected));output.write_text(json.dumps(reports,indent=2))
    print('NCE experiment complete; independent candidate audit required.',flush=True)
