"""Select warm-sequence/contextual factor models on the unchanged local split."""
from imp7_sequence_core import *
from imp3_lib import history_weights
from tst2_candidate import influence,weighted_influence
import argparse

NAMES=['U','V','H','bu','bi','B','T']
ANAMES=['au','av','ah','abu','abi','aB','aT']


def sequence_data(day):
    path=ROOT/f'imp7_sequence_data_cut{day}.npz'
    if path.exists():
        with np.load(path) as f:
            assert int(f['cutoff'])==start+day*DAY
            return [f[k] for k in ['users','items','history','buckets','types']]
    a=np.asarray(train[train[:,2]<start+day*DAY]);e=a[::10]
    assert np.array_equal(e[:,3],e[:,2]%86400//21600)
    assert np.array_equal(e[:,4],(((e[:,2]//86400+3)%7)>=5).astype(np.int32))
    _,ix=np.unique(e[:,0],return_index=True);e=e[ix]
    assert len(np.unique(e[:,0]))==len(e)
    h=np.asarray(hist[e[:,6]]);ts=np.asarray(np.load(ROOT/'train_times.npy',mmap_mode='r')[e[:,6]])
    valid=h>=0
    assert np.all((ts<e[:,2,None])|~valid)
    assert np.all((ts[:,1:]>=ts[:,:-1])|~(valid[:,1:]&valid[:,:-1]))
    arrays=sequence_examples(e[:,0],h,ts)
    np.savez_compressed(path,**dict(zip(['users','items','history','buckets','types'],arrays)),cutoff=start+day*DAY)
    print('Warm sequence examples',day,len(arrays[0]),'duplicates removed',int(valid.sum())-len(arrays[0]),flush=True)
    return arrays


def negatives(rng,positive):
    result=np.empty((len(positive),11),np.int32)
    for j in range(11):
        result[:,j]=rng.integers(0,ni,size=len(positive),dtype=np.int32)
        bad=(result[:,j]==positive)|(result[:,:j]==result[:,j,None]).any(1)
        while bad.any():
            result[bad,j]=rng.integers(0,ni,size=int(bad.sum()),dtype=np.int32)
            bad=(result[:,j]==positive)|(result[:,:j]==result[:,j,None]).any(1)
    return result


def fit(day,baseline_prefix,chosen=None):
    a=np.asarray(train[train[:,2]<start+day*DAY])
    users,items,history,target=event_data(a,hist);buckets=a[::10,3].copy();types=a[::10,4].copy()
    wu,wi,wh,wb,wt=sequence_data(day);del a;gc.collect()
    pop=np.bincount(wi,minlength=ni)+1;pop_bias=np.log(pop/pop.mean()).astype(np.float32)
    ps,acc=init32(start+max(day-2,2)*DAY,16)
    ps.extend([np.zeros((ni,4),np.float32),np.zeros((ni,2),np.float32)])
    acc.extend([np.full_like(ps[-2],.1),np.full_like(ps[-1],.1)])
    _,val=split(day);vu,vi,vh,_=event_data(val,hist)
    ref=np.load(ROOT/f'predictions_{baseline_prefix}_day{day}.npz')
    assert np.array_equal(ref['events'],val[:,6]) and np.array_equal(ref['y'],val[:,5])
    sw,_=history_weights(ref['events']);y=ref['y'];base=ref['pred']
    base_auc=float(roc_auc_score(y,base));base_weighted=float(roc_auc_score(y,base,sample_weight=sw))
    rng=np.random.default_rng(151);best=None;all_trials=[]
    for epoch in range(1,(chosen['epoch'] if chosen else 10)+1):
        t=time.time();neg=negatives(rng,wi)
        warm_loss=sequence_epoch(rng.permutation(len(wu)),wu,wi,wh,wb,wt,neg,*ps,*acc,.045,.002,.01,1.,pop_bias)
        event_loss=contextual_epoch(rng.permutation(len(users)),users,items,history,target,buckets,types,*ps,*acc,.045,.002,.01)
        print('Sequence fit',day,epoch,'loss',warm_loss,event_loss,'seconds',time.time()-t,flush=True)
        if (chosen is None and epoch not in [2,4,6,8,10]) or (chosen is not None and epoch!=chosen['epoch']):continue
        raw=contextual_predict(vu,vi,vh,val[::10,3],val[::10,4],*ps).ravel()
        assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
        assert np.all(raw.reshape(-1,10).sum(1)<=1+1e-12)
        weights=[chosen['weight']] if chosen else [0.,.1,.2,.3,.5,.75,1.]
        for weight in weights:
            pred=(1-weight)*base+weight*raw
            auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
            trial=dict(day=day,epoch=epoch,weight=weight,auc=auc,reweighted=weighted,
                       base_auc=base_auc,base_reweighted=base_weighted,baseline_prefix=baseline_prefix,
                       raw_auc=float(roc_auc_score(y,raw)))
            all_trials.append(trial)
            eligible=(auc>base_auc and weighted>base_weighted)
            if chosen is not None or (eligible and (best is None or auc>best['auc'])):
                best=trial.copy()
                se=np.linalg.norm(influence(y,pred)-influence(y,base))
                sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
                best.update(delta=auc-base_auc,delta_reweighted=weighted-base_weighted,
                            paired_interval=[float(auc-base_auc-1.96*se),float(auc-base_auc+1.96*se)],
                            weighted_paired_interval=[float(weighted-base_weighted-1.96*sew),float(weighted-base_weighted+1.96*sew)])
                np.savez_compressed(ROOT/f'imp7_sequence_day{day}.npz',y=y,events=ref['events'],raw=raw,pred=pred)
                state=dict(zip(NAMES,ps));state.update(dict(zip(ANAMES,acc)))
                state.update(epoch=epoch,cutoff=start+day*DAY,pop_bias=pop_bias,rng_state=json.dumps(rng.bit_generator.state))
                np.savez_compressed(ROOT/f'imp7_sequence_model_cut{day}.npz',**state)
        print('Sequence validation',day,epoch,'raw AUC',float(roc_auc_score(y,raw)),'best',json.dumps(best),flush=True)
        (ROOT/f'imp7_sequence_trials_day{day}.json').write_text(json.dumps(all_trials,indent=2))
    return best


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--baseline-prefix',default='imp5_aligned');args=ap.parse_args()
    output=ROOT/'imp7_sequence_report.json'
    assert not output.exists(),'Inspect the existing report; do not restart a live or completed experiment.'
    selected=fit(6,args.baseline_prefix)
    reports=[selected];output.write_text(json.dumps(reports,indent=2))
    if selected is None or selected['weighted_paired_interval'][0]<=0:
        print('No confirmed selection gain; preserve baseline.',flush=True)
    else:
        confirmation=fit(5,args.baseline_prefix,selected)
        reports.append(confirmation);output.write_text(json.dumps(reports,indent=2))
        print('Confirmation complete; independent audit required. Goal0.83 still requires verification.',flush=True)
