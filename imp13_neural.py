"""A compact CPU neural learner on the unchanged causal CTR features."""
from imp6_warmcontext import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

CHUNK=65536
SETTINGS=dict(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=.1,
              batch_size=4096,learning_rate_init=.001,max_iter=1,shuffle=True,
              random_state=42,early_stopping=False)


def transform(scaler,x):
    out=scaler.transform(np.array(x,dtype=np.float32,copy=True))
    np.clip(out,-8,8,out=out)
    assert np.isfinite(out).all()
    return out


def neural_predict(model,scaler,x):
    parts=[]
    for off in range(0,len(x),CHUNK):
        parts.append(model.predict_proba(transform(scaler,x[off:off+CHUNK]))[:,1])
    return np.concatenate(parts)


def fit(day,chosen=None):
    configure();engine.TAG='neural'
    x,y,path=engine.assemble_aligned([day-3,day-2,day-1],f'd{day}')
    xv,prior,sw=engine.aligned_validation(day)
    ref=np.load(ROOT/f'imp11_bilinear_day{day}.npz')
    assert np.array_equal(prior['events'],ref['events']) and np.array_equal(prior['y'],ref['y'])
    scaler=StandardScaler()
    for off in range(0,len(x),CHUNK):scaler.partial_fit(x[off:off+CHUNK])
    assert int(scaler.n_samples_seen_)==len(x)
    model=MLPClassifier(**SETTINGS);rng=np.random.default_rng(42)
    base=ref['pred'];yv=ref['y'];old=float(roc_auc_score(yv,base));oldw=float(roc_auc_score(yv,base,sample_weight=sw))
    best=None;trials=[];calls=0
    for epoch in range(1,(chosen['epoch'] if chosen else 8)+1):
        t=time.time();offsets=np.arange(0,len(x),CHUNK);rng.shuffle(offsets)
        for off in offsets:
            stop=min(off+CHUNK,len(x))
            model.partial_fit(transform(scaler,x[off:stop]),y[off:stop],classes=np.array([0,1]))
            calls+=1
        print('Neural epoch',day,epoch,'seconds',time.time()-t,'last chunk loss',model.loss_,flush=True)
        if (chosen is None and epoch not in [2,4,6,8]) or (chosen is not None and epoch!=chosen['epoch']):continue
        raw=neural_predict(model,scaler,xv)
        assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
        for weight in ([chosen['weight']] if chosen else [.1,.25,.5,.75,1.]):
            pred=(1-weight)*base+weight*raw
            auc=float(roc_auc_score(yv,pred));weighted=float(roc_auc_score(yv,pred,sample_weight=sw))
            result=dict(day=day,epoch=epoch,weight=weight,auc=auc,reweighted=weighted,
                        baseline_auc=old,baseline_reweighted=oldw,raw_auc=float(roc_auc_score(yv,raw)))
            trials.append(result);print(json.dumps(result),flush=True)
            if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
                best=result.copy();se=np.linalg.norm(influence(yv,pred)-influence(yv,base))
                sew=np.linalg.norm(weighted_influence(yv,pred,sw)-weighted_influence(yv,base,sw))
                best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
                np.savez_compressed(ROOT/f'imp13_neural_day{day}.npz',y=yv,events=ref['events'],pred=pred,raw=raw)
                joblib.dump(dict(model=model,scaler=scaler,epoch=epoch,partial_fit_calls=calls,cutoff=cutoff_of(day),training_rows=len(x)),ROOT/f'imp13_neural_d{day}.joblib')
        (ROOT/f'imp13_neural_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    del x,y,xv,model;gc.collect();path.unlink()
    return best


if __name__=='__main__':
    output=ROOT/'imp13_neural_report.json';assert not output.exists(),'Do not restart a completed experiment.'
    with threadpool_limits(limits=4):
        selected=fit(6);reports=[selected];output.write_text(json.dumps(reports,indent=2))
        if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
            reports.append(fit(5,selected));output.write_text(json.dumps(reports,indent=2))
    print('Neural experiment complete; independent audit required for any gain.',flush=True)
