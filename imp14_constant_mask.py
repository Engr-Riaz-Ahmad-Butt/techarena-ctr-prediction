"""Ignore unlearnable constant training inputs during frozen neural inference."""
from imp13_neural import *
import hashlib


def masked_predict(model,scaler,x):
    mask=scaler.var_<=1e-12
    parts=[]
    for off in range(0,len(x),CHUNK):
        value=transform(scaler,x[off:off+CHUNK])
        value[:,mask]=0
        parts.append(model.predict_proba(value)[:,1])
    return np.concatenate(parts)


if __name__=='__main__':
    output=ROOT/'imp14_constant_mask_report.json'
    assert not output.exists(),'Preserve the existing experiment.'
    reports=[]
    with threadpool_limits(limits=4):
        configure()
        for day in [6,5]:
            modelpath=ROOT/f'imp13_neural_d{day}.joblib'
            digest=hashlib.sha256(modelpath.read_bytes()).hexdigest()
            bundle=joblib.load(modelpath)
            assert bundle['epoch']==6 and int(bundle['cutoff'])==cutoff_of(day)
            xv,prior,sw=engine.aligned_validation(day)
            baseline=np.load(ROOT/f'imp11_bilinear_day{day}.npz')
            assert np.array_equal(prior['events'],baseline['events']) and np.array_equal(prior['y'],baseline['y'])
            raw=masked_predict(bundle['model'],bundle['scaler'],xv)
            y=baseline['y'];base=baseline['pred'];pred=.75*base+.25*raw
            assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
            old=float(roc_auc_score(y,base));auc=float(roc_auc_score(y,pred))
            oldw=float(roc_auc_score(y,base,sample_weight=sw));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
            se=np.linalg.norm(influence(y,pred)-influence(y,base))
            sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
            result=dict(day=day,epoch=6,weight=.25,auc=auc,reweighted=weighted,baseline_auc=old,baseline_reweighted=oldw,
                        raw_auc=float(roc_auc_score(y,raw)),delta=auc-old,delta_reweighted=weighted-oldw,
                        paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew],
                        masked_columns=np.flatnonzero(bundle['scaler'].var_<=1e-12).tolist(),frozen_model_sha256=digest)
            assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==digest
            reports.append(result);output.write_text(json.dumps(reports,indent=2));print(json.dumps(result),flush=True)
            np.savez_compressed(ROOT/f'imp14_constant_mask_day{day}.npz',y=y,events=baseline['events'],pred=pred,raw=raw)
            del xv,bundle;gc.collect()
    print('Inference-only test complete; require independent both-date audit for acceptance.',flush=True)
