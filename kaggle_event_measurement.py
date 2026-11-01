"""Portable rerun of Idea2 measurement ONLY on frozen causal input features."""
from pathlib import Path
import json,time,hashlib,platform
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,brier_score_loss

def run(bundle,output):
    tic=time.monotonic();bundle=Path(bundle);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    d=np.load(bundle);spec=json.loads(str(d['spec']));assert int(d['train_max_timestamp'])<int(d['validation_min_timestamp']);assert len(np.unique(d['validation_users']))==len(d['y']);assert ((d['y']==0)|(d['y']==1)).all()
    model=lgb.LGBMClassifier(**spec['model_params']);model.fit(d['train_features'],d['train_y'])
    # Freeze all predictions before examining validation outcomes for metrics.
    p=model.predict_proba(d['validation_features'])[:,1];assert np.isfinite(p).all();np.save(output/'event_probability.npy',p);model.booster_.save_model(str(output/'event_model.txt'))
    y=d['y'];w=d['event_weight'];arrival=d['arrival'];results=[]
    def plan(y,p,w):
        order=np.argsort(p,kind='stable');s=p[order];return order,y[order],w[order],np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    def value(pl,m):
        order,yy,ww,starts=pl;ww=ww*m[order];pos=np.add.reduceat(ww*yy,starts);neg=np.add.reduceat(ww*(1-yy),starts);return float(pos@(np.cumsum(neg)-.5*neg)/(pos.sum()*neg.sum()))
    def report(name,mask,yy=y,pp=p,ww=w,base=d['baseline_event_max']):
        yy,pp,ww,base=yy[mask],pp[mask],ww[mask],base[mask];pl=plan(yy,pp,ww);bl=plan(yy,base,ww);rng=np.random.default_rng(542001);a=[];delta=[]
        for _ in range(400):
            m=np.bincount(rng.integers(0,len(yy),len(yy)),minlength=len(yy));v=value(pl,m);a.append(v);delta.append(v-value(bl,m))
        return dict(group=name,events=len(yy),clicked_events=int(yy.sum()),weighted_event_auc=float(roc_auc_score(yy,pp,sample_weight=ww)),unweighted_event_auc=float(roc_auc_score(yy,pp)),auc95=np.quantile(a,[.025,.975]).tolist(),baseline_max_event_auc=float(roc_auc_score(yy,base,sample_weight=ww)),paired_delta95=np.quantile(delta,[.025,.975]).tolist(),weighted_brier=float(brier_score_loss(yy,pp,sample_weight=ww)))
    for name,mask in [('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',arrival==k) for k in range(4)]+[('arrival3plus',arrival>=2)]:results.append(report(name,mask))
    s=d['sample'];results.append(report('screen20_overall',np.ones(int(s.sum()),bool),y[s],p[s],d['sample_event_weight'],d['baseline_event_max'][s]))
    result=dict(stage='Idea2 step1 only; no combined item model',runtime_seconds=time.monotonic()-tic,python=platform.python_version(),lightgbm=lgb.__version__,bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),prediction_max_abs_difference_from_local=float(np.max(np.abs(p-d['local_event_probability']))),results=results,gate_passed=results[0]['weighted_event_auc']>=.53,holdout_in_bundle=False,submissions_created=False,protected_originals_present=False,code_origin='Own pipeline, no repository code copied',limitations=['Frozen causal features prepared locally; only small event-click classifier is retrained on Kaggle.','Training arrival horizon2days; validation4days; evaluation tail days5to7 unsupported.','Event AUC gain is not global candidate row-AUC gain.','Full V53 not rebuilt.'])
    (output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print('MEASUREMENT_COMPLETE_PAUSE_BEFORE_PROBABILITY_PRODUCT',flush=True);return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('bundle');ap.add_argument('--output',default='/kaggle/working/event_measurement_results');args=ap.parse_args();run(args.bundle,args.output)
