"""Approved Idea2 step2 only. Own code; fixed, causal probability comparison."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import time,json,hashlib,pickle
import numpy as np
from scipy.special import expit,logit,logsumexp,softmax
from scipy.optimize import minimize,minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
import lightgbm as lgb
from threadpoolctl import threadpool_limits

def ecdf(sorted_p,x):return (np.searchsorted(sorted_p,x,'left')+np.searchsorted(sorted_p,x,'right'))/(2*len(sorted_p))
def platt_fit(p,y):
    s=logit(np.clip(p,1e-6,1-1e-6))
    def loss(ab):
        z=ab[0]*s+ab[1];return float(np.mean(np.logaddexp(0,z)-y*z)+.001*(ab[0]-1)**2)
    opt=minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(0,10),(-10,10)]);assert opt.success;return opt.x
def platt(ab,p):return expit(ab[0]*logit(np.clip(p,1e-6,1-1e-6))+ab[1])
def clf(kind,plan):return make_pipeline(StandardScaler(),LogisticRegression(**plan['logistic'])) if kind=='logistic' else lgb.LGBMClassifier(**plan['boosted'])
class Plan:
    def __init__(self,y,p,w):
        self.o=np.argsort(p,kind='stable');self.y=y[self.o];self.w=w[self.o];s=p[self.o];self.starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    def value(self,m):
        w=self.w*m[self.o];p=np.add.reduceat(w*self.y,self.starts);n=np.add.reduceat(w*(1-self.y),self.starts);return float(p@(np.cumsum(n)-.5*n)/(p.sum()*n.sum()))
def compare(y,c,p,w,users,mask,boots,seed):
    yy=y[mask].ravel();cc=c[mask].ravel();pp=p[mask].ravel();ww=np.repeat(w[mask],10);_,u=np.unique(users[mask],return_inverse=True);nu=u.max()+1;cp=Plan(yy,cc,ww);qp=Plan(yy,pp,ww);one=np.ones(len(yy));cv=cp.value(one);pv=qp.value(one);assert abs(cv-roc_auc_score(yy,cc,sample_weight=ww))<1e-10
    rng=np.random.default_rng(seed);ds=[]
    for _ in range(boots):
        count=np.bincount(rng.integers(0,nu,nu),minlength=nu);mult=np.repeat(count[u],10);ds.append(qp.value(mult)-cp.value(mult))
    return dict(events=int(mask.sum()),users=int(nu),positive_rows=int(yy.sum()),control_auc=cv,candidate_auc=pv,delta=pv-cv,unweighted_control_auc=float(roc_auc_score(yy,cc)),unweighted_candidate_auc=float(roc_auc_score(yy,pp)),paired95=np.quantile(ds,[.025,.975]).tolist(),bootstrap_replicates=boots,effective_events=float(w[mask].sum()**2/np.square(w[mask]).sum()))

def run_case(file,out,plan,measurement_model=None):
    start=time.monotonic();d=np.load(file);out=Path(out);out.mkdir(parents=True,exist_ok=True);xf=d['xfit'];xv=d['xval'];yf=d['yfit'];event_y=yf.sum(1);cal=d['cal'];fit=~cal;assert d['fit_ts'][fit].max()<d['fit_ts'][cal].min()<=d['fit_ts'].max()<d['val_ts'].min();assert (yf.sum(1)<=1).all()
    mappings=[]
    if 'fit_score' in d:
        prior=np.sort(d['fit_score'][fit].ravel());sf=ecdf(prior,d['fit_score']);sv=ecdf(prior,d['val_score']);mappings=[prior]
    else:
        for name in ['warm','als']:mappings.append(np.sort(d['fit_'+name][fit].ravel()))
        sf=.95*ecdf(mappings[0],d['fit_warm'])+.05*ecdf(mappings[1],d['fit_als']);sv=.95*ecdf(mappings[0],d['val_warm'])+.05*ecdf(mappings[1],d['val_als'])
    selected=cal&(event_y==1);target=yf[selected].argmax(1);z=sf[selected]
    def choice_loss(log_a):
        a=np.exp(log_a);return float(np.mean(logsumexp(a*z,axis=1)-a*z[np.arange(len(z)),target]))
    opt=minimize_scalar(choice_loss,bounds=(np.log(.01),np.log(100)),method='bounded');assert opt.success;a=float(np.exp(opt.x));q=softmax(a*sv,axis=1);assert np.allclose(q.sum(1),1.)
    ef=np.column_stack([sf.max(1),sf.mean(1),xf[:,4],xf[:,3]]);ev=np.column_stack([sv.max(1),sv.mean(1),xv[:,4],xv[:,3]]);allf=np.column_stack([xf,ef[:,:2]]);allv=np.column_stack([xv,ev[:,:2]]);features={'basic':(xf[:,plan['basic_features']],xv[:,plan['basic_features']]),'existing':(ef,ev),'augmented':(allf,allv)}
    pred={};models={};calibrators={}
    # Logistic first, then boosted; all choices/settings fixed in the plan.
    for kind in ['logistic','boosted']:
        for name,(x,xx) in features.items():
            if kind=='boosted' and name=='basic':x,xx=xf,xv
            key=kind+'_'+name;temporary=clf(kind,plan);temporary.fit(x[fit],event_y[fit]);oof=temporary.predict_proba(x[cal])[:,1];ab=platt_fit(oof,event_y[cal]);calibrators[key]=ab.tolist()
            if kind=='boosted' and name=='basic' and measurement_model is not None:
                model=lgb.Booster(model_file=str(measurement_model));pv=model.predict(xx);model.save_model(str(out/(key+'.txt')))
                assert np.max(np.abs(pv-d['measurement_val_probability']))<1e-12
            else:
                model=clf(kind,plan);model.fit(x,event_y);pv=model.predict_proba(xx)[:,1]
            pred[key]=platt(ab,pv);models[key]=model
            print(json.dumps(dict(stage='event_model_frozen',case=Path(file).stem,model=key,calibration=ab.tolist(),elapsed=time.monotonic()-start)),flush=True)
    prior=float(event_y.mean());scores={'choice_constant':prior*q};scores.update({k:v[:,None]*q for k,v in pred.items()});assert all(np.isfinite(v).all() and (v>=0).all() and (v<=1).all() for v in scores.values())
    for k,p in pred.items():assert np.allclose(scores[k].sum(1),p)
    np.savez_compressed(out/'frozen_predictions.npz',conditional_choice=q,**scores,**{'event_'+k:v for k,v in pred.items()});(out/'models.pkl').write_bytes(pickle.dumps(dict(models=models,calibrators=calibrators,cdfs=mappings,inverse_temperature=a,prior=prior,plan=plan)))
    # Only now load held-out labels; no evaluation outcomes participate in fits.
    y=d['yval'];assert (y.sum(1)<=1).all();w=d['weight'];user=d['val_user'];hist=d['history_length'];groups=[('overall',np.ones(len(y),bool))]
    if 'arrival' in d:groups += [('arrival1',d['arrival']==0),('arrival2',d['arrival']==1),('arrival3plus',d['arrival']>=2)]
    groups += [('history0',hist==0),('history1to5',(hist>=1)&(hist<=5)),('history6to20',hist>=6)]
    comparisons=[('logistic_vs_choice','choice_constant','logistic_basic'),('boosted_vs_choice','choice_constant','boosted_basic'),('logistic_beyond_existing','logistic_existing','logistic_augmented'),('boosted_beyond_existing','boosted_existing','boosted_augmented')];metrics=[]
    for label,c,p in comparisons:
        for group,mask in groups:
            r=compare(y,scores[c],scores[p],w,user,mask,plan['bootstrap_full'],542201);r.update(comparison=label,group=group,passes_point_gate=r['delta']>=plan['gate']);metrics.append(r)
        print(json.dumps(dict(stage='paired_metrics_complete',case=Path(file).stem,comparison=label,overall=metrics[-len(groups)],elapsed=time.monotonic()-start)),flush=True)
    screen=[]
    if 'sample' in d:
        s=d['sample']
        for label,c,p in comparisons:
            r=compare(y[s],scores[c][s],scores[p][s],d['sample_weight'],user[s],np.ones(s.sum(),bool),plan['bootstrap_screen'],542201);r.update(comparison=label,group='screen20_overall');screen.append(r)
    calibration_metrics={}
    for k,p in scores.items():
        if k!='choice_constant':
            e=pred[k];calibration_metrics[k]=dict(event_auc=float(roc_auc_score(y.sum(1),e,sample_weight=w)),event_brier=float(brier_score_loss(y.sum(1),e,sample_weight=w)),row_brier=float(brier_score_loss(y.ravel(),p.ravel(),sample_weight=np.repeat(w,10))),row_logloss=float(log_loss(y.ravel(),p.ravel(),sample_weight=np.repeat(w,10),labels=[0,1])),weighted_predicted_event_click_share=float(np.average(e,weights=w)),weighted_actual_event_click_share=float(np.average(y.sum(1),weights=w)))
    refs=dict(original_saved_score_auc=float(roc_auc_score(y.ravel(),d['original_val_score'].ravel(),sample_weight=np.repeat(w,10))),conditional_choice_only_auc=float(roc_auc_score(y.ravel(),q.ravel(),sample_weight=np.repeat(w,10))))
    result=dict(case=Path(file).stem,metrics=metrics,screen=screen,calibration_metrics=calibration_metrics,references=refs,inverse_softmax_temperature=a,choice_calibration_events=int(selected.sum()),choice_calibration_nll=choice_loss(opt.x),event_platt_parameters=calibrators,training_events=len(yf),event_model_initial_fit_events=int(fit.sum()),calibration_events=int(cal.sum()),last_fit_label_timestamp=int(d['fit_ts'].max()),first_validation_timestamp=int(d['val_ts'].min()),elapsed_seconds=time.monotonic()-start,validation_labels_used_for_fitting=False,external_data=False,holdout_opened=False,submission_created=False)
    (out/'report.json').write_text(json.dumps(result,indent=2));return result

def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True);tic=time.monotonic();plan=json.loads((root/'frozen_plan.json').read_text());manifest=json.loads((root/'manifest.json').read_text());assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==m['sha256'] for n,m in manifest.items())
    with threadpool_limits(limits=1):
        horizon=run_case(root/'horizon_product.npz',out/'horizon',plan,root/'measurement_event_model.txt');saved=run_case(root/'saved_v53_product.npz',out/'saved_v53',plan)
    result=dict(horizon=horizon,saved_v53=saved,total_seconds=time.monotonic()-tic,stage='Step2 complete; pause; no evaluation scoring',limitations=['Horizon uses causal simplified V53-component blend, not full V53.','Historical training horizon2days versus4validation days; evaluation arrivaldays5to7 unsupported.','Saved V53 diagnostic tests day6 only and uses day5 user-crossfit calibration scores; it is not a second independent temporal validation or a horizon curve.','Saved V53 rank inputs inherit legacy cohort rank transforms; any future stateless application requires replacing rank operations with frozen historical CDFs and verifying the whole pipeline.','Bootstrap conditions on fitted models, calibrators, fixed weights and reused cohorts; no multiple-testing adjustment.'])
    (out/'report.json').write_text(json.dumps(result,indent=2));print('TWO_PART_MEASUREMENT_COMPLETE_PAUSE',json.dumps(dict(total_seconds=result['total_seconds'],horizon=[v for v in horizon['metrics'] if v['group']=='overall'],saved_v53=[v for v in saved['metrics'] if v['group']=='overall'])),flush=True);return result
