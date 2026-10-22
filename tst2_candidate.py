"""Independently audit saved candidates against the verified V11 baseline."""
import argparse,json,sys
from pathlib import Path
from config import ARTIFACT_DIR as ROOT
import numpy as np
from sklearn.metrics import roc_auc_score
def influence(y,p):
    positive=np.sort(p[y==1]);negative=np.sort(p[y==0]);auc=roc_auc_score(y,p)
    out=np.empty(len(y));mask=y==1
    out[mask]=((np.searchsorted(negative,p[mask],'left')+np.searchsorted(negative,p[mask],'right'))/(2*len(negative))-auc)/len(positive)
    out[~mask]=(1-(np.searchsorted(positive,p[~mask],'left')+np.searchsorted(positive,p[~mask],'right'))/(2*len(positive))-auc)/len(negative)
    return out.reshape(-1,10).sum(axis=1)
def weighted_influence(y,p,w):
    def cdf(values,weights,query):
        ix=np.argsort(values);values=values[ix];weights=weights[ix]
        cumulative=np.r_[0.,np.cumsum(weights)]
        return (cumulative[np.searchsorted(values,query,'left')]+cumulative[np.searchsorted(values,query,'right')])/(2*weights.sum())
    mask=y==1;auc=roc_auc_score(y,p,sample_weight=w);out=np.empty(len(y))
    out[mask]=w[mask]/w[mask].sum()*(cdf(p[~mask],w[~mask],p[mask])-auc)
    out[~mask]=w[~mask]/w[~mask].sum()*(1-cdf(p[mask],w[mask],p[~mask])-auc)
    return out.reshape(-1,10).sum(axis=1)
def audit(path,day,weight,gate=False,raw_prefix=None):
    d=np.load(path);baseline=np.load(ROOT/f'predictions_v11_day{day}.npz')
    assert np.array_equal(d['events'],baseline['events']) and np.array_equal(d['y'],baseline['y'])
    ev=d['events'].reshape(-1,10);assert (ev==ev[:,:1]).all() and len(np.unique(ev[:,0]))==len(ev)
    raw=d['raw'] if 'raw' in d.files else np.load(ROOT/f'{raw_prefix}_day{day}.npz')['raw']
    for p in [d['pred'],raw]:
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    if gate:
        hist=np.load(ROOT/'train_hist.npy',mmap_mode='r')
        weight=weight*((hist[d['events']]>=0).sum(axis=1)<=4)
    error=float(np.max(np.abs(d['pred']-((1-weight)*baseline['pred']+weight*raw))))
    assert error<1e-12
    old=float(roc_auc_score(d['y'],baseline['pred']));new=float(roc_auc_score(d['y'],d['pred']))
    delta=new-old;se=float(np.sqrt(np.sum((influence(d['y'],d['pred'])-influence(d['y'],baseline['pred']))**2)))
    hist=np.load(ROOT/'train_hist.npy',mmap_mode='r');eh=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
    vg=np.digitize((hist[d['events']]>=0).sum(1),[0,4,9,19],right=True)
    eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
    sw=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
    weighted_old=float(roc_auc_score(d['y'],baseline['pred'],sample_weight=sw));weighted_new=float(roc_auc_score(d['y'],d['pred'],sample_weight=sw))
    weighted_delta=weighted_new-weighted_old
    weighted_se=float(np.sqrt(np.sum((weighted_influence(d['y'],d['pred'],sw)-weighted_influence(d['y'],baseline['pred'],sw))**2)))
    return {'day':day,'path':path.name,'rows':len(d['y']),'baseline_auc':old,'candidate_auc':new,'delta':delta,'recipe_max_difference':error,'paired_event_95pct_interval':[delta-1.96*se,delta+1.96*se],'history_weighted_baseline_auc':weighted_old,'history_weighted_candidate_auc':weighted_new,'history_weighted_delta':weighted_delta,'weighted_paired_event_95pct_interval':[weighted_delta-1.96*weighted_se,weighted_delta+1.96*weighted_se],'note':'Interval conditional on recipe and reweighting; excludes repeated-selection bias and evaluation-distribution uncertainty.'}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('prefix');ap.add_argument('--weight',type=float,required=True)
    ap.add_argument('--gate',action='store_true');ap.add_argument('--raw-prefix');args=ap.parse_args()
    reports=[audit(ROOT/f'{args.prefix}_day{day}.npz',day,args.weight,args.gate,args.raw_prefix) for day in [6,5]]
    out={'reports':reports,'improves_both':all(r['delta']>0 for r in reports),'target_reached_both':all(r['candidate_auc']>=.85 for r in reports)}
    (ROOT/f'tst2_{args.prefix}_report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
