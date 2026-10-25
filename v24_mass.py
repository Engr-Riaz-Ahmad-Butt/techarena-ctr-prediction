"""Lightweight event click-mass correction; freeze day6 choice before day5."""
import hashlib, json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as ROOT
from tst2_candidate import influence, weighted_influence

def rank(x): return rankdata(x, method='average')/(len(x)+1)
def prediction(z, alpha):
    mass=np.repeat(z['raw'].reshape(-1,10).sum(1),10)
    return rank(np.log(np.maximum(z['pred'],1e-12))+alpha*np.log(np.maximum(mass,1e-12)))
def measure(z,p):
    y=z['y']; old=z['pred']; gain=float(roc_auc_score(y,p)-roc_auc_score(y,old))
    se=float(np.linalg.norm(influence(y,p)-influence(y,old)))
    hist=np.load(ROOT/'train_hist.npy',mmap_mode='r'); eh=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
    vg=np.digitize((hist[z['events']]>=0).sum(1),[0,4,9,19],right=True)
    eg=np.digitize((eh>=0).sum(1),[0,4,9,19],right=True)
    w=((np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg)))[vg]
    wg=float(roc_auc_score(y,p,sample_weight=w)-roc_auc_score(y,old,sample_weight=w))
    ws=float(np.linalg.norm(weighted_influence(y,p,w)-weighted_influence(y,old,w)))
    return dict(auc=float(roc_auc_score(y,p)),baseline_auc=float(roc_auc_score(y,old)),gain=gain,conditional_interval=[gain-1.96*se,gain+1.96*se],weighted_gain=wg,weighted_interval=[wg-1.96*ws,wg+1.96*ws])
def main():
    paths=[ROOT/f'v23_outside_day{d}.npz' for d in (6,5)]
    frozen={'alphas':[-.5,-.25,-.1,.1,.25,.5], 'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'note':'Reused local validation; intervals exclude model-selection uncertainty.'}
    (ROOT/'v24_mass_frozen.json').open('x').write(json.dumps(frozen,indent=2))
    z=np.load(paths[0]); trials=[]
    for a in frozen['alphas']:
        trials.append(dict(alpha=a,auc=float(roc_auc_score(z['y'],prediction(z,a)))))
    best=max(trials,key=lambda r:r['auc'])
    (ROOT/'v24_mass_selection.json').open('x').write(json.dumps(best,indent=2))
    reports=[]
    for day,path in zip((6,5),paths):
        z=np.load(path); p=prediction(z,best['alpha']); r=measure(z,p); r['day']=day; reports.append(r)
        print(json.dumps(r),flush=True)
        np.savez_compressed(ROOT/f'v24_mass_day{day}.npz',y=z['y'],events=z['events'],pred=p)
        if r['conditional_interval'][0]<=0 or r['weighted_interval'][0]<=0: break
    accepted=len(reports)==2 and all(r['conditional_interval'][0]>0 and r['weighted_interval'][0]>0 for r in reports)
    result=dict(trials=trials,selected=best,reports=reports,accepted=accepted,note=frozen['note'],evaluation_csv_ready=False)
    (ROOT/'v24_mass_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__': main()
