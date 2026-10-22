"""Idea 3: importance-weighted stacked CatBoost (same 190 cached V6 features, same params as stack_experiment.py).

Per-row sample weight = evaluation share / fit-block share of the row's history-length bin
(bins 0, 1-4, 5-9, 10-19, 20 clicks; evaluation shares from the unlabeled evaluation_hist.npy), so the trees
focus on the evaluation-like (short-history) population. The day-6 validation pool carries the protocol's
history reweighting so early stopping selects trees by reweighted AUC. Trees chosen on day 6 only; day 5 and
the final model (days 4-6, evaluation features cut at day 7) reuse that count unchanged.
Outputs: predictions_imp3_<tag>_day{6,5}.npz, submission_imp3_<tag>_raw.csv (component), imp3_<tag>_report.json.
Feature matrices are assembled into an on-disk memmap to keep RAM use low.
"""
from imp3_lib import *
import argparse
from catboost import CatBoostClassifier,Pool
from threadpoolctl import threadpool_limits
threadpool_limits(limits=12)
train=np.load(ROOT/'train.npy',mmap_mode='r');hist=np.load(ROOT/'train_hist.npy',mmap_mode='r')
start=int(train[:,2].min());DAY=86400;FEAT=ROOT/'features'
PARAMS=dict(depth=7,learning_rate=.08,loss_function='Logloss',thread_count=12,random_seed=42,allow_writing_files=False)
_eb=history_bins_eval();esh=np.bincount(_eb,minlength=5)/len(_eb)  # one row per evaluation event; weights are normalised to mean 1 anyway

def cutoff_of(day):return start+day*DAY
def day_mask(day):return (train[:,2]>=cutoff_of(day))&(train[:,2]<cutoff_of(day+1))
def bins_of(event_ids):return np.digitize((np.asarray(hist[event_ids])>=0).sum(1),EDGES,right=True)

def assemble(days,tag,log):
    """Concatenate cached blocks into one on-disk float32 memmap; return (x memmap, y, bins)."""
    path=FEAT/f'imp3_fit_{tag}.npy';ys=[];bs=[];n=0
    for d in days:
        m=np.asarray(day_mask(d));rows=np.asarray(train[m]);assert len(rows)%10==0
        ys.append(rows[:,5].astype(np.float32));bs.append(bins_of(rows[:,6]));n+=len(rows);del rows,m
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,190));off=0
    for d in days:
        b=np.load(FEAT/f'stack_day{d}_c{d}.npy',mmap_mode='r');assert b.shape[1]==190
        for s in range(0,len(b),200000):x[off+s:off+min(s+200000,len(b))]=b[s:s+200000]
        off+=len(b);del b;gc.collect()
    assert off==n;x.flush();del x;gc.collect();log(f'assembled {path.name} rows {n}')
    return np.load(path,mmap_mode='r'),np.concatenate(ys),np.concatenate(bs)

def fit_weights(bins):
    share=np.bincount(bins,minlength=5)/len(bins);w=(esh/share)[bins].astype(np.float32)
    return w/w.mean()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='wstack');ap.add_argument('--power',type=float,default=1.,help='weight exponent (1 = full importance weights)')
    ap.add_argument('--stages',default='6,5,final');ap.add_argument('--trees',type=int,default=None);args=ap.parse_args()
    logf=open(ROOT/f'imp3_{args.tag}.log','a')
    def log(s):print(s,flush=True);logf.write(s+'\n');logf.flush()
    rp=ROOT/f'imp3_{args.tag}_report.json';report=json.loads(rp.read_text()) if rp.exists() else {}
    trees=args.trees or report.get('trees')
    for stage in args.stages.split(','):
        if stop_requested():log('STOP3 found');break
        t0=time.time()
        if stage in('6','5'):
            day=int(stage);x,y,b=assemble([day-3,day-2,day-1],f'{args.tag}_d{day}',log);w=fit_weights(b)**args.power
            log(f'day {day} fit rows {len(y)} bin shares {np.round(np.bincount(b,minlength=5)/len(b),4).tolist()} eval shares {np.round(esh,4).tolist()} weights by bin {[round(float(w[b==k][0]),3) for k in range(5)]}')
            ref=np.load(ROOT/f'predictions_v6_day{day}.npz');ev=ref['events'];yv=ref['y']
            xv=np.load(FEAT/f'stack_val{day}_c{day}.npy',mmap_mode='r');sw,_=history_weights(ev)
            if day==6 and trees is None:
                m=CatBoostClassifier(iterations=2500,eval_metric='AUC',verbose=100,**PARAMS)
                m.fit(Pool(x,y,weight=w),eval_set=Pool(xv,yv,weight=sw),early_stopping_rounds=100,use_best_model=True);trees=int(m.tree_count_)
                report['trees']=trees;log(f'chosen trees {trees}')
            else:
                m=CatBoostClassifier(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,weight=w))
            pred=m.predict_proba(xv)[:,1];s=scores(yv,pred,sw)
            old=np.load(ROOT/f'predictions_stack_day{day}.npz');so=scores(yv,old['pred'],sw)
            report[f'day{day}']=dict(auc=s[0],reweighted=s[1],stack_auc=so[0],stack_reweighted=so[1],trees=trees,minutes=round((time.time()-t0)/60,1))
            log(f'day {day} weighted stack auc/reweighted {s} vs stack {so}')
            save_predictions(args.tag,day,yv,pred,ev)
            del m,x,y,b,w,xv;gc.collect();(FEAT/f'imp3_fit_{args.tag}_d{day}.npy').unlink()
        elif stage=='final':
            assert trees;x,y,b=assemble([4,5,6],f'{args.tag}_final',log);w=fit_weights(b)**args.power
            m=CatBoostClassifier(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,weight=w))
            m.save_model(str(ROOT/f'imp3_model_{args.tag}.cbm'))
            xe=np.load(FEAT/'stack_eval_c7.npy',mmap_mode='r');prob=m.predict_proba(xe)[:,1]
            ids,_=read_submission('stack');path=write_submission(f'{args.tag}_raw',ids,prob)
            report['final']=dict(trees=trees,minutes=round((time.time()-t0)/60,1),submission=str(path))
            log(f'final written {path}');del m,x,y,b,w,xe;gc.collect();(FEAT/f'imp3_fit_{args.tag}_final.npy').unlink()
        rp.write_text(json.dumps(report,indent=2))
