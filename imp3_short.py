"""Idea 5: short-history specialist. A CatBoost model (same 190 cached features, same params as the stacked
model) fitted only on rows whose user history has <= 9 clicks (bins 0, 1-4, 5-9), with importance weights
eval share / fit share within those bins. It is mixed into the stacked prediction only on such rows:
  stack_mod = stack                      for history >= 10 clicks
            = (1-g)*stack + g*specialist for history <= 9 clicks
Trees (early stopping, reweighted eval pool) and g are selected on day 6 by the reweighted AUC of the full V15
recipe with stack replaced by stack_mod; day 5 and the final model reuse them unchanged.
Outputs: predictions_imp3_short_day{6,5}.npz (stack_mod), submission_imp3_short_raw.csv (component), report JSON.
"""
from imp3_lib import *
import argparse
from catboost import CatBoostClassifier,Pool
from threadpoolctl import threadpool_limits
threadpool_limits(limits=12)
train=np.load(ROOT/'train.npy',mmap_mode='r');hist=np.load(ROOT/'train_hist.npy',mmap_mode='r');ehist=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')
start=int(train[:,2].min());DAY=86400;FEAT=ROOT/'features';MAXBIN=2
PARAMS=dict(depth=7,learning_rate=.08,loss_function='Logloss',thread_count=12,random_seed=42,allow_writing_files=False)
_eb=history_bins_eval();esh=np.bincount(_eb,minlength=5)/len(_eb)
V15=dict(stack=.6,v6=.3,choice=0.,v7=.1);A15=.3

def cutoff_of(day):return start+day*DAY
def bins_of(event_ids):return np.digitize((np.asarray(hist[event_ids])>=0).sum(1),EDGES,right=True)

def short_blocks(days,log):
    xs=[];ys=[];bs=[]
    for d in days:
        m=np.asarray((train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1)));rows=np.asarray(train[m]);b=bins_of(rows[:,6])
        keep=b<=MAXBIN;x=np.load(FEAT/f'stack_day{d}_c{d}.npy',mmap_mode='r')
        idx=np.flatnonzero(keep);xs.append(np.asarray(x[idx]));ys.append(rows[idx,5].astype(np.float32));bs.append(b[idx]);del x,rows,m;gc.collect()
        log(f'day {d}: {len(idx)} short-history rows')
    x=np.concatenate(xs);y=np.concatenate(ys);b=np.concatenate(bs);del xs,ys,bs;gc.collect()
    share=np.bincount(b,minlength=5)/len(b);w=(esh/np.maximum(share,1e-9))[b].astype(np.float32);w/=w.mean()
    return x,y,w

def full_blend(parts,stack_mod):
    L=np.array([logit(stack_mod),logit(parts['v6']),logit(parts['choice']),logit(parts['v7'])])
    return (1-A15)*competitive(sigmoid(np.tensordot(np.array([V15['stack'],V15['v6'],V15['choice'],V15['v7']]),L,1)))+A15*parts['warm16']

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='short');ap.add_argument('--stages',default='6,5,final');args=ap.parse_args()
    logf=open(ROOT/f'imp3_{args.tag}.log','a')
    def log(s):print(s,flush=True);logf.write(s+'\n');logf.flush()
    rp=ROOT/f'imp3_{args.tag}_report.json';report=json.loads(rp.read_text()) if rp.exists() else {}
    trees=report.get('trees');g=report.get('g')
    for stage in args.stages.split(','):
        if stop_requested():log('STOP3 found');break
        t0=time.time()
        if stage in('6','5'):
            day=int(stage);x,y,w=short_blocks([day-3,day-2,day-1],log)
            yv,ev,parts,base=load_components(day);sw,vg=history_weights(ev);short=vg<=MAXBIN
            xv=np.load(FEAT/f'stack_val{day}_c{day}.npy',mmap_mode='r');xs=np.asarray(xv[np.flatnonzero(short)]);del xv
            if day==6 and trees is None:
                m=CatBoostClassifier(iterations=2500,eval_metric='AUC',verbose=100,**PARAMS)
                m.fit(Pool(x,y,weight=w),eval_set=Pool(xs,yv[short],weight=sw[short]),early_stopping_rounds=100,use_best_model=True);trees=int(m.tree_count_)
                report['trees']=trees;log(f'chosen trees {trees}')
            else:
                m=CatBoostClassifier(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,weight=w))
            spec=m.predict_proba(xs)[:,1];stack=parts['stack']
            log(f'day {day} short rows {int(short.sum())}: specialist auc {roc_auc_score(yv[short],spec):.6f} (rw {roc_auc_score(yv[short],spec,sample_weight=sw[short]):.6f}) vs stack {roc_auc_score(yv[short],stack[short]):.6f} (rw {roc_auc_score(yv[short],stack[short],sample_weight=sw[short]):.6f})')
            def mod(gg):
                s=stack.copy();s[short]=(1-gg)*stack[short]+gg*spec;return s
            if day==6 and g is None:
                cand={gg:scores(yv,full_blend(parts,mod(gg)),sw) for gg in [0.,.25,.5,.75,1.]}
                log('g grid (auc, reweighted): '+json.dumps({str(k):v for k,v in cand.items()}))
                g=max(cand,key=lambda k:cand[k][1]);report['g']=g;log(f'chosen g {g}')
            sm=mod(g);p=full_blend(parts,sm);s=scores(yv,p,sw);b=scores(yv,base,sw)
            report[f'day{day}']=dict(auc=s[0],reweighted=s[1],base_auc=b[0],base_reweighted=b[1],trees=trees,g=g,minutes=round((time.time()-t0)/60,1))
            log(f'day {day} full blend auc/reweighted {s} vs V15 {b}')
            save_predictions(args.tag,day,yv,sm,ev);m.save_model(str(ROOT/f'imp3_model_{args.tag}_day{day}.cbm'))
            del m,x,y,w,xs,parts;gc.collect()
        elif stage=='final':
            assert trees is not None and g is not None;x,y,w=short_blocks([4,5,6],log)
            m=CatBoostClassifier(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,weight=w));m.save_model(str(ROOT/f'imp3_model_{args.tag}.cbm'))
            eb=np.repeat(_eb,10);short=eb<=MAXBIN;xe=np.load(FEAT/'stack_eval_c7.npy',mmap_mode='r');xs=np.asarray(xe[np.flatnonzero(short)]);del xe
            spec=m.predict_proba(xs)[:,1];ids,stack=read_submission('stack');sm=stack.copy();sm[short]=(1-g)*stack[short]+g*spec
            path=write_submission(f'{args.tag}_raw',ids,sm);report['final']=dict(trees=trees,g=g,short_rows=int(short.sum()),submission=str(path),minutes=round((time.time()-t0)/60,1))
            log(f'final written {path}');del m,x,y,w,xs;gc.collect()
        rp.write_text(json.dumps(report,indent=2))
