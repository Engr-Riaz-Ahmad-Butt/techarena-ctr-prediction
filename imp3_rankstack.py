"""Own idea C: pairwise-ranking CatBoost on the cached stacked features, as a new blend member.

Same 190 features and day blocks as the stacked model, but the objective is PairLogit with group_id = event
(each event has 10 candidates, at most one clicked). Events without a click contribute no pairs, so the model
learns within-event ordering only; the event-level 'any click' signal stays with the Logloss members.
Trees chosen on day 6 by (unweighted, as CatBoost computes it) PairLogit eval early stopping with the reweighted
AUC reported; day 5 and the final model reuse the count. Scores are mapped to (0,1) by sigmoid for blending.
Outputs: predictions_imp3_rankstack_day{6,5}.npz, submission_imp3_rankstack_raw.csv, imp3_rankstack_report.json.
"""
from imp3_lib import *
import argparse
from catboost import CatBoostRanker,Pool
from threadpoolctl import threadpool_limits
threadpool_limits(limits=12)
train=np.load(ROOT/'train.npy',mmap_mode='r');start=int(train[:,2].min());DAY=86400;FEAT=ROOT/'features'
PARAMS=dict(depth=7,learning_rate=.08,loss_function='PairLogit',thread_count=12,random_seed=42,allow_writing_files=False)
def cutoff_of(day):return start+day*DAY

def assemble(days,tag,log):
    path=FEAT/f'imp3_fit_{tag}.npy';ys=[];gs=[];n=0
    for d in days:
        rows=np.asarray(train[(train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1))]);assert len(rows)%10==0 and (rows[:,6].reshape(-1,10)==rows[::10,6,None]).all()
        ys.append(rows[:,5].astype(np.float32));gs.append(rows[:,6].astype(np.int64));n+=len(rows);del rows
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,190));off=0
    for d in days:
        b=np.load(FEAT/f'stack_day{d}_c{d}.npy',mmap_mode='r')
        for s in range(0,len(b),200000):x[off+s:off+min(s+200000,len(b))]=b[s:s+200000]
        off+=len(b);del b;gc.collect()
    assert off==n;x.flush();del x;gc.collect();log(f'assembled {path.name} rows {n}')
    return np.load(path,mmap_mode='r'),np.concatenate(ys),np.concatenate(gs)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='rankstack');ap.add_argument('--stages',default='6,5,final');ap.add_argument('--trees',type=int,default=None);args=ap.parse_args()
    logf=open(ROOT/f'imp3_{args.tag}.log','a')
    def log(s):print(s,flush=True);logf.write(s+'\n');logf.flush()
    rp=ROOT/f'imp3_{args.tag}_report.json';report=json.loads(rp.read_text()) if rp.exists() else {}
    trees=args.trees or report.get('trees')
    for stage in args.stages.split(','):
        if stop_requested():log('STOP3 found');break
        t0=time.time()
        if stage in('6','5'):
            day=int(stage);x,y,g=assemble([day-3,day-2,day-1],f'{args.tag}_d{day}',log)
            ref=np.load(ROOT/f'predictions_v6_day{day}.npz');ev=ref['events'];yv=ref['y'];sw,_=history_weights(ev)
            xv=np.load(FEAT/f'stack_val{day}_c{day}.npy',mmap_mode='r')
            if day==6 and trees is None:
                m=CatBoostRanker(iterations=2500,verbose=100,**PARAMS)
                m.fit(Pool(x,y,group_id=g),eval_set=Pool(xv,yv,group_id=ev.astype(np.int64)),early_stopping_rounds=100,use_best_model=True);trees=int(m.tree_count_)
                report['trees']=trees;log(f'chosen trees {trees}')
            else:
                m=CatBoostRanker(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,group_id=g))
            pred=sigmoid(m.predict(xv));s=scores(yv,pred,sw);old=np.load(ROOT/f'predictions_stack_day{day}.npz');so=scores(yv,old['pred'],sw)
            report[f'day{day}']=dict(auc=s[0],reweighted=s[1],stack_auc=so[0],stack_reweighted=so[1],trees=trees,minutes=round((time.time()-t0)/60,1))
            log(f'day {day} rankstack auc/reweighted {s} vs stack {so}');save_predictions(args.tag,day,yv,pred,ev)
            del m,x,y,g,xv;gc.collect();(FEAT/f'imp3_fit_{args.tag}_d{day}.npy').unlink()
        elif stage=='final':
            assert trees;x,y,g=assemble([4,5,6],f'{args.tag}_final',log)
            m=CatBoostRanker(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,group_id=g));m.save_model(str(ROOT/f'imp3_model_{args.tag}.cbm'))
            xe=np.load(FEAT/'stack_eval_c7.npy',mmap_mode='r');prob=sigmoid(m.predict(xe))
            ids,_=read_submission('stack');path=write_submission(f'{args.tag}_raw',ids,prob)
            report['final']=dict(trees=trees,minutes=round((time.time()-t0)/60,1),submission=str(path));log(f'final written {path}')
            del m,x,y,g,xe;gc.collect();(FEAT/f'imp3_fit_{args.tag}_final.npy').unlink()
        rp.write_text(json.dumps(report,indent=2))
