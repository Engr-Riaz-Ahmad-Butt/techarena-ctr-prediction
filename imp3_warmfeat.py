"""Idea 4: warm16 model score as extra CatBoost features for the stacked model.

For each day block d the warm16 model (V14's frozen recipe: dim 16, aux 1.0, pop_correct, 10 epochs, seed 151)
is fitted with labels strictly before the start of day d, then scores block-d rows. Block 5 / 6 / val5 / val6 /
evaluation reuse the already-fitted imp2 models (cutoffs day 5, day 6 and day 7 = final), which were fitted under
exactly the same rule. Fits for cutoffs 2, 3, 4 are new (init32 at the day-2 statistics, the earliest cached).
Extra features per row: raw softmax score, logit(score), score - event mean, within-event rank/9.
Then the stacked CatBoost (same params; optional importance weights --power) is refitted on 194 features.
Trees chosen on day 6 only. Outputs mirror imp3_wstack.py with tag 'warmfeat'.
"""
from imp3_lib import *
import argparse

from catboost import CatBoostClassifier,Pool
from threadpoolctl import threadpool_limits
threadpool_limits(limits=12)
from imp2_warm_choice import train,hist,evaluation,ehist,start,DAY,ni,event_data,warm_data,init32,warm_epoch,train_epoch,predict_events,split
FEAT=ROOT/'features';EXTRA=4
PARAMS=dict(depth=7,learning_rate=.08,loss_function='Logloss',thread_count=12,random_seed=42,allow_writing_files=False)
_eb=history_bins_eval();esh=np.bincount(_eb,minlength=5)/len(_eb)
def cutoff_of(day):return start+day*DAY
def bins_of(event_ids):return np.digitize((np.asarray(hist[event_ids])>=0).sum(1),EDGES,right=True)

SEEDS=[151];SUFFIX=''  # --seeds: warm scores averaged over these seeds (151 = V14's model; others from imp3_warm_seeds.py)

def model_path(cut,seed=151):
    if seed==151:
        if cut==5:return ROOT/'imp2_model_warm1.0_d16_pc1_day5.npz'
        if cut==6:return ROOT/'imp2_model_warm1.0_d16_pc1_day6.npz'
        if cut==7:return ROOT/'imp2_final_model_warm1.0_d16_pc1.npz'
        return ROOT/f'imp3_model_warm16_cut{cut}.npz'
    if cut in(5,6):return ROOT/f'imp3_model_warm16_s{seed}_day{cut}.npz'
    if cut==7:return ROOT/f'imp3_model_warm16_s{seed}_final.npz'
    return ROOT/f'imp3_model_warm16_s{seed}_cut{cut}.npz'

def fit_cut(cut,log,seed=151):
    out=model_path(cut,seed)
    if out.exists():return
    a=np.asarray(train[train[:,2]<cutoff_of(cut)]);users,items,history,target=event_data(a,hist);wu,wi=warm_data(a);del a;gc.collect()
    counts=np.bincount(wi,minlength=ni)+1;pop_bias=np.log(counts/counts.mean()).astype('float32')
    ps,acc=init32(cutoff_of(max(cut-2,2)),16);rng=np.random.default_rng(seed)
    for ep in range(1,11):
        t=time.time();neg=rng.integers(0,ni,size=(len(wu),11),dtype=np.int32);collision=neg==wi[:,None]
        while collision.any():
            neg[collision]=rng.integers(0,ni,size=int(collision.sum()),dtype=np.int32);collision=neg==wi[:,None]
        warm_epoch(rng.permutation(len(wu)),wu,wi,neg,ps[0],ps[1],acc[0],acc[1],.045,.002,1.,pop_bias)
        loss=train_epoch(rng.permutation(len(users)),users,items,history,target,*ps,*acc,.045,.002)
        log(f'cutoff day {cut} epoch {ep} loss {loss:.5f} {time.time()-t:.1f}s')
    np.savez_compressed(out,U=ps[0],V=ps[1],H=ps[2],bu=ps[3],bi=ps[4]);del users,items,history,target,wu,wi;gc.collect()

def score(rows,histories,cut):
    raws=[]
    for seed in SEEDS:
        m=np.load(model_path(cut,seed));ps=[m[k] for k in ['U','V','H','bu','bi']]
        raws.append(predict_events(rows[::10,0].copy(),rows[:,1].reshape(-1,10).copy(),np.asarray(histories[rows[::10,6]]),*ps))
    raw=np.mean(raws,axis=0)
    r=raw.ravel();rel=(raw-raw.mean(1,keepdims=True)).ravel();rank=(np.argsort(np.argsort(raw,1),1)/9).ravel()
    return np.column_stack([r,logit(r),rel,rank]).astype(np.float32)

def feat_file(name):return FEAT/f'imp3_wf{SUFFIX}_{name}.npy'

def build_features(log):
    for d in [2,3,4,5,6]:
        if feat_file(f'day{d}').exists():continue
        rows=np.asarray(train[(train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1))]);np.save(feat_file(f'day{d}'),score(rows,hist,d));log(f'scored block {d}');del rows
    for d in [5,6]:
        if feat_file(f'val{d}').exists():continue
        ref=np.load(ROOT/f'predictions_v6_day{d}.npz');_,rows=split(d)  # split() orders first events by user, as the saved predictions do
        assert np.array_equal(rows[:,6],ref['events']);np.save(feat_file(f'val{d}'),score(rows,hist,d));log(f'scored val {d}')
    if not feat_file('eval').exists():np.save(feat_file('eval'),score(np.asarray(evaluation),ehist,7));log('scored eval')

def assemble(days,tag,log):
    path=FEAT/f'imp3_fit_{tag}.npy';ys=[];bs=[];n=0
    for d in days:
        rows=np.asarray(train[(train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1))]);ys.append(rows[:,5].astype(np.float32));bs.append(bins_of(rows[:,6]));n+=len(rows);del rows
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,190+EXTRA));off=0
    for d in days:
        b=np.load(FEAT/f'stack_day{d}_c{d}.npy',mmap_mode='r');e=np.load(feat_file(f'day{d}'));assert len(e)==len(b)
        for s in range(0,len(b),200000):x[off+s:off+min(s+200000,len(b)),:190]=b[s:s+200000]
        x[off:off+len(b),190:]=e;off+=len(b);del b,e;gc.collect()
    assert off==n;x.flush();del x;gc.collect();log(f'assembled {path.name} rows {n}')
    return np.load(path,mmap_mode='r'),np.concatenate(ys),np.concatenate(bs)

def with_extra(base,name):return np.column_stack([np.asarray(base),np.load(feat_file(name))])

def fit_weights(bins,power):
    share=np.bincount(bins,minlength=5)/len(bins);w=((esh/share)[bins]**power).astype(np.float32);return w/w.mean()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='warmfeat');ap.add_argument('--power',type=float,default=0.)
    ap.add_argument('--stages',default='warm,feat,6,5,final');ap.add_argument('--seeds',default='151')
    ap.add_argument('--lr',type=float,default=.08);ap.add_argument('--depth',type=int,default=7);args=ap.parse_args()
    SEEDS=[int(s) for s in args.seeds.split(',')];SUFFIX='' if SEEDS==[151] else '_s'+'-'.join(str(s) for s in SEEDS)
    PARAMS.update(learning_rate=args.lr,depth=args.depth)
    logf=open(ROOT/f'imp3_{args.tag}.log','a')
    def log(s):print(s,flush=True);logf.write(s+'\n');logf.flush()
    rp=ROOT/f'imp3_{args.tag}_report.json';report=json.loads(rp.read_text()) if rp.exists() else {'power':args.power,'seeds':SEEDS,'lr':args.lr,'depth':args.depth}
    trees=report.get('trees');log(f'settings seeds {SEEDS} lr {args.lr} depth {args.depth} power {args.power} tag {args.tag}')
    for stage in args.stages.split(','):
        if stop_requested():log('STOP3 found');break
        t0=time.time()
        if stage=='warm':
            for seed in SEEDS:
                for cut in [2,3,4]:fit_cut(cut,log,seed)
        elif stage=='feat':build_features(log)
        elif stage in('6','5'):
            day=int(stage);x,y,b=assemble([day-3,day-2,day-1],f'{args.tag}_d{day}',log);w=fit_weights(b,args.power)
            ref=np.load(ROOT/f'predictions_v6_day{day}.npz');ev=ref['events'];yv=ref['y'];sw,_=history_weights(ev)
            xv=with_extra(np.load(FEAT/f'stack_val{day}_c{day}.npy',mmap_mode='r'),f'val{day}')
            if day==6 and trees is None:
                m=CatBoostClassifier(iterations=2500,eval_metric='AUC',verbose=100,**PARAMS)
                m.fit(Pool(x,y,weight=w),eval_set=Pool(xv,yv,weight=sw),early_stopping_rounds=100,use_best_model=True);trees=int(m.tree_count_);report['trees']=trees;log(f'chosen trees {trees}')
            else:
                m=CatBoostClassifier(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,weight=w))
            pred=m.predict_proba(xv)[:,1];s=scores(yv,pred,sw);old=np.load(ROOT/f'predictions_stack_day{day}.npz');so=scores(yv,old['pred'],sw)
            report[f'day{day}']=dict(auc=s[0],reweighted=s[1],stack_auc=so[0],stack_reweighted=so[1],trees=trees,minutes=round((time.time()-t0)/60,1))
            log(f'day {day} warmfeat stack auc/reweighted {s} vs stack {so}');save_predictions(args.tag,day,yv,pred,ev)
            imp=m.get_feature_importance();log(f'importance of extra features {imp[190:].round(2).tolist()} (max base {imp[:190].max():.2f})')
            del m,x,y,b,w,xv;gc.collect();(FEAT/f'imp3_fit_{args.tag}_d{day}.npy').unlink()
        elif stage=='final':
            assert trees;x,y,b=assemble([4,5,6],f'{args.tag}_final',log);w=fit_weights(b,args.power)
            m=CatBoostClassifier(iterations=trees,verbose=200,**PARAMS);m.fit(Pool(x,y,weight=w));m.save_model(str(ROOT/f'imp3_model_{args.tag}.cbm'))
            xe=with_extra(np.load(FEAT/'stack_eval_c7.npy',mmap_mode='r'),'eval');prob=m.predict_proba(xe)[:,1]
            ids,_=read_submission('stack');path=write_submission(f'{args.tag}_raw',ids,prob)
            report['final']=dict(trees=trees,minutes=round((time.time()-t0)/60,1),submission=str(path));log(f'final written {path}')
            del m,x,y,b,w,xe;gc.collect();(FEAT/f'imp3_fit_{args.tag}_final.npy').unlink()
        rp.write_text(json.dumps(report,indent=2))
