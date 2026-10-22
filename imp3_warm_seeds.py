"""Idea 2: extra seeds of the accepted warm16 recipe (dim 16, auxiliary weight 1.0, pop_correct, 10 epochs).

Nothing is selected here: the recipe is V14's frozen one, only the RNG seed changes. For each seed the
script fits day 6 (cutoff day 6), day 5 (cutoff day 5) and the final model (all seven days) and saves the
raw predictions: imp3_warm16_s<seed>_day{6,5}.npz (events, y, raw) and imp3_warm16_s<seed>_final_raw.npy.
Blending of the seed average is done later by imp3_seed_blend.py (selected on day 6 only).
imp2_warm_choice.py is imported, never modified.
"""
from imp2_warm_choice import *
import argparse

PREFIX='warm16';RANDOM_INIT=False  # own idea B: replace the deterministic SVD initialisation by seed-specific random factors

def fit(seed,users,items,history,target,wu,wi,pop_bias,cutoff,epochs,log):
    ps,acc=init32(cutoff,16);rng=np.random.default_rng(seed)
    if RANDOM_INIT:
        r=np.random.default_rng(seed+1000)
        V=r.standard_normal(ps[1].shape).astype('float32');V/=np.linalg.norm(V,axis=1,keepdims=True)
        U=r.standard_normal(ps[0].shape).astype('float32');U*=.25/np.linalg.norm(U,axis=1,keepdims=True)
        ps[0],ps[1],ps[2]=U,V,.25*V.copy();acc[0],acc[1],acc[2]=[np.full_like(p,.1) for p in (U,V,ps[2])]
    for ep in range(1,epochs+1):
        t=time.time();neg=rng.integers(0,ni,size=(len(wu),11),dtype=np.int32)
        collision=neg==wi[:,None]
        while collision.any():
            neg[collision]=rng.integers(0,ni,size=int(collision.sum()),dtype=np.int32)
            collision=neg==wi[:,None]
        wl=warm_epoch(rng.permutation(len(wu)),wu,wi,neg,ps[0],ps[1],acc[0],acc[1],.045,.002,1.,pop_bias)
        loss=train_epoch(rng.permutation(len(users)),users,items,history,target,*ps,*acc,.045,.002)
        log(f'seed {seed} cutoff {cutoff} epoch {ep} warm_loss {wl:.5f} loss {loss:.5f} seconds {time.time()-t:.1f}')
    return ps

def day_run(seed,day,epochs,log):
    out=ROOT/f'imp3_{PREFIX}_s{seed}_day{day}.npz'
    if out.exists():log(f'{out.name} exists, skipping');return
    a=np.asarray(train[train[:,2]<start+day*DAY])
    users,items,history,target=event_data(a,hist);wu,wi=warm_data(a);del a;gc.collect()
    counts=np.bincount(wi,minlength=ni)+1;pop_bias=np.log(counts/counts.mean()).astype('float32')
    _,val=split(day);vu,vi,vh,_=event_data(val,hist)
    reference=np.load(ROOT/f'predictions_v6_day{day}.npz');assert np.array_equal(reference['events'],val[:,6])
    ps=fit(seed,users,items,history,target,wu,wi,pop_bias,start+(day-2)*DAY,epochs,log)
    raw=predict_events(vu,vi,vh,*ps).ravel()
    log(f'seed {seed} day {day} raw auc {roc_auc_score(val[:,5],raw):.7f}')
    np.savez_compressed(out,events=val[:,6],y=val[:,5],raw=raw)
    np.savez_compressed(ROOT/f'imp3_model_warm16_s{seed}_day{day}.npz',U=ps[0],V=ps[1],H=ps[2],bu=ps[3],bi=ps[4])
    del users,items,history,target,wu,wi,ps;gc.collect()

def final_run(seed,epochs,log):
    out=ROOT/f'imp3_{PREFIX}_s{seed}_final_raw.npy'
    if out.exists():log(f'{out.name} exists, skipping');return
    assert int(train[:,2].max())<start+7*DAY
    users,items,history,target=event_data(train,hist);wu,wi=warm_data(train)
    counts=np.bincount(wi,minlength=ni)+1;pop_bias=np.log(counts/counts.mean()).astype('float32')
    ps=fit(seed,users,items,history,target,wu,wi,pop_bias,start+6*DAY,epochs,log)
    raw=predict_events(evaluation[::10,0],evaluation[:,1].reshape(-1,10),ehist,*ps).ravel()
    assert len(raw)==len(evaluation) and np.isfinite(raw).all() and np.all((raw>=0)&(raw<=1))
    np.save(out,raw);np.savez_compressed(ROOT/f'imp3_model_warm16_s{seed}_final.npz',U=ps[0],V=ps[1],H=ps[2],bu=ps[3],bi=ps[4])
    log(f'seed {seed} final raw saved')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True)
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--stages',default='6,5,final')
    ap.add_argument('--random-init',action='store_true');ap.add_argument('--prefix',default='warm16');args=ap.parse_args()
    RANDOM_INIT=args.random_init;PREFIX=args.prefix
    logf=open(ROOT/'imp3_warm_seeds.log','a')
    def log(s):print(s,flush=True);logf.write(s+'\n');logf.flush()
    for seed in args.seeds:
        for stage in args.stages.split(','):
            if (ROOT/'STOP3').exists():log('STOP3 found, exiting');sys.exit(0)
            if stage=='final':final_run(seed,args.epochs,log)
            else:day_run(seed,int(stage),args.epochs,log)
