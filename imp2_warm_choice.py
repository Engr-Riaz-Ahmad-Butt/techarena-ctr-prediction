"""Joint event choice and pre-window positive history factorization.

Only the earliest observed history of each user is used as auxiliary positives.
No held-out history or label participates in parameter fitting. Day 6 selects
the auxiliary weight/epoch/blend; day 5 confirms the exact same settings.
"""
from choice64_core import *
import argparse

@njit
def warm_epoch(order, users, positive, negatives, U,V,au,av,lr,reg,weight,pop_bias):
    d=U.shape[1];loss=0.
    for q in order:
        u=users[q];items=negatives[q].copy();items[0]=positive[q]
        z=U[u].copy();logits=np.empty(len(items));maximum=-1e30
        for j in range(len(items)):
            v=pop_bias[items[j]]
            for k in range(d):v+=z[k]*V[items[j],k]
            logits[j]=max(-25.,min(25.,v));maximum=max(maximum,logits[j])
        p=np.exp(logits-maximum);denom=p.sum();loss+=math.log(denom)+maximum-logits[0]
        grad=np.zeros(d,np.float32)
        for j in range(len(items)):
            i=items[j];err=weight*(p[j]/denom-(1. if j==0 else 0.))
            for k in range(d):
                grad[k]+=err*V[i,k]
                g=err*z[k]+weight*reg*V[i,k]
                av[i,k]+=g*g;V[i,k]-=lr*g/math.sqrt(av[i,k])
        for k in range(d):
            g=grad[k]+weight*reg*U[u,k]
            au[u,k]+=g*g;U[u,k]-=lr*g/math.sqrt(au[u,k])
    return loss/max(len(order),1)

def warm_data(a):
    ev=a[::10];_,ix=np.unique(ev[:,0],return_index=True)
    e=ev[ix];h=np.asarray(hist[e[:,6]])
    u=np.repeat(e[:,0],20);i=h.ravel();keep=i>=0
    stamps=np.load(ROOT/'train_times.npy',mmap_mode='r')[e[:,6]].ravel()
    assert np.all(stamps[keep]<np.repeat(e[:,2],20)[keep]),'History must precede first event'
    keys=np.rec.fromarrays([u[keep],i[keep],stamps[keep]],names='user,item,timestamp')
    _,unique=np.unique(keys,return_index=True)
    print('warm history duplicate events removed',int(keep.sum())-len(unique),flush=True)
    return u[keep][unique].astype(np.int32),i[keep][unique].astype(np.int32)

def init32(cutoff,dim=32):
    with (ROOT/f'preference_stats_{cutoff}.pkl').open('rb') as f:s=pickle.load(f)
    V=normalize(s[3][:,:dim]).astype('float32')
    U=.25*normalize(s[4][:,:dim]).astype('float32');H=.25*V.copy()
    c=s[0][1];g=c['global'];ur=(c['user_click']+50*g)/(c['user_impr']+50)
    ir=(c['item_click']+50*g)/(c['item_impr']+50)
    t=np.clip(ur*10,.02,.98)
    bu=np.clip(np.log(t/(1-t)/10)+1.8,-2,2).astype('float32')
    bi=np.clip(np.log(ir/g),-2,2).astype('float32')
    ps=[U,V,H,bu,bi];return ps,[np.full_like(p,.1) for p in ps]

def run(day,auxweight,epochs=10,chosen=None,dim=32,pop_correct=False):
    tag=f'warm{auxweight}_d{dim}_pc{int(pop_correct)}'
    a=np.asarray(train[train[:,2]<start+day*DAY])
    users,items,history,target=event_data(a,hist);wu,wi=warm_data(a);del a
    counts=np.bincount(wi,minlength=ni)+1
    pop_bias=np.log(counts/counts.mean()).astype('float32') if pop_correct else np.zeros(ni,np.float32)
    _,val=split(day);vu,vi,vh,_=event_data(val,hist)
    old=np.load(ROOT/f'predictions_weighted_day{day}.npz')
    reference=np.load(ROOT/f'predictions_v6_day{day}.npz')
    assert np.array_equal(reference['events'],val[:,6]) and np.array_equal(old['y'],val[:,5])
    eg=np.digitize((ehist>=0).sum(axis=1),[0,4,9,19],right=True)
    vg=np.digitize((hist[val[:,6]]>=0).sum(axis=1),[0,4,9,19],right=True)
    ratios=(np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg))
    sampleweights=ratios[vg]
    ps,acc=init32(start+(day-2)*DAY,dim);rng=np.random.default_rng(151)
    best=None;stale=0;baseline=float(roc_auc_score(val[:,5],old['pred']))
    baseline_weighted=float(roc_auc_score(val[:,5],old['pred'],sample_weight=sampleweights))
    for ep in range(1,epochs+1):
        t=time.time();neg=rng.integers(0,ni,size=(len(wu),11),dtype=np.int32)
        collision=neg==wi[:,None]
        while collision.any():
            neg[collision]=rng.integers(0,ni,size=int(collision.sum()),dtype=np.int32)
            collision=neg==wi[:,None]
        wl=warm_epoch(rng.permutation(len(wu)),wu,wi,neg,ps[0],ps[1],acc[0],acc[1],.045,.002,auxweight,pop_bias)
        loss=train_epoch(rng.permutation(len(users)),users,items,history,target,*ps,*acc,.045,.002)
        raw=predict_events(vu,vi,vh,*ps).ravel()
        weights=[.05,.1,.2,.3,.5] if chosen is None else [chosen['weight']]
        scores={w:float(roc_auc_score(val[:,5],(1-w)*old['pred']+w*raw)) for w in weights}
        weighted={w:float(roc_auc_score(val[:,5],(1-w)*old['pred']+w*raw,sample_weight=sampleweights)) for w in weights}
        # Later compact-rank arm predeclares evaluation-history adjusted AUC.
        selection=weighted if pop_correct else scores
        w=max(selection,key=selection.get)
        result={'day':day,'auxiliary_weight':auxweight,'dimension':dim,'pop_correct':pop_correct,'epoch':ep,'weight':w,'auc':scores[w],
                'raw_auc':float(roc_auc_score(val[:,5],raw)),'baseline_auc':baseline,'warm_pairs':len(wu),
                'history_weighted_auc':weighted[w],'selection_score':selection[w],
                'baseline_history_weighted_auc':baseline_weighted}
        print(json.dumps(result),'seconds',round(time.time()-t,1),flush=True)
        if (chosen is None and (best is None or selection[w]>best['selection_score'])) or (chosen is not None and ep==epochs):
            best=result;stale=0;np.savez_compressed(ROOT/f'imp2_{tag}_day{day}.npz',
                events=val[:,6],y=val[:,5],raw=raw,pred=(1-w)*old['pred']+w*raw)
            np.savez_compressed(ROOT/f'imp2_model_{tag}_day{day}.npz',
                U=ps[0],V=ps[1],H=ps[2],bu=ps[3],bi=ps[4])
        elif chosen is None:stale+=1
        if chosen is not None:best=result
        if chosen is None and stale>=3:
            print('Stopping selection after three stale epochs.',flush=True);break
    return best

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--weight',type=float,default=.15)
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--dim',type=int,default=32)
    ap.add_argument('--pop-correct',action='store_true');args=ap.parse_args()
    best=run(6,args.weight,args.epochs,dim=args.dim,pop_correct=args.pop_correct)
    tag=f'warm{args.weight}_d{args.dim}_pc{int(args.pop_correct)}'
    (ROOT/f'imp2_{tag}_selection.json').write_text(json.dumps(best,indent=2))
    if best['history_weighted_auc']<=best['baseline_history_weighted_auc']:
        print('Rejected on selection date; no confirmation fit required.',flush=True)
        sys.exit(0)
    stronger=ROOT/'imp2_warm1.0_d16_pc1_selection.json'
    if args.dim!=16 and args.pop_correct and stronger.exists():
        reference=json.loads(stronger.read_text())
        if best['history_weighted_auc']<reference['history_weighted_auc']+.0001:
            print('Not a material improvement over accepted rank16; no confirmation fit.',flush=True)
            sys.exit(0)
    confirmation=run(5,args.weight,best['epoch'],best,dim=args.dim,pop_correct=args.pop_correct)
    (ROOT/f'imp2_{tag}_report.json').write_text(json.dumps([best,confirmation],indent=2))
