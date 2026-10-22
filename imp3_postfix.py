"""Own idea A: cheap post-hoc transforms on top of the V15 recipe (no training).

V15 = (1-.3)*competitive(sigmoid(.6 logit(stack)+.3 logit(v6)+.1 logit(v7)), .5) + .3*warm16.
Searched on day 6 (reweighted AUC), coordinate-wise, then frozen for day 5:
  1. competition strength s in {0,.25,.5,.75,1} and temperature T in {.8,.9,1,1.1,1.25} of the 4-member logit blend;
  2. per-history-bin additive logit offsets o_k (k = 0,1-4,5-9,10-19,20 clicks; o_4 fixed 0) applied to the final
     probability: p' = sigmoid(logit(p)+o[bin]). Within-bin ranking is untouched; only cross-bin ordering moves,
     which is exactly what the global AUC (and the reweighting) is sensitive to. Grid -1..1 step .1 per bin.
Bins use only history lengths (available for evaluation rows too).
"""
from imp3_lib import *
import argparse,itertools,hashlib
W=np.array([.6,.3,.1]);A=.3

def blend(parts,s,T):
    L=np.array([logit(parts['stack']),logit(parts['v6']),logit(parts['v7'])])
    return (1-A)*competitive(sigmoid((W@L)/T),s)+A*parts['warm16']

def offset(p,vg,o):return sigmoid(logit(p)+np.asarray(o)[vg])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='postfix');ap.add_argument('--chosen',default=None);args=ap.parse_args()
    logf=open(ROOT/f'imp3_{args.tag}.log','a')
    def say(*a):
        s=' '.join(str(x) for x in a);print(s,flush=True);logf.write(s+'\n');logf.flush()
    report={'tag':args.tag};chosen=json.loads(args.chosen) if args.chosen else None
    for day in [6,5]:
        y,ev,parts,base=load_components(day);sw,vg=history_weights(ev);b=scores(y,base,sw);say(f'day{day} {BASE_NAME}',b)
        if day==6 and chosen is None:
            grid={}
            for s in [0,.25,.5,.75,1.]:
                for T in [.8,.9,1.,1.1,1.25]:grid[(s,T)]=float(roc_auc_score(y,blend(parts,s,T),sample_weight=sw))
            top=sorted(grid.items(),key=lambda kv:-kv[1])[:5];say('strength/temperature top5',top)
            s,T=top[0][0];p=blend(parts,s,T);o=[0.,0.,0.,0.,0.];cur=top[0][1]
            for sweep in range(3):
                for k in [0,1,2,3]:
                    best=o[k]
                    for v in np.arange(-1,1.01,.1):
                        oo=list(o);oo[k]=float(round(v,2));sc=float(roc_auc_score(y,offset(p,vg,oo),sample_weight=sw))
                        if sc>cur+1e-9:cur,best=sc,float(round(v,2))
                    o[k]=best
                say(f'sweep {sweep} offsets {o} reweighted {cur:.6f}')
            chosen=dict(strength=s,temperature=T,offsets=o);report['chosen']=chosen;report['grid_top5']=[[list(k),v] for k,v in top]
        p=offset(blend(parts,chosen['strength'],chosen['temperature']),vg,chosen['offsets']);sc=scores(y,p,sw)
        ci,mean=bootstrap_delta(y,p,base,sw) if day==6 else (None,None)
        r=dict(auc=sc[0],reweighted=sc[1],base_auc=b[0],base_reweighted=b[1],baseline=BASE_NAME,delta_auc=sc[0]-b[0],delta_reweighted=sc[1]-b[1])
        if day==6:report['bootstrap_95_day6']=ci
        report[f'day{day}']=r;say(f'day{day}',json.dumps(r),ci);save_predictions(args.tag,day,y,p,ev)
    report['accepted_locally']=accept(report);(ROOT/f'imp3_{args.tag}_report.json').write_text(json.dumps(report,indent=2))
    say('ACCEPTED' if report['accepted_locally'] else 'NOT accepted',json.dumps(chosen))
    if report['accepted_locally']:
        ids=None;sub={}
        for n in ['stack','v6','v7']:
            i,pp=read_submission(n)
            if ids is None:ids=i
            assert i==ids;sub[n]=pp
        sub['warm16']=np.load(ROOT/'imp2_final_raw_warm1.0_d16_pc1.npy');eb=np.repeat(history_bins_eval(),10)
        p=offset(blend(sub,chosen['strength'],chosen['temperature']),eb,chosen['offsets'])
        path=write_submission(args.tag,ids,p,versioned=True);ver=path.name.split('_')[1]
        manifest=dict(tag=args.tag,candidate=path.name,baseline=BASE_NAME,recipe=dict(v15_weights=dict(stack=.6,v6=.3,v7=.1,warm16_a=.3),**chosen),
            sources=['submission_stack.csv','submission_v6.csv','submission_v7.csv','imp2_final_raw_warm1.0_d16_pc1.npy','evaluation_hist.npy (bins)'],
            report=report,submission=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (ROOT/f'imp3_manifest_{ver}.json').write_text(json.dumps(manifest,indent=2))
        queue_line(f'{path.name} (imp3_{args.tag})',report,f'V15 + strength/temperature + per-bin logit offsets {chosen}',f"imp3_postfix.py --tag {args.tag} --chosen '{json.dumps(chosen)}'")
        say('wrote',path)
