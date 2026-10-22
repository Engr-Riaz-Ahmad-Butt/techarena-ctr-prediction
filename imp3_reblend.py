"""Idea 1: component-level re-blend including the warm16 raw model.

Family A ('logit5'): sigmoid(sum w_i*logit(member_i)) over {stack,v6,choice,v7,warm16}, then the 50%
  event-competition adjustment. Weights on a 0.1 grid summing to 1.
Family B ('v14form'): competitive(sigmoid(4-member logit blend)) then (1-a)*that + a*warm16 (V14's form,
  with the four weights re-selected). a in {.1,...,.5}.
Selection: day-6 history-reweighted AUC only. Day 5 applies the chosen settings unchanged.
Optional --extra v1stack adds V1-stacked as a sixth member (day 6 only unless day-5 file exists).
"""
from imp3_lib import *
import itertools,argparse,hashlib

def simplex(k,step=10):
    for c in itertools.product(range(step+1),repeat=k-1):
        if sum(c)<=step:yield tuple(v/step for v in c)+((step-sum(c))/step,)

def pred_a(L,w):return competitive(sigmoid(np.tensordot(np.array(w),L,1)))
def pred_b(L4,warm,w,a):return (1-a)*competitive(sigmoid(np.tensordot(np.array(w),L4,1)))+a*warm

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extra',nargs='*',default=[]);ap.add_argument('--tag',default='reblend')
    ap.add_argument('--family',default='both');ap.add_argument('--chosen',default=None,help='JSON of a previously selected recipe (skips the day-6 grid)')
    ap.add_argument('--override',nargs='*',default=[],help='member=imp3tag: use predictions_imp3_<tag>_day*.npz / submission_imp3_<tag>_raw.csv for that member')
    ap.add_argument('--warm-seeds',default='151',help='comma list; 151 = V14 warm16, others = imp3_warm16_s<seed> files; raw averaged');args=ap.parse_args()
    log=open(ROOT/f'imp3_{args.tag}.log','a')
    def say(*a):
        s=' '.join(str(x) for x in a);print(s,flush=True);log.write(s+'\n');log.flush()
    members=['stack','v6','choice','v7','warm16']+list(args.extra)
    chosen={};report={'members':members,'tag':args.tag}
    days=[6] if any(not (ROOT/f'predictions_{n}_day5.npz').exists() for n in args.extra) else [6,5]
    if days==[6]:print('day-5 file missing for an extra member: day-6 exploration only',flush=True)
    for day in days:
        y,ev,parts,v14=load_components(day,extra=args.extra)
        seeds=[int(x) for x in args.warm_seeds.split(',')]
        for o in args.override:
            m,tg=o.split('=');d=np.load(ROOT/f'predictions_imp3_{tg}_day{day}.npz');assert np.array_equal(d['events'],ev) and np.array_equal(d['y'],y)
            parts[m]=d['pred'];say(f'day{day} member {m} overridden by imp3_{tg}')
        if seeds!=[151]:
            raws=[]
            for sd in seeds:
                if sd==151:raws.append(parts['warm16'])
                else:
                    d=np.load(ROOT/f'imp3_warm16_s{sd}_day{day}.npz');assert np.array_equal(d['events'],ev);raws.append(d['raw'])
            parts['warm16']=np.mean(raws,axis=0);say(f'day{day} warm16 = mean of seeds {seeds}; raw auc',scores(y,parts['warm16'],history_weights(ev)[0]))
        sw,vg=history_weights(ev)
        L=np.array([logit(parts[m]) for m in members]);warm=parts['warm16']
        base=scores(y,v14,sw);say(f'day{day} {BASE_NAME} auc/reweighted',base)
        if day==6 and args.chosen:
            chosen=json.loads(args.chosen);report['chosen']=chosen;say('using given recipe',chosen)
        elif day==6:
            t=time.time();trials=[]
            if args.family in('both','A'):
                for w in simplex(len(members)):
                    p=pred_a(L,w);trials.append((float(roc_auc_score(y,p,sample_weight=sw)),'A',w,None))
                say('family A done',len(trials),'in',round(time.time()-t),'s')
            if args.family in('both','B'):
                L4=L[[0,1,2,3]]
                for w in simplex(4):
                    for a in [.1,.15,.2,.25,.3,.4,.5]:
                        p=pred_b(L4,warm,w,a);trials.append((float(roc_auc_score(y,p,sample_weight=sw)),'B',w,a))
                say('family B done',len(trials),'in',round(time.time()-t),'s')
            trials.sort(key=lambda r:-r[0]);say('top 10:',*[f'\n  {r}' for r in trials[:10]])
            bestA=[r for r in trials if r[1]=='A'][:1];bestB=[r for r in trials if r[1]=='B'][:1]
            say('best A',bestA,'best B',bestB)
            fam,w,a=trials[0][1],trials[0][2],trials[0][3];chosen=dict(family=fam,weights=dict(zip(members,w)),a=a)  # family B: 4 weights
            report['selection_trials_top20']=[dict(score=r[0],family=r[1],weights=r[2],a=r[3]) for r in trials[:20]]
            report['chosen']=chosen
        w=[chosen['weights'].get(m,0.) for m in members]
        p=pred_a(L,w) if chosen['family']=='A' else pred_b(L[[0,1,2,3]],warm,w[:4],chosen['a'])
        s=scores(y,p,sw);ci,mean=bootstrap_delta(y,p,v14,sw) if day==6 else (None,None)
        r=dict(auc=s[0],reweighted=s[1],base_auc=base[0],base_reweighted=base[1],baseline=BASE_NAME,delta_auc=s[0]-base[0],delta_reweighted=s[1]-base[1])
        if day==6:report['bootstrap_95_day6']=ci;report['bootstrap_mean_day6']=mean
        report[f'day{day}']=r;say(f'day{day}',json.dumps(r),'ci' if day==6 else '',ci)
        save_predictions(args.tag,day,y,p,ev)
        del L,parts;gc.collect()
    if days==[6]:
        (ROOT/f'imp3_{args.tag}_report.json').write_text(json.dumps(report,indent=2));say('exploration only; no day-5 check');sys.exit(0)
    report['accepted_locally']=accept(report)
    (ROOT/f'imp3_{args.tag}_report.json').write_text(json.dumps(report,indent=2))
    say('ACCEPTED' if report['accepted_locally'] else 'NOT accepted',json.dumps(chosen))
    if report['accepted_locally'] and not args.extra:  # extras lack submissions
        ids=None;sub={}
        for n in ['stack','v6','v8','v7']:
            i,p=read_submission(n)
            if ids is None:ids=i
            assert i==ids;sub[n]=p
        sub['choice']=np.clip((sub['v8']-.75*sub['v6'])/.25,0,1)
        for o in args.override:
            m,tg=o.split('=');i,p=read_submission(f'imp3_{tg}_raw');assert i==ids;sub[m]=p
        raws=[np.load(ROOT/'imp2_final_raw_warm1.0_d16_pc1.npy') if sd==151 else np.load(ROOT/f'imp3_warm16_s{sd}_final_raw.npy') for sd in seeds]
        sub['warm16']=np.mean(raws,axis=0);assert len(sub['warm16'])==len(ids)
        L=np.array([logit(sub[m]) for m in members]);w=[chosen['weights'].get(m,0.) for m in members]
        p=pred_a(L,w) if chosen['family']=='A' else pred_b(L[[0,1,2,3]],sub['warm16'],w[:4],chosen['a'])
        path=write_submission(args.tag,ids,p,versioned=True);ver=path.name.split('_')[1]
        manifest=dict(tag=args.tag,candidate=path.name,baseline=BASE_NAME,recipe=chosen,members=members,overrides=args.override,warm_seeds=seeds,competition_strength=.5,
            sources=['submission_stack.csv','submission_v6.csv','submission_v8.csv (choice=(v8-.75*v6)/.25)','submission_v7.csv','warm16 raw = mean over seeds '+str(seeds)+' of imp2_final_raw_warm1.0_d16_pc1.npy / imp3_warm16_s<seed>_final_raw.npy'],
            report=report,submission=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (ROOT/f'imp3_manifest_{ver}.json').write_text(json.dumps(manifest,indent=2))
        cmd=f"imp3_reblend.py --tag {args.tag} --warm-seeds {args.warm_seeds} {' '.join('--override '+o for o in args.override)} --chosen '{json.dumps(chosen)}'"
        queue_line(f'{path.name} (imp3_{args.tag})',report,f'log-odds re-blend {chosen} warm seeds {seeds}',cmd)
        say('wrote',path)
