"""Three-way blend of the stacked model, V6 and the V8 choice component.

The choice component is recovered exactly from the saved V8 blend: V8 = 0.25*choice + 0.75*V6.
Weights are chosen on January 26 over a 0.1-step simplex grid and then fixed for January 25,
where the blend must beat V8 for the file to be written. Output: submission_blend.csv.
"""
from core import *
import itertools

def load(day):
    s=np.load(ROOT/('predictions_stack_day'+str(day)+'.npz'))
    v6=np.load(ROOT/('predictions_v6_day'+str(day)+'.npz'))
    v8=np.load(ROOT/('predictions_v8_day'+str(day)+'.npz'))
    assert np.array_equal(s['events'],v6['events']) and np.array_equal(s['events'],v8['events'])
    assert np.array_equal(s['y'],v6['y']) and np.array_equal(s['y'],v8['y'])
    choice=(v8['pred']-.75*v6['pred'])/.25
    return s['y'],s['pred'],v6['pred'],choice,v8['pred']

def mix(w,s,v6,c):return w[0]*s+w[1]*v6+w[2]*c

grid=[(a/10,b/10,(10-a-b)/10) for a in range(11) for b in range(11-a)]
reports=[];weights=None
for day in [6,5]:
    y,s,v6,c,v8=load(day)
    if day==6:
        scores={w:float(roc_auc_score(y,mix(w,s,v6,c))) for w in grid}
        weights=max(scores,key=scores.get)
        top=sorted(scores.items(),key=lambda kv:-kv[1])[:5]
        print('January 26 top weights (stack, v6, choice):',[(w,round(a,6)) for w,a in top],flush=True)
    auc=float(roc_auc_score(y,mix(weights,s,v6,c)));v8_auc=float(roc_auc_score(y,v8))
    reports.append({'validation_day':day,'rows':int(len(y)),'v8_auc':v8_auc,'blend_auc':auc,'improvement':auc-v8_auc,
                    'weights':{'stack':weights[0],'v6':weights[1],'choice':weights[2]}})
    print(reports[-1],flush=True)
(ROOT/'validation_blend.json').write_text(json.dumps(reports,indent=2))
if not all(r['improvement']>0 for r in reports):
    print('Blend did not beat V8 on both splits. Keep submission_v8.csv.');sys.exit(0)
stack=ROOT/'submission_stack.csv'
if not stack.exists():
    print('Blend passed. submission_stack.csv not built yet; rerun after stack_final.py.');sys.exit(0)
seen=set();n=0
with stack.open(newline='',encoding='utf-8') as fs,(ROOT/'submission_v6.csv').open(newline='',encoding='utf-8') as f6,\
     (ROOT/'submission_v8.csv').open(newline='',encoding='utf-8') as f8,(ROOT/'submission_blend.csv').open('w',newline='',encoding='utf-8') as out:
    w=csv.writer(out);w.writerow(['rowid','predicted_probability'])
    for a,b,d in zip(csv.DictReader(fs),csv.DictReader(f6),csv.DictReader(f8),strict=True):
        assert a['rowid']==b['rowid']==d['rowid'] and a['rowid'] not in seen
        p6=float(b['predicted_probability']);p8=float(d['predicted_probability']);pc=(p8-.75*p6)/.25
        p=weights[0]*float(a['predicted_probability'])+weights[1]*p6+weights[2]*pc
        p=min(max(p,0.0),1.0);assert np.isfinite(p)
        seen.add(a['rowid']);w.writerow([a['rowid'],format(p,'.10g')]);n+=1
assert n==529390
(ROOT/'blend_weights.json').write_text(json.dumps({'stack':weights[0],'v6':weights[1],'choice':weights[2]},indent=2))
print('Wrote submission_blend.csv',n,'rows with weights',weights)
