"""Select V7/V8 blend on January 26, confirm without retuning on January 25."""
from config import TRAIN_CSV, EVALUATION_CSV
from core import *

reports=[];weight=None
for day in [6,5]:
    a=np.load(ROOT/('predictions_v7_day'+str(day)+'.npz'))
    b=np.load(ROOT/('predictions_v8_day'+str(day)+'.npz'))
    assert np.array_equal(a['events'],b['events']) and np.array_equal(a['y'],b['y'])
    if day==6:
        grid={str(w):float(roc_auc_score(a['y'],w*a['pred']+(1-w)*b['pred'])) for w in [0,.25,.5,.75,1.]}
        weight=float(max(grid,key=grid.get));print('First-date blend grid:',grid,'V7 weight:',weight)
    old=float(roc_auc_score(b['y'],b['pred']))
    auc=float(roc_auc_score(a['y'],weight*a['pred']+(1-weight)*b['pred']))
    reports.append({'validation_day':day,'v8_auc':old,'combined_auc':auc,'improvement':auc-old,'v7_weight':weight})
print(json.dumps(reports,indent=2))
(ROOT/'validation_ensemble.json').write_text(json.dumps(reports,indent=2))
if weight==0 or not all(r['improvement']>0 for r in reports):
    print('Recommendation: submission_v8.csv');sys.exit(0)
seen=set()
with (ROOT/'submission_v7.csv').open(newline='',encoding='utf-8') as f7,(ROOT/'submission_v8.csv').open(newline='',encoding='utf-8') as f8,(ROOT/'submission_v9.csv').open('w',newline='',encoding='utf-8') as out:
    w=csv.writer(out);w.writerow(['rowid','predicted_probability'])
    for a,b in zip(csv.DictReader(f7),csv.DictReader(f8),strict=True):
        assert a['rowid']==b['rowid'] and a['rowid'] not in seen
        p=weight*float(a['predicted_probability'])+(1-weight)*float(b['predicted_probability'])
        assert np.isfinite(p) and 0<=p<=1
        seen.add(a['rowid']);w.writerow([a['rowid'],format(p,'.10g')])
assert len(seen)==529390
(ROOT/'v9_blend.json').write_text(json.dumps({'v7_weight':weight,'v8_weight':1-weight},indent=2))
print('Recommendation: submission_v9.csv')
