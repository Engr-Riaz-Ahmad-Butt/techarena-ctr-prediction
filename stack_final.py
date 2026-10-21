"""Fit the final stacked model (days 4-6, evaluation features cut off at day 7) and write submission_stack.csv.

The standalone stacked model failed the two-split gate, but its blend with V6 passed on both
splits, so the final component is still needed for blending. Tree count comes from validation_stack.json.
"""
from stack_experiment import *
from config import EVALUATION_CSV

if __name__=='__main__':
    t0=time.time()
    chosen=json.loads((ROOT/'validation_stack.json').read_text())[0]['iterations']
    print('Fitting final stacked model on days 4-6 with',chosen,'trees',flush=True)
    xf,yf=fit_blocks([4,5,6])
    xe=block('eval',np.asarray(evaluation),ehist,7)
    print('Fit rows',len(yf),'elapsed min',round((time.time()-t0)/60,1),flush=True)
    model=CatBoostClassifier(iterations=chosen,verbose=200,**PARAMS);model.fit(xf,yf)
    model.save_model(str(ROOT/'model_stack.cbm'));del xf,yf
    prob=model.predict_proba(xe)[:,1]
    assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    source=EVALUATION_CSV;seen=set()
    with source.open(newline='',encoding='utf-8') as srcfile,(ROOT/'submission_stack.csv').open('w',newline='',encoding='utf-8') as dest:
        writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
        for row,p in zip(csv.DictReader(srcfile),prob,strict=True):
            assert row['rowid'] not in seen
            seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')])
    assert len(seen)==529390
    print('Verified stacked submission',len(seen),'rows, minutes',round((time.time()-t0)/60,1),flush=True)
