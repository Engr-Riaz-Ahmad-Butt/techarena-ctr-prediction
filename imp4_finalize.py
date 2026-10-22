"""Refit the decomposition candidate only after both validation dates pass."""
from imp4_decompose import *
import hashlib


def run_final():
    report=json.loads((ROOT/'imp4_decompose_report.json').read_text())
    audit=json.loads((ROOT/'imp4_test_decomposition_report.json').read_text())
    assert {r['day'] for r in report}=={5,6}
    assert {r['day'] for r in audit['reports']}=={5,6}
    assert all(r['delta']>0 and r['delta_reweighted']>0 for r in report)
    assert report[0]['weighted_paired_interval'][0]>0
    assert all(r['recipe_max_error']<1e-12 for r in audit['reports'])
    assert all(r[k]['delta']>0 for r in audit['reports'] for k in ['auc','weighted_auc'])
    mix=report[0]['mix']
    assert all(r['mix']==mix and r['trees']==TREES for r in report)
    destination=ROOT/'submission_v18_candidate.csv'
    assert not destination.exists(), 'Do not overwrite an existing candidate'
    x,y,path=assemble_new([4,5,6],'decompose_final')
    model=CatBoostClassifier(iterations=TREES,**PARAMS,verbose=100)
    model.fit(x,y)
    model.save_model(str(ROOT/'imp4_decompose_final.cbm'))
    del x,y;gc.collect();path.unlink()
    a=np.asarray(evaluation)
    new=build('eval',a,ehist,7)
    base=np.load(FEAT/'stack_eval_c7.npy',mmap_mode='r')
    warm=np.load(FEAT/'imp3_wf_eval.npy',mmap_mode='r')
    assert len(a)==len(base)==len(warm)==len(new)==529390
    component=model.predict_proba(np.column_stack([base,warm,new]))[:,1]
    np.save(ROOT/'imp4_decompose_final_raw.npy',component)
    ids,old=read_submission('submission_imp3_warmfeat_raw.csv')
    ids6,v6=read_submission('submission_v6.csv')
    ids7,v7=read_submission('submission_v7.csv')
    ids17,old17=read_submission('submission_v17.csv')
    assert ids==ids6==ids7==ids17 and len(set(ids))==len(ids)
    raw_warm=np.mean([np.load(ROOT/f'{name}.npy').ravel() for name in
        ['imp2_final_raw_warm1.0_d16_pc1','imp3_warm16_s7_final_raw','imp3_warm16_s1009_final_raw']],axis=0)
    replay17=.9*competitive(sigmoid(.7*logit(old)+.2*logit(v6)+.1*logit(v7)))+.1*raw_warm
    assert np.allclose(replay17,old17,rtol=0,atol=1e-9), float(np.max(np.abs(replay17-old17)))
    member=(1-mix)*old+mix*component
    pred=.9*competitive(sigmoid(.7*logit(member)+.2*logit(v6)+.1*logit(v7)))+.1*raw_warm
    assert len(pred)==529390 and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    with destination.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['rowid','predicted_probability']);writer.writerows(zip(ids,pred))
    checkids,checkpred=read_submission(destination.name)
    assert checkids==ids and np.allclose(checkpred,pred,rtol=0,atol=1e-15)
    manifest=dict(candidate=destination.name,rows=len(pred),trees=TREES,mix=mix,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  validation=report,official_score=None)
    (ROOT/'imp4_final_report.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':run_final()
