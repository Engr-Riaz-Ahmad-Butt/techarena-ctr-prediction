"""Freeze an independently accepted feature variant and prepare V19."""
from imp4_decompose import *
import imp5_aligned as experiment
import argparse,hashlib


def finalize(tag,version):
    assert tag in ['aligned','timecontext'] and version>=19
    report=json.loads((ROOT/f'imp5_{tag}_report.json').read_text())
    audit=json.loads((ROOT/f'imp5_test_{tag}_report.json').read_text())
    assert {r['day'] for r in report}=={5,6}
    assert {r['day'] for r in audit['reports']}=={5,6}
    assert all(r['delta']>0 and r['delta_reweighted']>0 for r in report)
    assert next(r for r in report if r['day']==6)['weighted_paired_interval'][0]>0
    assert all(r['recipe_max_error']<1e-12 for r in audit['reports'])
    assert all(r[k]['delta']>0 for r in audit['reports'] for k in ['auc','weighted_auc'])
    mix=report[0]['mix'];assert all(r['mix']==mix and r['trees']==TREES for r in report)
    destination=ROOT/f'submission_v{version}_candidate.csv'
    assert not destination.exists(),'Never overwrite a candidate'
    experiment.TAG=tag
    if tag=='timecontext':
        from imp5_timecontext import context_features
        experiment.aligned_features=context_features
    x,y,path=experiment.assemble_aligned([4,5,6],'final')
    model=CatBoostClassifier(iterations=TREES,**PARAMS,verbose=100)
    model.fit(x,y);model.save_model(str(ROOT/f'imp5_{tag}_final.cbm'))
    del x,y;gc.collect();path.unlink()
    a=np.asarray(evaluation)
    base=np.load(FEAT/'stack_eval_c7.npy',mmap_mode='r')
    warm=np.load(FEAT/'imp3_wf_eval.npy',mmap_mode='r')
    decomp=build('eval',a,ehist,7)
    new=experiment.aligned_features('eval',a,ehist,7)
    assert len(a)==len(base)==len(warm)==len(decomp)==len(new)==529390
    component=model.predict_proba(np.column_stack([base,warm,decomp,new]))[:,1]
    np.save(ROOT/f'imp5_{tag}_final_raw.npy',component)
    ids18,old18=read_submission('submission_v18.csv')
    ids6,v6=read_submission('submission_v6.csv');ids7,v7=read_submission('submission_v7.csv')
    assert ids18==ids6==ids7 and len(set(ids18))==len(ids18)
    old=np.load(ROOT/'imp4_decompose_final_raw.npy')
    raw_warm=np.mean([np.load(ROOT/f'{name}.npy').ravel() for name in
        ['imp2_final_raw_warm1.0_d16_pc1','imp3_warm16_s7_final_raw','imp3_warm16_s1009_final_raw']],axis=0)
    def combine(member):
        return .9*competitive(sigmoid(.7*logit(member)+.2*logit(v6)+.1*logit(v7)))+.1*raw_warm
    assert np.allclose(combine(old),old18,rtol=0,atol=1e-15)
    pred=combine((1-mix)*old+mix*component)
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    with destination.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['rowid','predicted_probability']);writer.writerows(zip(ids18,pred))
    checkids,checkpred=read_submission(destination.name)
    assert checkids==ids18 and np.allclose(checkpred,pred,rtol=0,atol=1e-15)
    manifest=dict(candidate=destination.name,variant=tag,rows=len(pred),trees=TREES,mix=mix,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  validation=report,baseline_official_auc=.809189,official_score=None)
    (ROOT/f'imp5_{tag}_final_report.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',choices=['aligned','timecontext'],required=True)
    ap.add_argument('--version',type=int,default=19);args=ap.parse_args();finalize(args.tag,args.version)
