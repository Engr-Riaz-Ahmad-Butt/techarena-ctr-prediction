"""Independent V20 export audit. No training or finalizer imports."""
import argparse,ast,csv,hashlib,json,os,sys
from pathlib import Path
SOURCE_ROOT=Path(__file__).resolve().parent
from config import ARTIFACT_DIR,EVALUATION_CSV
ROOT=ARTIFACT_DIR
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
N=529390

def read(name):
    with (ROOT/name).open(newline='',encoding='utf-8') as f:
        reader=csv.reader(f);assert next(reader)==['rowid','predicted_probability'];rows=list(reader)
    assert len(rows)==N and all(len(r)==2 for r in rows)
    return [r[0] for r in rows],np.array([float(r[1]) for r in rows])

def array(name):
    a=np.load(ROOT/f'{name}.npy').ravel()
    assert a.shape==(N,) and np.isfinite(a).all()
    return a

def compare(a,b,tol=1e-15):
    error=float(np.max(np.abs(a-b)));assert error<tol,error
    return error

def rank(a):return rankdata(a,method='average')/(len(a)+1)
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def check_bilinear(rows,indices):
    with np.load(ROOT/'imp11_bilinear_final_model.npz') as m:
        theta=m['theta'];assert theta.shape==(515,) and int(m['epoch'])==6
        assert int(m['cutoff'])==int(np.load(ROOT/'train.npy',mmap_mode='r')[:,2].min())+7*86400
        for key in ['theta','first','second']:assert np.isfinite(m[key]).all()
        train=np.load(ROOT/'train.npy',mmap_mode='r');cutoff=int(m['cutoff'])
        events=((train[::10,2]>=cutoff-3*86400)&(train[::10,2]<cutoff)).sum()
        assert int(m['step'])==6*((int(events)+1023)//1024)
    with np.load(ROOT/'imp2_final_model_warm1.0_d16_pc1.npz') as m:
        U,V,H,bu,bi=[m[k].astype(float) for k in ['U','V','H','bu','bi']]
    with np.load(ROOT/'imp3_model_warm16_cut2.npz') as m:anchor=unit(m['V'].astype(float))
    left,_,right=np.linalg.svd(unit(V).T@anchor,full_matrices=False);rotation=left@right
    a=rows[indices];e=a[::10];items=a[:,1].reshape(-1,10)
    h=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r')[e[:,6]]
    ts=np.load(ROOT/'evaluation_times.npy',mmap_mode='r')[e[:,6]];valid=h>=0
    assert np.all((ts<e[:,2,None])|~valid)
    rawh=(H[np.maximum(h,0)]*valid[:,:,None]).sum(1)
    z=U[e[:,0]]+rawh/np.sqrt(np.maximum(valid.sum(1)[:,None],1))
    offset=np.clip(-1.8+bu[e[:,0],None]+bi[items]+np.einsum('ed,ecd->ec',z,V[items]),-30,30).astype(np.float32)
    u=unit(U[e[:,0]]@rotation).astype(np.float32);h=unit(rawh@rotation).astype(np.float32)
    v=unit(V@rotation)[items].astype(np.float32)
    extra=np.stack([np.ones_like(offset),np.broadcast_to(bu[e[:,0],None],offset.shape),bi[items]],axis=2).astype(np.float32)
    logits=offset+np.einsum('ed,ecd->ec',u@theta[:256].reshape(16,16)+h@theta[256:512].reshape(16,16),v)+np.einsum('ecf,f->ec',extra,theta[512:])
    maximum=np.maximum(0,logits.max(1,keepdims=True));exp=np.exp(logits-maximum)
    p=(exp/(np.exp(-maximum)+exp.sum(1,keepdims=True))).ravel()
    return compare(p,array('imp11_bilinear_final_raw')[indices],1e-7)

def check_als(rows):
    with np.load(ROOT/'imp17_als_cut7.npz') as m:
        U,V,bu,bi,exposures=[m[k] for k in ['U','V','bu','bi','exposures']];g=float(m['g'])
        assert int(m['iteration'])==6
        train=np.load(ROOT/'train.npy',mmap_mode='r');cutoff=int(train[:,2].min())+7*86400
        assert int(m['cutoff'])==cutoff and (train[:,2]<cutoff).all()
        assert np.array_equal(np.bincount(train[:,0],minlength=len(U)),exposures)
        assert abs(g-float(train[:,5].mean()))<1e-15
    assert all(np.isfinite(x).all() for x in [U,V,bu,bi,exposures])
    e=rows[::10];items=rows[:,1].reshape(-1,10);known=exposures[e[:,0]]>0
    expected=g+bu[e[known,0],None]+bi[items[known]]+np.einsum('ed,ecd->ec',U[e[known,0]],V[items[known]])
    raw=array('imp15_als_final_raw').reshape(-1,10)
    known_error=compare(expected,raw[known],1e-12)
    cold=np.flatnonzero(~known);chosen=cold[np.linspace(0,len(cold)-1,min(100,len(cold)),dtype=int)] if len(cold) else []
    hist=np.load(ROOT/'evaluation_hist.npy',mmap_mode='r');times=np.load(ROOT/'evaluation_times.npy',mmap_mode='r')
    design=np.column_stack([np.ones(len(V)),V]);gram=design.T@design;rhs=-design.T@(g+bi)
    errors=[]
    for event in chosen:
        row=e[event];h=hist[row[6]];ts=times[row[6]];valid=h>=0
        assert (ts[valid]<row[2]).all()
        pairs=sorted(set(zip(h[valid].tolist(),ts[valid].tolist())))
        ids=np.array([p[0] for p in pairs],int);x=design[ids];confidence=9*len(ids)/len(V)
        answer=np.linalg.solve(confidence*gram+x.T@x+np.diag(np.r_[20.,np.ones(16)]),confidence*rhs+x.T@(1-g-bi[ids]))
        expected=g+answer[0]+bi[items[event]]+V[items[event]]@answer[1:]
        errors.append(compare(expected,raw[event],1e-12))
    return dict(known_rows=int(known.sum()*10),known_max_error=known_error,cold_events=len(chosen),cold_max_error=max(errors,default=0))

def run(evaluation_csv):
    frozenpath=ROOT/'imp19_export_frozen.json';frozen=json.loads(frozenpath.read_text())
    for group in ['source_sha256','prerequisite_sha256']:
        for name,digest in frozen[group].items():assert hashlib.sha256(((SOURCE_ROOT if group=='source_sha256' else ROOT)/name).read_bytes()).hexdigest()==digest,(group,name)
    assert frozen['version']==20
    audit=json.loads((ROOT/'imp19_test_candidate_report.json').read_text())
    assert audit['status']=='PASS' and audit['acceptance_gates_passed']
    ids,pred=read('submission_v20_candidate.csv');oldids,v19=read('submission_v19.csv')
    ids6,v6=read('submission_v6.csv');ids7,v7=read('submission_v7.csv')
    assert ids==oldids==ids6==ids7 and len(set(ids))==N
    with Path(evaluation_csv).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.reader(stream);header=next(reader);column=header.index('rowid')
        original_ids=[row[column] for row in reader]
    assert ids==original_ids and len(original_ids)==N
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    warm=np.mean([array(n) for n in ['imp2_final_raw_warm1.0_d16_pc1','imp3_warm16_s7_final_raw','imp3_warm16_s1009_final_raw']],axis=0)
    def logit(p):p=np.clip(p,1e-8,1-1e-8);return np.log(p)-np.log1p(-p)
    def combine(member):
        z=.7*logit(member)+.2*logit(v6)+.1*logit(v7)
        p=np.clip(1/(1+np.exp(-np.clip(z,-35,35))),1e-8,1-1e-8).reshape(-1,10);odds=p/(1-p)
        return .9*(.5*p+.5*odds/(1+odds.sum(1,keepdims=True))).ravel()+.1*warm
    stages={};old=array('imp5_aligned_final_raw');stages['v19']=compare(combine(old),v19)
    base=combine(.5*old+.5*array('imp5_warmcontext_final_raw'))
    stages['warmcontext']=compare(base,array('imp5_warmcontext_final_pred'))
    base=.5*base+.5*array('imp10_leafwise_final_raw');stages['leafwise']=compare(base,array('imp10_leafwise_final_pred'))
    base=.9*base+.1*array('imp11_bilinear_final_raw');stages['bilinear']=compare(base,array('imp11_bilinear_final_pred'))
    for prefix,w in [('imp15_als',.05),('imp17_tree',.75),('imp19_tree',.5)]:
        base=(1-w)*rank(base)+w*rank(array(f'{prefix}_final_raw'))
        stages[prefix]=compare(base,array(f'{prefix}_final_pred'))
    stages['csv']=compare(base,pred)
    rows=np.load(ROOT/'evaluation.npy',mmap_mode='r');assert len(rows)==N
    sampled=(np.linspace(0,N//10-1,100,dtype=int)[:,None]*10+np.arange(10)).ravel()
    names=['stack_eval_c7.npy','imp3_wf_eval.npy','imp4_decompose_eval.npy','imp5_aligned_eval.npy','imp6_context_eval.npy','imp17_als_eval.npy','imp19_als_aligned_eval.npy']
    parts=[np.load(ROOT/'features'/name,mmap_mode='r') for name in names]
    assert [p.shape for p in parts]==[(N,w) for w in [190,4,32,32,32,32,32]]
    x=np.column_stack([p[sampled] for p in parts]);models=[]
    tree=ast.parse((SOURCE_ROOT/'imp10_leafwise.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SETTINGS' for t in n.targets)]
    ns={};exec(compile(tree,'settings','exec'),ns);settings=ns['SETTINGS']
    for prefix,width,trees in [('imp5_warmcontext',290,599),('imp10_leafwise',290,641),('imp17_tree',322,1063),('imp19_tree',354,735)]:
        if prefix=='imp5_warmcontext':
            m=CatBoostClassifier();m.load_model(str(ROOT/f'{prefix}_final.cbm'));params=m.get_all_params()
            assert m.tree_count_==trees and len(m.feature_names_)==width and params['depth']==7 and abs(params['learning_rate']-.08)<1e-7
            replay=m.predict_proba(x[:,:width])[:,1]
        else:
            m=lgb.Booster(model_file=str(ROOT/f'{prefix}_final.txt'));assert m.num_trees()==trees and m.num_feature()==width
            for key in settings:
                if key in ['verbosity']:continue
                if isinstance(settings[key],str):assert settings[key] in m.params[key] if isinstance(m.params[key],list) else settings[key]==m.params[key]
                else:assert abs(float(settings[key])-float(m.params[key]))<1e-12,(key,m.params[key])
            replay=m.predict(x[:,:width],num_threads=1)
        models.append(dict(prefix=prefix,features=width,trees=trees,replay_rows=len(sampled),max_error=compare(replay,array(f'{prefix}_final_raw')[sampled],1e-12)))
    report=dict(status='PASS',rows=N,unique_ids=N,ids_exact_against_verified_v19=True,ids_exact_against_original_evaluation=True,original_evaluation_csv=str(Path(evaluation_csv).resolve()),recipe_stage_errors=stages,models=models,bilinear_sample_max_error=check_bilinear(rows,sampled),als=check_als(rows),sha256=hashlib.sha256((ROOT/'submission_v20_candidate.csv').read_bytes()).hexdigest(),local_auc=.8215282579633503,confirmation_auc=.814599936313616,official_score=None)
    manifest=json.loads((ROOT/'imp19_final_report.json').read_text());assert manifest['sha256']==report['sha256'] and manifest['rows']==N
    assert manifest['export_manifest_sha256']==hashlib.sha256(frozenpath.read_bytes()).hexdigest()
    report['export_manifest_sha256']=manifest['export_manifest_sha256']
    (ROOT/'imp19_test_final_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation-csv',default=os.environ.get('CTR_EVALUATION_CSV',str(EVALUATION_CSV)))
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.evaluation_csv)
