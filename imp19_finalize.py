"""Refit the exact accepted imp19 dependency chain and export V20 candidate."""
from imp19_tree import *
import imp11_bilinear as bilinear
from imp11_bilinear_core import loss_gradient,adam_step
import hashlib
SOURCE_ROOT=Path(__file__).resolve().parent


def accepted():
    audit=json.loads((ROOT/'imp19_test_candidate_report.json').read_text())
    assert audit['status']=='PASS' and audit['both_dates_improved'] and audit['acceptance_gates_passed']
    for prefix in ['imp10_leafwise','imp11_bilinear','imp15_als','imp17_tree','imp19_tree']:
        records=json.loads((ROOT/f'{prefix}_report.json').read_text())
        assert {r['day'] for r in records}=={5,6}
        assert all(r['delta']>0 and r['delta_reweighted']>0 for r in records)
        assert all(r['paired_interval'][0]>0 and r['weighted_paired_interval'][0]>0 for r in records)
    for prefix,key,value,weight in [('imp10_leafwise','trees',641,.5),('imp11_bilinear','epoch',6,.1),('imp15_als','iteration',6,.05),('imp17_tree','trees',1063,.75),('imp19_tree','trees',735,.5)]:
        records=json.loads((ROOT/f'{prefix}_report.json').read_text())
        assert all(r[key]==value and r['weight']==weight for r in records)
    context=json.loads((ROOT/'imp5_warmcontext_report.json').read_text())
    assert {r['day'] for r in context}=={5,6} and all(r['trees']==599 and r['mix']==.5 for r in context)
    return audit


def freeze_export():
    sources=['imp19_finalize.py','imp19_tree.py','imp19_als_aligned.py','imp17_tree.py','imp17_als_features.py','imp15_als.py','imp15_als_core.py','imp11_bilinear.py','imp11_bilinear_core.py','imp10_leafwise.py','imp6_warmcontext.py','imp5_aligned.py','imp4_decompose.py','imp3_warmfeat.py','imp3_lib.py','choice64_core.py']
    prerequisites=['submission_v19.csv','submission_v6.csv','submission_v7.csv','imp5_aligned_final_raw.npy','imp2_final_raw_warm1.0_d16_pc1.npy','imp3_warm16_s7_final_raw.npy','imp3_warm16_s1009_final_raw.npy','imp2_final_model_warm1.0_d16_pc1.npz','imp3_model_warm16_cut2.npz','imp19_test_candidate_report.json']
    prerequisites += [f'{p}_report.json' for p in ['imp5_warmcontext','imp10_leafwise','imp11_bilinear','imp15_als','imp17_tree','imp19_tree']]
    frozen=dict(source_sha256={n:hashlib.sha256((SOURCE_ROOT/n).read_bytes()).hexdigest() for n in sources},prerequisite_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in prerequisites},version=20,cutoff=cutoff_of(7))
    manifest=ROOT/'imp19_export_frozen.json'
    if manifest.exists():assert json.loads(manifest.read_text())==frozen,'Export sources/inputs changed; cached results cannot be reused'
    else:
        for prefix in ['imp5_warmcontext','imp10_leafwise','imp11_bilinear','imp15_als','imp17_tree','imp19_tree']:
            assert not (ROOT/f'{prefix}_final_raw.npy').exists(),'Untracked final raw cache'
            assert not (ROOT/f'{prefix}_final.txt').exists() and not (ROOT/f'{prefix}_final.cbm').exists()
        assert not (ROOT/'imp11_bilinear_final_model.npz').exists()
        manifest.write_text(json.dumps(frozen,indent=2))
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def selected(prefix):
    records=json.loads((ROOT/f'{prefix}_report.json').read_text())
    return next(r for r in records if r['day']==6)


def ranked(p):return rankdata(p,method='average')/(len(p)+1)


def evaluation_matrix(width):
    a=np.asarray(evaluation)
    parts=[np.load(FEAT/'stack_eval_c7.npy',mmap_mode='r'),
           np.load(FEAT/'imp3_wf_eval.npy',mmap_mode='r'),build('eval',a,ehist,7),
           combined_features('eval',a,ehist,7)]
    if width>=322:
        from imp17_als_features import als_features
        parts.append(als_features('eval',a,ehist,7))
    if width>=354:parts.append(interactions('eval',a,ehist,7))
    result=np.column_stack(parts)
    assert result.shape==(529390,width) and np.isfinite(result).all()
    return result


def tree_final(prefix,width,trees,cat=False):
    rawpath=ROOT/f'{prefix}_final_raw.npy'
    modelpath=ROOT/f'{prefix}_final.{"cbm" if cat else "txt"}'
    if not rawpath.exists():
        if not modelpath.exists():
            if width==290:configure()
            elif width==322:
                from imp17_tree import setup as configure17
                configure17()
            else:setup()
            engine.TAG=f'export_{prefix}';engine.WIDTH=width
            x,y,path=engine.assemble_aligned([4,5,6],'final')
            if cat:
                model=CatBoostClassifier(iterations=trees,**PARAMS,verbose=100)
                model.fit(x,y);model.save_model(str(modelpath))
            else:
                data=lgb.Dataset(x,label=y,free_raw_data=True)
                model=lgb.train(SETTINGS,data,num_boost_round=trees)
                model.save_model(str(modelpath));del data
            del model,x,y;gc.collect();path.unlink()
        if cat:
            model=CatBoostClassifier();model.load_model(str(modelpath))
            assert model.tree_count_==trees and len(model.feature_names_)==width
        else:
            model=lgb.Booster(model_file=str(modelpath))
            assert model.current_iteration()==trees and model.num_feature()==width
        x=evaluation_matrix(width)
        raw=model.predict_proba(x)[:,1] if cat else model.predict(x)
        assert raw.shape==(529390,) and np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
        np.save(rawpath,raw);del x,model;gc.collect()
    return np.load(rawpath)


def eval_bilinear_pack():
    with np.load(model_path(7)) as m:U,V,H,bu,bi=[m[k].astype(np.float64) for k in ['U','V','H','bu','bi']]
    unit=lambda a:a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    with np.load(model_path(2)) as m:anchor=unit(m['V'].astype(np.float64))
    left,_,right=np.linalg.svd(unit(V).T@anchor,full_matrices=False);rotation=left@right
    a=np.asarray(evaluation);e=a[::10];ids=a[:,1].reshape(-1,10)
    h=np.asarray(ehist[e[:,6]]);valid=h>=0
    times=np.load(ROOT/'evaluation_times.npy',mmap_mode='r')[e[:,6]]
    assert np.all((times<e[:,2,None])|~valid)
    raw_h=(H[np.maximum(h,0)]*valid[:,:,None]).sum(1)
    profile=U[e[:,0]]+raw_h/np.sqrt(np.maximum(valid.sum(1)[:,None],1))
    offset=np.clip(-1.8+bu[e[:,0],None]+bi[ids]+np.einsum('ed,ecd->ec',profile,V[ids]),-30,30)
    extra=np.stack([np.ones_like(offset),np.broadcast_to(bu[e[:,0],None],offset.shape),bi[ids]],axis=2)
    return dict(u=unit(U[e[:,0]]@rotation).astype(np.float32),h=unit(raw_h@rotation).astype(np.float32),
                v=unit(V@rotation)[ids].astype(np.float32),offset=offset.astype(np.float32),extra=extra.astype(np.float32),events=e[:,6].copy())


def bilinear_final():
    path=ROOT/'imp11_bilinear_final_raw.npy';statepath=ROOT/'imp11_bilinear_final_model.npz'
    if not path.exists():
        if not statepath.exists():
            parts=[]
            for d in [4,5,6]:
                rows=np.asarray(train[(train[:,2]>=cutoff_of(d))&(train[:,2]<cutoff_of(d+1))])
                parts.append(bilinear.pack(f'day{d}',rows,d))
            data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};del parts;gc.collect()
            theta=np.zeros(515,np.float32);first=theta.copy();second=theta.copy();step=0;rng=np.random.default_rng(42)
            for epoch in range(1,selected('imp11_bilinear')['epoch']+1):
                order=rng.permutation(len(data['u']));losses=[]
                for off in range(0,len(order),1024):
                    ix=order[off:off+1024]
                    loss,gradient,_=loss_gradient(theta,*[data[k][ix] for k in ['u','h','v','offset','extra','target']])
                    step+=1;adam_step(theta,gradient,first,second,step);losses.append(loss)
                print('Final bilinear epoch',epoch,'loss',float(np.mean(losses)),flush=True)
            np.savez_compressed(statepath,theta=theta,first=first,second=second,epoch=epoch,step=step,cutoff=cutoff_of(7))
            del data;gc.collect()
        with np.load(statepath) as saved:theta=saved['theta']
        raw=bilinear.predict(theta,eval_bilinear_pack());np.save(path,raw)
    return np.load(path)


def run():
    accepted();destination=ROOT/'submission_v20_candidate.csv'
    assert not destination.exists() and not (ROOT/'submission_v20.csv').exists(),'Never overwrite a submission'
    frozen_sha256=freeze_export()
    # Reconstruct the accepted probability baseline using the same frozen recipe.
    context=tree_final('imp5_warmcontext',290,599,True)
    ids,v19=read_submission('submission_v19.csv');ids6,v6=read_submission('submission_v6.csv');ids7,v7=read_submission('submission_v7.csv')
    assert ids==ids6==ids7 and len(ids)==len(set(ids))==529390
    warm=np.mean([np.load(ROOT/f'{n}.npy').ravel() for n in ['imp2_final_raw_warm1.0_d16_pc1','imp3_warm16_s7_final_raw','imp3_warm16_s1009_final_raw']],axis=0)
    old=np.load(ROOT/'imp5_aligned_final_raw.npy')
    def combine(member):return .9*competitive(sigmoid(.7*logit(member)+.2*logit(v6)+.1*logit(v7)))+.1*warm
    assert np.max(np.abs(combine(old)-v19))<1e-12
    mix=json.loads((ROOT/'imp5_warmcontext_report.json').read_text())[0]['mix']
    base=combine((1-mix)*old+mix*context);np.save(ROOT/'imp5_warmcontext_final_pred.npy',base)
    r=selected('imp10_leafwise');raw=tree_final('imp10_leafwise',290,r['trees'])
    base=(1-r['weight'])*base+r['weight']*raw;np.save(ROOT/'imp10_leafwise_final_pred.npy',base)
    r=selected('imp11_bilinear');raw=bilinear_final()
    base=(1-r['weight'])*base+r['weight']*raw;np.save(ROOT/'imp11_bilinear_final_pred.npy',base)
    from imp17_als_features import ensure_model
    from imp15_als import score
    with np.load(ensure_model(7)) as m:
        raw=score(np.asarray(evaluation),m['U'],m['V'],m['bu'],m['bi'],float(m['g']),m['exposures'],histories=ehist,history_times=np.load(ROOT/'evaluation_times.npy',mmap_mode='r'))
    np.save(ROOT/'imp15_als_final_raw.npy',raw)
    r=selected('imp15_als');base=(1-r['weight'])*ranked(base)+r['weight']*ranked(raw);np.save(ROOT/'imp15_als_final_pred.npy',base)
    for prefix,width in [('imp17_tree',322),('imp19_tree',354)]:
        r=selected(prefix);raw=tree_final(prefix,width,r['trees'])
        base=(1-r['weight'])*ranked(base)+r['weight']*ranked(raw)
        np.save(ROOT/f'{prefix}_final_pred.npy',base)
    assert base.shape==(529390,) and np.isfinite(base).all() and ((base>=0)&(base<=1)).all()
    with destination.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['rowid','predicted_probability']);writer.writerows(zip(ids,base))
    manifest=dict(candidate=destination.name,rows=len(base),experiment='imp19_tree',local_auc=.8215282579633503,confirmation_auc=.814599936313616,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),export_manifest_sha256=frozen_sha256,official_score=None,ranking_scores_only=True,final_training_days=[4,5,6],cutoff=cutoff_of(7))
    (ROOT/'imp19_final_report.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
