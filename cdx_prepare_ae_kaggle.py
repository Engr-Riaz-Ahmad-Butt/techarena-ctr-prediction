"""Package only current causal AE inputs for the user's requested Kaggle run."""
from pathlib import Path
import ast,hashlib,json,pickle,shutil,zipfile
import numpy as np
from scipy.stats import rankdata
import cdx_policy_hpf as m
B,A,H=m.B,m.A,m.H;R=H/'cdx_catalog_ae_001';P=R/'kaggle_payload'
P.mkdir(exist_ok=True)
for source,name in [(R/'clicks.npz','clicks.npz'),(R/'active_users.npy','active_users.npy'),
    (H/'cdx_collective_001/control_model.npz','initial_model.npz'),(R/'frozen.json','frozen.json'),(R/'data_info.json','data_info.json')]:
    shutil.copyfile(source,P/name)
rows=np.load(H/'score_rows.npy',mmap_mode='r');co=np.load(H/'cohort.npz');legacy=np.load(H/'matched_refresh_001/predictions.npz')
assert np.all(rows[:,5]==-1) and len(np.unique(rows[::10,0]))==49420
hist=np.load(A/'train_hist.npy',mmap_mode='r')[rows[::10,6]]
ts=np.load(A/'train_times.npy',mmap_mode='r')[rows[::10,6]]
assert np.all((ts<rows[::10,2,None])|(hist<0))
items=np.asarray(rows[:,1]).reshape(-1,10)
policy=pickle.loads((H/'cdx_policy_hpf_001/prefix3_models.pkl').read_bytes())
raw=m.raw_svd(policy['svd'],rows[::10,0],items)
e3=np.mean([m.cdf(policy['cdfs']['svd'][k],raw[k]) for k in range(3)],axis=0)
base=legacy['control_blend'].reshape(-1,10)+.015*m.centered(e3)
assert np.allclose(base[co['sample']],np.load(H/'cdx_policy_hpf_001/screen_predictions.npz')['control'],atol=1e-12)
np.savez_compressed(P/'prediction_features.npz',history=hist,items=items,control=base,
    point=np.load(H/'cdx_collective_001/frozen_predictions.npz')['raw_control'],
    old_rank=rankdata(legacy['other_expert'])/(len(rows)+1),events=co['events'],
    sample=co['sample'],sample_weight=co['sample_event_weight'],weight=co['event_weight'],arrival=co['arrival'],history_group=co['history_group'])
# Separate permitted cohort outcomes, not part of fit or score inputs.
np.save(P/'validation_labels.npy',np.load(H/'validation_labels.npy').astype(np.int8))
source=(B/'cdx_click_autoencoder.py').read_text();tree=ast.parse(source)
names={'CatalogAE','batch','reconstruction','checks','score'}
parts=[('@torch.no_grad()\n' if n.name=='score' else '')+ast.get_source_segment(source,n)
    for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
core='\n\n'.join(parts).replace('conditional=torch.softmax(logits,1).numpy()','conditional=torch.softmax(logits,1).cpu().numpy()')
source2=(B/'cdx_policy_hpf.py').read_text();tree2=ast.parse(source2)
compare=next(ast.get_source_segment(source2,n) for n in tree2.body if isinstance(n,ast.FunctionDef) and n.name=='compare')
header='''from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import json,hashlib,time,math
SEED=8711;EPOCHS=20;BATCH=256;LR=.001;MAX_KL=.2;WARMUP=5;REPLACE=.05
'''
body='''
def run(folder,output='/kaggle/working/cdx_catalog_ae_results'):
    folder=Path(folder);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((folder/'upload_manifest.json').read_text())
    for name,digest in manifest['hashes'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    assert torch.cuda.is_available(),'GPU required: please enable the Kaggle accelerator; CPU training will not be started.'
    device='cuda';torch.set_default_device(device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    print('KAGGLE_ACCELERATOR',torch.cuda.get_device_name(0),flush=True)
    x=sp.load_npz(folder/'clicks.npz');active=np.load(folder/'active_users.npy');initial=dict(np.load(folder/'initial_model.npz'))
    features=np.load(folder/'prediction_features.npz');info=json.loads((folder/'data_info.json').read_text())
    assert info['max_source_timestamp']<info['cutoff'] and checks(initial,x)
    torch.manual_seed(SEED);pilot=CatalogAE(initial,x,True);opt=torch.optim.Adam(pilot.parameters(),lr=LR)
    ir=torch.Generator(device=device).manual_seed(SEED+1);zr=torch.Generator(device=device).manual_seed(SEED+2);timings=[]
    for j in range(8):
        data=batch(x,active[j*BATCH:(j+1)*BATCH]);torch.cuda.synchronize();start=time.monotonic()
        opt.zero_grad(set_to_none=True);logits,kl=pilot(*data[:3],ir,zr)
        loss=reconstruction(logits,data[0],data[3],data[4])+.02*kl;loss.backward();opt.step();torch.cuda.synchronize()
        timings.append(time.monotonic()-start)
    estimate=float(np.median(timings[2:])*math.ceil(len(active)/BATCH)*EPOCHS*2*1.5+45)
    print('GPU_PILOT',json.dumps(dict(step_seconds=timings,estimated_seconds=estimate)),flush=True)
    assert estimate<1800,'Pilot exceeds30minutes; stop for review before training.'
    del pilot,opt,logits,kl,loss;torch.cuda.empty_cache()
    predictions={};logs={};tic=time.monotonic()
    for name,variational in [('dae',False),('vae',True)]:
        torch.manual_seed(SEED);model=CatalogAE(initial,x,variational);opt=torch.optim.Adam(model.parameters(),lr=LR)
        rng=np.random.default_rng(SEED);ir=torch.Generator(device=device).manual_seed(SEED+1);zr=torch.Generator(device=device).manual_seed(SEED+2)
        logs[name]=[]
        for epoch in range(EPOCHS):
            st=time.monotonic();model.train();order=rng.permutation(active);losses=[];kls=[]
            for off in range(0,len(order),BATCH):
                data=batch(x,order[off:off+BATCH]);opt.zero_grad(set_to_none=True)
                logits,kl=model(*data[:3],ir,zr);nll=reconstruction(logits,data[0],data[3],data[4])
                beta=MAX_KL*min(1,(epoch+off/len(order))/WARMUP) if variational else 0
                loss=nll+beta*kl;assert torch.isfinite(loss);loss.backward();opt.step()
                losses.append(float(nll.detach()));kls.append(float(kl.detach()))
            record=dict(arm=name,epoch=epoch+1,nll=float(np.mean(losses)),kl=float(np.mean(kls)),seconds=time.monotonic()-st)
            logs[name].append(record);print(json.dumps(record),flush=True)
        predictions[name]=score(model,features['history'],features['items'],features['point'],initial['bi'])
        torch.save(model.state_dict(),out/f'{name}.pt')
        del model,opt;torch.cuda.empty_cache()
    torch.set_default_device('cpu');base=features['control'];candidates={}
    for name,q in predictions.items():
        assert np.isfinite(q).all() and np.allclose(q.sum(1),features['point'].sum(1),atol=2e-7)
        candidates[name]=base+REPLACE*(rankdata(q.ravel())/(q.size+1)-features['old_rank']).reshape(-1,10)
    np.savez_compressed(out/'frozen_predictions.npz',control=base,dae=candidates['dae'],vae=candidates['vae'],events=features['events'])
    # Both models/predictions saved before accessing any permitted validation outcomes.
    sample=features['sample'];y=np.load(folder/'validation_labels.npy')[sample];w=features['sample_weight'];ar=features['arrival'][sample];hg=features['history_group'][sample]
    masks=[('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',ar==k) for k in range(4)]+[('arrival3plus',ar>=2)]+[(f'history{k}',hg==k) for k in range(3)]
    reports={};passed={};full={}
    with threadpool_limits(limits=1):
        for arm in ['dae','vae']:
            reports[arm]=[]
            for label,mask in masks:
                z=compare(y,base[sample],candidates[arm][sample],w,mask);z['group']=label;reports[arm].append(z)
                print('SCREEN',arm,json.dumps(z),flush=True)
            passed[arm]=all(z['gain']>=.0002 for z in reports[arm][:5]);full[arm]=[]
            if passed[arm]:
                fy=np.load(folder/'validation_labels.npy')
                for label,mask in [('overall',np.ones(len(fy),bool))]+[(f'arrival{k+1}',features['arrival']==k) for k in range(4)]:
                    z=compare(fy,base,candidates[arm],features['weight'],mask);z['group']=label;full[arm].append(z)
        mechanism=compare(y,candidates['dae'][sample],candidates['vae'][sample],w,np.ones(len(y),bool))
    result=dict(status='complete',device=torch.cuda.get_device_name(0),gpu_pilot_estimated_seconds=estimate,
        screen=reports,screen_passed=passed,full_confirmation=full,
        confirmation_passed={k:bool(v) and all(z['gain']>=.0005 for z in v) for k,v in full.items()},
        vae_vs_dae=mechanism,training_logs=logs,seconds=time.monotonic()-tic,source_manifest=manifest,
        holdout_present=False,evaluation_present=False,submission_created=False,validation_used_for_fitting=False,
        scope='Simplified exposure-enhanced causal horizon control; NOT fullV53/nxa_h1 or official score')
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print('CATALOG_AE_COMPLETE',json.dumps(result),flush=True);return result
'''
runner=header+core+'\n\n'+compare+'\n\n'+body
compile(runner,'cdx_kaggle_catalog_ae.py','exec');(P/'cdx_kaggle_catalog_ae.py').write_text(runner,encoding='utf-8')
manifest=dict(purpose='User expressly requested current test on Kaggle2026-10-30; fresh private experiment',
    model_recipe='Same preregistered20epoch DAE/VAE; only device changesCPUtoCUDA, not hyperparameters',
    source_code='Own project code, no repository implementation copied',no_holdout=True,no_evaluation=True,no_submission=True,
    local_run='Stopped by user request after12DAEepochs; no validation scored; both arms restart on Kaggle',
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file() and p.name!='upload_manifest.json'})
(P/'upload_manifest.json').write_text(json.dumps(manifest,indent=2))
zpath=R/'cdx_catalog_ae_kaggle.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
    for p in P.iterdir():z.write(p,p.name)
cell="""# Codex causal catalog autoencoder: matched DAE/VAE. GPU only; no holdout/submissions.
from pathlib import Path
import importlib.util,zipfile
roots=list(Path('/kaggle/input').rglob('cdx_kaggle_catalog_ae.py'))
if not roots:
    archives=list(Path('/kaggle/input').rglob('cdx_catalog_ae_kaggle.zip'))
    assert len(archives)==1,'Attach the private TechArena Codex Catalog AE Inputs dataset.'
    target=Path('/kaggle/working/cdx_catalog_ae_inputs');target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:z.extractall(target)
    roots=[target/'cdx_kaggle_catalog_ae.py']
assert len(roots)==1
runner=roots[0];spec=importlib.util.spec_from_file_location('cdx_catalog_ae',runner)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.run(runner.parent)
"""
(R/'kaggle_cell.txt').write_text(cell)
nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Causal catalog AE matched test\nFrozen20%user screen; exact catalog likelihood; same original event click mass. No holdout or submission.\n']),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(keepends=True))],metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=4)
(R/'cdx_catalog_ae_kaggle.ipynb').write_text(json.dumps(nb,indent=2))
protected=json.loads((H/'protected_before.json').read_text());assert all(m.sha(B/k)==v['sha256'] for k,v in protected.items())
print(json.dumps(dict(zip=str(zpath),megabytes=zpath.stat().st_size/1e6,files=[p.name for p in P.iterdir()],protected_unchanged=len(protected)),indent=2))
