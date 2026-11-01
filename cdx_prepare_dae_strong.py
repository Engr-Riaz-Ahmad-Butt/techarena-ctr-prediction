"""Freeze the surviving catalog DAE against saved stronger daily controls.

No fits locally; no evaluation or reserved labels. New private Kaggle payload.
"""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
B=Path(__file__).resolve().parent;A=B/'artifacts';R=A/'v54_horizon/cdx_dae_strong_001';P=R/'kaggle_payload'
SEED=8711
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
 R.mkdir(exist_ok=True);P.mkdir(exist_ok=True);assert not (R/'frozen.json').exists()
 paths=[B/'cdx_prepare_dae_strong.py',B/'cdx_click_autoencoder.py',A/'v54_horizon/cdx_catalog_ae_001/postfit_audit.json']
 for day in [5,6]:paths += [A/f'selection_day{day}.npz',A/f'nxa_h1_day{day}.npz',A/f'imp3_model_warm16_s7_day{day}.npz',A/f'imp3_warm16_s7_day{day}.npz']
 plan=dict(date='2026-10-30',hypothesis='Surviving unordered catalog denoising preference signal adds value beyond frozen full daily H1/V53 and H1/V39 scores',
  scope='Direct daily stronger-baseline test of one already surviving DAE; no new seed/dimension/weight/epoch searches',
  initialization='Existing warm16 seed7 checkpoint for each scored day; trained strictly before that day; preserve the same H/V warm-start principle',
  architecture='Same16Dcatalog DAE,50%inputcorruption,20fixed epochs,Adam.001,batch256,seed8711',
  training='Deduplicated actual pre-cutoff clicked triples plus earliest supplied history per pre-cutoff user; no artificial exposure negatives',
  bridge='rank(frozenparent)+.05*(rank(DAE probability)-rank(same-day warm16 probability)), joint across complete ALLOWED daily cohort; same fixed5percentagepoint displacement as horizon',
  rank_scope='Separate dates have separate cutoff-fitted pipelines; no per-arrival or sample-only ranks. Validation diagnostic only; stateless deployment CDF and calibrated output not established',
  primary='H1 with V53 parent; primary supplied3bin mix, historical5bin scores also mandatory guard',
  secondary='H1 with V39 parent plus unheaded V53; identical recipe, no model selection on secondary outputs',
  gate='Screen+.0002 BOTH day5/day6 on primary under BOTH3bin/5bin weights; only survivor full+.0005 BOTH days/BOTHweightings',
  sample='Same frozen user hash(u*2654435761%100)<20;3bin weights fitted without labels separately for sample/full',
  baseline_caveat='Inherited day5H1 is2fold user-cross-fitted on day5; day6H1 fitted on earlier day5. No new head fitting. This is not a new fully causal reproduction of V53',
  source='Own surviving DAE implementation; no copied repository code or external data',no_holdout=True,no_evaluation=True,no_submissions=True,
  input_hashes={str(p.relative_to(B)):sha(p) for p in paths})
 (R/'frozen.json').open('x').write(json.dumps(plan,indent=2));shutil.copyfile(R/'frozen.json',P/'frozen.json')
 t=np.load(A/'train.npy',mmap_mode='r');h=np.load(A/'train_hist.npy',mmap_mode='r');ht=np.load(A/'train_times.npy',mmap_mode='r');head=t[::10]
 assert np.all(np.diff(head[:,2])>=0);start=int(head[0,2]);metadata={}
 for day in [5,6]:
  cutoff=start+day*86400;n=int(np.searchsorted(head[:,2],cutoff));past=np.asarray(t[:n*10]);eh=np.asarray(head[:n])
  _,first=np.unique(eh[:,0],return_index=True);first=eh[first];hist=np.asarray(h[first[:,6]]);stamp=np.asarray(ht[first[:,6]]);valid=hist>=0
  assert np.all((stamp<first[:,2,None])|~valid)
  clicks=past[past[:,5]==1];u=np.r_[clicks[:,0],np.broadcast_to(first[:,0,None],hist.shape)[valid]];ii=np.r_[clicks[:,1],hist[valid]];ts=np.r_[clicks[:,2],stamp[valid]]
  order=np.lexsort((ts,ii,u));u,ii,ts=u[order],ii[order],ts[order]
  keep=np.r_[True,(u[1:]!=u[:-1])|(ii[1:]!=ii[:-1])|(ts[1:]!=ts[:-1])];assert ts.max()<cutoff
  init=A/f'imp3_model_warm16_s7_day{day}.npz';initial=dict(np.load(init));shutil.copyfile(init,P/f'initial_day{day}.npz')
  x=sp.csr_matrix((np.ones(int(keep.sum()),np.float32),(u[keep],ii[keep])),shape=(len(initial['U']),50000))
  sp.save_npz(P/f'clicks_day{day}.npz',x);active=np.flatnonzero(np.asarray(x.sum(1)).ravel()>0);np.save(P/f'active_day{day}.npy',active)
  allowed=np.load(A/f'selection_day{day}.npz');old=np.load(A/f'imp3_warm16_s7_day{day}.npz');saved=np.load(A/f'nxa_h1_day{day}.npz')
  # Never read old/saved['y']; only the explicit allowed selection file owns outcomes.
  index=allowed['full_indices'] if day==6 else np.arange(len(allowed['events']))
  ev=allowed['events'];assert np.array_equal(old['events'][index],ev) and np.array_equal(saved['events'][index],ev)
  assert np.all(ev.reshape(-1,10)==ev[::10,None]);events=ev[::10];rows=np.asarray(t[events*10]);users=rows[:,0]
  assert len(np.unique(users))==len(events) and rows[:,2].min()>=cutoff and rows[:,2].max()<cutoff+86400
  history=np.asarray(h[events]);stamp=np.asarray(ht[events]);assert np.all((stamp<rows[:,2,None])|(history<0))
  items=np.asarray(t[events[:,None]*10+np.arange(10),1]);point=old['raw'][index].reshape(-1,10)
  assert np.isfinite(point).all() and np.all(point>=0) and np.all(point.sum(1)<=1+1e-6)
  sample=(users.astype(np.uint64)*np.uint64(2654435761)%100)<20
  length=(history>=0).sum(1);group=np.where(length==0,0,np.where(length<=5,1,2))
  def weights(g):return (np.array([.0126,.1637,.8237])/(np.bincount(g,minlength=3)/len(g)))[g]
  w3=weights(group);sw3=weights(group[sample]);w5=allowed['weights'].reshape(-1,10)
  assert np.all(w5==w5[:,0,None]);w5=w5[:,0]
  def rank(z):return (rankdata(z.ravel())/(z.size+1)).reshape(-1,10)
  np.savez_compressed(P/f'features_day{day}.npz',history=history,items=items,point=point,old_rank=rank(point),
   control=rank(saved['pred'][index]),export_parent=rank(saved['pred_v39'][index]),v53=rank(allowed['baseline']),
   events=events,sample=sample,w3=w3,sw3=sw3,w5=w5,history_group=group)
  y=allowed['y'].reshape(-1,10);assert np.all(y.sum(1)<=1);np.save(P/f'labels_day{day}.npy',y.astype(np.int8))
  metadata[str(day)]=dict(cutoff=cutoff,max_fit_timestamp=int(ts.max()),training_events=n,active_users=len(active),deduplicated_click_occurrences=int(keep.sum()),validation_events=len(events),screen_events=int(sample.sum()),history_group_counts=np.bincount(group,minlength=3).tolist(),screen_history_group_counts=np.bincount(group[sample],minlength=3).tolist())
  print('PREPARED',day,json.dumps(metadata[str(day)]),flush=True)
 (P/'data_info.json').write_text(json.dumps(metadata,indent=2))
 s=(B/'cdx_click_autoencoder.py').read_text();tree=ast.parse(s);names={'CatalogAE','batch','reconstruction','checks','score'}
 core='\n\n'.join(('@torch.no_grad()\n' if n.name=='score' else '')+ast.get_source_segment(s,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
 core=core.replace('conditional=torch.softmax(logits,1).numpy()','conditional=torch.softmax(logits,1).cpu().numpy()')
 s=(B/'cdx_policy_hpf.py').read_text();compare=next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='compare')
 header='''from pathlib import Path
import json,time,math,hashlib
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
SEED=8711;EPOCHS=20;BATCH=256;LR=.001
'''
 body='''
def run(folder,output='/kaggle/working/cdx_dae_strong_results'):
 folder=Path(folder);out=Path(output);out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((folder/'upload_manifest.json').read_text())
 for name,digest in manifest['hashes'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
 assert torch.cuda.is_available(),'GPU required; do not launch a CPU training fallback'
 torch.set_default_device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 print('KAGGLE_ACCELERATOR',torch.cuda.get_device_name(0),flush=True);tic=time.monotonic();predictions={};logs={};info=json.loads((folder/'data_info.json').read_text())
 for day in [5,6]:
  x=sp.load_npz(folder/f'clicks_day{day}.npz');active=np.load(folder/f'active_day{day}.npy');initial=dict(np.load(folder/f'initial_day{day}.npz'));f=np.load(folder/f'features_day{day}.npz')
  assert info[str(day)]['max_fit_timestamp']<info[str(day)]['cutoff'] and checks(initial,x)
  torch.manual_seed(SEED);model=CatalogAE(initial,x,False);opt=torch.optim.Adam(model.parameters(),lr=LR)
  rng=np.random.default_rng(SEED);ir=torch.Generator(device='cuda').manual_seed(SEED+1);zr=torch.Generator(device='cuda').manual_seed(SEED+2)
  logs[str(day)]=[]
  for epoch in range(EPOCHS):
   st=time.monotonic();model.train();order=rng.permutation(active);losses=[]
   for off in range(0,len(order),BATCH):
    data=batch(x,order[off:off+BATCH]);opt.zero_grad(set_to_none=True);logits,_=model(*data[:3],ir,zr)
    loss=reconstruction(logits,data[0],data[3],data[4]);assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
   rec=dict(day=day,epoch=epoch+1,nll=float(np.mean(losses)),seconds=time.monotonic()-st);logs[str(day)].append(rec);print('FIT',json.dumps(rec),flush=True)
   if epoch==0:
    estimate=rec['seconds']*EPOCHS*2*1.5+60;print('FIRST_EPOCH_RUNTIME_ESTIMATE_SECONDS',estimate,flush=True)
    assert estimate<1800,'Runtime exceeds30minutes; stop before remaining epochs'
  q=score(model,f['history'],f['items'],f['point'],initial['bi'])
  assert np.isfinite(q).all() and np.allclose(q.sum(1),f['point'].sum(1),atol=2e-7)
  delta=.05*((rankdata(q.ravel())/(q.size+1)).reshape(-1,10)-f['old_rank'])
  predictions[day]={k:f[k]+delta for k in ['control','export_parent','v53']}
  torch.save(model.state_dict(),out/f'dae_day{day}.pt')
  np.savez_compressed(out/f'predictions_day{day}.npz',**predictions[day],events=f['events'],delta=delta,raw_dae=q)
  del model,opt,x;torch.cuda.empty_cache()
 # Both fixed checkpoints and complete ALLOWED daily predictions precede outcome access.
 torch.set_default_device('cpu');screen=[];full=[]
 with threadpool_limits(limits=1):
  for day in [5,6]:
   f=np.load(folder/f'features_day{day}.npz');take=f['sample'];y=np.load(folder/f'labels_day{day}.npy')[take]
   for k in ['control','export_parent','v53']:
    for weight in ['3bin','5bin']:
     w=f['sw3'] if weight=='3bin' else f['w5'][take]
     z=compare(y,f[k][take],predictions[day][k][take],w,np.ones(len(y),bool));z.update(day=day,parent=k,weight=weight);screen.append(z);print('SCREEN',json.dumps(z),flush=True)
  passed=all(z['gain']>=.0002 for z in screen if z['parent']=='control')
  if passed:
   for day in [5,6]:
    f=np.load(folder/f'features_day{day}.npz');y=np.load(folder/f'labels_day{day}.npy')
    for k in ['control','export_parent','v53']:
     for weight in ['3bin','5bin']:
      w=f['w3'] if weight=='3bin' else f['w5'];z=compare(y,f[k],predictions[day][k],w,np.ones(len(y),bool));z.update(day=day,parent=k,weight=weight);full.append(z);print('CONFIRM',json.dumps(z),flush=True)
  confirmed=bool(full) and all(z['gain']>=.0005 for z in full if z['parent']=='control')
 result=dict(status='complete',device=torch.cuda.get_device_name(0),seconds=time.monotonic()-tic,screen=screen,screen_passed=passed,full_confirmation=full,confirmation_passed=confirmed,training_logs=logs,source_manifest=manifest,
  inherited_baseline_caveat='Day5H1uses user crossfit onday5; day6headfit onearlierday5. NewDAEmodels strictlyprecutoff; baseline heads not refit.',
  no_hyperparameter_search=True,labels_accessed_after_predictions_frozen=True,holdout_scored=False,evaluation_read=False,submission_created=False,
  scope='Daily selection-cohort stronger-parent incremental test, not official score or evaluation export')
 (out/'report.json').write_text(json.dumps(result,indent=2));print('DAE_STRONG_COMPLETE',json.dumps(result),flush=True);return result
'''
 runner=header+core+'\n\n'+compare+'\n\n'+body;compile(runner,'cdx_kaggle_dae_strong.py','exec');(P/'cdx_kaggle_dae_strong.py').write_text(runner)
 manifest=dict(purpose='Previously surviving DAE now tested against existing stronger day5/day6controls onprivateKaggle',source='Own code, original recipe and20%user rule',no_holdout=True,no_evaluation=True,no_submission=True,
  hashes={p.name:sha(p) for p in P.iterdir() if p.is_file() and p.name!='upload_manifest.json'})
 (P/'upload_manifest.json').write_text(json.dumps(manifest,indent=2));zpath=R/'cdx_dae_strong_kaggle.zip'
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
  for p in P.iterdir():z.write(p,p.name)
 cell="""# Surviving fixed DAE vs frozen stronger daily controls. GPU only. No holdout/evaluation/submissions.
from pathlib import Path
import importlib.util,zipfile
roots=list(Path('/kaggle/input').rglob('cdx_kaggle_dae_strong.py'))
if not roots:
    archives=list(Path('/kaggle/input').rglob('cdx_dae_strong_kaggle.zip'));assert len(archives)==1
    target=Path('/kaggle/working/cdx_dae_strong_inputs');target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:z.extractall(target)
    roots=[target/'cdx_kaggle_dae_strong.py']
assert len(roots)==1
runner=roots[0];spec=importlib.util.spec_from_file_location('cdx_dae_strong',runner)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.run(runner.parent)
"""
 nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# DAE against the stronger daily baselines\nFixed survivor recipe, strict pre-cutoff DAE fitting, existing20%user rule and both weight definitions. No holdout/evaluation/submissions.\n']),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(keepends=True))],metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=4)
 (R/'cdx_dae_strong_kaggle.ipynb').write_text(json.dumps(nb,indent=2))
 protected=json.loads((A/'v54_horizon/protected_before.json').read_text());assert all(sha(B/k)==v['sha256'] for k,v in protected.items())
 print(json.dumps(dict(bundle=str(zpath),megabytes=zpath.stat().st_size/1e6,metadata=metadata,protected_unchanged=len(protected)),indent=2))

if __name__=='__main__':prepare()
