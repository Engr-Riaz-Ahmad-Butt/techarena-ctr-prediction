"""Freeze first-post-cutoff cohorts and isolate the causal V53 rebuild inputs."""
import ast,hashlib,json,os,time,shutil
from pathlib import Path
import numpy as np
from config import ARTIFACT_DIR as A

ROOT=Path(__file__).resolve().parent
OUT=A/'v54_horizon'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def rake(arrival,history):
    targeta=np.array([67,17,7,4],float)/95;targeth=np.array([.0126,.1637,.8237]);w=np.ones(len(arrival))
    for iteration in range(10000):
        for g,target in ((arrival,targeta),(history,targeth)):
            share=np.bincount(g,weights=w,minlength=len(target))/w.sum()
            if np.any((share==0)&(target>0)):raise ValueError('Unsupported target margin')
            w*=np.divide(target,share)[g]
        error=max(np.max(abs(np.bincount(arrival,weights=w,minlength=4)/w.sum()-targeta)),np.max(abs(np.bincount(history,weights=w,minlength=3)/w.sum()-targeth)))
        if error<1e-12:break
    else:raise RuntimeError('Raking failed')
    w/=w.mean();return w,dict(iterations=iteration+1,max_margin_error=float(error),minimum=float(w.min()),maximum=float(w.max()),effective_events=float(w.sum()**2/(w@w)))
def main():
    tic=time.monotonic();OUT.mkdir(exist_ok=True)
    if (OUT/'cohort_manifest.json').exists():raise FileExistsError('Frozen cohort already exists; inspect and reuse it')
    t=np.load(A/'train.npy',mmap_mode='r');h=np.load(A/'train_hist.npy',mmap_mode='r');ht=np.load(A/'train_times.npy',mmap_mode='r');head=t[::10];start=int(head[:,2].min());cut=start+3*86400
    future=np.flatnonzero(head[:,2]>=cut);_,ix=np.unique(head[future,0],return_index=True);first=future[np.sort(ix)]
    allowed6=set(np.load(A/'selection_day6.npz')['events'][::10].tolist())
    excluded=np.array([(head[ev,2]>=start+6*86400 and int(ev) not in allowed6) for ev in first]);events=first[~excluded]
    assert len(events)==49420 and len(np.unique(head[events,0]))==len(events)
    # First events determined from metadata before any validation label is read.
    day=((head[events,2]-cut)//86400).astype(int);length=(h[events]>=0).sum(1);groups=np.select([length==0,length<=5],[0,1],default=2)
    assert np.all((ht[events]<head[events,2,None])|(h[events]<0))
    sample=(head[events,0].astype(np.uint64)*np.uint64(2654435761)%100)<20
    assert sample.sum()==9878
    w,meta=rake(day,groups);sw,smeta=rake(day[sample],groups[sample])
    # Model-scoring data contain deliberately unavailable labels.
    rowix=(events[:,None]*10+np.arange(10)).ravel();scoring=np.asarray(t[rowix]).copy();labels=scoring[:,5].astype(np.int8).reshape(-1,10);scoring[:,5]=-1
    assert np.all(labels.sum(1)<=1)
    np.save(OUT/'score_rows.npy',scoring);np.save(OUT/'validation_labels.npy',labels)
    np.savez_compressed(OUT/'cohort.npz',events=events,arrival=day,history_length=length,history_group=groups,sample=sample,event_weight=w,sample_event_weight=sw)
    prior=np.asarray(t[t[:,2]<cut]);assert prior[:,2].max()<cut
    np.save(OUT/'train.npy',prior)
    for src,dest in [('train_hist.npy','train_hist.npy'),('train_times.npy','train_times.npy'),('train_hist.npy','evaluation_hist.npy'),('train_times.npy','evaluation_times.npy'),('ids.json','ids.json')]:
        target=OUT/dest
        if not target.exists():
            # History arrays stay read-only; hard links avoid copying source data.
            shutil.copyfile(A/src,target)
    np.save(OUT/'evaluation.npy',scoring)
    stat=[]
    for d in range(4):
        m=day==d;uu=head[events[m],0];last=np.zeros(int(t[:,0].max())+1,dtype=np.int32);np.maximum.at(last,prior[:,0],prior[:,2]);seen=last[uu]>0;gap=(head[events[m],2][seen]-last[uu[seen]])/86400
        stat.append(dict(arrival_day=d+1,events=int(m.sum()),sample_events=int((m&sample).sum()),click_events=int(labels[m].sum()),click_event_share=float(labels[m].sum()/m.sum()),history_counts=np.bincount(groups[m],minlength=3).tolist(),no_prior_impression_users=int((~seen).sum()),median_activity_gap_at_event_days=float(np.median(gap)),effective_events=float(w[m].sum()**2/(w[m]@w[m]))))
    protected=list(A.glob('v53*'))+list(ROOT.glob('submission*.csv'))+list((ROOT/'baseline_backup').rglob('*'))
    protected={str(p.relative_to(ROOT)):dict(size=p.stat().st_size,sha256=sha(p)) for p in protected if p.is_file()}
    (OUT/'protected_before.json').write_text(json.dumps(protected,indent=2))
    # Record the actual import/feature dependencies, not just the version name.
    names=['v53_causal_propensity','v49_expert_calibration','v48_history_calibration','v39_regularized_als32','v36_recent_context','v23_outside','v21_fast_seeds','v21_fast_din','v21_din','imp19_finalize','imp19_tree','imp17_tree','imp15_als','imp11_bilinear','imp10_leafwise','imp6_warmcontext','imp5_aligned','imp4_decompose','imp3_warmfeat','core_v6','core_v5','core_v4','core_v2','core']
    inventory={}
    for name in names:
        p=ROOT/(name+'.py');tree=ast.parse(p.read_text());inventory[name]=dict(sha256=sha(p),imports=[ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))],functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))])
    (OUT/'dependency_inventory.json').write_text(json.dumps(inventory,indent=2))
    manifest=dict(status='cohort_frozen_models_not_yet_fitted',cutoff=cut,training_max_timestamp=int(prior[:,2].max()),validation_rows=len(scoring),excluded_reserved_first_events=int(excluded.sum()),target_arrival=(np.array([67,17,7,4])/95).tolist(),target_history=[.0126,.1637,.8237],full_raking=meta,sample_raking=smeta,arrival_summary=stat,click_event_share=float(labels.sum()/len(labels)),holdout_opened=False,first_event_substitution=False,score_label_sentinel=-1,unsupported_evaluation_days=[5,6,7],elapsed_seconds=time.monotonic()-tic,source_sha256=sha(Path(__file__)))
    (OUT/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
    s=json.loads((ROOT/'agent_state.json').read_text());s['status']='horizon_baseline_rebuild_in_progress';s['next_action']='Rebuild causal V53 dependencies before testing warm-user ALS refresh';s['horizon']={'directory':str(OUT),'cohort_frozen':True,'baseline_fitted':False,'idea1_tested':False};(ROOT/'agent_state.json').write_text(json.dumps(s,indent=2))
    with (ROOT/'results_log.md').open('a',encoding='utf-8') as f:f.write(f'\n## 2026-10-29 — Horizon baseline cohort frozen\n\n- Cutoff {cut}; {len(events)} first-user events, {int(sample.sum())} sampled; 735 reserved-first events excluded without substitution. Joint raking margins verified. Global click-containing share {manifest["click_event_share"]:.6%}.\n- Baseline models not yet refitted; no horizon AUC or Idea1 gain claimed. Sources/protected hashes and dependency inventory isolated under artifacts/v54_horizon. No holdout opened; no submission.\n')
if __name__=='__main__':main()
