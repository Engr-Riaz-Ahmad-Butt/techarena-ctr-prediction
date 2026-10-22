"""No-fit regression and independent scalar ALS feature replay."""
import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
from config import ARTIFACT_DIR
import numpy as np
from scipy.sparse import coo_matrix
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from imp15_als_core import solve_rows
set_num_threads(4)


def functions(filename,names,ns):
    tree=ast.parse((ROOT/filename).read_text())
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    exec(compile(tree,filename,'exec'),ns)


if __name__=='__main__':
    train=np.load(ARTIFACT_DIR/'train.npy',mmap_mode='r')
    hist=np.load(ARTIFACT_DIR/'train_hist.npy',mmap_mode='r')
    times=np.load(ARTIFACT_DIR/'train_times.npy',mmap_mode='r')
    start=int(train[:,2].min())
    ns=dict(np=np,ROOT=ARTIFACT_DIR,train=train,hist=hist,solve_rows=solve_rows,coo_matrix=coo_matrix,
            cutoff_of=lambda cut:start+cut*86400)
    functions('imp15_als.py',{'paired_matrices','augmented','score'},ns)
    regressions=[]
    with threadpool_limits(limits=1):
        for day in [6,5]:
            reference=np.load(ARTIFACT_DIR/f'imp15_als_day{day}.npz')
            with np.load(ARTIFACT_DIR/f'imp15_model_day{day}.npz') as f:model={k:f[k] for k in f.files}
            assert int(model['cutoff'])==start+day*86400 and int(model['iteration'])==6
            U,V,bu,bi,exposures=[model[k] for k in ['U','V','bu','bi','exposures']]
            g=float(model['g']);ns['ni']=len(V)
            events=reference['events'].reshape(-1,10)[:,0]
            rows=train[(events[:,None]*10+np.arange(10)).ravel()]
            default=ns['score'](rows,U,V,bu,bi,g,exposures)
            error=float(np.max(np.abs(default-reference['raw'])))
            assert error==0,error
            profiles,bias=ns['score'](rows,U,V,bu,bi,g,exposures,return_profile=True,histories=hist,history_times=times)
            reconstructed=(g+bias[:,None]+bi[rows[:,1].reshape(-1,10)]+
                           np.einsum('ed,ecd->ec',profiles,V[rows[:,1].reshape(-1,10)])).ravel()
            profile_error=float(np.max(np.abs(reconstructed-reference['raw'])))
            assert profile_error==0
            regressions.append(dict(day=day,rows=len(rows),default_raw_max_error=error,
                                    optional_profile_raw_max_error=profile_error))
        # Independent feature replay on 100 complete day6 events, including edge cases.
        day=6
        reference=np.load(ARTIFACT_DIR/'imp15_als_day6.npz')
        with np.load(ARTIFACT_DIR/'imp15_model_day6.npz') as f:model={k:f[k] for k in f.files}
        U,V,bu,bi,exposures=[model[k] for k in ['U','V','bu','bi','exposures']]
        g=float(model['g']);ns['ni']=len(V)
        events=reference['events'].reshape(-1,10)[:,0]
        event_rows=train[events*10]
        picks=np.linspace(0,len(events)-1,98,dtype=int).tolist()
        cold=np.flatnonzero(exposures[event_rows[:,0]]==0)
        empty=np.flatnonzero((hist[events]>=0).sum(1)==0)
        for candidates in [cold,empty]:
            if len(candidates):picks.append(int(candidates[0]))
        picks=np.unique(picks)
        if len(picks)<100:
            picks=np.unique(np.r_[picks,np.setdiff1d(np.arange(len(events)),picks)[:100-len(picks)]])
        picks=picks[:100]
        ids=events[picks]
        selected=train[(ids[:,None]*10+np.arange(10)).ravel()]
        e=selected[::10]
        assert ((e[:,2]>=start+6*86400)&(e[:,2]<start+7*86400)).all()
        assert ((times[ids]<e[:,2,None])|(hist[ids]<0)).all()
        with tempfile.TemporaryDirectory(prefix='imp17_test_',dir=ARTIFACT_DIR) as directory:
            feature_dir=Path(directory)
            assert feature_dir.resolve().is_relative_to(ARTIFACT_DIR.resolve())
            ns['FEAT']=feature_dir
            ns['ensure_model']=lambda cut:ARTIFACT_DIR/f'imp15_model_day{cut}.npz'
            functions('imp17_als_features.py',{'als_features'},ns)
            actual=ns['als_features']('test_scalar_val6',selected,hist,6)
            assert actual.shape==(1000,32) and np.isfinite(actual).all()
            item_exposure=np.bincount(train[train[:,2]<start+6*86400,1],minlength=len(V))
            design=np.column_stack([np.ones(len(V)),V])
            gram=design.T@design
            rhs=-design.T@(g+bi)
            scalar=[]
            for event,row in enumerate(e):
                history=hist[row[6]]
                valid=history>=0
                history=history[valid]
                stamps=times[row[6]][valid]
                assert (stamps<row[2]).all()
                user=int(row[0]);n=len(history)
                if exposures[user]:u,bias=U[user],bu[user]
                else:
                    unique=sorted(set(zip(history.tolist(),stamps.tolist())))
                    hi=np.array([item for item,stamp in unique],dtype=int)
                    confidence=9*len(hi)/len(V)
                    x=design[hi]
                    a=confidence*gram+x.T@x+np.diag(np.r_[20.,np.ones(16)])
                    values=np.linalg.solve(a,confidence*rhs+x.T@(1-g-bi[hi]))
                    bias,u=values[0],values[1:]
                un=np.linalg.norm(u)
                profile=V[history].mean(0) if n else np.zeros(16)
                hn=np.linalg.norm(profile)
                unit_history=V[history]/np.maximum(np.linalg.norm(V[history],axis=1,keepdims=True),1e-12)
                for candidate in selected[event*10:event*10+10,1]:
                    v=V[candidate];vn=np.linalg.norm(v)
                    direct=float(u@v);hd=float(profile@v)
                    sim=unit_history@(v/max(vn,1e-12))
                    maximum=float(sim.max()) if n else 0.
                    std=float(sim.std()) if n else 0.
                    last3=float(sim[-3:].mean()) if n else 0.
                    last5=float(sim[-5:].mean()) if n else 0.
                    scalar.append([bias,bi[candidate],direct,un,vn,direct/max(un*vn,1e-12),
                        g+bias+bi[candidate]+direct,np.log1p(exposures[user]),hd,hn,
                        hd/max(hn*vn,1e-12),maximum,std,last3,last5,
                        int((history==candidate).sum()),np.log1p(n),np.log1p(item_exposure[candidate])])
            scalar=np.array(scalar)
            relative=[]
            for col in [1,2,5,6,8,10,11]:
                v=scalar[:,col].reshape(-1,10)
                relative.extend([(v-v.mean(1,keepdims=True)).ravel(),(v-v.max(1,keepdims=True)).ravel()])
            expected=np.column_stack([scalar,*relative]).astype(np.float32)
            feature_error=float(np.max(np.abs(actual-expected)))
            assert feature_error<2e-6,feature_error
            positions=(picks[:,None]*10+np.arange(10)).ravel()
            names=['stack_val6_c6.npy','imp3_wf_val6.npy','imp4_decompose_val6.npy','imp5_aligned_val6.npy','imp6_context_val6.npy']
            base=np.column_stack([np.load(ARTIFACT_DIR/'features'/name,mmap_mode='r')[positions] for name in names])
            ns['_base_combined']=lambda name,rows,histories,cut:base[:,226:]
            functions('imp17_tree.py',{'combined'},ns)
            suffix=ns['combined']('test_scalar_val6',selected,hist,6)
            full=np.column_stack([base[:,:226],suffix])
            assert full.shape==(1000,322) and np.array_equal(full[:,:290],base)
    result=dict(status='PASS',score_regressions=regressions,manual_feature_events=len(e),
                scalar32_manual_max_error=feature_error,first290_columns_unchanged=True,
                sampled_cold_events=int((exposures[e[:,0]]==0).sum()),
                sampled_empty_histories=int(((hist[ids]>=0).sum(1)==0).sum()),
                source_cutoff_chronology=True)
    (ARTIFACT_DIR/'imp17_test_features_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
