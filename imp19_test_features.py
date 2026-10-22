"""Independent aligned ALS interactions and unchanged-prefix audit; no fit."""
import ast
import hashlib
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
SOURCES=['imp19_als_aligned.py','imp19_tree.py','imp17_als_features.py','imp15_als.py','imp15_als_core.py']
HASHES={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def load_functions(filename,names,ns):
    tree=ast.parse((ROOT/filename).read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(tree,filename,'exec'),ns)


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


if __name__=='__main__':
    train=np.load(ARTIFACT_DIR/'train.npy',mmap_mode='r')
    hist=np.load(ARTIFACT_DIR/'train_hist.npy',mmap_mode='r')
    times=np.load(ARTIFACT_DIR/'train_times.npy',mmap_mode='r')
    start=int(train[:,2].min())
    with np.load(ARTIFACT_DIR/'imp15_model_day6.npz') as f:model={k:f[k] for k in f.files}
    U,V,bu,bi,exposures=[model[k] for k in ['U','V','bu','bi','exposures']]
    g=float(model['g']);ni=len(V)
    reference=np.load(ARTIFACT_DIR/'imp17_tree_day6.npz')
    events=reference['events'].reshape(-1,10)[:,0]
    all_e=train[events*10]
    picks=np.linspace(0,len(events)-1,98,dtype=int).tolist()
    for candidates in [np.flatnonzero(exposures[all_e[:,0]]==0),np.flatnonzero((hist[events]>=0).sum(1)==0)]:
        if len(candidates):picks.append(int(candidates[0]))
    picks=np.unique(picks)
    if len(picks)<100:picks=np.unique(np.r_[picks,np.setdiff1d(np.arange(len(events)),picks)[:100-len(picks)]])
    picks=picks[:100]
    eids=events[picks]
    rows=train[(eids[:,None]*10+np.arange(10)).ravel()]
    e=rows[::10]
    assert np.array_equal(rows[:,6].reshape(-1,10),np.broadcast_to(eids[:,None],(100,10)))
    assert ((times[eids]<e[:,2,None])|(hist[eids]<0)).all()
    calls=[]
    def ensure(cut):
        calls.append(cut)
        assert cut in [2,3,4,5,6]
        name=f'imp15_model_day{cut}.npz' if cut in [5,6] else f'imp17_als_cut{cut}.npz'
        path=ARTIFACT_DIR/name
        with np.load(path) as f:
            assert int(f['cutoff'])==start+cut*86400 and int(f['iteration'])==6
        return path
    with tempfile.TemporaryDirectory(prefix='imp19_test_',dir=ARTIFACT_DIR) as directory:
        temporary=Path(directory)
        assert temporary.resolve().is_relative_to(ARTIFACT_DIR.resolve())
        class RoutedRoot:
            def __truediv__(self,name):
                return (temporary/name) if str(name).startswith('imp19_rotation_') else (ARTIFACT_DIR/name)
        ns=dict(np=np,ROOT=RoutedRoot(),FEAT=temporary,ni=ni,hist=hist,
                solve_rows=solve_rows,coo_matrix=coo_matrix,ensure_model=ensure,
                cutoff_of=lambda cut:start+cut*86400)
        load_functions('imp15_als.py',{'paired_matrices','augmented','score'},ns)
        load_functions('imp19_als_aligned.py',{'unit','alignment','interactions'},ns)
        with threadpool_limits(limits=1):
            actual=ns['interactions']('test_val6',rows,hist,6)
            assert actual.shape==(1000,32) and np.isfinite(actual).all()
            with np.load(temporary/'imp19_rotation_cut6.npz') as f:
                rotation=f['rotation'];singular=f['singular_values']
                assert int(f['anchor_cutoff'])==start+2*86400<=int(f['cutoff'])==start+6*86400
            with np.load(ensure(2)) as f:anchor=unit(f['V'].astype(float))
            cross=unit(V).T@anchor
            left,s,right=np.linalg.svd(cross,full_matrices=False)
            rotation_error=float(np.max(np.abs(rotation-left@right)))
            orth_error=float(np.max(np.abs(rotation.T@rotation-np.eye(16))))
            objective_error=float(abs(np.trace(rotation.T@cross)-s.sum()))
            assert rotation_error<1e-12 and orth_error<1e-10 and objective_error<1e-8
            assert np.allclose(singular,s,rtol=0,atol=1e-10)
            design=np.column_stack([np.ones(ni),V]);gram=design.T@design
            rhs=-design.T@(g+bi)
            expected=[];dot_errors=[];cosine_errors=[];empty=0;cold=0
            for event,row in enumerate(e):
                h=hist[row[6]];valid=h>=0;h=h[valid];t=times[row[6]][valid]
                assert (t<row[2]).all()
                user=int(row[0])
                if exposures[user]:u=U[user]
                else:
                    cold+=1
                    unique=sorted(set(zip(h.tolist(),t.tolist())))
                    ids=np.array([item for item,stamp in unique],int)
                    x=design[ids];confidence=9*len(ids)/ni
                    value=np.linalg.solve(confidence*gram+x.T@x+np.diag(np.r_[20.,np.ones(16)]),
                                          confidence*rhs+x.T@(1-g-bi[ids]))
                    u=value[1:]
                profile=V[h].mean(0) if len(h) else np.zeros(16)
                if not len(h):empty+=1
                for item in rows[event*10:event*10+10,1]:
                    v=V[item]
                    rotated_u,rotated_h,rotated_v=u@rotation,profile@rotation,v@rotation
                    dot_errors.extend([abs(u@v-rotated_u@rotated_v),abs(profile@v-rotated_h@rotated_v)])
                    products=np.r_[unit(rotated_u)*unit(rotated_v),unit(rotated_h)*unit(rotated_v)]
                    expected.append(products)
                    cosine_errors.extend([abs(products[:16].sum()-unit(u)@unit(v)),
                                          abs(products[16:].sum()-unit(profile)@unit(v))])
            expected=np.array(expected,dtype=np.float32)
            product_error=float(np.max(np.abs(expected-actual)))
            assert product_error<1e-6 and max(dot_errors)<1e-10 and max(cosine_errors)<1e-10
            no_history=(hist[eids]>=0).sum(1)==0
            assert np.all(actual.reshape(-1,10,32)[no_history,:,16:]==0)
            pos=(picks[:,None]*10+np.arange(10)).ravel()
            files=['stack_val6_c6.npy','imp3_wf_val6.npy','imp4_decompose_val6.npy',
                   'imp5_aligned_val6.npy','imp6_context_val6.npy','imp17_als_val6.npy']
            old=np.column_stack([np.load(ARTIFACT_DIR/'features'/name,mmap_mode='r')[pos] for name in files])
            assert old.shape==(1000,322)
            ns['_base_combined']=lambda name,rows,histories,cut:old[:,226:]
            load_functions('imp19_tree.py',{'combined'},ns)
            full=np.column_stack([old[:,:226],ns['combined']('test_val6',rows,hist,6)])
            assert full.shape==(1000,354) and np.array_equal(full[:,:322],old)
            count_before=len(calls)
            rejected_cut=False
            try:ns['alignment'](1)
            except AssertionError:rejected_cut=True
            assert rejected_cut and len(calls)==count_before
            rejected_rows=False
            bad=rows.copy();bad[:10,2]=start+6*86400-1
            try:ns['interactions']('test_val6',bad,hist,6)
            except AssertionError:rejected_rows=True
            assert rejected_rows and len(calls)==count_before
    assert HASHES=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    result=dict(status='PASS',source_sha256=HASHES,events=100,rows=1000,
                manual_product_max_error=product_error,rotation_replay_max_error=rotation_error,
                orthogonality_max_error=orth_error,procrustes_objective_error=objective_error,
                dot_invariance_max_error=max(dot_errors),cosine_sum_max_error=max(cosine_errors),
                cold_dense_profiles_tested=cold,empty_history_profiles_tested=empty,
                empty_history_sixteen_zeros=True,first322_columns_unchanged=True,
                negative_cut_rejected_before_load=True,earlier_rows_rejected_before_cache=True,
                source_cutoffs_and_strict_past_history=True,
                runner_review='Accepted imp17 baseline, unchanged SETTINGS/rolling cohort/ordinary selection/frozen confirmation gates.')
    (ARTIFACT_DIR/'imp19_portable_test_features_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
