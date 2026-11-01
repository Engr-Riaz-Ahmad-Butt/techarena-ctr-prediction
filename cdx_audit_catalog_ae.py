"""Read-only audit of the downloaded Kaggle survivor; no training or promotion."""
from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import torch
import cdx_policy_hpf as common
B,A,H=common.B,common.A,common.H;R=H/'cdx_catalog_ae_001';P=R/'kaggle_payload';O=R/'kaggle_outputs/cdx_catalog_ae_results'
tic=time.monotonic();torch.set_num_threads(2);torch.set_default_device('cpu')
spec=importlib.util.spec_from_file_location('own_kaggle_ae',P/'cdx_kaggle_catalog_ae.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.score=torch.no_grad()(mod.score)
fp=np.load(P/'prediction_features.npz');pred=np.load(O/'frozen_predictions.npz');co=np.load(H/'cohort.npz');rows=np.load(H/'score_rows.npy',mmap_mode='r')
y=np.load(H/'validation_labels.npy');report=json.loads((O/'report.json').read_text())
assert np.array_equal(pred['events'],co['events']) and np.array_equal(fp['events'],co['events'])
assert len(np.unique(rows[::10,0]))==len(co['events'])==49420 and len(rows)==494200
assert np.all(rows[:,5]==-1) and np.all(y.sum(1)<=1) and np.array_equal(y,np.load(P/'validation_labels.npy'))
assert np.array_equal(pred['control'],fp['control'])
assert all(np.isfinite(pred[k]).all() for k in ['control','dae','vae'])
assert not {'label','labels','y','target'}.intersection(fp.files)
ts=np.load(A/'train_times.npy',mmap_mode='r')[rows[::10,6]];hist=fp['history'];info=json.loads((P/'data_info.json').read_text())
assert np.all((ts<rows[::10,2,None])|(hist<0)) and info['max_source_timestamp']<info['cutoff']<=rows[::10,2].min()
for mask,w in [(np.ones(len(y),bool),co['event_weight']),(co['sample'],co['sample_event_weight'])]:
    assert np.all(w>0) and np.allclose([w[co['history_group'][mask]==j].sum()/w.sum() for j in range(3)],[.0126,.1637,.8237],atol=1e-9)
    assert np.allclose([w[co['arrival'][mask]==j].sum()/w.sum() for j in range(4)],np.array([67,17,7,4])/95,atol=1e-9)
initial=dict(np.load(P/'initial_model.npz'));x=sp.load_npz(P/'clicks.npz');n=128;sl=slice(0,n);replay={}
for arm in ['dae','vae']:
    model=mod.CatalogAE(initial,x,arm=='vae')
    model.load_state_dict(torch.load(O/f'{arm}.pt',map_location='cpu',weights_only=True));model.eval()
    def predict(input_rows,input_hist=hist[sl],ii=None,point=None):
        # No label argument is selected from the mock rows.
        return mod.score(model,input_hist,input_rows[:,1].reshape(-1,10) if ii is None else ii,
            fp['point'][sl] if point is None else point,initial['bi'])
    mock=np.array(rows[:n*10]);q=predict(mock)
    relabeled=mock.copy();relabeled[:,5]=np.random.default_rng(5823).integers(0,2,len(mock))
    assert np.array_equal(q,predict(relabeled)) and np.array_equal(q,predict(mock))
    assert np.allclose(q.sum(1),fp['point'][sl].sum(1),atol=2e-7)
    permutation=np.random.default_rng(8513).permutation(10)
    assert np.allclose(q[:,permutation],predict(mock,ii=fp['items'][sl][:,permutation],point=fp['point'][sl][:,permutation]),atol=1e-10)
    reordered=hist[sl].copy();rng=np.random.default_rng(6171)
    for v in reordered:rng.shuffle(v)
    assert np.array_equal(q,predict(mock,input_hist=reordered))
    auc=roc_auc_score(y.ravel(),pred[arm].ravel(),sample_weight=np.repeat(co['event_weight'],10))
    auc_difference=auc-report['full_confirmation'][arm][0]['candidate_auc']
    # Same1e-10 tolerance as the original independent sklearn check: weighted
    # tie aggregation and sklearn cumulative sums differ by2.7e-12 here.
    assert abs(auc_difference)<1e-10
    replay[arm]=dict(weighted_auc_reproduced=auc,auc_difference=auc_difference,label_invariant=True,deterministic=True,candidate_permutation=True,history_order_invariant=True,event_mass_preserved=True)
history=[]
for j in range(3):
    mask=co['history_group']==j;w=np.repeat(co['event_weight'][mask],10)
    row=dict(history_group=j,events=int(mask.sum()),clicked_events=int(y[mask].sum()),control_auc=roc_auc_score(y[mask].ravel(),pred['control'][mask].ravel(),sample_weight=w))
    for arm in ['dae','vae']:
        row[arm+'_auc']=roc_auc_score(y[mask].ravel(),pred[arm][mask].ravel(),sample_weight=w);row[arm+'_gain']=row[arm+'_auc']-row['control_auc']
    history.append(row)
protected=json.loads((H/'protected_before.json').read_text());assert all(common.sha(B/k)==v['sha256'] for k,v in protected.items())
out=dict(status='passed',replay=replay,full_cohort_history=history,protected_files_unchanged=len(protected),
    one_event_per_user=True,ten_candidates=True,at_most_one_click=True,histories_strictly_earlier=True,
    fitted_inputs_before_cutoff=True,weight_marginals_match=True,control_matches_frozen_upload=True,
    validation_labels_not_inputs=True,holdout_scored=False,evaluation_scored=False,submission_created=False,
    seconds=time.monotonic()-tic,training_performed=False,model_promotion=False,
    checks_scope='Read-only local CPU inference/metric verification of Kaggle artifacts; heavy training remains on Kaggle')
(R/'postfit_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
