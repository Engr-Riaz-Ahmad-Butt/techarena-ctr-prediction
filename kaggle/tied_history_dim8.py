"""Predeclared lower-dimensional factor pilot; compare a preserved 32D control."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import tied_history_gpu_bag as bag

DIMENSIONS = 8
MINIMUM_RAW_ADJUSTED_GAIN = .002


def numpy_predict(state, histories, candidates):
    vectors = state['items.weight'].cpu().numpy()
    bias = state['bias'].cpu().numpy()
    global_bias = float(state['global_bias'])
    parts = []
    for off in range(0, len(histories), 1024):
        h, c = histories[off:off+1024], candidates[off:off+1024]
        valid = h >= 0
        profile = (vectors[h.clip(0)]*valid[:,:,None]).sum(1)/np.maximum(valid.sum(1),1)[:,None]
        logits = (profile[:,None,:]*vectors[c]).sum(-1)+bias[c]+global_bias
        parts.append(expit(logits).ravel())
    return np.concatenate(parts)


def run():
    assert torch.cuda.is_available(), 'CUDA required; do not run a long local CPU fallback'
    started = time.monotonic()
    roots = list(Path('/kaggle/input').rglob('neural_manifest.json'))
    controls = list(Path('/kaggle/input').rglob('factors6/seed7201_pred.npy'))
    assert len(roots) == len(controls) == 1, (roots, controls)
    root, control_root = roots[0].parent, controls[0].parent
    output = Path('/kaggle/working/tied-history-dim8-001')
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    original_recipe = dict(bag.RECIPE)
    assert original_recipe['dimensions'] == 32
    bag.RECIPE = dict(original_recipe, dimensions=DIMENSIONS)
    frozen = dict(recipe=bag.RECIPE, changed_setting='dimensions32_to8_only',
                  raw_adjusted_gain_gate=MINIMUM_RAW_ADJUSTED_GAIN,
                  source_sha256=bag.digest(__file__), factor_source_sha256=bag.digest(bag.__file__),
                  gpu=torch.cuda.get_device_name(0), fresh_holdout=False, promoted=False,
                  official_auc=None, per_seed_validation_selection=False)
    (output/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print(json.dumps(frozen),flush=True)
    data, stamps = bag.load(root)
    with np.load(root/'val6.npz') as z:
        val = {key:z[key].copy() for key in z.files}
    prepared = bag.prepare(data, stamps, val, int(data['timestamp'][0])+6*86400)
    hashes = bag.verify_prepared(prepared,6)
    (output/'prepared_hashes.json').write_text(json.dumps(hashes,indent=2))
    histories, candidates = prepared['val_h'].copy(), prepared['val_i'].copy()
    for name in ['fit_h','fit_i','fit_y','aux_h','aux_target','val_h','val_i']:
        prepared[name] = torch.as_tensor(prepared[name],device='cuda:0')
    predictions, errors, control = [], [], []
    for seed in bag.RECIPE['seeds']:
        p = bag.one_seed(prepared,seed,output,torch.device('cuda:0'))
        state = torch.load(output/f'seed{seed}.pt',weights_only=True,map_location='cpu')
        replay = numpy_predict(state,histories,candidates)
        error = float(np.max(np.abs(p-replay)))
        assert error < 1e-6, error
        predictions.append(p);errors.append(error)
        control.append(np.load(control_root/f'seed{seed}_pred.npy'))
        print(json.dumps(dict(stage='independent_numpy_replay',seed=seed,max_absolute_error=error)),flush=True)
        del state,replay;gc.collect()
    raw8 = np.mean(np.stack(predictions).astype('float64'),axis=0)
    raw32 = np.mean(np.stack(control).astype('float64'),axis=0)
    rank = lambda p:rankdata(np.asarray(p,dtype='float64'),method='average')/(len(p)+1)
    blend = lambda p:.95*rank(val['baseline'])+.05*rank(p)
    p8, p32 = blend(raw8), blend(raw32)
    def metric(p,mask=None):
        if mask is None: mask=np.ones(len(p),dtype=bool)
        return dict(ordinary_auc=float(roc_auc_score(val['y'][mask],p[mask])),
                    adjusted_auc=float(roc_auc_score(val['y'][mask],p[mask],sample_weight=val['weights'][mask])))
    report = dict(baseline=metric(val['baseline']),raw8=metric(raw8),raw32=metric(raw32),
                  blend8=metric(p8),blend32=metric(p32),numpy_replay_max_errors=errors,
                  frozen_dimensions=8,training_steps_per_seed=2000,seed_count=5,
                  fresh_holdout=False,promoted=False,csv_ready=False,official_auc=None,target_achieved=False)
    assert abs(report['raw32']['ordinary_auc']-.7825589859284416)<1e-9, 'Preserved 32D control changed'
    report['raw_adjusted_gain']=report['raw8']['adjusted_auc']-report['raw32']['adjusted_auc']
    report['blend_adjusted_gain']=report['blend8']['adjusted_auc']-report['blend32']['adjusted_auc']
    report['full_oof_worth_testing']=(report['raw_adjusted_gain']>=MINIMUM_RAW_ADJUSTED_GAIN
                                    and report['blend_adjusted_gain']>0)
    report['history_segments']={}
    groups=np.digitize((data['hist'][val['indices']]>=0).sum(1),[0,4,9,19],right=True)
    for group in range(5):
        mask=np.repeat(groups==group,10)
        report['history_segments'][str(group)]=dict(events=int((groups==group).sum()),
                                                   raw8=metric(raw8,mask),raw32=metric(raw32,mask))
    report['elapsed_seconds']=time.monotonic()-started
    np.savez_compressed(output/'predictions.npz',y=val['y'],weights=val['weights'],
                        events=np.repeat(val['indices'],10),baseline=val['baseline'],
                        raw8=raw8,raw32=raw32,blend8=p8,blend32=p32)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    bag.RECIPE=original_recipe
    return report
