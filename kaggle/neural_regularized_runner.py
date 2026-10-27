"""Regularized warm joint embeddings with sampled anchor penalties and dropout.

The only held-out labels used are for final scoring. No epoch, blend weight,
or recipe selection uses validation labels. This is exploratory on reused dates.
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

RECIPE = dict(seed=6503, dim=16, epochs=6, batch_size=2048, learning_rate=.0003,
              weight_decay=.0001, anchor_penalty=.2, auxiliary_weight=1.,
              rank_residual_weight=.1, negatives=10, training="all events before cutoff",
              user_dropout=.25, history_dropout=.1, anchor_scope="sampled vector sum of squares")


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def expand(factors, context, rng, dim):
    old = factors["V"].shape[1]
    assert dim >= old
    values = {k:v.copy() for k,v in factors.items()}
    for k in ["U","V","H"]:
        extra=np.zeros((len(values[k]),dim-old),dtype="float32")
        if k=="V":extra[:]=rng.normal(0,.05,extra.shape)
        values[k]=np.column_stack([values[k],extra])
    matrices={}
    for name, v in context.items():
        a=torch.zeros((*v.shape[:-2],dim,dim),dtype=v.dtype)
        a[..., :old, :old]=v
        matrices[name]=a
    return values,matrices


def warm_pairs(data,fitting,n_items):
    # First pre-cutoff event per user, including supplied warm-up history.
    _,first=np.unique(data["users"][fitting],return_index=True)
    ix=fitting[first]
    h=data["hist"][ix];valid=h>=0
    users=np.repeat(data["users"][ix],h.shape[1])[valid.ravel()]
    items=h.ravel()[valid.ravel()]
    gaps=data["gaps"][ix].ravel()[valid.ravel()]
    keys=np.rec.fromarrays([users,items,gaps],names="user,item,gap")
    _,unique=np.unique(keys,return_index=True)
    users,items=users[unique],items[unique]
    counts=np.bincount(items,minlength=n_items)+1
    return users,items,np.log(counts/counts.mean()).astype("float32")


def make_model(base,factors,context):
    model=base.JointModel(factors,context)
    model.U.register_forward_hook(lambda module,args,out: nn.functional.dropout(out,p=RECIPE["user_dropout"],training=module.training))
    model.H.register_forward_hook(lambda module,args,out: nn.functional.dropout(out,p=RECIPE["history_dropout"],training=module.training))
    return model


def fit(root, output, day, data, manifest, base, metrics, device):
    started=time.monotonic()
    rng=np.random.default_rng(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    torch.cuda.manual_seed_all(RECIPE["seed"])
    output.mkdir(parents=True,exist_ok=False)
    cut=manifest["start"]+day*86400
    fitting=np.flatnonzero(data["timestamp"]<cut)
    val=dict(np.load(root/f"val{day}.npz"));vi=val["indices"]
    assert data["timestamp"][fitting].max()<data["timestamp"][vi].min()
    assert not np.intersect1d(fitting,vi).size
    with np.load(root/f"factors{day}.npz") as z:
        factors={k:z[k].astype("float32") for k in ["U","V","H","bu","bi"]}
    context=torch.load(root/f"context{day}.pt",weights_only=True,map_location="cpu")
    factors,context=expand(factors,context,rng,RECIPE["dim"])
    model=make_model(base,factors,context).to(device)
    initial=base.predict(model,data,vi,device)
    expected=np.load(root/f"initial{day}.npy")
    error=float(np.max(np.abs(initial-expected)))
    assert error<3e-6,error
    wu,wi,pop_bias=warm_pairs(data,fitting,len(factors["V"]))
    assert len(wu)>0
    pop=torch.as_tensor(pop_bias,device=device)
    anchors={k:p.detach().clone() for k,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE["learning_rate"],weight_decay=RECIPE["weight_decay"])
    recipe=dict(RECIPE,day=day,cutoff=cut,fit_events=len(fitting),warm_pairs=len(wu),
                last_fit_timestamp=int(data["timestamp"][fitting].max()),initial_replay_error=error,
                input_manifest_sha256=base.digest(root/"neural_manifest.json"),
                source_sha256=base.digest(Path(__file__)),torch=torch.__version__)
    (output/"recipe.json").write_text(json.dumps(recipe,indent=2))
    print(dict(stage="prepared",day=day,fit_events=len(fitting),warm_pairs=len(wu),initial_replay_error=error),flush=True)
    for epoch in range(1,RECIPE["epochs"]+1):
        model.train();order=rng.permutation(fitting);loss_total=0.;warm_total=0.
        for off in range(0,len(order),RECIPE["batch_size"]):
            ix=order[off:off+RECIPE["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits=model(*base.batch(data,ix,device))
            loss=nn.functional.cross_entropy(logits,torch.as_tensor(data["target"][ix],device=device,dtype=torch.long))
            chosen=rng.integers(0,len(wu),size=len(ix))
            users=torch.as_tensor(wu[chosen],device=device,dtype=torch.long)
            positives=wi[chosen]
            negative=rng.integers(0,len(pop_bias),size=(len(ix),RECIPE["negatives"]),dtype=np.int64)
            same=negative==positives[:,None]
            while same.any():
                negative[same]=rng.integers(0,len(pop_bias),size=int(same.sum()))
                same=negative==positives[:,None]
            candidates=torch.as_tensor(np.column_stack([positives,negative]),device=device,dtype=torch.long)
            scores=(model.U(users)[:,None]*model.V(candidates)).sum(-1)+pop[candidates]
            warm_loss=nn.functional.cross_entropy(scores,torch.zeros(len(ix),device=device,dtype=torch.long))
            hist=torch.as_tensor(data["hist"][ix],device=device,dtype=torch.long)
            valid=hist>=0
            drift=(model.U.weight[users]-anchors["U.weight"][users]).square().sum(-1).mean()
            drift=drift+(model.V.weight[candidates]-anchors["V.weight"][candidates]).square().sum(-1).mean()
            hd=(model.H.weight[hist.clamp(min=0)]-anchors["H.weight"][hist.clamp(min=0)]).square().sum(-1)
            drift=drift+(hd*valid).sum()/valid.sum().clamp(min=1)
            drift=drift+(model.bu.weight[users]-anchors["bu.weight"][users]).square().mean()
            drift=drift+(model.bi.weight[candidates]-anchors["bi.weight"][candidates]).square().mean()
            drift=drift+.01*sum((p-anchors[name]).square().sum() for name,p in model.named_parameters() if "matrix" in name)
            total=loss+RECIPE["auxiliary_weight"]*warm_loss+RECIPE["anchor_penalty"]*drift
            assert torch.isfinite(total), "Nonfinite training loss"
            total.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_total+=float(loss.detach())*len(ix);warm_total+=float(warm_loss.detach())*len(ix)
        p=dict(day=day,epoch=epoch,epochs=RECIPE["epochs"],event_loss=loss_total/len(order),
               auxiliary_loss=warm_total/len(order),elapsed_s=round(time.monotonic()-started,1))
        (output/"progress.json").write_text(json.dumps(p));print(p,flush=True)
    torch.save(model.state_dict(),output/"model.pt")
    raw=base.predict(model,data,vi,device)
    pred=metrics.rank(metrics.rank(val["baseline"])+RECIPE["rank_residual_weight"]*(metrics.rank(raw)-metrics.rank(initial)))
    m=metrics.metrics(val["y"],pred,val["baseline"],val["weights"])
    m.update(day=day,raw_auc=float(metrics.roc_auc_score(val["y"],raw)),initial_raw_auc=float(metrics.roc_auc_score(val["y"],initial)),
             initial_replay_error=error,model_sha256=base.digest(output/"model.pt"),official_auc=None,csv_ready=False)
    restored=make_model(base,factors,context).to(device)
    restored.load_state_dict(torch.load(output/"model.pt",weights_only=True,map_location=device))
    replay=base.predict(restored,data,vi,device)
    assert np.array_equal(replay,raw), "Saved checkpoint prediction changed"
    m["saved_checkpoint_replay"]="PASS"
    np.savez_compressed(output/"predictions.npz",raw=raw,initial=initial,pred=pred,**val)
    (output/"report.json").write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2),flush=True)
    return m


def run(root,output,device="cuda"):
    root,output=Path(root),Path(output)
    assert torch.cuda.is_available() if device=="cuda" else True
    torch.set_num_threads(4)
    base=load_module("joint_base",root/"neural_runner.py")
    metrics=load_module("joint_metrics",root/"runner.py")
    manifest=json.loads((root/"neural_manifest.json").read_text())
    for name,info in manifest["files"].items():
        assert base.digest(root/name)==info["sha256"],name
    output.mkdir(parents=True,exist_ok=False)
    (output/"neural_regularized_runner.py").write_text(Path(__file__).read_text())
    data=dict(np.load(root/"events.npz"))
    reports=[fit(root,output/"day6",6,data,manifest,base,metrics,device)]
    if reports[0]["positive_gate"]:
        reports.append(fit(root,output/"day5",5,data,manifest,base,metrics,device))
    summary=dict(reports=reports,eligible_for_review=len(reports)==2 and all(m["positive_gate"] for m in reports),
                 target=.83,target_achieved=False,csv_ready=False,official_auc=None,
                 note="Full-prefix training, pre-cutoff warm histories only. Reused dates; no official forecast.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary
