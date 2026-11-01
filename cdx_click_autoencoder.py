"""Own compact warm-started catalog denoising/variational click autoencoder.

Order-free user click counts, exact full-catalog multinomial reconstruction;
fixed matched DAE/VAE arms. No copied implementation, new heads or searches.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,time,argparse,math
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import cdx_policy_hpf as metric
B,A,H=metric.B,metric.A,metric.H
R=H/'cdx_catalog_ae_001'
SEED=8711;EPOCHS=20;BATCH=256;LR=.001;MAX_KL=.2;WARMUP=5;REPLACE=.05

def prepare():
    t=np.load(A/'train.npy',mmap_mode='r');h=np.load(A/'train_hist.npy',mmap_mode='r');ht=np.load(A/'train_times.npy',mmap_mode='r')
    cut=int(t[0,2])+3*86400;n=int(np.searchsorted(t[::10,2],cut))*10;past=np.asarray(t[:n]);head=past[::10]
    _,first=np.unique(head[:,0],return_index=True);e=head[first];hh=np.asarray(h[e[:,6]]);ts=np.asarray(ht[e[:,6]]);valid=hh>=0
    assert np.all((ts<e[:,2,None])|~valid)
    clicked=past[past[:,5]==1]
    u=np.r_[clicked[:,0],np.broadcast_to(e[:,0,None],hh.shape)[valid]]
    i=np.r_[clicked[:,1],hh[valid]];stamp=np.r_[clicked[:,2],ts[valid]]
    assert stamp.max()<cut
    order=np.lexsort((stamp,i,u));u,i,stamp=u[order],i[order],stamp[order]
    unique=np.r_[True,(u[1:]!=u[:-1])|(i[1:]!=i[:-1])|(stamp[1:]!=stamp[:-1])]
    model=dict(np.load(H/'cdx_collective_001/control_model.npz'))
    x=sp.csr_matrix((np.ones(int(unique.sum()),np.float32),(u[unique],i[unique])),shape=(len(model['U']),50000))
    active=np.flatnonzero(np.asarray(x.sum(1)).ravel()>0)
    return x,active,model,dict(cutoff=cut,max_source_timestamp=int(stamp.max()),click_occurrences=int(unique.sum()),
        fitted_users=len(active),exposure_zeros_not_used_as_click_labels=True)

class CatalogAE(nn.Module):
    def __init__(self,initial,clicks,variational):
        super().__init__();d=initial['V'].shape[1];ni=len(initial['V']);self.variational=variational
        self.encoder=nn.EmbeddingBag(ni,d,mode='sum',include_last_offset=True)
        self.mean=nn.Linear(d,d);self.logvar=nn.Linear(d,d);self.decoder=nn.Linear(d,ni)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor(initial['H']))
            self.mean.weight.copy_(torch.eye(d));self.mean.bias.zero_()
            self.logvar.weight.zero_();self.logvar.bias.fill_(-4.)
            self.decoder.weight.copy_(torch.tensor(initial['V']))
            marginal=np.asarray(clicks.sum(0)).ravel()+1
            self.decoder.bias.copy_(torch.tensor(np.log(marginal/marginal.sum())))

    def encode(self,items,offsets,weights,input_rng=None):
        if self.training:
            assert input_rng is not None
            weights=weights*(torch.rand(weights.shape,generator=input_rng)>=.5)*2
        hidden=torch.tanh(self.encoder(items,offsets,per_sample_weights=weights))
        return self.mean(hidden),self.logvar(hidden).clamp(-12,8)

    def forward(self,items,offsets,weights,input_rng,latent_rng):
        mean,logvar=self.encode(items,offsets,weights,input_rng)
        if self.variational:
            z=mean+torch.exp(.5*logvar)*torch.randn(mean.shape,generator=latent_rng)
        else:z=mean
        return self.decoder(z),.5*(mean.square()+logvar.exp()-1-logvar).sum(1).mean()

def batch(x,rows):
    sub=x[rows].tocsr();norm=np.sqrt(np.asarray(sub.multiply(sub).sum(1)).ravel())
    owner=np.repeat(np.arange(len(rows)),np.diff(sub.indptr))
    weights=sub.data/np.maximum(norm[owner],1e-8)
    return (torch.tensor(sub.indices,dtype=torch.long),torch.tensor(sub.indptr,dtype=torch.long),
        torch.tensor(weights,dtype=torch.float32),torch.tensor(owner,dtype=torch.long),torch.tensor(sub.data,dtype=torch.float32))

def reconstruction(logits,items,owner,count):
    return -(nn.functional.log_softmax(logits,dim=1)[owner,items]*count).sum()/len(logits)

def checks(initial,x):
    # Sparse target likelihood equals independent dense multinomial likelihood.
    logits=torch.tensor([[.2,-.4,.6],[.8,.1,-.2]],dtype=torch.float64)
    target=torch.tensor([[0.,2.,1.],[1.,0.,0.]],dtype=torch.float64)
    r,c=torch.where(target>0)
    sparse=reconstruction(logits,c,r,target[r,c]);dense=-(target*(logits-torch.logsumexp(logits,1,keepdim=True))).sum()/2
    assert torch.allclose(sparse,dense,atol=1e-12)
    a=CatalogAE(initial,x,False).eval();b=CatalogAE(initial,x,True).eval();inputs=batch(x,np.array([0,1,2]))[:3]
    with torch.no_grad():
        am,_=a.encode(*inputs);bm,_=b.encode(*inputs)
        assert torch.equal(am,bm) and torch.equal(a.decoder(am),b.decoder(bm))
    return True

@torch.no_grad()
def score(model,history,items,point,item_bias):
    model.eval();ne=len(history);valid=history>=0;owner=np.broadcast_to(np.arange(ne)[:,None],history.shape)[valid]
    x=sp.csr_matrix((np.ones(int(valid.sum()),np.float32),(owner,history[valid])),shape=(ne,50000))
    out=[]
    for start in range(0,ne,1024):
        take=np.arange(start,min(start+1024,ne));ii,off,w,_,_=batch(x,take);z,_=model.encode(ii,off,w)
        candidates=torch.tensor(items[take],dtype=torch.long)
        # The density decoder's marginal item bias is replaced with the SAME frozen CTR bias.
        logits=(z[:,None,:]*model.decoder.weight[candidates]).sum(2)+torch.tensor(item_bias[items[take]])
        conditional=torch.softmax(logits,1).numpy();out.append(conditional*point[take].sum(1)[:,None])
    return np.concatenate(out)

def freeze():
    R.mkdir(exist_ok=True);assert not (R/'frozen.json').exists()
    plan=dict(date='2026-10-30',hypothesis='Full-catalog denoising and a learned variational bottleneck recover preference missed by event-level factor training',
        source='https://arxiv.org/abs/1802.05814',authors='Liang, Krishnan, Hoffman, Jebara2018',
        source_scope='Own compact warm-started adaptation of multinomial DAE/VAE principles; not a paper reproduction',
        data='Deduplicated actual pre-cutoff clicked triples plus each pre-cutoff user earliest supplied history; counts across items, no exposure negatives',
        architecture='50000x16 embedding-bag encoder -> tanh ->16Dmean/logvar ->50000-way decoder, exact catalog likelihood',
        initialization='Cutoff3 archived warm16 H,V; mean layer identity,logvar bias-4; learned decoder starts at training click marginal',
        epochs=EPOCHS,batch_size=BATCH,lr=LR,seed=SEED,input_dropout=.5,max_kl=MAX_KL,kl_warmup_epochs=WARMUP,
        optimizer='Adam, no validation early stopping',matched='Same initialization, batches and input corruption RNG; independent latent RNG only in VAE; DAE omits sampling/KL',
        scoring='Mean latent, personal decoder dot+same frozen original item CTR bias; original event-click mass retained; no learned calibration/temperature',
        bridge='Same exposure-enhanced horizon reference; fixed5percentagepoints of original warm rank replaced by DAE or VAE rank, joint complete horizon',
        candidates='DAE and VAE each preregistered vs same frozen reference; VAE-vs-DAE also reported for mechanism',
        gate='Each candidate needs+.0002 overall/all4arrivals; only survivor full+.0005 overall/all4; no holdout',
        no_search=True,no_evaluation_statistics=True,no_holdout=True,no_submission=True,
        hashes={str(p.relative_to(B)):metric.sha(p) for p in [B/'cdx_click_autoencoder.py',H/'cohort.npz',H/'cdx_collective_001/control_model.npz',H/'cdx_collective_001/frozen_predictions.npz',H/'cdx_policy_hpf_001/screen_predictions.npz']})
    (R/'frozen.json').open('x').write(json.dumps(plan,indent=2));return plan

def pilot():
    freeze();x,active,initial,info=prepare();sp.save_npz(R/'clicks.npz',x);np.save(R/'active_users.npy',active)
    (R/'data_info.json').write_text(json.dumps(info,indent=2));assert checks(initial,x)
    torch.manual_seed(SEED);m=CatalogAE(initial,x,True);opt=torch.optim.Adam(m.parameters(),lr=LR)
    ir=torch.Generator().manual_seed(SEED+1);lrng=torch.Generator().manual_seed(SEED+2);times=[]
    for k in range(8):
        data=batch(x,active[k*BATCH:(k+1)*BATCH]);tic=time.monotonic();opt.zero_grad(set_to_none=True)
        logits,kl=m(*data[:3],ir,lrng);loss=reconstruction(logits,data[0],data[3],data[4])+.02*kl
        loss.backward();opt.step();times.append(time.monotonic()-tic)
    estimate=float(np.median(times[2:]) * math.ceil(len(active)/BATCH)*EPOCHS*2*1.5+45)
    out=dict(stage='training_only_pilot',step_seconds=times,estimated_seconds=estimate,numerical_checks=True,
        fitted_users=len(active),validation_scored=False,device='cpu',threads=2)
    (R/'pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

def run():
    assert not (R/'report.json').exists();plan=json.loads((R/'frozen.json').read_text())
    assert all(metric.sha(B/p)==v for p,v in plan['hashes'].items())
    assert json.loads((R/'pilot.json').read_text())['estimated_seconds']<1800
    tic=time.monotonic();x=sp.load_npz(R/'clicks.npz');active=np.load(R/'active_users.npy');info=json.loads((R/'data_info.json').read_text())
    initial=dict(np.load(H/'cdx_collective_001/control_model.npz'));rows=np.load(H/'score_rows.npy',mmap_mode='r');co=np.load(H/'cohort.npz')
    history=np.asarray(np.load(A/'train_hist.npy',mmap_mode='r')[rows[::10,6]]);items=rows[:,1].reshape(-1,10)
    timestamps=np.load(A/'train_times.npy',mmap_mode='r')[rows[::10,6]]
    assert np.all((timestamps<rows[::10,2,None])|(history<0)) and rows[::10,2].min()>=info['cutoff']>info['max_source_timestamp']
    point=np.load(H/'cdx_collective_001/frozen_predictions.npz')['raw_control']
    predictions={};logs={}
    for name,variational in [('dae',False),('vae',True)]:
        torch.manual_seed(SEED);model=CatalogAE(initial,x,variational);opt=torch.optim.Adam(model.parameters(),lr=LR)
        rng=np.random.default_rng(SEED);ir=torch.Generator().manual_seed(SEED+1);zr=torch.Generator().manual_seed(SEED+2)
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
        predictions[name]=score(model,history,items,point,initial['bi']);torch.save(model.state_dict(),R/f'{name}.pt')
    legacy=np.load(H/'matched_refresh_001/predictions.npz');base=legacy['control_blend'].reshape(-1,10)
    # Reproduce the established label-free exposure signal with frozen historical CDFs.
    policy=__import__('pickle').loads((H/'cdx_policy_hpf_001/prefix3_models.pkl').read_bytes())
    raw=metric.raw_svd(policy['svd'],rows[::10,0],items)
    e3=np.mean([metric.cdf(policy['cdfs']['svd'][k],raw[k]) for k in range(3)],axis=0)
    base=base+.015*metric.centered(e3)
    sample=co['sample'];prior=np.load(H/'cdx_policy_hpf_001/screen_predictions.npz')
    assert np.allclose(base[sample],prior['control'],atol=1e-12)
    old_rank=rankdata(legacy['other_expert'])/(len(rows)+1);candidate={}
    for name,q in predictions.items():
        assert np.isfinite(q).all() and np.allclose(q.sum(1),point.sum(1),atol=2e-7)
        candidate[name]=base+REPLACE*(rankdata(q.ravel())/(len(rows)+1)-old_rank).reshape(-1,10)
    np.savez_compressed(R/'frozen_predictions.npz',control=base,dae=candidate['dae'],vae=candidate['vae'],raw_dae=predictions['dae'],raw_vae=predictions['vae'],events=co['events'])
    # Both final fixed checkpoints and all predictions precede validation outcome access.
    y=np.load(H/'validation_labels.npy')[sample];w=co['sample_event_weight'];ar=co['arrival'][sample];hg=co['history_group'][sample]
    masks=[('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',ar==k) for k in range(4)]+[('arrival3plus',ar>=2)]+[(f'history{k}',hg==k) for k in range(3)]
    reports={};passes={};full={}
    with threadpool_limits(limits=1):
        for arm in ['dae','vae']:
            reports[arm]=[]
            for label,mask in masks:
                z=metric.compare(y,base[sample],candidate[arm][sample],w,mask);z['group']=label;reports[arm].append(z)
            passes[arm]=all(z['gain']>=.0002 for z in reports[arm][:5]);full[arm]=[]
            if passes[arm]:
                fy=np.load(H/'validation_labels.npy')
                for label,mask in [('overall',np.ones(len(fy),bool))]+[(f'arrival{k+1}',co['arrival']==k) for k in range(4)]:
                    z=metric.compare(fy,base,candidate[arm],co['event_weight'],mask);z['group']=label;full[arm].append(z)
        mechanism=metric.compare(y,candidate['dae'][sample],candidate['vae'][sample],w,np.ones(len(y),bool))
    protected=json.loads((H/'protected_before.json').read_text());assert all(metric.sha(B/k)==v['sha256'] for k,v in protected.items())
    result=dict(status='complete',screen=reports,screen_passed=passes,full_confirmation=full,
        confirmation_passed={k:bool(v) and all(z['gain']>=.0005 for z in v) for k,v in full.items()},
        vae_vs_dae=mechanism,training_logs=logs,data_info=info,seconds=time.monotonic()-tic,
        protected_files_unchanged=len(protected),holdout_scored=False,evaluation_scored=False,submission_created=False,
        fixed_final_epoch=True,no_validation_parameter_selection=True,event_click_mass_preserved=True,
        scope='Compact catalog AE diagnostic atop the simplified exposure-enhanced horizon; not fullV53/nxa_h1 or official score')
    (R/'report.json').write_text(json.dumps(result,indent=2));print('AE_RESULT_BEGIN',json.dumps(result),'AE_RESULT_END',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):pilot() if a.pilot else run()
