"""Own gated LightGCN-inspired matched preference screen; no copied repository code."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib,copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

class Preference(nn.Module):
    def __init__(self,initial,graph_enabled):
        super().__init__();self.U=nn.Parameter(torch.tensor(initial['U']));self.V=nn.Parameter(torch.tensor(initial['V']));self.H=nn.Parameter(torch.tensor(initial['H']));self.bu=nn.Parameter(torch.tensor(initial['bu']));self.bi=nn.Parameter(torch.tensor(initial['bi']));self.gate=nn.Parameter(torch.tensor(0.),requires_grad=graph_enabled);self.graph_enabled=graph_enabled
    def embedding(self,adj):
        e=torch.cat([self.U,self.V])
        if self.graph_enabled:
            one=torch.sparse.mm(adj,e);two=torch.sparse.mm(adj,one);e=e+.25*torch.tanh(self.gate)*.5*(one+two)
        return e
    def forward(self,u,i,h,adj,embedded=None):
        nu=len(self.U);e=self.embedding(adj) if embedded is None else embedded;valid=h>=0;hh=self.H[h.clamp_min(0)]*valid[:,:,None];z=e[u]+hh.sum(1)/valid.sum(1).clamp_min(1).sqrt()[:,None];logits=-1.8+self.bu[u,None]+self.bi[i]+torch.einsum('bd,bjd->bj',z,e[nu+i]);return torch.cat([logits,torch.zeros((len(u),1),device=u.device)],1)

def arrays(d,device):return [torch.as_tensor(d[k],dtype=torch.long,device=device) for k in ['users','items','history']]
def graph(path,n,device):
    d=np.load(path);assert int(d['max_source_time'])<int(d['cutoff']);return torch.sparse_coo_tensor(torch.tensor(d['indices'],device=device),torch.tensor(d['values'],device=device),size=(n,n)).coalesce()
def batch_step(model,opt,inputs,target,adj,regularization):
    u,i,h=inputs;opt.zero_grad(set_to_none=True);logits=model(u,i,h,adj);loss=nn.functional.cross_entropy(logits,target);reg=(model.U[u].square().mean()+model.V[i].square().mean()+model.H[h.clamp_min(0)].square().mean());(loss+regularization*reg).backward();opt.step();return float(loss.detach())
@torch.no_grad()
def predict(model,data,adj,device):
    model.eval();xx=arrays(data,device);embedded=model.embedding(adj);out=[]
    for off in range(0,len(xx[0]),2048):out.append(torch.softmax(model(*[x[off:off+2048] for x in xx],adj,embedded),1)[:,:10].cpu().numpy())
    return np.concatenate(out)

def pilot(root):
    root=Path(root);plan=json.loads((root/'frozen_plan.json').read_text());torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');d=np.load(root/'training.npz');initial=dict(np.load(root/'initial.npz'));n=len(initial['U'])+len(initial['V']);adj=graph(root/'graph2.npz',n,device);m=Preference(initial,True).to(device);opt=torch.optim.Adagrad(m.parameters(),lr=plan['learning_rate'],initial_accumulator_value=.1);xx=arrays(d,device);y=torch.as_tensor(d['target'],dtype=torch.long,device=device);batch=plan['batch_size'];timing=[]
    day2=np.flatnonzero(d['day']==2)
    for k in range(4):
        if device.type=='cuda':torch.cuda.synchronize()
        tic=time.monotonic();ix=torch.as_tensor(day2[k*batch:(k+1)*batch],dtype=torch.long,device=device);batch_step(m,opt,[x[ix] for x in xx],y[ix],adj,plan['regularization'])
        if device.type=='cuda':torch.cuda.synchronize()
        timing.append(time.monotonic()-tic)
    steps=int(np.ceil(len(y)/batch))*plan['epochs'];estimate=float(np.median(timing[1:])*steps*1.5+45);r=dict(stage='training_only_timing_pilot',device=str(device),step_seconds=timing,graph_steps_per_arm=steps,estimated_matched_run_seconds=estimate,validation_scored=False);print(json.dumps(r),flush=True);return r

def compare(y,b,p,w,mask,boots=200):
    yy=y[mask].ravel();bb=b[mask].ravel();pp=p[mask].ravel();ww=np.repeat(w[mask],10)
    def prep(z):
        o=np.argsort(z,kind='stable');v=z[o];return o,yy[o],ww[o],np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    def auc(z,m):
        o,y,w,s=z;w=w*m[o];pos=np.add.reduceat(w*y,s);neg=np.add.reduceat(w*(1-y),s);return float(pos@(np.cumsum(neg)-.5*neg)/(pos.sum()*neg.sum()))
    bc=prep(bb);pc=prep(pp);one=np.ones(len(yy));base=auc(bc,one);cand=auc(pc,one);assert abs(base-roc_auc_score(yy,bb,sample_weight=ww))<1e-10;rng=np.random.default_rng(8141);ne=int(mask.sum());ds=[]
    for _ in range(boots):
        m=np.repeat(np.bincount(rng.integers(0,ne,ne),minlength=ne),10);ds.append(auc(pc,m)-auc(bc,m))
    return dict(events=ne,control_auc=base,candidate_auc=cand,gain=cand-base,paired95=np.quantile(ds,[.025,.975]).tolist())

def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(exist_ok=True,parents=True);assert not (out/'report.json').exists(),'Do not repeat a completed fit';plan=json.loads((root/'frozen_plan.json').read_text());manifest=json.loads((root/'manifest.json').read_text());assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v['sha256'] for k,v in manifest.items());torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tic=time.monotonic();d=np.load(root/'training.npz');v=np.load(root/'screen.npz');initial=dict(np.load(root/'initial.npz'));n=len(initial['U'])+len(initial['V']);adj={k:graph(root/f'graph{k}.npz',n,device) for k in [1,2,3]};xx=arrays(d,device);target=torch.as_tensor(d['target'],dtype=torch.long,device=device);raw={};logs={};assert int(d['timestamp'].max())<plan['validation_min_timestamp'];assert len(np.unique(v['users']))==len(v['users'])
    # Both arms initially reproduce the same warm bag model exactly.
    zero=Preference(initial,False).to(device);extra=Preference(initial,True).to(device)
    with torch.no_grad():
        take=[x[:128] for x in xx];assert torch.allclose(zero(*take,adj[1]),extra(*take,adj[1]),rtol=0,atol=0)
    del zero,extra
    with threadpool_limits(limits=2):
        for name,enabled in [('bag_control',False),('graph_candidate',True)]:
            torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed']);model=Preference(initial,enabled).to(device);opt=torch.optim.Adagrad(model.parameters(),lr=plan['learning_rate'],initial_accumulator_value=plan['initial_adagrad_accumulator']);logs[name]=[]
            for epoch in range(plan['epochs']):
                model.train();ls=[]
                for day in [1,2]:
                    index=np.flatnonzero(d['day']==day);rng.shuffle(index)
                    for off in range(0,len(index),plan['batch_size']):
                        ix=torch.as_tensor(index[off:off+plan['batch_size']],dtype=torch.long,device=device);ls.append(batch_step(model,opt,[x[ix] for x in xx],target[ix],adj[day],plan['regularization']))
                row=dict(arm=name,epoch=epoch+1,loss=float(np.mean(ls)),gate=float(.25*torch.tanh(model.gate).detach()),seconds=time.monotonic()-tic);logs[name].append(row);print(json.dumps(row),flush=True)
            # No validation metrics or epoch selection; only final fixed epoch.
            raw[name]=predict(model,v,adj[3],device);torch.save(model.cpu().state_dict(),out/(name+'.pt'))
    control=v['baseline'];candidate=control+.05*(raw['graph_candidate']-raw['bag_control']);assert np.isfinite(candidate).all();np.savez_compressed(out/'frozen_predictions.npz',control=control,candidate=candidate,bag=raw['bag_control'],graph=raw['graph_candidate'],events=v['events'])
    y=v['y'];weights=v['weight'];arrival=v['arrival'];history=v['history_group'];metrics=[]
    with threadpool_limits(limits=1):
        for name,mask in [('overall',np.ones(len(y),bool))]+[(f'arrival{k+1}',arrival==k) for k in range(4)]+[(f'history{k}',history==k) for k in range(3)]:
            row=compare(y,control,candidate,weights,mask);row['group']=name;metrics.append(row)
    result=dict(stage='graph_screen_complete',device=str(device),logs=logs,screen=metrics,screen_passed=all(m['gain']>=.0002 for m in metrics[:5]),standalone_bag_auc=float(roc_auc_score(y.ravel(),raw['bag_control'].ravel(),sample_weight=np.repeat(weights,10))),standalone_graph_auc=float(roc_auc_score(y.ravel(),raw['graph_candidate'].ravel(),sample_weight=np.repeat(weights,10))),seconds=time.monotonic()-tic,initial_scores_identical=True,labels_used_for_selection=False,epoch_selection=False,holdout_opened=False,evaluation_scored=False,submission_created=False,limitations=['This is a fixed low-capacity graph residual diagnostic, not a full LightGCN reproduction.','Reference is the simplified horizon blend; not an official or full-V53 gain.','Only9878 user screen scored; no full-cohort confirmation or holdout.'])
    (out/'report.json').write_text(json.dumps(result,indent=2));print('GRAPH_SCREEN_JSON_BEGIN',json.dumps(result),'GRAPH_SCREEN_JSON_END',flush=True);return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--pilot',action='store_true');ap.add_argument('--output',default='/kaggle/working/cdx_graph_results');a=ap.parse_args();pilot(a.root) if a.pilot else run(a.root,a.output)
