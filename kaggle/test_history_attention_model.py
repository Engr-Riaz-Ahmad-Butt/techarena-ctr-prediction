"""Core invariants before any production history-attention training."""
import copy
import json
from pathlib import Path
import torch
import tabular_choice_setwise_runner as base
from history_attention_model import HistoryChoice


def main():
    torch.set_num_threads(2);torch.manual_seed(6902)
    parent=base.ChoiceNet(torch.zeros(354),torch.ones(354),base.RECIPE).eval()
    vectors=torch.randn(32,16);history=torch.randn(32,16)
    model=HistoryChoice(parent,vectors,history).eval()
    x=torch.randn(6,10,354);items=torch.randint(0,32,(6,10))
    hist=torch.randint(0,32,(6,20));hist[0,:]=-1
    gaps=torch.rand(6,20)*20+.01;bucket=torch.arange(6)%4;kind=torch.arange(6)%2
    timestamp=torch.full((6,),1737849600,dtype=torch.long)+bucket*21600
    args=(x,items,hist,gaps,bucket,kind,timestamp)
    with torch.no_grad():
        logits,delta=model(*args)
        assert torch.equal(logits,parent(x)) and not delta.any()
    original={k:v.clone() for k,v in parent.state_dict().items()}
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
    target=torch.tensor([0,2,10,9,10,1])
    for _ in range(2):
        model.train();optimizer.zero_grad();logits,delta=model(*args)
        loss=torch.nn.functional.cross_entropy(logits,target)+.05*delta.square().mean()
        loss.backward();optimizer.step()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.attention.parameters())
    assert all(torch.equal(parent.state_dict()[k],v) for k,v in original.items())
    model.eval();perm=torch.tensor([9,4,0,3,6,5,7,1,8,2]);hp=torch.randperm(20)
    with torch.no_grad():
        a,d=model(*args)
        b,_=model(x[:,perm],items[:,perm],hist,gaps,bucket,kind,timestamp)
        c,_=model(x,items,hist[:,hp],gaps[:,hp],bucket,kind,timestamp)
        assert torch.allclose(a[:,:10][:,perm],b[:,:10],atol=2e-6)
        assert torch.allclose(a[:,10],b[:,10],atol=2e-6)
        assert torch.allclose(a,c,atol=2e-6)
        assert not d[0].any() and d.abs().max()<=.5
        restored=HistoryChoice(copy.deepcopy(parent),vectors,history).eval()
        restored.load_state_dict(model.state_dict())
        assert torch.equal(restored(*args)[0],a)
    report=dict(status='PASS',scope='Synthetic core model only; production training not tested',
                checks=['zero-initialized parent equivalence','frozen parent unchanged after optimizer updates',
                        'nonzero attention gradients','candidate and history permutation invariance',
                        'empty history fallback','bounded residual','checkpoint state replay'])
    destination=Path(__file__).resolve().parents[1]/'results/history_attention_model_test.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
