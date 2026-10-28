"""Setwise feature invariance, permutation and checkpoint integration."""
import numpy as np
import torch
import tabular_choice_setwise_runner as setwise
import test_tabular_choice_runner as integration

def main():
    torch.manual_seed(6802)
    model=setwise.ChoiceNet(torch.zeros(354),torch.ones(354),setwise.RECIPE).eval()
    x=torch.randn(8,10,354)
    slope=torch.rand(8,1,354)+.5
    offset=torch.randn(8,1,354)*2
    with torch.no_grad():
        a=model(x)
        b=model(x*slope+offset)
    error=float((a-b).abs().max())
    assert error<2e-6,error
    integration.choice=setwise
    integration.main()
    print("PASS: per-event positive affine feature transforms; max logit error",error)

if __name__=="__main__":main()
