"""Regression checks for masking, candidate symmetry and auxiliary cutoff use."""
import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import neural_aux_runner as aux
import neural_runner as base
import neural_regularized_runner as regularized


class NeuralInvariants(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(4)
        self.f={k:rng.normal(0,.15,s).astype("float32") for k,s in
                {"U":(4,3),"V":(6,3),"H":(6,3),"bu":(4,),"bi":(6,)}.items()}
        self.c={k:torch.tensor(rng.normal(0,.02,s),dtype=torch.float32) for k,s in
                {"bucket":(4,3,3),"kind":(2,3,3),"recent":(3,3)}.items()}
        h=torch.full((3,20),-1);h[1,0]=4;h[2]=torch.arange(20)%6
        self.inputs=(torch.tensor([0,1,2]),torch.tensor([[0,1,2,3,4,5,0,1,2,3]]*3),
                     h,torch.zeros(3,20),torch.tensor([0,1,2]),torch.tensor([0,1,0]))

    def test_candidate_symmetry_and_outside_option(self):
        model=base.JointModel(self.f,self.c)
        out=model(*self.inputs)
        permutation=torch.tensor([9,0,8,1,7,2,6,3,5,4])
        changed=list(self.inputs);changed[1]=changed[1][:,permutation]
        p=model(*changed)
        torch.testing.assert_close(out[:,:10][:,permutation],p[:,:10])
        self.assertTrue(torch.equal(out[:,10],p[:,10]))
        self.assertTrue(bool(torch.all(out.softmax(1)[:,:10].sum(1)<1)))

    def test_empty_history_mask_and_embedding_gradients(self):
        model=base.JointModel(self.f,self.c)
        model(*(x[:1] for x in self.inputs)).sum().backward()
        self.assertEqual(float(model.H.weight.grad.abs().sum()),0.)
        model.zero_grad()
        torch.nn.functional.cross_entropy(model(*self.inputs),torch.tensor([10,0,3])).backward()
        self.assertGreater(float(model.H.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.V.weight.grad.abs().sum()),0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_capacity_expansion_preserves_initial_predictions(self):
        original=base.JointModel(self.f,self.c)(*self.inputs)
        f,c=aux.expand(self.f,self.c,np.random.default_rng(6502),64)
        expanded=base.JointModel(f,c)(*self.inputs)
        torch.testing.assert_close(original,expanded,atol=1e-6,rtol=1e-6)

    def test_auxiliary_pairs_only_use_supplied_fitting_events(self):
        data=dict(users=np.array([0,0,1]),hist=np.array([[1,-1],[4,-1],[2,2]]),
                  gaps=np.array([[2,0],[1,0],[2,3]],dtype="float32"))
        users,items,_=aux.warm_pairs(data,np.array([0,2]),6)
        self.assertEqual(sorted(zip(users,items)),[(0,1),(1,2),(1,2)])
        self.assertNotIn(4,items)

    def test_dropout_does_not_change_inference_or_checkpoint_replay(self):
        expected=base.JointModel(self.f,self.c).eval()(*self.inputs)
        model=regularized.make_model(base,self.f,self.c).eval()
        torch.testing.assert_close(expected,model(*self.inputs))
        restored=regularized.make_model(base,self.f,self.c).eval()
        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(model(*self.inputs),restored(*self.inputs)))
        torch.manual_seed(12)
        model.train()
        self.assertFalse(torch.equal(expected,model(*self.inputs)))

    def test_one_step_regularized_training_and_saved_checkpoint(self):
        # Exercise the new sampled anchor loss and optimizer on synthetic data.
        # These fixture scores have no relation to competition performance.
        workspace=Path(__file__).resolve().parents[1]/"artifacts"
        workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="neural_smoke_",dir=workspace) as temporary:
            root=Path(temporary).resolve()
            self.assertTrue(root.is_relative_to(workspace.resolve()))
            data=dict(users=np.arange(12)%4,
                      items=np.tile(np.arange(10)%6,(12,1)),
                      hist=np.tile(self.inputs[2].numpy(),(4,1)),gaps=np.zeros((12,20),dtype="float32"),
                      bucket=np.arange(12)%4,kind=np.arange(12)%2,
                      timestamp=np.r_[np.arange(8)*3600,np.full(4,6*86400)],
                      target=np.array([0,10,1,10,2,10,3,10,0,10,2,10]))
            indices=np.arange(8,12)
            initial=base.predict(base.JointModel(self.f,self.c),data,indices,"cpu")
            y=np.zeros((4,10),dtype="int8");y[0,0]=1;y[2,2]=1
            np.savez(root/"factors6.npz",**self.f)
            torch.save(self.c,root/"context6.pt")
            np.save(root/"initial6.npy",initial)
            np.savez(root/"val6.npz",indices=indices,y=y.ravel(),baseline=initial,weights=np.ones(40))
            (root/"neural_manifest.json").write_text(json.dumps(dict(start=0)))
            helpers=regularized.load_module("smoke_metrics",Path(__file__).with_name("runner.py"))
            frozen=dict(regularized.RECIPE)
            try:
                regularized.RECIPE.update(epochs=1,batch_size=4)
                with contextlib.redirect_stdout(io.StringIO()):
                    report=regularized.fit(root,root/"output",6,data,dict(start=0),base,helpers,"cpu")
            finally:
                regularized.RECIPE.clear();regularized.RECIPE.update(frozen)
            self.assertEqual(report["saved_checkpoint_replay"],"PASS")
            saved=torch.load(root/"output"/"model.pt",weights_only=True)
            self.assertFalse(torch.equal(saved["V.weight"][:,:3],torch.from_numpy(self.f["V"])))
            metadata=json.loads((root/"output"/"recipe.json").read_text())
            self.assertLess(metadata["last_fit_timestamp"],6*86400)


if __name__=="__main__":
    unittest.main()
