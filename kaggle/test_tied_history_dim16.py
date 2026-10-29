import unittest
import numpy as np
import torch
from tied_history_gpu_bag import BagFactor
from tied_history_dim16 import numpy_predict


class ReplayTest(unittest.TestCase):
    def test_independent_replay_with_empty_and_duplicate_contexts(self):
        torch.manual_seed(812)
        model=BagFactor(12,16).double()
        history=np.array([[1,1,2,-1],[-1,-1,-1,-1],[4,2,4,5]])
        items=np.array([[1,2,3],[4,5,6],[7,8,9]])
        with torch.no_grad():
            expected=model(torch.tensor(history),torch.tensor(items)).sigmoid().numpy().ravel()
        actual=numpy_predict(model.state_dict(),history,items)
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-14)


if __name__=='__main__':
    unittest.main()
