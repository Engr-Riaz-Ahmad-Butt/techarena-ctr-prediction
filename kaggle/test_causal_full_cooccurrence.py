import unittest
import numpy as np
from causal_full_cooccurrence import graph,features,prefix_ledger,contexts


class GraphTests(unittest.TestCase):
    def test_binary_profiles_oracle_and_normalization(self):
        ledger={0:{(1,1),(2,2),(3,2)},1:{(1,1),(2,3)},
                2:{(1,2),(2,3)},3:{(1,4),(2,5)}}
        matrix,freq,users,checks=graph(ledger,12)
        self.assertEqual(users,4)
        self.assertEqual(freq[2],2)
        self.assertEqual(matrix[1,2],1)
        self.assertEqual(matrix[2,2],0)
        self.assertTrue(checks)
        history=np.array([[4,4,-1],[-1,-1,-1]])
        candidates=np.tile(np.arange(10),(2,1))
        x=features(matrix,freq,users,history,candidates)
        self.assertEqual(x.shape,(20,12))
        np.testing.assert_allclose(x[5,:6],[1,1,np.log(4)/6,np.log(4)/6,np.log(2),np.log(2)],rtol=1e-6)
        self.assertEqual(x[4,9],1)
        np.testing.assert_array_equal(x[10:,:7],0)
        self.assertEqual(x[15,7],np.float32(np.log(2)))

    def test_prefix_graph_ignores_heldout_targets(self):
        data=dict(timestamp=np.array([10,20,40]),users=np.array([0,1,0]),
                  hist=np.array([[1,-1],[2,-1],[3,4]]),
                  items=np.tile(np.arange(10),(3,1)),target=np.array([3,5,7]))
        stamps=np.array([[1,0],[2,0],[10,35]])
        first=prefix_ledger(data,stamps,30)
        data['target'][-1]=9
        self.assertEqual(first,prefix_ledger(data,stamps,30))
        np.testing.assert_array_equal(contexts(first,data,stamps,[2])[0,:3],[4,3,1])
        stamps[2,0]=40
        with self.assertRaises(AssertionError):contexts(first,data,stamps,[2])


if __name__=='__main__':unittest.main()
