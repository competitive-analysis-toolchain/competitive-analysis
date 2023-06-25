import unittest
from TaylorModel import Monomial, TaylorModel,contains_zero
from Flowpipes import FlowPipe, FlowpipeList
from Traces import TraceTree
import json


class FPA_GENERAL_TEST_CASE(unittest.TestCase):

    def test_contains_zero_success(self):
        self.assertTrue(contains_zero(-1, 2))
        self.assertFalse(contains_zero(2, 3))
        self.assertFalse(contains_zero(-3, -1))

    def test_monomial_interval_success(self):
        monomial = Monomial()
        monomial.coeff = [1, 2]
        monomial.degrees = [1, 2]

        intervals = [[1, 2], [1, 2]]
        result = monomial.get_interval(intervals)
        self.assertEqual(result, [1, 16])

        intervals = [[1, 2], [-1, 2]]
        result = monomial.get_interval(intervals)
        self.assertEqual(result, [0, 16])

        intervals = [[-2, -1], [1, 2]]
        result = monomial.get_interval(intervals)
        self.assertEqual(result, [-16, -1])

        intervals = [[-2, 1], [1, 2]]
        result = monomial.get_interval(intervals)
        self.assertEqual(result, [-16, 8])

    def test_taylor_interval_success(self):
        m1 = Monomial()
        m1.coeff = [1, 2]
        m1.degrees = [1, 2]
        m2 = Monomial()
        m2.coeff = [1, 2]
        m2.degrees = [1, 2]
        m3 = Monomial()
        m3.coeff = [1, 2]
        m3.degrees = [1, 2]

        tm = TaylorModel()
        tm.expansion = [m1, m2, m3]
        tm.remainder = [0, 1]

        intervals = [[1, 2], [1, 2]]

        result = tm.get_interval(intervals)

        self.assertEqual(result, [3, 48])

    def test_fp_interval_success(self):
        m1 = Monomial()
        m1.coeff = [1, 2]
        m1.degrees = [1, 2]
        m2 = Monomial()
        m2.coeff = [1, 2]
        m2.degrees = [1, 2]
        m3 = Monomial()
        m3.coeff = [1, 2]
        m3.degrees = [1, 2]

        tm1 = TaylorModel()
        tm1.expansion = [m1, m2]
        tm1.remainder = [0, 1]

        tm2 = TaylorModel()
        tm2.expansion = [m3]
        tm2.remainder = [0, 1]

        fp = FlowPipe()
        fp.statevars = {"var1": tm1, "var2": tm2}
        fp.domains = [[1, 2], [1, 2]]
        result = fp.get_interval()
        self.assertEqual(result["var1"], [2, 32])
        self.assertEqual(result["var2"], [1, 16])



    def test_mono_notenoughdom(self):
        m1 = Monomial()
        m1.coeff = [1, 2]

        with self.assertRaises(RuntimeError):
            m1.get_interval([[1,2]])


    def test_fpl_get_counter_example(self):
        fp1 = FlowPipe()
        fp1.unsafe = True
        fp1.modeID = 1

        fp2 = FlowPipe()
        fp2.modeID = 2

        fp3 = FlowPipe()
        fp3.modeID = 3

        fpl1 = FlowpipeList()
        fpl1.flowpipes = [fp1, fp2, fp3]
        self.assertEqual(fpl1.get_counter_example()[0], 1)
        fpl2 = FlowpipeList()
        fpl2.flowpipes = [fp2, fp3]
        self.assertEqual(fpl2.get_counter_example()[0], -1)

    def test_fpl_prune(self):
        fp1 = FlowPipe()
        fp1.unsafe = True
        fp1.modeID = 1

        fp2 = FlowPipe()
        fp2.modeID = 2

        fp3 = FlowPipe()
        fp3.modeID = 3

        fpl1 = FlowpipeList()
        fpl1.flowpipes = [fp1, fp2, fp3]
        fpl1.flowpipenames = [ "1", "2", "3"]

        tp = TraceTree()
        tp.modeID = 1
        tp.branchID = -1
        tp.jumpsExecuted = -1

        pruned = fpl1.prune([tp])
        
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned.flowpipes[0].modeID, 1)
        self.assertEqual(pruned.flowpipes[0].branchID, -1)
        self.assertEqual(pruned.flowpipes[0].jumpsexecuted, -1)


if __name__ == '__main__':
    unittest.main()
