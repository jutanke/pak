import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from math import sqrt
import numpy.linalg as la

from pak.evaluation import MOTM

class TestEvaluation(unittest.TestCase):

    def test_motm(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [1, 2, 10, 10],
            [2, 1, 0, 0]
        ])

        Hy = np.array([
            [1, 2, 10, 10],
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])

        cost = lambda a, b: la.norm(a-b)

        fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, 10, cost)

        self.assertEqual(len(fp), 2)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(mme), 2)
        self.assertEqual(len(c), 2)
        self.assertEqual(len(d), 2)
        self.assertEqual(len(g), 2)

        self.assertEqual(np.sum(fp), 0)
        self.assertEqual(np.sum(m), 0)
        self.assertEqual(np.sum(mme), 0)
        self.assertEqual(np.sum(d), 0)
        self.assertEqual(np.sum(g), 3)
        self.assertEqual(np.sum(c), 3)

    def test_motm_correct_hyp_1elem(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])

        Hy = np.array([
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])

        cost = lambda a, b: la.norm(a-b)

        fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, 10, cost)

        self.assertEqual(len(fp), 2)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(mme), 2)
        self.assertEqual(len(c), 2)
        self.assertEqual(len(d), 2)
        self.assertEqual(len(g), 2)

        self.assertEqual(np.sum(fp), 0)
        self.assertEqual(np.sum(m), 0)
        self.assertEqual(np.sum(mme), 0)
        self.assertEqual(np.sum(d), 0)
        self.assertEqual(np.sum(g), 2)
        self.assertEqual(np.sum(c), 2)

    def test_motm_wrong_hyp(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])

        Hy = np.array([
            [1, 1, 0, 0],
            [1, 2, 10, 0],
            [2, 1, 0, 0]
        ])

        cost = lambda a, b: la.norm(a-b)

        fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, 10, cost)

        self.assertEqual(len(fp), 2)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(mme), 2)
        self.assertEqual(len(c), 2)
        self.assertEqual(len(d), 2)
        self.assertEqual(len(g), 2)

        self.assertEqual(np.sum(fp), 1)
        self.assertEqual(np.sum(m), 0)
        self.assertEqual(np.sum(mme), 0)
        self.assertEqual(np.sum(d), 0)
        self.assertEqual(np.sum(g), 2)
        self.assertEqual(np.sum(c), 2)

    def test_motm_1miss(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [1, 2, 10, 0],
            [2, 1, 0, 0]
        ])

        Hy = np.array([
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])

        cost = lambda a, b: la.norm(a-b)

        fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, 10, cost)

        self.assertEqual(len(fp), 2)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(mme), 2)
        self.assertEqual(len(c), 2)
        self.assertEqual(len(d), 2)
        self.assertEqual(len(g), 2)

        self.assertEqual(np.sum(fp), 0)
        self.assertEqual(np.sum(m), 1)
        self.assertEqual(np.sum(mme), 0)
        self.assertEqual(np.sum(d), 0)
        self.assertEqual(np.sum(g), 3)
        self.assertEqual(np.sum(c), 2)

    def test_motm_1mme(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])

        Hy = np.array([
            [1, 1, 0, 0],
            [2, 2, 0, 0]
        ])

        cost = lambda a, b: la.norm(a-b)

        fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, 10, cost)

        self.assertEqual(len(fp), 2)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(mme), 2)
        self.assertEqual(len(c), 2)
        self.assertEqual(len(d), 2)
        self.assertEqual(len(g), 2)

        self.assertEqual(np.sum(fp), 0)
        self.assertEqual(np.sum(m), 0)
        self.assertEqual(np.sum(mme), 1)
        self.assertEqual(np.sum(d), 0)
        self.assertEqual(np.sum(g), 2)
        self.assertEqual(np.sum(c), 2)


    def test_motm_cross(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [1, 2, 2, 2],
            [2, 1, 1, 1],
            [2, 2, 1, 1],
            [3, 1, 2, 2],
            [3, 2, 0, 0]
        ])

        Hy = np.array([
            [1, 2, 2, 2],
            [1, 1, 0, 0],
            [2, 2, 1, 1],
            [2, 1, 1, 1],
            [3, 1, 2, 2],
            [3, 2, 0, 0]
        ])

        cost = lambda a, b: la.norm(a-b)

        fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, 1.3, cost)

        self.assertEqual(len(fp), 3)
        self.assertEqual(len(m), 3)
        self.assertEqual(len(mme), 3)
        self.assertEqual(len(c), 3)
        self.assertEqual(len(d), 3)
        self.assertEqual(len(g), 3)

        self.assertEqual(np.sum(fp), 0)
        self.assertEqual(np.sum(m), 0)
        self.assertEqual(np.sum(mme), 0)
        self.assertEqual(np.sum(d), 0)
        self.assertEqual(np.sum(g), 6)
        self.assertEqual(np.sum(c), 6)


    def test_motm_interrupt(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [2, 1, 0, 0],
            [5, 1, 0, 0],
            [6, 1, 0, 0],
            [7, 1, 0, 0],
            [8, 1, 0, 0]
        ])

        Hy = np.array([
            [1, 1, 0, 0],
            [2, 1, 0, 0],
            [5, 1, 0, 0],
            [6, 1, 0, 0],
            [7, 1, 0, 0],
            [8, 1, 0, 0]
        ])

        cost = lambda a, b: la.norm(a-b)

        fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, 10, cost)

        self.assertEqual(len(fp), 8)
        self.assertEqual(len(m), 8)
        self.assertEqual(len(mme), 8)
        self.assertEqual(len(c), 8)
        self.assertEqual(len(d), 8)
        self.assertEqual(len(g), 8)

        self.assertEqual(np.sum(fp), 0)
        self.assertEqual(np.sum(m), 0)
        self.assertEqual(np.sum(mme), 0)
        self.assertEqual(np.sum(d), 0)
        self.assertEqual(np.sum(g), 6)
        self.assertEqual(np.sum(c), 6)


# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
