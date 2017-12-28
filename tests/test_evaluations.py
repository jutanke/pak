import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from math import sqrt
import numpy.linalg as la

from pak.evaluation import MOTP, MOTM

class TestEvaluation(unittest.TestCase):

    def test_motm(self):
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
        self.assertEqual(np.sum(g), 2)
        self.assertEqual(np.sum(c), 2)


    def test_simple_motp(self):
        """ simple MOTP
        """

        # (frame, pid, x, y, w, h)
        Gt = np.array([
            [1, 1, 10, 10, 10, 10],
            [2, 1, 10, 10, 10, 10],
            [3, 1, 10, 10, 10, 10],
            [4, 1, 10, 10, 10, 10],
            [5, 1, 10, 10, 10, 10]
        ])

        Hy = np.array([
            [1, 5, 10, 10, 10, 10],
            [2, 5, 10, 10, 10, 10],
            [3, 5, 10, 10, 10, 10],
            [4, 5, 10, 10, 10, 10],
            [5, 5, 10, 10, 10, 10]
        ])


        MOTP.evaluate(Gt, Hy, 2)

        self.assertTrue(True)


# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
