import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from math import sqrt
import numpy.linalg as la

from pak.evaluation import MOTP, MOTA

class TestMOTM_algorithms(unittest.TestCase):

    def test_simple_MOTA_small_dev(self):
        Gt = np.array([
            [1, 1, 0.1, -0.1],
            [1, 2, 9.7, 10.1],
            [2, 1, 0.2, 0.3]
        ])
        Hy = np.array([
            [1, 2, 10, 10],
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])
        T = 1

        result = MOTA.evaluate(Gt, Hy, T)
        print('RES', result)
        #self.assertEqual(result, 0, 0.)

    def test_simple_MOTA(self):
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
        T = 1

        result = MOTA.evaluate(Gt, Hy, T)
        self.assertEqual(result, 1)

    def test_simple_MOTA_different_ids(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [1, 2, 10, 10],
            [2, 1, 0, 0]
        ])
        Hy = np.array([
            [1, 20, 10, 10],
            [1, 10, 0, 0],
            [2, 10, 0, 0]
        ])
        T = 1

        result = MOTA.evaluate(Gt, Hy, T)
        self.assertEqual(result, 1)

    def test_simple_MOTP(self):
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
        T = 1

        result = MOTP.evaluate(Gt, Hy, T)
        self.assertEqual(result, 0)

    def test_simple_MOTP_small_dev(self):
        Gt = np.array([
            [1, 1, 0.1, -0.1],
            [1, 2, 9.7, 10.1],
            [2, 1, 0.2, 0.3]
        ])
        Hy = np.array([
            [1, 2, 10, 10],
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])
        T = 1

        result = MOTP.evaluate(Gt, Hy, T)
        #self.assertEqual(result, 0, 0.)


# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
