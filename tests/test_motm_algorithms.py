import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from math import sqrt
import numpy.linalg as la

from pak.evaluation import MOTP, MOTA

class TestMOTM_algorithms(unittest.TestCase):


    def test_MOTA_aabb_idsw(self):
        Gt = np.array([
            [1, 1, 0, 0, 1, 1],
            [1, 2, 1, 1, 1, 1],
            [2, 1, 0, 0, 1, 1],
            [2, 2, 1, 1, 1, 1],
        ])
        Hy = np.array([
            [1, 2, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [2, 2, 0, 0, 1, 1],
            [2, 1, 1, 1, 1, 1]
        ])
        T = 0.5
        result, info = MOTA.evaluate_aabb(Gt, Hy, T, info=True)

        self.assertEqual(info['IDSW'], 2)
        self.assertEqual(result, 0.5)

    def test_MOTA_aabb_complete_lost(self):
        Gt = np.array([
            [1, 1, 0, 0, 0.4, 0.4],
            [1, 2, 10, 10, 0.4, 0.4],
            [2, 1, 0, 0, 0.4, 0.4]
        ])
        Hy = np.array([
            [1, 2, 10, 10, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [2, 1, 0, 0, 1, 1]
        ])
        T = 0.5

        result, info = MOTA.evaluate_aabb(Gt, Hy, T, info=True)
        self.assertEqual(result, -1)

    def test_simple_MOTA_aabb(self):
        Gt = np.array([
            [1, 1, 0, 0, 1, 1],
            [1, 2, 10, 10, 1, 1],
            [2, 1, 0, 0, 1, 1]
        ])
        Hy = np.array([
            [1, 2, 10, 10, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [2, 1, 0, 0, 1, 1]
        ])
        T = 0.5

        result = MOTA.evaluate_aabb(Gt, Hy, T)
        self.assertEqual(result, 1)

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
        self.assertEqual(result, 1)

    def test_simple_MOTA_with_info(self):
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

        result, info = MOTA.evaluate(Gt, Hy, T, info=True)
        self.assertEqual(result, 1)

    def test_simple_MOTA_with_info_and_debug(self):
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

        result, info, debug = MOTA.evaluate(Gt, Hy, T, info=True, debug_info=True)
        self.assertEqual(result, 1)
        self.assertEqual(len(debug['FN']), 0)
        self.assertEqual(len(debug['FP']), 0)
        self.assertEqual(len(debug['IDSW']), 0)

    def test_simple_MOTA_with_debug(self):
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

        result, debug = MOTA.evaluate(Gt, Hy, T, info=False, debug_info=True)
        self.assertEqual(result, 1)
        self.assertEqual(len(debug['FN']), 0)
        self.assertEqual(len(debug['FP']), 0)
        self.assertEqual(len(debug['IDSW']), 0)


    def test_simple_MOTA_with_info_with_FP(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [1, 2, 10, 10],
            [2, 1, 0, 0]
        ])
        Hy = np.array([
            [1, 2, 10, 10],
            [2, 2, 10, 10],
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])
        T = 1

        result, info = MOTA.evaluate(Gt, Hy, T, info=True)
        self.assertEqual(info['FP'], 1)
        self.assertEqual(info['FN'], 0)
        self.assertEqual(info['IDSW'], 0)

    def test_simple_MOTA_with_info_with_FN(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [1, 2, 10, 10],
            [2, 2, 10, 10],
            [2, 1, 0, 0]
        ])
        Hy = np.array([
            [1, 2, 10, 10],
            [1, 1, 0, 0],
            [2, 1, 0, 0]
        ])
        T = 1

        result, info = MOTA.evaluate(Gt, Hy, T, info=True)
        self.assertEqual(info['FP'], 0)
        self.assertEqual(info['FN'], 1)
        self.assertEqual(info['IDSW'], 0)

    def test_simple_MOTA_with_info_with_IDSW(self):
        Gt = np.array([
            [1, 1, 0, 0],
            [1, 2, 10, 10],
            [2, 1, 0, 0]
        ])
        Hy = np.array([
            [1, 2, 10, 10],
            [1, 1, 0, 0],
            [2, 2, 0, 0]
        ])
        T = 1

        result, info = MOTA.evaluate(Gt, Hy, T, info=True)
        self.assertEqual(info['FP'], 0)
        self.assertEqual(info['FN'], 0)
        self.assertEqual(info['IDSW'], 1)

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
