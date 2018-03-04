import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from pak.evaluation import one_hot_classification as ohc


class TestClassification(unittest.TestCase):

    def test_ohc_simple_float(self):
        Y = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        Y_ = np.array([
            [0.8, 0.2],
            [0.4, 0.6],
            [0.55, 0.45]
        ])
        acc = ohc.accuracy(Y, Y_)
        self.assertEqual(acc, 1)

    def test_ohc_simple(self):
        Y = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        Y_ = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        acc = ohc.accuracy(Y, Y_)
        self.assertEqual(acc, 1)

    def test_ohc_simple0(self):
        Y = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        Y_ = np.array([
            [0, 1],
            [1, 0],
            [0, 1]
        ])
        acc = ohc.accuracy(Y, Y_)
        self.assertEqual(acc, 0)

    def test_ohc_simple_half(self):
        Y = np.array([
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ])
        Y_ = np.array([
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1]
        ])
        acc = ohc.accuracy(Y, Y_)
        self.assertEqual(acc, 0.5)


# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
