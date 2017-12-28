import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from math import sqrt
import numpy.linalg as la

from pak.evaluation import MOTM

class TestSingleFrameData(unittest.TestCase):

    def test_empty(self):
        Data = np.array([])

        data = MOTM.SingleFrameData(Data)

        self.assertEqual(data.total_elements, 0)
        self.assertEqual(data.elements_left, 0)
        A = data.as_list()
        self.assertEqual(len(A), 0)


    def test_setup(self):
        Data = np.array([
            [1, 1, None],
            [1, 2, None],
            [1, 3, None],
            [1, 4, None],
        ])

        data = MOTM.SingleFrameData(Data)

        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 4)


    def test_complex_find(self):
        cost_fun = lambda a, b: abs(a-b)
        Data = np.array([
            [1, 1, 10],
            [1, 2, 12],
            [1, 2, 100],
            [1, 2, -19],
        ])

        data = MOTM.SingleFrameData(Data)
        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 4)

        obj = data.find_best(2, [1,10], cost_fun)
        self.assertEqual(obj[0], 2)
        self.assertEqual(obj[1], 12)


    def test_find(self):
        Data = np.array([
            [1, 1, 66],
            [1, 2, 77],
            [1, 3, 88],
            [1, 4, 99],
        ])

        data = MOTM.SingleFrameData(Data)

        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 4)

        obj = data.find(1)
        self.assertEqual(obj[0], 1)
        self.assertEqual(obj[1], 66)

        obj = data.find(2)
        self.assertEqual(obj[0], 2)
        self.assertEqual(obj[1], 77)


        obj = data.find(88)
        self.assertEqual(obj, None)



    def test_multiple_delete(self):
        Data = np.array([
            [1, 1, 99],
            [1, 2, 88],
            [1, 2, 99],
            [1, 3, 77],
            [1, 2, 77]
        ])

        data = MOTM.SingleFrameData(Data)
        self.assertEqual(data.total_elements, 5)
        self.assertEqual(data.elements_left, 5)

        data.remove([2,99])
        self.assertEqual(data.elements_left, 4)

        data.remove([1, 99])
        self.assertEqual(data.elements_left, 3)

        A = data.as_list()
        self.assertEqual(len(A), 3)
        self.assertEqual(A[0][0], 2)
        self.assertEqual(A[0][1], 88)
        self.assertEqual(A[1][0], 2)
        self.assertEqual(A[1][1], 77)
        self.assertEqual(A[2][0], 3)
        self.assertEqual(A[2][1], 77)


    def test_listing(self):
        Data = np.array([
            [1, 1, None],
            [1, 2, None],
            [1, 3, None],
            [1, 4, None],
        ])

        data = MOTM.SingleFrameData(Data)

        data.remove([3,None])
        data.remove([2,None])

        A = data.as_list()
        self.assertEqual(len(A), 2)
        self.assertEqual(A[0][0], 1)
        self.assertEqual(A[1][0], 4)


    def test_delete(self):
        Data = np.array([
            [1, 1, None],
            [1, 2, None],
            [1, 3, None],
            [1, 4, None],
        ])

        data = MOTM.SingleFrameData(Data)

        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 4)

        data.remove([1,None])
        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 3)

        data.remove([4,None])
        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 2)

        data.remove([3,None])
        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 1)

        data.remove([2,None])
        self.assertEqual(data.total_elements, 4)
        self.assertEqual(data.elements_left, 0)

        self.assertEqual(len(data.as_list()), 0)

# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
