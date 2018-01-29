import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from pak.evaluation import MOTM

class TestLocalToGlobal(unittest.TestCase):

    def test_basics(self):
        Data = np.array([
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2)
        ])

        L = MOTM.LocalFrameIdToGlobal(Data)

        self.assertEqual(L.get_true_idx(1, 0), 0)
        self.assertEqual(L.get_true_idx(1, 2), 2)
        self.assertEqual(L.get_true_idx(3, 1), 8)


    def test_out_of_order(self):
        Data = np.array([
            (2, 3),
            (1, 2),
            (3, 1),
            (1, 4),
            (1, 3),
            (2, 2),
            (1, 1),
            (3, 2),
            (2, 1)
        ])

        L = MOTM.LocalFrameIdToGlobal(Data)

        self.assertEqual(L.get_true_idx(2, 0), 0)
        self.assertEqual(L.get_true_idx(1, 2), 4)
        self.assertEqual(L.get_true_idx(3, 1), 7)




# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
