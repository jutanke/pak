import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
import numpy.linalg as la

from pak.post_processing import MOT

class TestMOT_postProcessing(unittest.TestCase):


    def test_generic_MOT_simple(self):

        X = np.array([
            (1, 1, 10),
            (1, 1, 3),
            (1, 1, 99),
            (1, 2, 1),
            (1, 2, 0),
            (2, 2, 4),
            (2, 2, 3),
            (2, 1, 1),
            (2, 1, 99)
        ])

        X_hat = MOT.remove_duplicates(X, lambda x: x[2])

        self.assertEqual(X_hat.shape[0], 4)
        self.assertEqual(X_hat[0,2], 99)
        self.assertEqual(X_hat[1,2], 1)
        self.assertEqual(X_hat[2,2], 4)
        self.assertEqual(X_hat[3,2], 99)




# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
