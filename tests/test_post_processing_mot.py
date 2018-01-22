import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
import numpy.linalg as la

from pak.post_processing import MOT

class TestMOT_postProcessing(unittest.TestCase):


    def test_remove_short_tracks_with_interrupt(self):
        X = np.array([
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 6],
            [3, 4],
            [4, 4],
            [4, 6],
            [5, 1],
            [5, 4],
            [5, 6],
            [6, 1],
            [6, 4],
            [7, 1],
            [7, 4],
            [8, 4]
        ])

        X_hat = MOT.remove_short_tracks(X, min_length=2)

        unique_pids = np.unique(X_hat[:,1])

        self.assertEqual(len(unique_pids), 3)
        self.assertEqual(unique_pids[0], 1)
        self.assertEqual(unique_pids[1], 4)
        self.assertEqual(unique_pids[2], 6)
        self.assertEqual(len(X_hat), 6 + 6 + 3)


    def test_remove_short_tracks(self):
        X = np.array([
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 6],
            [3, 4],
            [4, 1],
            [4, 4],
            [4, 6],
            [5, 1],
            [5, 4],
            [5, 6],
            [6, 1],
            [6, 4],
            [7, 1],
            [7, 4],
            [8, 4]
        ])

        X_hat = MOT.remove_short_tracks(X, min_length=3)

        unique_pids = np.unique(X_hat[:,1])

        self.assertEqual(len(unique_pids), 2)
        self.assertEqual(unique_pids[0], 1)
        self.assertEqual(unique_pids[1], 4)
        self.assertEqual(len(X_hat), 7 + 6)  # 1th plus 4th


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
