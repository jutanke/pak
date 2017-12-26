import unittest
import sys
sys.path.insert(0, '../')

from pak.evaluation import MOTP

class TestEvaluation(unittest.TestCase):

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

        self.assertTrue(True)


# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
