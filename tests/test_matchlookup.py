import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from math import sqrt
import numpy.linalg as la

from pak.evaluation import MOTM

class TestMatchLookup(unittest.TestCase):

    def test_insert(self):
        M = MOTM.MatchLookup(1, 2)
        self.assertEqual(len(M.get_matches(0)), 0)

        o,h = (1, None), (99, None)
        M.insert_match(1, o, h)
        self.assertEqual(len(M.get_matches(1)), 1)

# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
