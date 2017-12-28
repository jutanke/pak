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

        self.assertFalse(M.has_mismatch(1, o, h))

        o_, h_ = M.get_matches(1)[0]
        self.assertEqual(o_[0], 1)
        self.assertEqual(h_[0], 99)

        o,h = (1, None), (99, None)
        M.insert_match(2, o, h)

        self.assertFalse(M.has_mismatch(2, o, h))

    def test_counting(self):
        M = MOTM.MatchLookup(1,2)

        o,h = (1, None), (95, None)
        M.insert_match(1, o, h)

        o,h = (2, None), (96, None)
        M.insert_match(1, o, h)

        o,h = (3, None), (97, None)
        M.insert_match(1, o, h)

        o,h = (3, None), (97, None)
        M.insert_match(2, o, h)

        self.assertEqual(M.count_matches(1), 3)
        self.assertEqual(M.count_matches(2), 1)

    def test_mismatch(self):
        M = MOTM.MatchLookup(1,2)
        o,h = (1, None), (99, None)
        M.insert_match(1, o, h)
        o,h = (1, None), (88, None)
        M.insert_match(2, o, h)
        self.assertFalse(M.has_mismatch(1, o, h))
        self.assertTrue(M.has_mismatch(2, o, h))

        self.assertEqual(M.count_matches(1), 1)
        self.assertEqual(M.count_matches(2), 1)

    def test_multiple_inserts(self):
        M = MOTM.MatchLookup(1,2)
        o,h = (1, None), (95, None)
        M.insert_match(1, o, h)
        o,h = (2, None), (96, None)
        M.insert_match(1, o, h)
        o,h = (3, None), (97, None)
        M.insert_match(1, o, h)

        self.assertEqual(len(M.get_matches(1)), 3)

        o,h = (2, None), (96, None)
        M.insert_match(2, o, h)
        o,h = (3, None), (97, None)
        M.insert_match(2, o, h)

        self.assertEqual(len(M.get_matches(2)), 2)



# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()
