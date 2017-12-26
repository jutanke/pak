import numpy as np
from pak.utils import extract_eq
from scipy.optimize import linear_sum_assignment

# https://github.com/justayak/pppr
from pppr import aabb



def evaluate(gt, hypothesis, threshold):
    """ Ground-truth vs hypothesis for the
        Multiple Object Tracking Precision

        gt: [
            (frame, pid, x, y, w, h)
        ]

        hypothesis: [
            (frame, pid, x, y, w, h)
        ]

        threshold: after which no correspondence is possible
    """
    pass
