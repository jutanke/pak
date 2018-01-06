import numpy as np
import numpy.linalg as la
from pppr import aabb
from pak.evaluation import MOTM


def evaluate_aabb(Gt, Hy, threshold, info=False):
    """ Ground-truth vs hypothesis for the
        Multiple Object Tracking Accuracy

        Gt: [
            (frame, pid, x, y, w, h)
        ]

        Hy: [
            (frame, pid, x, y, w, h)
        ]

        threshold: after which no correspondence is possible
        info: if info is True the FN, FP, and IDSW values are returned
              as well

        The objects are not points but axis-aligned bounding boxes and
        the distance is calculated by 1 - IoU

        The result values are in the range of [-infinity, 1)
    """
    cost_fun = lambda a, b: 1 - aabb.IoU(a, b)
    return _evaluate(Gt, Hy, threshold, cost_fun, info)
    

def evaluate(Gt, Hy, threshold, info=False):
    """ Ground-truth vs hypothesis for the
        Multiple Object Tracking Accuracy

        Gt: [
            (frame, pid, x, y)
        ]

        Hy: [
            (frame, pid, x, y)
        ]

        threshold: after which no correspondence is possible
        info: if info is True the FN, FP, and IDSW values are returned
              as well

        The result values are in the range of [-infinity, 1)
    """
    cost_fun = lambda a, b: la.norm(a-b)
    return _evaluate(Gt, Hy, threshold, cost_fun, info)


def _evaluate(Gt, Hy, threshold, cost_fun, info=False):
    fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, threshold, cost_fun)

    FN = np.sum(m)
    FP = np.sum(fp)
    IDSW = np.sum(mme)
    GT = np.sum(g)

    mota = 1 - (FN + FP + IDSW) / GT

    if info:
        return mota, {
            'FN': FN,
            'FP': FP,
            'IDSW': IDSW,
            'GT': GT
        }
    else:
        return mota
