import numpy as np
import numpy.linalg as la
from pak.evaluation import MOTM

def evaluate(Gt, Hy, threshold):
    """ Ground-truth vs hypothesis for the
        Multiple Object Tracking Accuracy

        Gt: [
            (frame, pid, x, y)
        ]

        Hy: [
            (frame, pid, x, y)
        ]

        threshold: after which no correspondence is possible


        The result values are in the range of [-infinity, 1)
    """
    cost_fun = lambda a, b: la.norm(a-b)
    fp, m, mme, c, d, g = MOTM.evaluate(Gt, Hy, threshold, cost_fun)

    FN = np.sum(m)
    FP = np.sum(fp)
    IDSW = np.sum(mme)
    GT = np.sum(g)

    return 1 - (FN + FP + IDSW) / GT
