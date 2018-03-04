import numpy as np


def accuracy(Y, Y_):
    """ calculate the one-hot vector accuracy

        Y: {one-hot-vector} ground-truth [ [1, 0], [0, 1] ]
        Y_: {one-hot vector} estimate [ [.8, .2], [.3, .7]]

        returns the accuracy of the prediction, the cut-off is made
        at 0.5

        return values are between (0 ... 1)
    """
    assert len(np.unique(Y)) == 2, 'Ground-truth data should only consist of 0-1'
    assert len(Y.shape) == 2 and len(Y_.shape) == 2, 'Y and Y_ must be matrices'
    n, m = Y.shape
    assert n == Y_.shape[0] and m == Y_.shape[1], 'Y and Y_ must have the same shape'
    assert n == np.sum(Y), 'Y does not represent a probability'
    assert n == np.sum(Y_), 'Y_ does not represent a probability'
    y = Y[:,0]
    y_ = (Y_[:,0] > 0.5) * 1
    total_same = np.sum((y == y_) * 1)
    return total_same/n
