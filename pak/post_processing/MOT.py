# ==== Multiple-Object-Tracking ====
#
# Given some tracking results we can apply some post-processing to make
# the results more smooth

import numpy as np
from pak import utils


def remove_duplicates(X, score_fun):
    """ this function runs over the data and selects the "best" (according to the
        score-function) for each frame per class

        X: {np.array} data with shape (n,m): n is the number of data points,
                m is the data dimension. The first element is the frame number,
                the second one the id. The rest is the data term

                e.g.: [
                    [frame, pid, ...DATA...],
                    ...
                ]

        score_fun: {function} gets a single data point and evaluates a score. The
                data point with the largest score is kept while all others are
                dropped

        returns {np.array} with the same structure as X but with only a single
                id per frame.
    """
    last_frame = np.max(X[:,0])
    first_frame = np.min(X[:,0])
    assert last_frame > 0 and first_frame > 0 and last_frame >= first_frame

    X_result = []
    for frame in range(first_frame, last_frame+1):
        X_frame = utils.extract_eq(X, 0, frame)
        if len(X_frame) > 0:
            lookup = {}
            for x in X_frame:
                pid, score = x[1], score_fun(x)
                if pid in lookup:
                    if score > lookup[pid][0]:
                        lookup[pid] = (score, x)
                else:
                    lookup[pid] = (score, x)

            for pid, x in lookup.values():
                X_result.append(x)

    return np.array(X_result)
