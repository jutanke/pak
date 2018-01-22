# ==== Multiple-Object-Tracking ====
#
# Given some tracking results we can apply some post-processing to make
# the results more smooth

import numpy as np
from pak import utils


def remove_short_tracks(X, min_length):
    """ very short tracklets are usually a sign of noisy detection and should
        be removed to get more stable results.
        ATTENTION: this function assumes that X has no duplicates! Please
            remove all duplicates prior to using this function

        X: {np.array} data with shape (n,m): n is the number of data points,
                m is the data dimension. The first element is the frame number,
                the second one the id. The rest is the data term

                e.g.: [
                    [frame, pid, ...DATA...],
                    ...
                ]

        min_length: {Integer} inclusive threshold after which a tracklet is being
                dropped
    """
    last_frame = int(np.max(X[:,0]))
    first_frame = int(np.min(X[:,0]))
    assert last_frame > 0 and first_frame > 0 and last_frame >= first_frame

    get_pid = lambda x: int(x[1])
    get_pid_from_tracklet = lambda lst: get_pid(lst[0])

    X_result = []

    # count the length of the current tracklet per id
    # structure:
    #       KEY:    pid
    #       DATA:   (occured {Boolean}, data {list})
    pid_tracklets = {}

    for frame in range(first_frame, last_frame):
        X_frame = utils.extract_eq(X, 0, frame)

        # reset the 'occurence' flag on all tracks
        for pid in pid_tracklets.keys():
            pid_tracklets[pid][0] = False

        if len(X_frame) > 0:
            lookup = set()
            for x in X_frame:
                pid = get_pid(x)
                assert pid not in lookup, "Frames MUST NOT have duplicate ids"
                lookup.add(pid)

                if pid in pid_tracklets:
                    pid_tracklets[pid][0] = True
                    pid_tracklets[pid][1].append(x)
                else:
                    pid_tracklets[pid] = [True, [x]]

        # check if a track was lost: if so see if we need to drop it or add it
        # to the True results
        drop_pids = []  # keys that will be dropped from current tracking
        for occured, tracklet in pid_tracklets.values():
            if not occured:
                drop_pids.append(get_pid_from_tracklet(tracklet))
                if len(tracklet) > min_length:
                    X_result.extend(tracklet)  # the tracklet is long enough

        for pid in drop_pids:
            del pid_tracklets[pid]


    # finish up all left-over tracks
    for _, tracklet in pid_tracklets.values():
        if len(tracklet) > min_length:
            X_result.extend(tracklet)

    return np.array(X_result)


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
    last_frame = int(np.max(X[:,0]))
    first_frame = int(np.min(X[:,0]))
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
