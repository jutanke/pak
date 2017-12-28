# Multiple Object Tracking Metrics
# This is the basic algorithm for most
# Multiple-Object tracking Metrics
import numpy as np
from pak.utils import extract_eq
from scipy.optimize import linear_sum_assignment

def evaluate(Gt, Hy, T, calc_cost):
    """ Runs the Multiple Object Tracking Metrics algorithm

        Gt: Ground-truth: [
            [frame, pid, ..DATA.. ],
            ...
        ]

        Hy: hypothesis: [
            [frame, pid, ..DATA.. ],
            ...
        ]

        T: cost threshold after which pairs cannot be
            connected anymore

        calc_cost: {function} that gets the ..DATA.. term
            as parameter: e.g. for points it could calculate
            the distance, for aabb's it could calculate IoU..

        return:
            fp: List<Integer> of false-positives
            m: List<Integer> of misses
            mme: List<Integer> of mismatches
            c: List<Integer> of matches
            d: List<Object> of distances beween o_i and h_j
            g: List<Integer> number of objects in t

            all lists have the same length of total number of
            frames
    """
    first_frame = np.min(Gt[:,0])
    last_frame = np.max(Gt[:,0])
    number_of_frames = last_frame - first_frame + 1

    fp = [0] * number_of_frames
    m = [0] * number_of_frames
    mme = [0] * number_of_frames
    c = [0] * number_of_frames
    d = [None] * number_of_frames
    g = [0] * number_of_frames

    M = [{}]

    # ----------------------------
    # "t" is the true frame number that can be any integer
    # "t_pos" is the respective integer that starts at 0 ...
    for t_pos, t in enumerate(range(first_frame, last_frame + 1)):
        Ot = extract_eq(Gt, col=0, value=t)
        Ht = extract_eq(Hy, col=0, value=t)
        g[t_pos], _ = Ot.shape  # count number of objects in t




    # ----------------------------
    return fp, m, mme, c, d, g


# ---
class MatchLookup:
    """ Lookup for the Matches
    """

    def __init__(self, first_frame, last_frame):
        self.first_frame = first_frame
        self.last_frame = last_frame
        number_of_frames = last_frame - first_frame + 1
        self.matches = [None] * number_of_frames
        self.lookups = [None] * number_of_frames


    def insert_match(self, t, o, h):
        """ insert a (o,h) match

            o: [pid, ..DATA..]
            h: [pid, ..DATA..]
        """
        assert t <= self.last_frame
        assert t >= self.first_frame
        if self.matches[t] is None:
            self.matches[t] = []

        if self.lookups[t] is None:
            # this is needed so that we can make
            # sure if we had a mismatch or not!
            self.lookups[t] = {}

        assert o[0] not in self.lookups[t]
        self.matches[t].append((o, h))
        self.lookups[t][o[0]] = h[0]


    def has_mismatch(self, t, o, h):
        """ checks if there is a mismatch in the t-1 time stemp
            "o" and "h" are samples from the t time step
        """
        if t == self.first_frame:
            return False  # first frame can never have mismatch
        assert t <= self.last_frame
        assert t > self.first_frame
        t_prev = t-1

        if o[0] in self.lookups[t_prev]:
            return self.lookups[t_prev][0] != h[0]


    def get_matches(self, t):
        """ gets all matches at frame t
        """
        if t == (self.first_frame - 1):
            return []
        assert t <= self.last_frame
        assert t >= self.first_frame
        assert self.matches[t] is not None
        return self.matches[t]

# ----
