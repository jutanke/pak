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

    M = MatchLookup(first_frame, last_frame)

    # ----------------------------
    def get_best_h(Ht, hid, o):
        """ tries to find the best hypothesis
            for the given observation
        """
        h_candidates = extract_eq(Ht, col=1, value=oid)
        if len(h_candidates) == 0:
            return []
        elif len(h_candidates) == 1:
            return np.squeeze(h_candidates)[1:]
        else:
            best_cost = 9999999999
            for h in h_candidates:
                pass

    # ----------------------------
    # "t" is the true frame number that can be any integer
    # "t_pos" is the respective integer that starts at 0 ...
    for t_pos, t in enumerate(range(first_frame, last_frame + 1)):
        Ot = extract_eq(Gt, col=0, value=t)
        Ht = extract_eq(Hy, col=0, value=t)
        g[t_pos], _ = Ot.shape  # count number of objects in t

        # ----------------------------------
        # verify if old match is still valid!
        # ----------------------------------
        for (o, h) in M.get_matches(t-1):
            oid, hid = o[0], h[0]
            o_cur = extract_eq(Ot, col=1, value=oid)
            assert len(o_cur) < 2
            if len(o_cur) == 1:
                # o also exists in the current frame..
                # ... keep going
                o_cur = np.squeeze(o_cur)[1:]



        print(Ot[0][1:])



    # ----------------------------
    return fp, m, mme, c, d, g


# ---
class SingleFrameData:
    """ handles the data for a single frame
        Data: [
            (frame, pid, ..DATA..),
            ...
        ]
    """
    def __init__(self, data):
        assert len(np.unique(data[:,0])) == 1  # ensure single frame
        self.lookup = {}

        n, _ = data.shape
        self.elements_left = n
        self.total_elements = n

        for d in data:
            dobj = d[1:]
            pid = d[1]
            if pid in self.lookup:
                self.lookup[pid].append(dobj)
            else:
                self.lookup[pid] = [dobj]


    def remove(self, o):
        """
        o: [pid, ..DATA..]
        """
        pid = o[0]
        assert pid in self.lookup
        self.elements_left -= 1  # count-down
        if len(self.lookup[pid]) > 1:
            # find closest and delete
            delete_index = -1
            for i, other in enumerate(self.lookup[pid]):
                delete_index = i
                for left,right in zip(o, other):  # check all positions
                    if left != right:
                        delete_index = -1
                        break
                if delete_index > -1:
                    break
            assert delete_index > -1
            self.lookup[pid].pop(delete_index)
        else:
            del self.lookup[pid]


    def as_list(self):
        """ returns the rest of the elements as list
        """
        #TODO this function is very inefficient
        result = []
        for key in self.lookup.keys():
            for item in self.lookup[key]:
                result.append(item)
        return result


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
        pos = t - self.first_frame
        if self.matches[pos] is None:
            self.matches[pos] = []

        if self.lookups[pos] is None:
            # this is needed so that we can make
            # sure if we had a mismatch or not!
            self.lookups[pos] = {}

        assert o[0] not in self.lookups[pos]
        self.matches[pos].append((o, h))
        self.lookups[pos][o[0]] = h[0]


    def has_mismatch(self, t, o, h):
        """ checks if there is a mismatch in the t-1 time stemp
            "o" and "h" are samples from the t time step
        """
        if t == self.first_frame:
            return False  # first frame can never have mismatch
        assert t <= self.last_frame
        assert t > self.first_frame
        t_prev = t-1
        pos = t_prev - self.first_frame

        if o[0] in self.lookups[pos]:
            return self.lookups[pos][o[0]] != h[0]


    def get_matches(self, t):
        """ gets all matches at frame t
        """
        if t == (self.first_frame - 1):
            return []
        assert t <= self.last_frame
        assert t >= self.first_frame
        pos = t - self.first_frame
        assert self.matches[pos] is not None
        return self.matches[pos]

# ----
