# Multiple Object Tracking Metrics
# This is the basic algorithm for most
# Multiple-Object tracking Metrics
import numpy as np
from pak.utils import extract_eq
from math import ceil, floor
from scipy.optimize import linear_sum_assignment

HIGH_VALUE = 999999999

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
            d: List<Double> of distances beween o_i and h_i (summed)
            g: List<Integer> number of objects in t

            all lists have the same length of total number of
            frames
    """
    global HIGH_VALUE
    first_frame = np.min(Gt[:,0])
    last_frame = np.max(Gt[:,0])
    assert floor(first_frame) == ceil(first_frame)  # make sure the
    assert floor(last_frame) == ceil(last_frame)    # values are integers
    first_frame = int(first_frame)
    last_frame = int(last_frame)

    number_of_frames = last_frame - first_frame + 1

    fp = [0] * number_of_frames
    m = [0] * number_of_frames
    mme = [0] * number_of_frames
    c = [0] * number_of_frames
    d = [0] * number_of_frames
    g = [0] * number_of_frames

    M = MatchLookup(first_frame, last_frame)

    # ----------------------------
    # "t" is the true frame number that can be any integer
    # "t_pos" is the respective integer that starts at 0 ...
    for t_pos, t in enumerate(range(first_frame, last_frame + 1)):
        Ot = SingleFrameData(extract_eq(Gt, col=0, value=t))
        Ht = SingleFrameData(extract_eq(Hy, col=0, value=t))
        g[t_pos] = Ot.total_elements  # count number of objects in t

        # ----------------------------------
        # verify if old match is still valid!
        # ----------------------------------
        is_empty = True
        for (o, h) in M.get_matches(t-1):
            oid, hid = o[0], h[0]
            if Ot.has(oid) and Ht.has(hid):
                o_cur = Ot.find(oid)
                h_cur = Ht.find_best(hid, o_cur, calc_cost)
                cost = calc_cost(o_cur[1:], h_cur[1:])
                if cost < T:
                    # the tracked object is still valid! :)
                    Ot.remove(o_cur)
                    Ht.remove(h_cur)
                    M.insert_match(t, o_cur, h_cur)
                    is_empty = False

        if is_empty:
            M.init(t)  # to allow frames with no matches!

        # ----------------------------------
        # match not-yet corresponding pairs
        # ----------------------------------
        Ot_ummatched = Ot.as_list()  # the already matched elements
        Ht_unmatched = Ht.as_list()  # were removed
        count_o, count_h = len(Ot_ummatched), len(Ht_unmatched)
        C = np.ones((count_o, count_h)) * HIGH_VALUE
        for i,o in enumerate(Ot_ummatched):
            for j,h in enumerate(Ht_unmatched):
                cost = calc_cost(o[1:], h[1:])
                C[i,j] = cost if cost < T else HIGH_VALUE

        row_ind, col_ind = linear_sum_assignment(C)

        for i, j in zip(row_ind, col_ind):
            o_cur, h_cur, cost = Ot_ummatched[j], Ht_unmatched[i], C[i,j]
            if cost < T:
                Ot.remove(o_cur)
                Ht.remove(h_cur)
                M.insert_match(t, o_cur, h_cur)
                if M.has_mismatch(t, o_cur, h_cur):
                    mme[t_pos] += 1

        # ----------------------------------
        # handle unmatched rest
        # ----------------------------------
        c[t_pos] = M.count_matches(t)
        fp[t_pos] = Ht.elements_left
        assert fp[t_pos] >= 0
        m[t_pos] = Ot.elements_left
        assert m[t_pos] >= 0

        # ----------------------------------
        # calculate cost between all matches
        # ----------------------------------
        cost_sum = 0
        for (o, h) in M.get_matches(t):
            cost_sum += calc_cost(o[1:], h[1:])
        d[t_pos] = cost_sum

    # ----------------------------
    return fp, m, mme, c, d, g


# =============================================
# Helper data structures
# =============================================
# ---
class SingleFrameData:
    """ handles the data for a single frame
        Data: [
            (frame, pid, ..DATA..),
            ...
        ]
    """
    def __init__(self, data):
        self.lookup = {}

        if len(data) == 0:
            n = 0
        else:
            assert len(np.unique(data[:,0])) == 1  # ensure single frame
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


    def has(self, pid):
        """ Tests if the dataframe has the given pid or not
        """
        return pid in self.lookup


    def find(self, pid):
        """ find the given object
        """
        if pid in self.lookup:
            assert len(self.lookup[pid]) == 1
            return self.lookup[pid][0]
        else:
            return None


    def find_best(self, pid, target, cost_fun):
        """ finds the best object with given pid for the target
            target: [pid, ..DATA..]
        """
        global HIGH_VALUE
        if pid in self.lookup:
            A = self.lookup[pid]
            if len(A) > 1:
                lowest_cost = HIGH_VALUE
                lowest = None
                a = target[1:]
                for other in A:
                    b = other[1:]
                    cost = cost_fun(a,b)
                    if cost < lowest_cost:
                        lowest_cost = cost
                        lowest = other

                assert lowest is not None
                return lowest
            else:
                return A[0]
        else:
            return None


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
        self.first_frame = int(first_frame)
        self.last_frame = int(last_frame)
        number_of_frames = last_frame - first_frame + 1
        self.matches = [None] * number_of_frames
        self.lookups = [None] * number_of_frames
        self.count_matches_in_t = [0] * number_of_frames


    def init(self, t):
        """ initializes the set in the case of no matches

            The function that accessing un-initialized lookups
            is considered an error is INTENTIONAL to hopefully
            prevent hard-to-debug logic-bugs
        """
        pos = t - self.first_frame
        assert self.matches[pos] is None
        self.matches[pos] = []

        assert self.lookups[pos] is None
        self.lookups[pos] = {}


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
        self.count_matches_in_t[pos] += 1


    def count_matches(self, t):
        """ counts the number of matches in frame t
        """
        pos = t - self.first_frame
        return self.count_matches_in_t[pos]


    def has_mismatch(self, t, o, h):
        """ checks if there is a mismatch in the t-1 time stemp
            "o" and "h" are samples from the t time step
        """
        if t == self.first_frame:
            return False  # first frame can never mismatch
        assert t <= self.last_frame
        assert t > self.first_frame
        t_prev = t-1
        pos = t_prev - self.first_frame

        oid = o[0]
        hid = h[0]
        if oid in self.lookups[pos]:
            return self.lookups[pos][oid] != hid


    def get_matches(self, t):
        """ gets all matches at frame t
        """
        if t == (self.first_frame - 1):
            return []
        assert t <= self.last_frame
        assert t >= self.first_frame
        pos = t - self.first_frame
        result = self.matches[pos]
        assert result is not None
        return result

# ----
