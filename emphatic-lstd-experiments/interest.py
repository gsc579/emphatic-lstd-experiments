"""
Implementations of different kinds of interest for emphatic reinforcement 
learning experiments.
"""
from itertools import accumulate, count

def start_indices(seqs):
    """
    Given a sequence of sequences, return the indices of the first element of 
    each sequence, were the sequences to be flattened.
    """
    ret = [0]
    ret.extend(accumulate(len(x) for x in seqs))
    return ret 




class FirstVisitInterest:
    def __init__(self, episodes):
        # Using sets might be non-ideal
        self.starts = set(accumulate(len(ep) for ep in episodes))
        self.starts.add(0) # Not necessary if reset() called in __init__()
        self._count = count()
        self.reset()

    def reset(self):
        """Reset at the end of an episode."""
        self.seen = set()

    def __call__(self, fvec):
        # should interest be zero in terminal states?
        if next(self._count) in self.starts:
            self.reset()

        fv = tuple(fvec)
        if fv in self.seen:
            return 0
        else:
            self.seen.add(fv)
            return 1


class StartStateInterest:
    def __init__(self, episodes):
        # Using sets might be non-ideal
        self.starts = set(accumulate(len(ep) for ep in episodes))
        self.starts.add(0) # Not necessary if reset() called in __init__()
        self._count = count()
        
    def __call__(self, fvec):
        if next(self._count) in self.starts:
            return 1
        else:
            return 0