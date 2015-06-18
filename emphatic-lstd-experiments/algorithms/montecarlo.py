"""
Implementation of Monte Carlo methods for estimating Value Functions.
"""
import numpy as np 
from collections import defaultdict


# Better to not treat these as algorithms of the same form as online learning
# algorithms, instead just given them the list of episodes, have them process
# it in reverse.

class EveryVisitMC:
    """Every visit Monte Carlo"""
    def __init__(self, n):
        self.n = n
        self.ndct = {}

    def terminate(self):
        """End of episode termination."""
        pass

    @property 
    def theta(self):
        pass


    def update(self, fvec, reward):
        pass


class FirstVisitMC:
    """First visit Monte Carlo."""
    def __init__(self, n):
        self.n = n
        self.retdct = defaultdict(list)
        self.epdct = defaultdict(list)

    def terminate(self):
        """End of episode termination."""
        for fv, rewards in self.epdct.items():
            self.retdct[fv].append(sum(rewards))

        # reset for start of next episode
        self.epdct = defaultdict(list)

    @property 
    def theta(self):
        # least squares solution
        ns = len(self.retdct)
        A  = np.zeros((ns, self.n))
        b  = np.zeros(ns)

        for i, k in enumerate(self.retdct):
            A[i] = k
            b[i] = np.mean(self.retdct[k])

        return np.dot(np.linalg.pinv(A), b)


    def update(self, fvec, reward):
        fvec = tuple(fvec)
        if fvec not in self.epdct:
            self.epdct[fvec] = []

        # add rewards to all feature vectors seen so far
        for fv in self.epdct.keys():
            self.epdct[fv].append(reward)
        