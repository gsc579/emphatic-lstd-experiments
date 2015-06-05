"""
Emphatic least-squares temporal difference learning implementation. 
"""
import numpy as np 

class ELSTD:
    """Emphatic least-squares temporal difference learning. """
    def __init__(self, n, epsilon=0, **kwargs):
        self.n = n
        self.z = np.zeros(n)
        self.A = np.eye(n) * epsilon
        self.b = np.zeros(n)
        self.F = 0
        self.M = 0

    def reset(self):
        """Perform end-of-episode reset."""
        self.z[:] = 0
        self.F = 0
        self.M = 0

    @property
    def theta(self):
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
        return _theta

    def update(self, fvec, reward, fvec_p, gm, gm_p, lm, I):
        self.F = gm * self.F + I
        self.M = (lm * I) + ((1 - lm) * self.F)
        self.z = (gm * lm * self.z + self.M * fvec)
        self.A += np.outer(self.z, (fvec - gm_p * fvec_p))
        self.b += self.z * reward