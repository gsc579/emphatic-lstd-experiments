#!python3
"""
Least Squares Temporal Difference Learning.
"""
import numpy as np 


class LSTD():
    def __init__(self, n, epsilon=0):
        self.n  = n                         # number of features
        self.t  = 0                         # timestep
        self.z  = np.zeros(n)               # traces 
        self.A  = np.eye(n,n) * epsilon     # A^-1 . b = theta^*
        self.b  = np.zeros(n) 

    def reset(self):
        """Perform end-of-episode reset."""
        self.z[:] = 0

    @property 
    def theta(self):
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
        return _theta 

    def update(self, fvec, reward, fvec_p, gamma, lmbda):
        self.z = gamma * lmbda * self.z + fvec 
        self.A += np.outer(self.z, (fvec - gamma*fvec_p))
        self.b += self.z * reward