"""Emphatic least-squares temporal difference learning implementation. """
import numpy as np 


class ELSTD:
    """Emphatic least-squares temporal difference learning. """
    def __init__(self, n, epsilon=0, **kwargs):
        self.n = n
        self.z = np.zeros(n, dtype=np.float)
        self.A = np.eye(n, dtype=np.float) * epsilon
        self.b = np.zeros(n)
        self.F = 0
        self.M = 0

    def reset(self):
        """Reset traces for the start of episode."""
        self.z[:] = 0
        self.F = 0
        self.M = 0

    @property
    def theta(self):
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
        #np.dot(a,b)矩阵ab相乘
        #在numpy中可以使用numpy.linalg.pinv求伪逆
        return _theta

    def update(self, fvec, reward, fvec_p, params):
        # Should include rho and gamma_p to be completely correct
        gamma = params['gamma']
        interest = params['interest']
        lmbda = params['lmbda']
        self.F = gamma * self.F + interest
        self.M = (lmbda * interest) + ((1 - lmbda) * self.F)
        self.z = (gamma * lmbda * self.z + self.M * fvec)
        self.A += np.outer(self.z, (fvec - gamma * fvec_p))
        self.b += self.z * reward
