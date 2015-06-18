"""
Temporal Difference Learning with Dutch Traces.
"""
import numpy as np


class TrueOnlineTD:
    def __init__(self, n, **kwargs):
        """ Initialize the agent. """
        self.n = n
        self.z = np.zeros(n)
        self.theta = np.zeros(n)

    def update(self, fvec, R, fvec_p, alpha, gamma, lmbda, rho=1):
        """ Perform an update for a single step of the algorithm. """
        self.z  = rho*gamma*lmbda*self.z + rho*alpha*(1 - rho*gamma*lmbda*np.dot(fvec, self.z))*fvec
        
        delta = R + gamma*np.dot(self.theta, fvec_p) - np.dot(self.theta, fvec)
        self.theta += alpha*delta*self.z
        return delta

    def reset(self):
        """Perform reset for end of episode."""
        self.z[:] = 0

    @classmethod
    def from_weights(cls, weights):
        """Create and initialize an agent from a weight vector"""
        weights = np.array(weights)
        assert(weights.ndim == 1)
        # Initialize the object
        fvec_length = len(weights)
        obj = cls(fvec_length)
        # Set the weights from the weight vector
        obj.theta[:] = weights
        return obj