"""
Template for reinforcement learning agents.
"""
import numpy as np 


class Agent:
    def __init__(self, n, **kwargs):
        """ Initialize the agent. """
        self.n = n

    def update(self, fvec, R, fvec_p, params):
        """ Perform an update for a single step of the algorithm. """
        pass

    def reset(self):
        """Reset traces for the start of episode."""
        pass

    @property 
    def theta(self):
        pass

