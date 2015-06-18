"""
Implementing policies for reinforcement learning environments.
"""
import numpy as np


class RandomPolicy:
    """
    A random policy, i.e., one that chooses an action uniformly at random from 
    those available in the current state.
    """
    def __init__(self, env, random_seed=None):
        """Initialize with an environment, and optionally a random seed."""
        self.env = env 
        self.RandomState = np.random.RandomState(random_seed)

    def __call__(self, state):
        actions = self.env.get_actions(state)
        return self.RandomState.choice(actions)
