"""
Feature mapping functions for function approximation.

In order for function approximation to work the same way in the episodic and
continuing settings, we need to be able to return a zero vector for terminal
states; this is achieved by specifying the keyword `terminals` when 
initializing the feature function and then checking if the state is terminal
when the function is called.
"""
import numpy as np


class Combination:
    """A feature from the combination of other features."""
    def __init__(self, features, terminals=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        self.length = sum(x.length for x in features)
        self._features = features

    @property 
    def terminals(self):
        return self._terminals

    def __call__(self, x):
        # if x in self.terminals:
        #     return np.zeros(self.length)
        return np.concatenate([f(x) for f in self._features])



class Bias:
    """A bias feature (always returns 1 except in terminal state)"""
    def __init__(self, terminals=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        self.length = 1

    def __call__(self, x):
        # if x in self.terminals:
        #     return np.zeros(self.length)
        return np.ones(1)

    @property 
    def terminals(self):
        return self._terminals


class Identity:
    """Identity mapping. Return the vector passed to it."""
    def __init__(self, length, terminals=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        self.length = length 

    def __call__(self, x):
        x = np.array(x)
        assert(len(x) == self.length)
        return x

    @property 
    def terminals(self):
        return self._terminals


class Int2Binary:
    """
    Convert integer to its bit vector representation.
    On initialization, it precomputes an array which is used to extract the 
    individual bits of each integer. 
    """
    def __init__(self, length, terminals=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        # Precompute the array for converting integers to bit vectors
        self.length = length
        self._array = (1 << np.arange(length))

    def __call__(self, x):
        # if x in self.terminals:
        #     return np.zeros(self.length)
        x = np.array(x)
        ret = []
        for i in x.flat:
            ret.append((i & self._array) > 0)
        return np.ravel(ret).astype(np.uint8)

    @property 
    def terminals(self):
        return self._terminals

class Int2Unary:
    """
    Convert integer to unary representation (e.g., for tabular case)
    """
    def __init__(self, length, terminals=None):
        if terminals is None:
            self._terminals = set()
        else:
            self._terminals = set(terminals)
        self.length = length
        self._array = np.eye(length)

    def __call__(self, x):
        # if x in self.terminals:
        #     return np.zeros(self.length)
        return self._array[x]

    @property 
    def terminals(self):
        return self._terminals