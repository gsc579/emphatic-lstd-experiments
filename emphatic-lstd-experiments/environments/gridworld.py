"""
Grid World.
"""
import numpy as np 


COMPASS = ('north', 'east', 'south', 'west')
DIRMAP  = {x: 2**i for i, x in enumerate(COMPASS)}
BASIS   = tuple(np.array(x) for x in [[0,1], [1,0], [0,-1], [-1,0]])
BASEMAP = {x: ix for ix, x in enumerate(COMPASS)}
EMPTY   = 0
BLOCKED = 1


class Gridworld:
    """
    A simple grid world environment.

    States are specified as (i,j) coordinate pairs, corresponding to the i-th 
    row and the j-th column.
    """
    def __init__(self, nx, ny, start=None, goal=None, **kwargs):
        if start is None:
            start = (0, 0)
        if goal is None:
            goal = (nx - 1, ny - 1)
        
        # Set values and initially empty gridworld
        self.nx = nx
        self.ny = ny 
        self.start = start
        self.goal = goal
        self._state = start
        self.shape = (nx, ny)
        self.array = np.zeros((nx, ny), dtype=np.int)


    # Grid/Array related
    def blocked(self, idx):
        return self.array[idx] == BLOCKED

    def unblocked(self, idx):
        return self.array[idx] == EMPTY        

    def vision(self, idx, distance=1):
        ret = []
        for x in BASIS:
            y = tuple(x + idx)
            if self.valid_index(y):
                ret.append(self.array[y])
            else:
                ret.append(BLOCKED)
        return ret

    def valid_index(self, idx):
        return all(0 <= x < y for x, y in zip(idx, self.shape)) 


    # Environment related
    def is_terminal(self, s=None):
        if s is None:
            s = self.state
        return s == self.goal

    def reset(self):
        self._state = self.start

    @property 
    def actions(self):
        return COMPASS

    @property 
    def state(self):
        return self._state

    @property 
    def states(self):
        return list(np.ndindex(self.shape))

    @property 
    def nonterminals(self):
        return [s for s in self.states if not self.is_terminal(s)]

    @property
    def terminals(self):
        return [s for s in self.states if self.is_terminal(s)]

    @property
    def num_states(self):
        return len(self.nonterminals) + 1

    def observe(self, s=None):
        if s is None:
            s = self.state 
        return self.vision(s)

    def get_actions(self, s=None):
        if s is None:
            s = self.state 
        return self.actions

    def do(self, action):
        # Allow for actions to be specified as either integers or their name
        if isinstance(action, str):
            move = BASEMAP[action]

        # Perform no action if in the goal state, as the goal is terminal
        if self.is_terminal():
            return 0

        # Compute destination and if valid, move there, else remain
        dest = tuple(self._state + BASIS[move])
        if self.valid_index(dest) and self.unblocked(dest):
            next_state = dest
        else:
            next_state = self.state
        
        ret = self.reward(self.state, action, next_state)
        self._state = next_state
        return ret 

    def reward(self, s, a, sp):
        if not self.is_terminal(s) and self.is_terminal(sp):
            return 0
        else:
            return -1


