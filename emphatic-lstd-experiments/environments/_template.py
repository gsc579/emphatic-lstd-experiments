"""

- The `state` property should return full state information
- `observe()` may not return the full state information
"""



class EnvironmentTemplate:
    def __init__(self, *args, **kwargs):
        pass 


    def is_terminal(self, s=None):
        if s is None:
            s = self.state 
        return True

    def reset(self):
        pass

    @property 
    def state(self):
        return self._state

    @property 
    def states(self):
        pass 

    @property 
    def nonterminals(self):
        return [s for s in self.states if not self.is_terminal(s)]

    @property
    def terminals(self):
        return [s for s in self.states if self.is_terminal(s)]

    @property
    def num_states(self):
        return len(self.nonterminals) + 1

    @property 
    def actions(self):
        pass

    def observe(self, s=None):
        if s is None:
            s = self.state 
        return s

    def get_actions(self, s=None):
        if s is None:
            s = self.state
        return self.actions

    def do(self, action):
        pass

    def reward(self, s, a, sp):
        return 0