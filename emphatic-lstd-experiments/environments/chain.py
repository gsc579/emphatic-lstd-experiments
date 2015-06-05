class Chain:
    """
    An environment consisting of a chain of states.
    Each state has two actions, `left` and `right`, except for the terminal 
    states at the "ends" of the chain. 
    """
    ACTIONS = ('left', 'right')
    def __init__(self, length, start=None):
        if start is None:
            start = (length - 1) // 2
        
        self.length = length
        self.start = start
        self.state = start 

    def reset(self):
        self.state = self.start

    def is_terminal(self, s=None):
        if s is None:
            s = self.state
        return s == self.LEFTMOST or s == self.RIGHTMOST

    @property 
    def LEFTMOST(self):
        return 0

    @property 
    def RIGHTMOST(self):
        return self.length - 1

    def observe(self, s=None):
        if s is None:
            s = self.state 
        return s

    def do(self, action):
        if self.is_terminal():
            next_state = self.state 
        elif action == 'left':
            next_state = self.state - 1
        elif action == 'right':
            next_state = self.state + 1
        else:
            raise Exception("Invalid action:", action)

        ret = self.reward(self.state, action, next_state)
        self.state = next_state
        return ret 

    def reward(self, s, a, sp):
        if sp == self.RIGHTMOST and not self.is_terminal(s) :
            return 1
        else:
            return 0

    @property
    def actions(self):
        return self.ACTIONS

    @property 
    def states(self):
        return list(range(self.LEFTMOST, self.RIGHTMOST + 1))

    @property
    def terminals(self):
        return [s for s in self.states if self.is_terminal(s)]

    @property
    def nonterminals(self):
        return [s for s in self.states if not self.is_terminal(s)]

    @property
    def num_states(self):
        return len(self.nonterminals) + 1