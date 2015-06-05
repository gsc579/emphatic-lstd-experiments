"""
Simulating a random walk (an episodic problem) for reinforcement learning, in
the on-policy setting.
"""
import numpy as np 

from environments.chain import Chain

# Specify experiment #############
random_seed = None
num_episodes = 5
num_states = 7
TERMINAL = -1


# Setup experiment
np.random.seed(random_seed)


# Setup Environment
env = Chain(num_states)


# Specify policy
def policy(s, actions):
    return np.random.choice(actions)


# Run the simulation
episodes = []
for i in range(num_episodes):
    env.reset()
    episode = []
    while not env.is_terminal():
        # Observe, take action, get next observation, and compute reward
        s  = env.observe()
        a  = policy(s, env.actions)
        env.do(a)
        sp = env.observe()
        r  = env.reward(s, a, sp)
        if not env.is_terminal():
            episode.append((s, a, r, sp))
        else:
            episode.append((s, a, r, TERMINAL))

    # Append trajectory to episodes
    episodes.append(episode)



# Perform learning
for episode in episodes:
    for step in episode:
        s, a, r, sp = step