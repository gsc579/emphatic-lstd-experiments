"""
A sort of integrative test, checking that TD works properly on the random walk
environment.

We should see convergence to `1/2` for a single "bias" feature, convergence to
`(1/n, 2/n, ..., (n-1)/n)` for the tabular representation, and and something
not too ridiculous for other representations (so long as they're not terrible).

It's also important to check that the addition of a bias feature doesn't ruin
the function approximation, as this indicates something is wrong with how the
feature vectors are computed, how the trajectories are processed/generated, or 
how the agent updates itself from new data.
"""
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib.cm as cm 

from algorithms.td import TD 
from features.features import Bias, Combination, Int2Binary, Int2Unary
from environments.chain import Chain

# Specify experiment #############
random_seed = None
num_episodes = 1000
num_states = 7


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
        s  = env.state
        a  = policy(s, env.actions)
        r  = env.do(a)
        sp = env.state

        # Append step to episode trajectory
        episode.append((s, a, r, sp))

    # Append trajectory to episodes
    episodes.append(episode)


# Set feature mapping
phi1 = Int2Unary(env.num_states, terminals=env.terminals)
phi2 = Bias(terminals=env.terminals)
phi  = Combination((phi1, phi2), terminals=env.terminals)
# phi = Int2Unary(env.num_states, terminals=env.terminals)

# Setup agent
agent = TD(phi.length)
alpha = 0.01
gamma = 1
lmbda = 0


# Perform learning
for episode in episodes:
    # Reset agent for start of episode
    agent.reset()
    for step in episode[:-1]:
        # Unpack timestep and perform function approximation
        s, a, r, sp = step
        fvec = phi(s)
        fvec_p = phi(sp)
        agent.update(fvec, r, fvec_p, alpha, gamma, lmbda)
    # Perform final step
    s, a, r, sp = episode[-1]
    fvec = phi(s)
    fvec_p = np.zeros_like(fvec)
    agent.update(fvec, r, fvec_p, alpha, gamma, lmbda)


# Determine the values of each state
values = {}
for s in env.nonterminals:
    obs = env.observe(s)
    values[obs] = np.dot(agent.theta, phi(obs))

print("Values:")
print(values)

# Problem specific
diff = np.array([(values[s] - s/env.num_states) for s in values])
mse  = np.sum(diff**2)
print("MSE:", mse)

# Show a heatmap of state values
fig, ax = plt.subplots()
mat = np.array([np.dot(agent.theta, phi(env.observe(x))) for x in sorted(env.nonterminals)]).reshape(1, -1)
cax = ax.matshow(mat, cmap=cm.Reds)
fig.colorbar(cax)