
# Path-fixing hack
import inspect, os, sys
filename = inspect.getframeinfo(inspect.currentframe()).filename
curdir = os.path.dirname(os.path.abspath(filename))
pardir = os.path.dirname(curdir)
sys.path.append(pardir)

import numpy as np 

import algorithms, features, environments
from algorithms import FirstVisitMC
from environments.policy import RandomPolicy
from environments.gridworld import Gridworld
from features.features import *


def interest_start():
    """Interest in start state only."""
    start = True
    def func(fvec):
        nonlocal start
        if start:
            start = False
            return 1.0
        else:
            return 0.0
    return func


def run_mc(episodes, agent):
    """
    Run Monte Carlo on `episodes` using algorithm `agent`.
    """
    for episode in episodes:
        # Initialize interest function for episode
        ifunc = interest_start()

        # Step through the episode
        for t, step in enumerate(episode):
            s, a, r, sp = step
            agent.update(s, r)

        # End of episode termination
        agent.terminate()

    return agent.theta


def apply_fa(episodes, phi_func):
    """Apply function approximation to a series of episodes."""
    ret = []
    for episode in episodes:
        tmp = []
        for step in episode[:-1]:
            s, a, r, sp = step
            fvec   = phi_func(s)
            fvec_p = phi_func(sp)
            tmp.append((fvec, a, r, fvec_p))
        # Account for final step of the episode
        s, a, r, sp = episode[-1]
        fvec   = phi_func(s)
        fvec_p = np.zeros(phi_func.length, dtype=np.float)
        tmp.append((fvec, a, r, fvec_p))
        ret.append(tmp)
    return ret


def run_episodes(num_episodes, env, policy):
    return [run_episode(env, policy) for i in range(num_episodes)]
    

def run_episode(env, policy):
    env.reset()
    ret = []
    while not env.is_terminal():
        # Observe, take action, get next observation, and compute reward
        s  = env.observe()
        a  = policy(s)
        r  = env.do(a)
        sp = env.observe()

        # Append step to episode trajectory
        ret.append((s, a, r, sp))
    return ret


def get_features(env, phi):
    return list(set(tuple(phi(env.observe(s))) for s in env.nonterminals))


def get_values(env, weights, phi_func):
    """Values of nonterminal states in env."""
    ret = {}
    for s in env.nonterminals:
        obs = tuple(env.observe(s))
        ret[s] = np.dot(weights, phi_func(obs))
    return ret


def avg_length(episodes):
    """Average episode length."""
    return np.mean([len(x) for x in episodes])


def rms_length(episodes):
    """Root mean squared episode length."""
    return np.sqrt(np.mean([len(x)**2 for x in episodes]))


if __name__ == "__main__":
    random_seed = 101
    num_episodes = 100
    nx, ny = 4, 2

    # Initialize RNG
    np.random.seed(random_seed)

    # Setup environment
    env = Gridworld(nx, ny, random_seed=random_seed)

    # Setup policy
    policy = RandomPolicy(env, random_seed=random_seed)

    # Generate data
    _episodes = run_episodes(num_episodes, env, policy)

    # Apply function approximation
    phi0 = Identity(len(env.observe()))
    phi1 = Bias()
    phi  = Combination((phi0, phi1))
    episodes = apply_fa(_episodes, phi)

    # Setup MC 
    mc_agent = FirstVisitMC(phi.length)

    # Run MC on episodes
    theta = run_mc(episodes, mc_agent)

    # Check that the values are right
    fvlst = get_features(env, phi)
    avg = avg_length(episodes)
    env.reset()
    start = phi(env.observe())

    print("Average length:", avg)
    print("MC Value:", np.dot(theta, start))

