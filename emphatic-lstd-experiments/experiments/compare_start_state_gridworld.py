"""
Script for running comparisons between LSTD and ELSTD.
"""
# Path-fixing hack
import inspect, os, sys
filename = inspect.getframeinfo(inspect.currentframe()).filename
curdir = os.path.dirname(os.path.abspath(filename))
pardir = os.path.dirname(curdir)
sys.path.append(pardir)

import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib.cm as cm 

import algorithms, features, environments
from algorithms import ELSTD, LSTD, TD
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


def on_policy():
    """The on-policy setting has rho=1 for all timesteps."""
    def func(s, a):
        return 1
    return func 

def constant(val):
    """A factory for constant values for parameters."""
    def func(*args):
        return val
    return func


def run_experiment(episodes, agent, update_params):
    """
    Run an experiment for an agent over a series of episodes with the given
    parameters.
    """
    for episode in episodes:
        # Reset for start of episode
        agent.reset()
        # Initialize interest function for episode
        ifunc = interest_start()

        # Step through the episode
        for t, step in enumerate(episode[:-1]):
            s, a, r, sp = step
            params = dict(**update_params)
            params['interest'] = ifunc(s)
            agent.update(s, r, sp, params)

        # Final timestep update
        s, a, r, sp = episode[-1]
        sp = np.zeros_like(s, dtype=np.float)
        params = dict(**update_params)
        params['interest'] = ifunc(s)
        agent.update(s, r, sp, params)

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


def get_values(env, weights, phi_func):
    """Values of nonterminal states in env."""
    ret = {}
    for s in env.nonterminals:
        obs = tuple(env.observe(s))
        ret[s] = np.dot(weights, phi_func(obs))
    return ret


def mat_values(env, weights, phi_func):
    """Values of gridworld cells, as a matrix."""
    ret = np.zeros(env.shape)
    for s in env.nonterminals:
        obs = tuple(env.observe(s))
        ret[s] = np.dot(weights, phi_func(obs))
    return ret


def obs_values(env, weights, phi_func):
    lst = list(set([tuple(env.observe(s)) for s in env.nonterminals]))
    return {x: np.dot(weights, phi_func(x)) for x in lst}


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

    # Setup agents
    elstd = algorithms.ELSTD(phi.length)
    lstd = algorithms.LSTD(phi.length)

    # Setup agent parameters
    params = {'gamma': 1.0, 'lmbda': 1.0}

    # Run the experiment
    elstd_theta = run_experiment(episodes, elstd, params)
    lstd_theta = run_experiment(episodes, lstd, params)

    # Compare the algorithms
    diff_theta  = elstd_theta - lstd_theta
    elstd_values = get_values(env, elstd_theta, phi)
    lstd_values = get_values(env, lstd_theta, phi)


    # print("ELSTD weights:", elstd_theta)
    # print("LSTD weights:", lstd_theta)
    # print("Difference:", diff_theta)
    # print("L1-norm of difference:", np.sum(np.abs(diff_theta)))

    print("ELSTD start state value:", elstd_values[env.start])
    print("LSTD start state value:", lstd_values[env.start])
    
    print("Average episode length:", avg_length(episodes))
    # print("RMS episode length:", rms_length(episodes))

    start_value = -1*(avg_length(episodes) - 1)
    print("ELSTD:", np.abs(start_value - elstd_values[env.start]))
    print("LSTD:", np.abs(start_value - lstd_values[env.start]))