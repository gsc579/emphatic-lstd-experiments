"""
Interest only in the start state of each episode.
"""
# Path-fixing hack
import inspect, os, sys
filename = inspect.getframeinfo(inspect.currentframe()).filename
curdir = os.path.dirname(os.path.abspath(filename))
pardir = os.path.dirname(curdir)
sys.path.append(pardir)

import click
import pickle
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib.cm as cm 

import algorithms
from interest import StartStateInterest



@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('--agent', type=str)
@click.option('--alpha', type=float)
@click.option('--gamma', type=float)
@click.option('--lmbda', type=float)
@click.option('-p', '--plot', default=False)
def cli(filename, agent, alpha, gamma, lmbda, plot):
    data = pickle.load(open(filename, 'rb'))
    episodes = data['episodes']
    fvec_length = len(episodes[0][0])

    # Initialize the agent
    agent_cls = getattr(algorithms, agent)
    agent = agent_cls(fvec_length)

    # Interest function is StartStateInterest
    ifunc = StartStateInterest(episodes)

    # Perform learning
    for episode in episodes:
        # Setup gamma values
        gmlst     = [gamma for i in episode[:-1]] + [0]
        # Reset agent for start of episode
        agent.reset()
        for i, step in enumerate(episode[:-1]):
            # Unpack timestep
            s, a, r, sp = step
            s = np.array(s)
            sp = np.array(sp)
            interest = ifunc(s)
            params = {
                'alpha': alpha, 
                'gamma': gmlst[i],
                'gamma_p': gmlst[i+1], 
                'lmbda': lmbda, 
                'interest': interest,
                'rho': 1.0}
            agent.update(s, r, sp, params)

        # Final timestep update
        s, a, r, sp = episode[-1]
        s = np.array(s)
        sp = np.zeros(fvec_length, dtype=np.float)
        interest = ifunc(s)

    # Plot the graph if so flagged
    # TODO 

    # Return information about the run
    return agent.theta

if __name__ == "__main__":
    cli()
