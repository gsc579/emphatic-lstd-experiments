"""
Generate data for an experiment.
"""
import click
import pickle
import numpy as np 


from environments.policy import *
from environments.gridworld import Gridworld




@click.command()
@click.option("-n", "--num_episodes", default=1, help="number of episodes")
@click.option('--shape', type=(int, int), help="shape of the gridworld")
@click.option("--random_seed", default=123, help="seed for pseudo RNG")
@click.option("-o", "--output", type=click.Path(exists=False))
def cli(num_episodes, shape, random_seed, output=None):
    data = simulate(num_episodes, shape, random_seed)
    if output is None:
        print(data)
    else:
        with open(output, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def simulate(num_episodes, shape, random_seed):
    if random_seed is None:
        random_seed = np.random.get_state()

    # Setup environment
    nx, ny = shape
    env = Gridworld(nx, ny, random_seed=random_seed)

    # Setup policy (currently only random policy is used)
    pol = RandomPolicy(env, random_seed=random_seed)

    # Run the simmulation
    episodes = []
    for i in range(num_episodes):
        env.reset()
        episode = []
        while not env.is_terminal():
            # Observe, take action, get next observation, and compute reward
            s  = env.observe()
            a  = pol(s)
            r  = env.do(a)
            sp = env.observe()

            # Append step to episode trajectory
            episode.append((s, a, r, sp))

        # Append trajectory to episodes
        episodes.append(episode)

    # Store and return the data
    data = {}
    data['environment'] = env
    data['num_episodes'] = num_episodes
    data['random_seed'] = random_seed
    data['episodes'] = episodes 
    
    return data


if __name__ == '__main__':
    cli()
