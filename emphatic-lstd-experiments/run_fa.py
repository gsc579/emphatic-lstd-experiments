"""
Apply function approximation to a series of episodes.
"""
import click
import pickle
import numpy as np 

from features import features


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('-f', '--feature', type=str, multiple=True)
def _cli(filename, feature):
    for f in feature:
        fexp = f.strip().split()
        name = fexp.pop(0)
        fcls = getattr(features, name)
        print(name)
        print(*fexp)
        # Will need to decorate functions to get appropriate type conversion
        fobj = fcls(*fexp) 


@click.command()
@click.argument('filename', type=click.Path(exists=True))
def cli(filename):
    data = pickle.load(open(filename, 'rb'))
    episodes = data['episodes']
    fvec_length = len(episodes[0][0])

    phi_cls = getattr(features, 'Identity')
    phi_obj = phi_cls('4')

    # Convert episode states to features
    for episode in episodes:
        for step in episode:
            s, a, r, sp = step
            fvec = phi_obj(s)
            fvec_p = phi_obj(sp)




if __name__ == "__main__":
    cli()
