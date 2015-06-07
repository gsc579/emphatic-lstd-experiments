"""
Utilities for working with and exploring interest/emphasis.
"""
from interest import * 


def seq_lengths(seqs):
    """Given a sequence of sequenecs, return a list of their lengths."""
    return [len(x) for x in seqs]


def total_length(seqs):
    """Given a sequence of sequences, return the sum of sequence lengths."""
    return sum(len(x) for x in seqs)


def avg_length(seqs):
    """Given a sequence of sequenecs, return their average length."""
    lst = [len(x) for x in seqs]
    return sum(lst)/len(lst)


def next_F(F, gm, I):
    return gm*F + I 


def next_M(F, gm, I, lm):
    return (lm*I) + (1 - lm) * F


def gen_gmseq(episodes, val=1):
    for ep in episodes:
        for step in ep[:-1]:
            yield val
        yield 0

def gen_iseq(episodes, ifunc):
    _ifunc = ifunc(episodes)
    for ep in episodes:
        for step in ep[:-1]:
            s, a, r, sp = step
            yield _ifunc(s)
        yield 0

def gen_lmseq(episodes, val):
    for ep in episodes:
        for step in ep:
            yield val


def emphseq(gmseq, iseq, lmseq):
    F = 0
    M = 0
    for gm, I, lm in zip(gmseq, iseq, lmseq):
        yield (F, M)
        F = next_F(F, gm, I)
        M = next_M(F, gm, I, lm)


# TESTING
if __name__ == "__main__":
    exec(open('experiments/start_state_gridworld.py', 'r').read())

if 'episodes' in vars():
    gmseq = list(gen_gmseq(episodes))
    iseq  = list(gen_iseq(episodes, StartStateInterest))
    # iseq  = list(gen_iseq(episodes, FirstVisitInterest))
    lmseq = list(gen_lmseq(episodes, 1.0))
    emseq = list(emphseq(gmseq, iseq, lmseq))
    fseq, mseq = zip(*emseq)


    fig, axes = plt.subplots(2)
    axes[0].plot(fseq)
    axes[1].plot(mseq)
    for ax in axes:
        ax.set_xlim(0, 20)
    plt.show()