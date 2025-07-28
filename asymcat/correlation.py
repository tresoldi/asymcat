"""
Compute series correlation.

It is by definition symmetric.
"""

# TODO: move scorer's computations to a different module (perhaps also
#       renaming scorer itself)

# Import Python standard libraries
from collections import Counter

# Import 3rd party libraries
import numpy as np

# Import local modules
from . import common, scorer


def cramers_v(series_x, series_y):
    cooccs = list(zip(series_x, series_y, strict=False))

    # Build a contingency table
    # TODO: use another library? pandas?
    alphabet_x, alphabet_y = common.collect_alphabets(cooccs)
    ct = []
    for x_val in alphabet_x:
        counter = Counter([y for x, y in cooccs if x == x_val])
        obs = [counter.get(y, 0) for y in alphabet_y]
        ct.append(obs)

    # Compute Cramers'V and return
    return scorer.compute_cramers_v(np.array(ct))


def conditional_entropy(series_x, series_y):
    return scorer.conditional_entropy(series_x, series_y)


def theil_u(series_x, series_y):
    return scorer.conditional_entropy(series_x, series_y)
