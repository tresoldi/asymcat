# encoding: utf-8

# Import Python standard libraries
from collections import Counter
from itertools import chain, product
import math

# Import 3rd party libraries
import numpy as np
import scipy.stats as ss

# import local libraries
from . import dataio

# TODO: in all cases, receive the co-occs (which can be properly checked)
# and compute the observations if needed and not provided (could be cached
# and given by the user)

# TODO: add a scorer with theil times npmi? this would probably mean having
# to return 0 instead of none in other cases for consistency


def get_alphabets(cooccs):
    """
    Return the `x` and `y` alphabets from a list of co-occurrences.
    """

    alphabet_x, alphabet_y = zip(*cooccs)

    return sorted(set(alphabet_x)), sorted(set(alphabet_y))


# TODO: better explanation on square/non-square
# TODO: expand to more than 5 elements? how? random?
def build_ct(obs, square=False):
    """
    Build a contingency table from a dictionary of observations.
    
    The contingency table can be either square or not.
    """

    if square:
        cont_table = np.array([[obs["11"], obs["12"]], [obs["21"], obs["22"]]])
    else:
        cont_table = np.array(
            [[obs["10"], obs["11"], obs["12"]], [obs["20"], obs["21"], obs["22"]]]
        )

    return cont_table


def comp_chi2(cont_table):
    """
    Computes the chi2 for a contigency table of observations.
    """

    return ss.chi2_contingency(cont_table)[0]


# TODO: rename to comp_cramer_v() or use other normalization
# from https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
# and https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(cont_table):
    """
    Compute Cramer's V
    """

    n = cont_table.sum()
    r, k = cont_table.shape

    chi2 = comp_chi2(cont_table)
    phi2 = chi2 / n

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# TODO: make sure it is 2x2
# TODO: what if it is infinite?
def fisher_exact_odds_ratio(cont_table):
    return ss.fisher_exact(cont_table)[0]


# TODO: can only call if it was smoothed, remove exception here
def comp_pmi(p_x, p_y, p_xy, normalized):

    if p_xy == 0.0:
        return None

    pmi = math.log(p_xy / (p_x * p_y))

    if normalized:
        pmi = pmi / -math.log(p_xy)

    return pmi


# TODO: rename MLE
def frequency_scorer(obs):
    """
    Return a simple assymetric scorer based on frequency.
    """

    scorer = {
        pair: (obs[pair]["11"] / obs[pair]["10"], obs[pair]["11"] / obs[pair]["01"])
        for pair in obs
    }

    return scorer


def pmi_scorer(obs, normalized):

    scorer = {}
    for pair in obs:
        p_xy = obs[pair]["11"] / obs[pair]["00"]
        p_x = obs[pair]["10"] / obs[pair]["00"]
        p_y = obs[pair]["01"] / obs[pair]["00"]

        scorer[pair] = (
            comp_pmi(p_x, p_y, p_xy, normalized),
            comp_pmi(p_y, p_x, p_xy, normalized),
        )

    return scorer


# NOTE: returning twice (v, v) as it is symmetric
def chi2_scorer(obs, square=True):
    scorer = {}
    for pair in obs:
        cont_table = build_ct(obs[pair], square)
        v = comp_chi2(cont_table)
        scorer[pair] = (v, v)

    return scorer


# NOTE: returning twice (v, v) as it is symmetric
def cramers_v_scorer(obs, square=True):
    scorer = {}
    for pair in obs:
        cont_table = build_ct(obs[pair], square)
        v = cramers_v(cont_table)
        scorer[pair] = (v, v)

    return scorer


# NOTE: returning twice (v, v) as it is symmetric
def fisher_exact_scorer(obs):
    scorer = {}
    for pair in obs:
        cont_table = build_ct(obs[pair], True)
        v = fisher_exact_odds_ratio(cont_table)
        scorer[pair] = (v, v)

    return scorer


def conditional_entropy(x, y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())

    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theil_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


# TODO: x/y or a/b?
def theil_u_scorer(cooccs):
    # Get the product from the alphabets
    alphabet_a, alphabet_b = zip(*cooccs)
    alphabet_a = list(sorted(set(alphabet_a)))
    alphabet_b = list(sorted(set(alphabet_b)))

    # build scorer
    scorer = {}
    for x, y in product(alphabet_a, alphabet_b):
        # Subset by taking the cooccurrences that have either
        sub_cooccs = [pair for pair in cooccs if any([pair[0] == x, pair[1] == y])]
        all_x, all_y = zip(*sub_cooccs)

        # run theil's
        scorer[(x, y)] = (theil_u(all_x, all_y), theil_u(all_y, all_x))

    return scorer


def scorer2matrix(scorer):
    """
    Builds a matrix from a scorer.
    """

    print(scorer)


# TODO: add normalization
# TODO: extend comment, also with what is returned
# TODO: smoothing?
def mle_scorer(cooccs, obs=None):
    """
    Return a simple assymetric scorer based on frequency.
    
    If `obs` is not provided, it will be computed from the co-occurrences.
    """

    # Collect the observations, if not provided
    if not obs:
        obs = dataio.get_observations(cooccs)

    # Obtain the alphabets from the co-occurrence pairs
    alphabet_x, alphabet_y = get_alphabets(cooccs)

    # Collect the scorer
    scorer = {}
    for pair in product(alphabet_x, alphabet_y):
        scorer[pair] = (
            obs[pair]["11"] / obs[pair]["10"],
            obs[pair]["11"] / obs[pair]["01"],
        )

    return scorer


def scorer2matrix(scorer):
    alphabet_x, alphabet_y = get_alphabets(scorer)

    xy = np.array(
        [np.array([scorer[(x, y)][0] for x in alphabet_x]) for y in alphabet_y]
    )
    yx = np.array(
        [np.array([scorer[(x, y)][1] for y in alphabet_y]) for x in alphabet_x]
    )

    return xy, yx, alphabet_x, alphabet_y
