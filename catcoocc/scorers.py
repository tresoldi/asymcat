# encoding: utf-8

# Import Python standard libraries
from collections import Counter
from itertools import chain, product
import math

# Import 3rd party libraries
import numpy as np
import scipy.stats as ss

# import local libraries
from . import utils

# TODO: always ask for `obs`, or even for cont_table?

# TODO: in all cases, receive the co-occs (which can be properly checked)
# and compute the observations if needed and not provided (could be cached
# and given by the user)

# TODO: add a scorer with theil times npmi? this would probably mean having
# to return 0 instead of none in other cases for consistency





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


def compute_chi2(cont_table):
    """
    Computes the chi2 for a contigency table of observations.
    """

    return ss.chi2_contingency(cont_table)[0]


# TODO: rename to comp_cramer_v() or use other normalization
# from https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
# and https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def compute_cramers_v(cont_table):
    """
    Compute Cramer's V
    """

    n = cont_table.sum()
    r, k = cont_table.shape

    chi2 = compute_chi2(cont_table)
    phi2 = chi2 / n

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# TODO: make sure it is 2x2
# TODO: what if it is infinite?
def compute_fisher_exact_odds_ratio(cont_table):
    return ss.fisher_exact(cont_table)[0]


# TODO: can only call if it was smoothed, remove exception here?
def compute_pmi(p_x, p_y, p_xy, normalized):

    # assumes independence
    # TODO: a `strict` argument could make it fail or return zero
    # this is like a limit, should get the lowest score?
    if p_xy == 0.0:
        p_xy = 0.000000001

    pmi = math.log(p_xy / (p_x * p_y))

    if normalized:
        pmi = pmi / -math.log(p_xy)

    return pmi

# TODO: rename to x_symbols and y_symbols
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


def compute_theil_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x





# TODO: allow independent normalization, not by default
# TODO: rename to scaling
def normalize_scorer(scorer, method, nrange=None):
    # Extract scores as a list, combining `xy` and `yx`
    scores = list(chain.from_iterable(scorer.values()))

    # normalizatoin is performed over xy and yx together
    if method == "minmax":
        # Set normalization range        
        if not nrange:
            range_low, range_high = (0, 1)
        else:
            range_low, range_high = nrange

        # Cache values for speeding computation
        min_score = min(scores)
        max_score = max(scores)
        score_diff = max_score - min_score
        range_diff = range_high - range_low

        norm_scorer = {
            pair : (
            
            range_low + (((value[0] - min_score)*range_diff) / score_diff), 
            range_low + (((value[1] - min_score)*range_diff) / score_diff)
            )
            
            for pair, value in scorer.items()
        }

    elif method == "mean":
        # Cache values for speeding computation
        mean = np.mean(scores)
        score_diff = max(scores) - min(scores)
        
        
        norm_scorer = {
            pair : ( (value[0] - mean) / score_diff, (value[1] - mean) / score_diff)
            for pair, value in scorer.items()
        }
    elif method == "stdev":
        print("not implemented")
    else:
        print("unknown normalization")

    return norm_scorer
    

def scorer2matrix(scorer):
    alphabet_x, alphabet_y = utils.collect_alphabets(scorer)

    xy = np.array(
        [np.array([scorer[(x, y)][0] for x in alphabet_x]) for y in alphabet_y]
    )
    yx = np.array(
        [np.array([scorer[(x, y)][1] for y in alphabet_y]) for x in alphabet_x]
    )

    return xy, yx, alphabet_x, alphabet_y

####################### SCORERS

# TODO: extend comment, also with what is returned
# TODO: smoothing?
def mle_scorer(cooccs, obs=None):
    """
    Return a simple assymetric scorer based on frequency.
    
    If `obs` is not provided, it will be computed from the co-occurrences.
    """

    # Collect the observations, if not provided
    if not obs:
        obs = utils.collect_observations(cooccs)

    # Obtain the alphabets from the co-occurrence pairs
    alphabet_x, alphabet_y = utils.collect_alphabets(cooccs)

    # Collect the scorer
    scorer = {
        pair : (
            obs[pair]["11"] / obs[pair]["10"],
            obs[pair]["11"] / obs[pair]["01"],
        )
        for pair in product(alphabet_x, alphabet_y)
    }
    
    return scorer

def pmi_scorer(cooccs, obs=None, normalized=False):
    # Collect the observations, if not provided
    if not obs:
        obs = utils.collect_observations(cooccs)
        
    scorer = {}
    for pair in obs:
        p_xy = obs[pair]["11"] / obs[pair]["00"]
        p_x = obs[pair]["10"] / obs[pair]["00"]
        p_y = obs[pair]["01"] / obs[pair]["00"]

        scorer[pair] = (
            compute_pmi(p_x, p_y, p_xy, normalized),
            compute_pmi(p_y, p_x, p_xy, normalized),
        )

    return scorer
    

def chi2_scorer(cooccs, obs=None, square=True):
    # Collect the observations, if not provided
    if not obs:
        obs = utils.collect_observations(cooccs)

    scorer = {}
    for pair in obs:
        cont_table = build_ct(obs[pair], square)
        chi2 = compute_chi2(cont_table)
        scorer[pair] = (chi2, chi2)

    return scorer
    
def cramers_v_scorer(cooccs, obs=None, square=True):
    # Collect the observations, if not provided
    if not obs:
        obs = utils.collect_observations(cooccs)

    scorer = {}
    for pair in obs:
        cont_table = build_ct(obs[pair], square)
        cramers_v = compute_cramers_v(cont_table)
        scorer[pair] = (cramers_v, cramers_v)

    return scorer

def fisher_exact_scorer(cooccs, obs=None):
    # Collect the observations, if not provided
    if not obs:
        obs = utils.collect_observations(cooccs)

    scorer = {}
    for pair in obs:
        cont_table = build_ct(obs[pair], True)
        fischer = compute_fisher_exact_odds_ratio(cont_table)
        scorer[pair] = (fischer, fischer)

    return scorer
    
def theil_u_scorer(cooccs):
    # Get the product from the alphabets
    alphabet_x, alphabet_y = zip(*cooccs)
    alphabet_x = set(alphabet_x)
    alphabet_y = set(alphabet_y)

    # build scorer
    scorer = {}
    for x, y in product(alphabet_x, alphabet_y):
        # Subset by taking the cooccurrences that have either
        sub_cooccs = [pair for pair in cooccs if any([pair[0] == x, pair[1] == y])]
        all_x, all_y = zip(*sub_cooccs)

        # run theil's
        scorer[(x, y)] = (compute_theil_u(all_x, all_y), compute_theil_u(all_y, all_x))

    return scorer
    
# TODO: allow normalized PMI
def tresoldi_scorer(cooccs, obs=None):
    # Collect the observations, if not provided
    if not obs:
        obs = utils.collect_observations(cooccs)
    
    # get the NPMI and theil_u scorer for all pairs
    npmi = pmi_scorer(cooccs, obs, normalized=False)
    theil_u = theil_u_scorer(cooccs)
    
    # build new scorer and return
    scorer = {
        pair : tuple([score * npmi[pair][0] for score in theil_u[pair]])
        for pair in obs.keys()
    }
    
    return scorer
