# encoding: utf-8

# Note: includes code modified from
# - https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792
# - https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

"""
Defines the various scorers for categorical co-occurrence analysis.
"""

# TODO: have a function for checking if we have enough data for a chi2, combined with smoothing
# TODO: improve the speed of theil's u computation
# TODO: extend scipy options in chi2_contingency
# TODO: investigate if it is worth reusing chi2 statistics for cramer
# TODO: include a logarithmic scaler (instead of percentile one)
# TODO: allow to combine scalers? log within range?

# Import Python standard libraries
from collections import Counter
from itertools import chain, product
import math

# Import 3rd party libraries
import numpy as np
import scipy.stats as ss

# import local modules
from . import utils


def conditional_entropy(x_symbols, y_symbols):
    """
    Computes the entropy of `x` given `y`.

    :param list x_symbols: A list of all observed `x` symbols.
    :param list y_symbols: A list of all observed `y` symbols.

    :return float entropy: The conditional entropy of `x` given `y`.
    """

    # Cache counters; while the xy_counter is already computed in other
    # parts, particularly in the scorers, it is worth repeating the code
    # here to have a more general function.
    y_counter = Counter(y_symbols)
    xy_counter = Counter(list(zip(x_symbols, y_symbols)))
    population = sum(y_counter.values())

    # Compute the entropy and return
    entropy = 0
    for xy_pair, xy_count in xy_counter.items():
        p_xy = xy_count / population
        p_y = y_counter[xy_pair[1]] / population
        entropy += p_xy * math.log(p_y / p_xy)

    return entropy


def compute_cramers_v(cont_table):
    """
    Compute Cramer's V from a contingency table.

    :param np.array cont_table: The contingency table for computation.

    :return float v: The Cramér's V measure for the given contingency table.
    """

    # Cache the shape and sum of the contingency table
    rows, cols = cont_table.shape
    population = cont_table.sum()

    # Compute chi2 and phi2
    chi2 = ss.chi2_contingency(cont_table)[0]
    phi2 = chi2 / population

    # Compute the correlations for Cramér's V
    phi2corr = max(0, phi2 - ((cols - 1) * (rows - 1)) / (population - 1))
    rcorr = rows - ((rows - 1) ** 2) / (population - 1)
    kcorr = cols - ((cols - 1) ** 2) / (population - 1)

    # Compute Cramér's from the correlations
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def compute_pmi(p_x, p_y, p_xy, normalized, limit=1e-6):
    """
    Compute the Pointwise Mutual Information.

    :param float p_x: The probability of `x`, p(x).
    :param float p_y: The probability of `y`, p(y).
    :param float p_xy: The probability of `xy`, p(xy).
    :param bool normalized: Whether to return the normalized Pointwise
        Mutual Information in range [-1, 1].
    :param float limit: The value to use for computation when `p_xy` is 0.0
        (i.e., non observed), as the limit of PMI to zero (default: 1e-6).

    :return float pmi: The Pointwise Mutual Information.
    """

    if p_xy == 0.0:
        pmi = math.log(limit / (p_x * p_y))
        if normalized:
            pmi = pmi / -math.log(limit)
    else:
        pmi = math.log(p_xy / (p_x * p_y))
        if normalized:
            pmi = pmi / -math.log(p_xy)

    return pmi


def compute_theil_u(x_symbols, y_symbols):
    """
    Compute the uncertainty coefficient Theil's U.

    :param list x_symbols: The list of observed symbols in series `x`.
    :param list y_symbols: The list of observed symbols in series `y`.

    :return float theil_u : The uncertainty coefficient given `x` and `y`.
    """

    # Compute the conditional entropy of `x` given `y`
    h_xy = conditional_entropy(x_symbols, y_symbols)

    # Cache a counter for the symbols of the `x` series and its length
    x_counter = Counter(x_symbols)
    population = len(x_symbols)

    # Compute the probability for each symbol and, from there, the entropy
    p_x = list(map(lambda n: n / population, x_counter.values()))
    h_x = ss.entropy(p_x)

    # If the entropy is zero (all items alike), the uncertainty coeffiecient
    # is by definition 1.0; otherwise, compute it in relation to the
    # conditional entropy
    if h_x == 0.0:
        theil_u = 1.0
    else:
        theil_u = (h_x - h_xy) / h_x

    return theil_u


# TODO: allow independent scaling over `x` and independent over `y` (currently doing all)
# TODO: allow scaling withing percentile borders
# TODO: see if we can vectorize numpy operations (now on dicts)
def scale_scorer(scorer, method="minmax", nrange=(0, 1)):
    """
    Scale a scorer.

    The function returns a scaled version of a scorer considering all
    the assymmetric scorers (i.e., both `x` given `y` and `y` given `x`).
    Implemented scoring methods are "minmax" (by default on range [0, 1],
    which can be modified by the `nrange` parameter) and "mean".

    :param dict scorer: A scoring dictionary.
    :param str method: The scoring method to be used, either `"minmax"` or
        `"mean"` or `"stdev"` (default: `"minmax"`).
    :param tuple nrange: A tuple with the scaling range, to be used when
        applicable (default: (0, 1)).

    :return dict scaled_scorer: A scaled version of the scorer.
    """

    # Extract scores as a list, combining `xy` and `yx`
    scores = list(chain.from_iterable(scorer.values()))

    # normalizatoin is performed over xy and yx together
    if method == "minmax":
        # Set normalization range
        range_low, range_high = nrange

        # Cache values for speeding computation
        min_score = min(scores)
        max_score = max(scores)
        score_diff = max_score - min_score
        range_diff = range_high - range_low

        scaled_scorer = {
            pair: (
                range_low + (((value[0] - min_score) * range_diff) / score_diff),
                range_low + (((value[1] - min_score) * range_diff) / score_diff),
            )
            for pair, value in scorer.items()
        }

    elif method == "mean":
        # Cache values for speeding computation
        mean = np.mean(scores)
        score_diff = max(scores) - min(scores)

        scaled_scorer = {
            pair: ((value[0] - mean) / score_diff, (value[1] - mean) / score_diff)
            for pair, value in scorer.items()
        }
    elif method == "stdev":
        mean = np.mean(scores)
        stdev = np.std(scores)

        scaled_scorer = {
            pair: ((value[0] - mean) / stdev, (value[1] - mean) / stdev)
            for pair, value in scorer.items()
        }
    else:
        raise ValueError("Unknown scaling method.")

    return scaled_scorer


def invert_scorer(scorer):
    """
    Inverts a scorer, so that the higher the affinity, the higher the score.

    It is recommended than only scorers in range [0..] are inverted.
    """

    # Collect the highest overall value
    scores = list(chain.from_iterable(scorer.values()))
    max_score = max(scores)

    inverted_scorer = {
        coocc: tuple([max_score - value for value in values])
        for coocc, values in scorer.items()
    }

    return inverted_scorer


def scorer2matrices(scorer):
    """
    Return the asymmetric matrices implied by a scorer and their alphabets.

    :param dict scorer: A scoring dictionary.

    :return np.array xy: A scoring matrix of `y` given `x`.
    :return np.array yx: A scoring matrix of `x` given `y`.
    :return list alphabet_x: The alphabet for matrix `x`.
    :return list alphabet_y: The alphabet for matrix `y`.
    """

    alphabet_x, alphabet_y = utils.collect_alphabets(scorer)

    xy = np.array(
        [np.array([scorer[(x, y)][0] for x in alphabet_x]) for y in alphabet_y]
    )
    yx = np.array(
        [np.array([scorer[(x, y)][1] for y in alphabet_y]) for x in alphabet_x]
    )

    return xy, yx, alphabet_x, alphabet_y


class CatScorer:
    """
    Class for computing catagorical co-occurrence scores.
    """

    def __init__(self, cooccs):
        """
        Initialization function.
        """

        # Store co-occs, observations, symbols and alphabets
        self.cooccs = cooccs
        self.obs = utils.collect_observations(cooccs)

        # Obtain the alphabets from the co-occurrence pairs
        self.alphabet_x, self.alphabet_y = utils.collect_alphabets(self.cooccs)

        # Initialize the square and non-square contingency table as None
        self._square_ct = None
        self._nonsquare_ct = None

        # Initialize the scorers as None, in a lazy fashion
        self._mle = None
        self._pmi = None
        self._npmi = None
        self._chi2 = None
        self._chi2_nonsquare = None
        self._cramersv = None
        self._cramersv_nonsquare = None
        self._fisher = None
        self._theil_u = None
        self._cond_entropy = None
        self._tresoldi = None

    def _compute_contingency_table(self, square):
        """
        Internal method for computing and caching contingency tables.

        :param bool square: Whether to compute a square (2x2) or non-square
            (3x2) contingency table.
        """

        if square and not self._square_ct:
            self._square_ct = {
                pair: utils.build_ct(self.obs[pair], True) for pair in self.obs
            }

        if not square and not self._nonsquare_ct:
            self._nonsquare_ct = {
                pair: utils.build_ct(self.obs[pair], False) for pair in self.obs
            }

    def mle(self):
        """
        Return an MLE scorer, computing it if necessary.
        """

        # Compute the scorer, if necessary
        if not self._mle:
            self._mle = {
                pair: (
                    self.obs[pair]["11"] / self.obs[pair]["10"],
                    self.obs[pair]["11"] / self.obs[pair]["01"],
                )
                for pair in product(self.alphabet_x, self.alphabet_y)
            }

        return self._mle

    def pmi(self, normalized=False):
        """
        Return a PMI scorer.

        :param bool normalize: Whether to return a normalized PMI or
            not (default: False)
        """

        # Compute the non-normalized scorer, if necessary
        if not normalized and not self._pmi:
            # Use (1/n)^2 as the limit for unobserved pairs
            limit = (1 / len(self.cooccs)) ** 2

            self._pmi = {}
            for pair in self.obs:
                p_xy = self.obs[pair]["11"] / self.obs[pair]["00"]
                p_x = self.obs[pair]["10"] / self.obs[pair]["00"]
                p_y = self.obs[pair]["01"] / self.obs[pair]["00"]

                self._pmi[pair] = (
                    compute_pmi(p_x, p_y, p_xy, False, limit),
                    compute_pmi(p_y, p_x, p_xy, False, limit),
                )

        # Compute the normalized scorer, if necessary
        if normalized and not self._npmi:
            # Use (1/n)^2 as the limit for unobserved pairs
            limit = (1 / len(self.cooccs)) ** 2

            self._npmi = {}
            for pair in self.obs:
                p_xy = self.obs[pair]["11"] / self.obs[pair]["00"]
                p_x = self.obs[pair]["10"] / self.obs[pair]["00"]
                p_y = self.obs[pair]["01"] / self.obs[pair]["00"]

                self._npmi[pair] = (
                    compute_pmi(p_x, p_y, p_xy, True, limit),
                    compute_pmi(p_y, p_x, p_xy, True, limit),
                )

        # Select the scorer to return (this allows easier refactoring later)
        if not normalized:
            ret = self._pmi
        else:
            ret = self._npmi

        return ret

    def chi2(self, square_ct=True):
        """
        Return a Chi2 scorer.

        :param bool square_ct: Whether to return the score over a squared or
            non squared contingency table (default: True)
        """

        # Compute the scorer over a square contingency table, if necessary
        if square_ct and not self._chi2:
            # Compute the square contingency table, if necessary
            self._compute_contingency_table(True)

            # Build the scorer
            self._chi2 = {}
            for pair in self.obs:
                chi2 = ss.chi2_contingency(self._square_ct[pair])[0]
                self._chi2[pair] = (chi2, chi2)

        # Compute the scorer over a non-square contingency table, if necessary
        if square_ct and not self._chi2_nonsquare:
            # Compute the non square contingency table, if necessary
            self._compute_contingency_table(False)

            # Build the scorer
            self._chi2_nonsquare = {}
            for pair in self.obs:
                chi2 = ss.chi2_contingency(self._nonsquare_ct[pair])[0]
                self._chi2_nonsquare[pair] = (chi2, chi2)

        # Select the scorer to return (this allows easier refactoring later)
        if square_ct:
            ret = self._chi2
        else:
            ret = self._chi2_nonsquare

        return ret

    def cramers_v(self, square_ct=True):
        """
        Return a Cramér's V scorer.

        :param bool square_ct: Whether to return the score over a squared or
            non squared contingency table (default: True)
        """

        # Compute the scorer over a square contingency table, if necessary
        if square_ct and not self._cramersv:
            # Compute the square contingency table, if necessary
            self._compute_contingency_table(True)

            # Build the scorer
            self._cramersv = {}
            for pair in self.obs:
                cramersv = compute_cramers_v(self._square_ct[pair])
                self._cramersv[pair] = (cramersv, cramersv)

        # Compute the scorer over a non-square contingency table, if necessary
        if square_ct and not self._cramersv_nonsquare:
            # Compute the non square contingency table, if necessary
            self._compute_contingency_table(False)

            # Build the scorer
            self._cramersv_nonsquare = {}
            for pair in self.obs:
                cramersv = compute_cramers_v(self._nonsquare_ct[pair])
                self._cramersv_nonsquare[pair] = (cramersv, cramersv)

        # Select the scorer to return (this allows easier refactoring later)
        if square_ct:
            ret = self._cramersv
        else:
            ret = self._cramersv_nonsquare

        return ret

    def fisher(self):
        """
        Return a Fisher Exact Odds Ratio scorer.

        Please note that in scipy's implementation the calculated odds ratio
        is different from the one found in R. While the latter returns the
        "conditional Maximum Likelihood Estimate", this implementation computes
        the more common "unconditional Maximum Likelihood Estimate", which
        is known to be very slow for contingency tables with large numbers.
        If the computation is too slow, the similar but inexact chi-square
        test the the chi2 contingency scorer can be used.
        """

        # Compute the square contingency table, if necessary
        self._compute_contingency_table(True)

        # Compute the scorer, if necessary
        if not self._fisher:
            self._fisher = {}
            for pair in self.obs:
                fisher = ss.fisher_exact(self._square_ct[pair])[0]
                self._fisher[pair] = (fisher, fisher)

        return self._fisher

    def theil_u(self):
        """
        Return a Theil's U uncertainty scorer.
        """

        # Compute theil u, if necessary; the code uses two nested loops
        # instead of a product(x, y) to gain some speed
        if not self._theil_u:
            self._theil_u = {}

            for x in self.alphabet_x:
                for y in self.alphabet_y:
                    subset = [
                        pair for pair in self.cooccs if pair[0] == x or pair[1] == y
                    ]
                    X = [pair[0] for pair in subset]
                    Y = [pair[1] for pair in subset]

                    # run theil's
                    self._theil_u[(x, y)] = (
                        compute_theil_u(Y, X),
                        compute_theil_u(X, Y),
                    )

        return self._theil_u

    def cond_entropy(self):
        """
        Return a corrected conditional entropy scorer.
        """

        if not self._cond_entropy:
            self._cond_entropy = {}

            for x in self.alphabet_x:
                for y in self.alphabet_y:
                    subset = [
                        pair for pair in self.cooccs if pair[0] == x or pair[1] == y
                    ]
                    X = [pair[0] for pair in subset]
                    Y = [pair[1] for pair in subset]

                    # run theil's
                    self._cond_entropy[(x, y)] = (
                        conditional_entropy(Y, X),
                        conditional_entropy(X, Y),
                    )

        return self._cond_entropy

    def tresoldi(self):
        """
        Return a `tresoldi` asymmetric uncertainty scorer.

        This is our intended scorer for alignments.
        """

        # Build the scorer, if necessary
        if not self._tresoldi:
            # Obtain the MLE and PMI scorers for all pairs (which will
            # trigger their computation, if necessary)
            mle = self.mle()
            pmi = self.pmi()

            # Build the new scorer
            self._tresoldi = {}
            for pair in self.obs:
                if pmi[pair][0] < 0:
                    xy = -((-pmi[pair][0]) ** (1 - mle[pair][0]))
                else:
                    xy = pmi[pair][0] ** (1 - mle[pair][0])

                if pmi[pair][1] < 0:
                    yx = -((-pmi[pair][1]) ** (1 - mle[pair][1]))
                else:
                    yx = pmi[pair][1] ** (1 - mle[pair][1])

                self._tresoldi[pair] = (xy, yx)

        return self._tresoldi
