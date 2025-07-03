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

import math

# Import Python standard libraries
from collections import Counter
from itertools import chain, product
from typing import Dict, List, Optional, Tuple, Union, Any

# Import 3rd party libraries
import numpy as np
import scipy.stats as ss  # type: ignore

# import local modules
from . import common


def conditional_entropy(x_symbols: List[Any], y_symbols: List[Any]) -> float:
    """
    Computes the entropy of `x` given `y`.

    Parameters
    ----------
    x_symbols : List[Any]
        A list of all observed `x` symbols.
    y_symbols : List[Any]
        A list of all observed `y` symbols.

    Returns
    -------
    float
        The conditional entropy of `x` given `y`.
    """

    # Cache counters; while the xy_counter is already computed in other
    # parts, particularly in the scorers, it is worth repeating the code
    # here to have a more general function.
    y_counter = Counter(y_symbols)
    xy_counter = Counter(list(zip(x_symbols, y_symbols)))
    population = sum(y_counter.values())

    # Compute the entropy and return
    entropy = 0.0
    for xy_pair, xy_count in xy_counter.items():
        p_xy = xy_count / population
        p_y = y_counter[xy_pair[1]] / population
        entropy += p_xy * math.log(p_y / p_xy)

    return entropy


def compute_cramers_v(cont_table: List[List[float]]) -> float:
    """
    Compute Cramer's V from a contingency table.

    Parameters
    ----------
    cont_table : np.array
        The contingency table for computation.

    Returns
    -------
    float
        The Cramér's V measure for the given contingency table.
    """

    # Cache the shape and sum of the contingency table
    rows = len(cont_table)
    cols = len(cont_table[0])
    population = sum([sum(r) for r in cont_table])

    # Compute chi2 and phi2
    chi2 = ss.chi2_contingency(cont_table)[0]
    phi2 = chi2 / population

    # Compute the correlations for Cramér's V
    phi2corr = max(0, phi2 - ((cols - 1) * (rows - 1)) / (population - 1))
    rcorr = rows - ((rows - 1) ** 2) / (population - 1)
    kcorr = cols - ((cols - 1) ** 2) / (population - 1)

    # Compute Cramér's from the correlations
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def compute_pmi(p_x: float, p_y: float, p_xy: float, normalized: bool, limit: float = 1e-6) -> float:
    """
    Compute the Pointwise Mutual Information.

    Parameters
    ----------
    p_x : float
        The probability of `x`, p(x).
    p_y : float
        The probability of `y`, p(y).
    p_xy : float
        The probability of `xy`, p(xy).
    normalized : bool
        Whether to return the normalized Pointwise Mutual Information in
        range [-1, 1].
    limit : float, optional
        The value to use for computation when `p_xy` is 0.0 (i.e., non
        observed), as the limit of PMI to zero (default: 1e-6).

    Returns
    -------
    float
        The Pointwise Mutual Information.
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


def compute_theil_u(x_symbols: List[Any], y_symbols: List[Any]) -> float:
    """
    Compute the uncertainty coefficient Theil's U.

    Parameters
    ----------
    x_symbols : List[Any]
        The list of observed symbols in series `x`.
    y_symbols : List[Any]
        The list of observed symbols in series `y`.

    Returns
    -------
    float
        The uncertainty coefficient given `x` and `y`.
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
def scale_scorer(scorer: Dict[Tuple[Any, Any], Tuple[float, float]], method: str = "minmax", nrange: Tuple[float, float] = (0, 1)) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
    """
    Scale a scorer.

    The function returns a scaled version of a scorer considering all
    the asymmetric scorers (i.e., both `x` given `y` and `y` given `x`).
    Implemented scoring methods are "minmax" (by default on range [0, 1],
    which can be modified by the `nrange` parameter) and "mean".

    Parameters
    ----------
    scorer : dict
        A scoring dictionary.
    method : str, optional
        The scoring method to be used, either `"minmax"` or `"mean"` or
        `"stdev"` (default: `"minmax"`).
    nrange : tuple, optional
        A tuple with the scaling range, to be used when applicable
        (default: (0, 1)).

    Returns
    -------
    dict
        A scaled version of the scorer.
    """

    # Extract scores as a list, combining `xy` and `yx`
    scores = list(chain.from_iterable(scorer.values()))

    # normalization is performed over xy and yx together
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
            pair: ((value[0] - mean) / score_diff, (value[1] - mean) / score_diff) for pair, value in scorer.items()
        }
    elif method == "stdev":
        mean = np.mean(scores)
        stdev = np.std(scores)

        scaled_scorer = {pair: ((value[0] - mean) / stdev, (value[1] - mean) / stdev) for pair, value in scorer.items()}
    else:
        raise ValueError("Unknown scaling method.")

    return scaled_scorer


def invert_scorer(scorer: Dict[Tuple[Any, Any], Tuple[float, float]]) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
    """
    Inverts a scorer, so that the higher the affinity, the higher the score.

    It is recommended than only scorers in range [0..] are inverted.

    Parameters
    ----------
    scorer : dict
        A scoring dictionary.

    Returns
    -------
    dict
        An inverted version of the scorer.
    """

    # Collect the highest overall value
    scores = list(chain.from_iterable(scorer.values()))
    max_score = max(scores)

    inverted_scorer = {coocc: tuple([max_score - value for value in values]) for coocc, values in scorer.items()}

    return inverted_scorer


def scorer2matrices(scorer: Dict[Tuple[Any, Any], Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, List[Any], List[Any]]:
    """
    Return the asymmetric matrices implied by a scorer and their alphabets.

    Parameters
    ----------
    scorer : dict
        A scoring dictionary.

    Returns
    -------
    tuple
        A tuple with the following elements: a scoring matrix of `y` given `x`,
        a scoring matrix of `x` given `y`, the alphabet for matrix `x`, and
        the alphabet for matrix `y`.
    """

    alphabet_x, alphabet_y = common.collect_alphabets(list(scorer))

    xy = np.array([np.array([scorer[(x, y)][0] for x in alphabet_x]) for y in alphabet_y])
    yx = np.array([np.array([scorer[(x, y)][1] for y in alphabet_y]) for x in alphabet_x])

    return xy, yx, alphabet_x, alphabet_y


class CatScorer:
    """
    Class for computing categorical co-occurrence scores.
    """

    def __init__(self, cooccs: List[Tuple[Any, Any]]):
        """
        Initialization function.

        Parameters
        ----------
        cooccs : List[Tuple[Any, Any]]
            A list of co-occurrence tuples.
        """

        # Store cooccs, observations, symbols and alphabets
        self.cooccs: List[Tuple[Any, Any]] = cooccs
        self.obs: Dict[Tuple[Any, Any], Dict[str, int]] = common.collect_observations(cooccs)

        # Obtain the alphabets from the co-occurrence pairs
        self.alphabet_x: List[Any]
        self.alphabet_y: List[Any]
        self.alphabet_x, self.alphabet_y = common.collect_alphabets(self.cooccs)

        # Initialize the square and non-square contingency table as None
        self._square_ct: Optional[Dict[Tuple[Any, Any], List[List[float]]]] = None
        self._nonsquare_ct: Optional[Dict[Tuple[Any, Any], List[List[float]]]] = None

        # Initialize the scorers as None, in a lazy fashion
        self._mle: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._pmi: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._npmi: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._chi2: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._chi2_nonsquare: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._cramersv: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._cramersv_nonsquare: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._fisher: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._theil_u: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._cond_entropy: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None
        self._tresoldi: Optional[Dict[Tuple[Any, Any], Tuple[float, float]]] = None

    def _compute_contingency_table_scorer(self, square: bool, computation_func, square_cache_attr: str, nonsquare_cache_attr: str) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
        """
        Helper method to compute scorers based on contingency tables.
        
        Parameters
        ----------
        square : bool
            Whether to compute square or non-square contingency table
        computation_func : callable
            Function to compute the score from contingency table
        square_cache_attr : str
            Name of the attribute to cache square results
        nonsquare_cache_attr : str
            Name of the attribute to cache non-square results
            
        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            The computed scorer dictionary
        """
        if square:
            cache_attr = square_cache_attr
            ct_attr = '_square_ct'
        else:
            cache_attr = nonsquare_cache_attr
            ct_attr = '_nonsquare_ct'
            
        if not getattr(self, cache_attr):
            # Compute the contingency table if necessary
            self._compute_contingency_table(square)
            
            # Build the scorer
            scorer = {}
            ct_dict = getattr(self, ct_attr)
            for pair in self.obs:
                score = computation_func(ct_dict[pair])
                scorer[pair] = (score, score)
            
            setattr(self, cache_attr, scorer)
        
        return getattr(self, cache_attr)

    def _compute_probabilities(self, pair: Tuple[Any, Any]) -> Tuple[float, float, float]:
        """
        Helper method to compute probabilities p(x), p(y), p(xy) for a given pair.
        
        Parameters
        ----------
        pair : Tuple[Any, Any]
            The symbol pair to compute probabilities for
            
        Returns
        -------
        Tuple[float, float, float]
            Tuple of (p_x, p_y, p_xy) probabilities
        """
        obs = self.obs[pair]
        total = obs["00"]
        p_x = obs["10"] / total
        p_y = obs["01"] / total  
        p_xy = obs["11"] / total
        return p_x, p_y, p_xy

    def _compute_contingency_table(self, square: bool) -> None:
        """
        Internal method for computing and caching contingency tables.

        Parameters
        ----------
        square : bool
            Whether to compute a square (2x2) or non-square (3x2) contingency
            table.
        """

        if square and not self._square_ct:
            self._square_ct = {pair: common.build_ct(self.obs[pair], True) for pair in self.obs}

        if not square and not self._nonsquare_ct:
            self._nonsquare_ct = {pair: common.build_ct(self.obs[pair], False) for pair in self.obs}

    def mle(self) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
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

    def pmi(self, normalized: bool = False) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
        """
        Return a PMI scorer.

        Parameters
        ----------
        normalized : bool
            Whether to return a normalized PMI or not (default: False)
        """

        # Compute the non-normalized scorer, if necessary
        if not normalized and not self._pmi:
            # Use (1/n)^2 as the limit for unobserved pairs
            limit = (1 / len(self.cooccs)) ** 2

            self._pmi = {}
            for pair in self.obs:
                p_x, p_y, p_xy = self._compute_probabilities(pair)
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
                p_x, p_y, p_xy = self._compute_probabilities(pair)
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

    def chi2(self, square_ct: bool = True) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
        """
        Return a Chi2 scorer.

        Parameters
        ----------
        square_ct : bool
            Whether to return the score over a squared or
            non squared contingency table (default: True)
        """
        return self._compute_contingency_table_scorer(
            square_ct,
            lambda ct: ss.chi2_contingency(ct)[0],
            '_chi2',
            '_chi2_nonsquare'
        )

    def cramers_v(self, square_ct: bool = True) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
        """
        Return a Cramér's V scorer.

        Parameters
        ----------
        square_ct : bool
            Whether to return the score over a squared or
            non squared contingency table (default: True)
        """
        return self._compute_contingency_table_scorer(
            square_ct,
            compute_cramers_v,
            '_cramersv',
            '_cramersv_nonsquare'
        )

    def fisher(self) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
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

    def theil_u(self) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
        """
        Return a Theil's U uncertainty scorer.
        """

        # Compute theil u, if necessary; optimize with numpy arrays when possible
        if not self._theil_u:
            self._theil_u = {}
            
            # Convert to numpy arrays for potentially faster filtering
            import numpy as np
            cooccs_array = np.array(self.cooccs)
            
            for x in self.alphabet_x:
                for y in self.alphabet_y:
                    # Use numpy boolean indexing for faster filtering on large datasets
                    if len(self.cooccs) > 1000:  # Only use numpy for larger datasets
                        mask = (cooccs_array[:, 0] == x) | (cooccs_array[:, 1] == y)
                        subset = cooccs_array[mask].tolist()
                    else:
                        subset = [pair for pair in self.cooccs if pair[0] == x or pair[1] == y]
                    
                    X = [pair[0] for pair in subset]
                    Y = [pair[1] for pair in subset]

                    # run theil's
                    self._theil_u[(x, y)] = (
                        compute_theil_u(Y, X),
                        compute_theil_u(X, Y),
                    )

        return self._theil_u

    def cond_entropy(self) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
        """
        Return a corrected conditional entropy scorer.
        """

        if not self._cond_entropy:
            self._cond_entropy = {}

            for x in self.alphabet_x:
                for y in self.alphabet_y:
                    subset = [pair for pair in self.cooccs if pair[0] == x or pair[1] == y]
                    X = [pair[0] for pair in subset]
                    Y = [pair[1] for pair in subset]

                    # run conditional entropy
                    self._cond_entropy[(x, y)] = (
                        conditional_entropy(Y, X),
                        conditional_entropy(X, Y),
                    )

        return self._cond_entropy

    def tresoldi(self) -> Dict[Tuple[Any, Any], Tuple[float, float]]:
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
