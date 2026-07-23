# Note: includes code modified from
# - https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792
# - https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

"""
Defines the various scorers for categorical co-occurrence analysis.
"""

# TODO: have a function for checking if we have enough data for a chi2, combined with smoothing
# TODO: extend scipy options in chi2_contingency
# TODO: investigate if it is worth reusing chi2 statistics for cramer
# TODO: include a logarithmic scaler (instead of percentile one)
# TODO: allow to combine scalers? log within range?

import math

# Import Python standard libraries
from collections import Counter
from itertools import chain, product
from typing import Any, NamedTuple

# Import 3rd party libraries
import numpy as np
import scipy.stats as ss  # type: ignore
from freqprob import MLE, Laplace, Lidstone

# import local modules
from . import common


def conditional_entropy(x_symbols: list[Any], y_symbols: list[Any]) -> float:
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
    xy_counter = Counter(list(zip(x_symbols, y_symbols, strict=False)))
    population = sum(y_counter.values())

    # Compute the entropy and return
    entropy = 0.0
    for xy_pair, xy_count in xy_counter.items():
        p_xy = xy_count / population
        p_y = y_counter[xy_pair[1]] / population
        entropy += p_xy * math.log(p_y / p_xy)

    return entropy


def compute_cramers_v(cont_table: list[list[float]]) -> float:
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


def compute_theil_u(x_symbols: list[Any], y_symbols: list[Any]) -> float:
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
    p_x = [n / population for n in x_counter.values()]
    h_x = ss.entropy(p_x)

    # If the entropy is zero (all items alike), the uncertainty coeffiecient
    # is by definition 1.0; otherwise, compute it in relation to the
    # conditional entropy
    theil_u = 1.0 if h_x == 0.0 else (h_x - h_xy) / h_x

    return theil_u


def compute_mutual_information(x_symbols: list[Any], y_symbols: list[Any]) -> float:
    """
    Compute the mutual information between X and Y.

    MI(X;Y) = Σ_x Σ_y p(x,y) * log(p(x,y) / (p(x) * p(y)))

    Parameters
    ----------
    x_symbols : List[Any]
        A list of all observed `x` symbols.
    y_symbols : List[Any]
        A list of all observed `y` symbols.

    Returns
    -------
    float
        The mutual information between X and Y.
    """
    if len(x_symbols) != len(y_symbols):
        raise ValueError("x_symbols and y_symbols must have the same length")

    if not x_symbols:
        return 0.0

    # Count joint and marginal frequencies
    xy_counter = Counter(list(zip(x_symbols, y_symbols, strict=False)))
    x_counter = Counter(x_symbols)
    y_counter = Counter(y_symbols)

    population = len(x_symbols)
    mi = 0.0

    # Compute mutual information
    for (x, y), xy_count in xy_counter.items():
        p_xy = xy_count / population
        p_x = x_counter[x] / population
        p_y = y_counter[y] / population

        # Add log term with proper handling of zero probabilities
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))

    return mi


def compute_normalized_mutual_information(x_symbols: list[Any], y_symbols: list[Any]) -> float:
    """
    Compute the normalized mutual information between X and Y.

    NMI(X;Y) = MI(X;Y) / H(X,Y)
    where H(X,Y) is the joint entropy.

    Parameters
    ----------
    x_symbols : List[Any]
        A list of all observed `x` symbols.
    y_symbols : List[Any]
        A list of all observed `y` symbols.

    Returns
    -------
    float
        The normalized mutual information between X and Y, in range [0, 1].
    """
    if len(x_symbols) != len(y_symbols):
        raise ValueError("x_symbols and y_symbols must have the same length")

    if not x_symbols:
        return 0.0

    # Compute mutual information
    mi = compute_mutual_information(x_symbols, y_symbols)

    # Compute joint entropy H(X,Y)
    xy_counter = Counter(list(zip(x_symbols, y_symbols, strict=False)))
    population = len(x_symbols)

    joint_entropy = 0.0
    for xy_count in xy_counter.values():
        p_xy = xy_count / population
        if p_xy > 0:
            joint_entropy -= p_xy * math.log(p_xy)

    # Normalize MI by joint entropy
    if joint_entropy == 0.0:
        return 0.0 if mi == 0.0 else 1.0  # Handle edge case

    return mi / joint_entropy


def compute_jaccard_index(x_contexts: list[Any], y_contexts: list[Any]) -> float:
    """
    Compute the Jaccard Index between two sets of contexts.

    J(X,Y) = |X ∩ Y| / |X ∪ Y|

    Parameters
    ----------
    x_contexts : List[Any]
        Contexts where symbol x appears.
    y_contexts : List[Any]
        Contexts where symbol y appears.

    Returns
    -------
    float
        The Jaccard Index between the two context sets, in range [0, 1].
    """
    set_x = set(x_contexts)
    set_y = set(y_contexts)

    intersection = len(set_x & set_y)
    union = len(set_x | set_y)

    if union == 0:
        return 0.0

    return intersection / union


def compute_goodman_kruskal_lambda(x_symbols: list[Any], y_symbols: list[Any], direction: str = "y_given_x") -> float:
    """
    Compute Goodman and Kruskal's Lambda for asymmetric association.

    λ(Y|X) = (Σ_x max(n_x,y) - max(n_y)) / (n - max(n_y))

    Parameters
    ----------
    x_symbols : List[Any]
        A list of all observed `x` symbols.
    y_symbols : List[Any]
        A list of all observed `y` symbols.
    direction : str
        Either "y_given_x" or "x_given_y" to specify direction.

    Returns
    -------
    float
        The Goodman-Kruskal Lambda coefficient, in range [0, 1].
    """
    if len(x_symbols) != len(y_symbols):
        raise ValueError("x_symbols and y_symbols must have the same length")

    if not x_symbols:
        return 0.0

    if direction == "y_given_x":
        # λ(Y|X): How much does knowing X reduce error in predicting Y
        predictors = x_symbols
        predicted = y_symbols
    else:
        # λ(X|Y): How much does knowing Y reduce error in predicting X
        predictors = y_symbols
        predicted = x_symbols

    # Count marginal frequencies of predicted variable
    predicted_counter = Counter(predicted)
    total_n = len(predicted)

    # Maximum marginal frequency (mode)
    max_marginal = max(predicted_counter.values())

    # Count conditional frequencies: for each predictor value,
    # find the maximum frequency of predicted values
    predictor_values = set(predictors)
    sum_max_conditional = 0

    for pred_val in predictor_values:
        # Get all predicted values when predictor = pred_val
        conditional_predicted = [predicted[i] for i, p in enumerate(predictors) if p == pred_val]
        if conditional_predicted:
            conditional_counter = Counter(conditional_predicted)
            max_conditional = max(conditional_counter.values())
            sum_max_conditional += max_conditional

    # Compute lambda
    denominator = total_n - max_marginal
    if denominator == 0:
        return 0.0  # Perfect prediction already exists

    lambda_val = (sum_max_conditional - max_marginal) / denominator
    return max(0.0, lambda_val)  # Ensure non-negative


def compute_log_likelihood_ratio(cont_table: list[list[float]]) -> float:
    """
    Compute the Log-Likelihood Ratio (G²) from a contingency table.

    G² = 2 * Σ O_ij * ln(O_ij / E_ij)
    where O_ij are observed frequencies and E_ij are expected frequencies.

    Parameters
    ----------
    cont_table : List[List[float]]
        The contingency table for computation.

    Returns
    -------
    float
        The Log-Likelihood Ratio statistic.
    """
    import numpy as np

    # Convert to numpy array for easier computation
    observed = np.array(cont_table, dtype=float)

    # Compute row and column totals
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()

    if total == 0:
        return 0.0

    # Compute expected frequencies
    expected = np.outer(row_totals, col_totals) / total

    # Compute G² statistic
    g2 = 0.0
    for i in range(observed.shape[0]):
        for j in range(observed.shape[1]):
            o_ij = observed[i, j]
            e_ij = expected[i, j]

            # Only add term if observed frequency > 0
            if o_ij > 0 and e_ij > 0:
                g2 += o_ij * math.log(o_ij / e_ij)

    return 2.0 * g2


def compute_log_likelihood_ratio_pvalue(cont_table: list[list[float]]) -> float:
    """
    Compute the p-value of the Log-Likelihood Ratio (G²) statistic.

    Under the null hypothesis of independence, G² is asymptotically
    chi-square distributed with ``(rows - 1) * (cols - 1)`` degrees of freedom.

    Parameters
    ----------
    cont_table : List[List[float]]
        The contingency table for computation.

    Returns
    -------
    float
        The p-value for the Log-Likelihood Ratio statistic, in range [0, 1].
    """
    g2 = compute_log_likelihood_ratio(cont_table)

    rows = len(cont_table)
    cols = len(cont_table[0]) if rows else 0
    dof = (rows - 1) * (cols - 1)

    # A degenerate table with no degrees of freedom carries no evidence
    # against independence.
    if dof <= 0:
        return 1.0

    return float(ss.chi2.sf(g2, dof))


# TODO: allow independent scaling over `x` and independent over `y` (currently doing all)
# TODO: allow scaling withing percentile borders
# TODO: see if we can vectorize numpy operations (now on dicts)
def scale_scorer(
    scorer: dict[tuple[Any, Any], tuple[float, float]], method: str = "minmax", nrange: tuple[float, float] = (0, 1)
) -> dict[tuple[Any, Any], tuple[float, float]]:
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
            pair: (float((value[0] - mean) / score_diff), float((value[1] - mean) / score_diff))
            for pair, value in scorer.items()
        }
    elif method == "stdev":
        mean = np.mean(scores)
        stdev = np.std(scores)

        scaled_scorer = {
            pair: (float((value[0] - mean) / stdev), float((value[1] - mean) / stdev)) for pair, value in scorer.items()
        }
    else:
        raise ValueError("Unknown scaling method.")

    return scaled_scorer


def invert_scorer(scorer: dict[tuple[Any, Any], tuple[float, float]]) -> dict[tuple[Any, Any], tuple[float, float]]:
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

    inverted_scorer = {
        coocc: (float(max_score - values[0]), float(max_score - values[1])) for coocc, values in scorer.items()
    }

    return inverted_scorer


def scorer2matrices(
    scorer: dict[tuple[Any, Any], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, list[Any], list[Any]]:
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

    xy = np.array([np.array([scorer[(x, y)][0] for y in alphabet_y]) for x in alphabet_x])
    yx = np.array([np.array([scorer[(x, y)][1] for x in alphabet_x]) for y in alphabet_y])

    return xy, yx, alphabet_x, alphabet_y


class _SubsetEntropies(NamedTuple):
    """
    Per-pair subset entropies shared by :meth:`CatScorer.theil_u` and
    :meth:`CatScorer.cond_entropy`.

    Each field is an ``|alphabet_x| x |alphabet_y|`` array indexed by ``[i, j]``,
    holding a quantity computed over the subset of co-occurrences selected for
    the pair ``(alphabet_x[i], alphabet_y[j])`` -- namely those whose first
    element is ``alphabet_x[i]`` or whose second element is ``alphabet_y[j]``.

    The ``*_zero`` fields are boolean masks flagging the pairs whose marginal
    entropy is exactly zero (a degenerate, single-symbol subset), derived from
    the integer counts so they match the scalar special cases exactly.
    """

    h_x: np.ndarray  # H(X) over the subset
    h_y: np.ndarray  # H(Y) over the subset
    h_xy: np.ndarray  # joint entropy H(X, Y) over the subset
    h_x_given_y: np.ndarray  # H(X | Y) over the subset
    h_y_given_x: np.ndarray  # H(Y | X) over the subset
    h_x_zero: np.ndarray  # True where H(X) == 0
    h_y_zero: np.ndarray  # True where H(Y) == 0


class CatScorer:
    """
    Class for computing categorical co-occurrence scores.
    """

    def __init__(self, cooccs: list[tuple[Any, Any]], smoothing_method: str = "mle", smoothing_alpha: float = 1.0):
        """
        Initialization function.

        Parameters
        ----------
        cooccs : List[Tuple[Any, Any]]
            A list of co-occurrence tuples.
        smoothing_method : str, optional
            Smoothing method for probability estimation. Options: 'mle', 'laplace', 'ele'.
            Default is 'mle'.
        smoothing_alpha : float, optional
            Smoothing parameter (alpha for Laplace/ELE). Default is 1.0.
        """

        # Store cooccs, observations, symbols and alphabets
        self.cooccs: list[tuple[Any, Any]] = cooccs
        self.obs: dict[tuple[Any, Any], dict[str, int]] = common.collect_observations(cooccs)

        # Obtain the alphabets from the co-occurrence pairs
        self.alphabet_x: list[Any]
        self.alphabet_y: list[Any]
        self.alphabet_x, self.alphabet_y = common.collect_alphabets(self.cooccs)

        # Store smoothing configuration
        self.smoothing_method = smoothing_method.lower()
        self.smoothing_alpha = smoothing_alpha

        # Store smoothing parameters - freqprob scorers will be created when needed
        # since they require frequency distributions which are computed per pair
        self._freqprob_scorer_class: type[MLE] | type[Laplace] | type[Lidstone] | None = None
        if self.smoothing_method == "mle":
            self._freqprob_scorer_class = MLE
        elif self.smoothing_method == "laplace":
            self._freqprob_scorer_class = Laplace
        elif self.smoothing_method == "lidstone":
            self._freqprob_scorer_class = Lidstone
        else:
            raise ValueError(f"Unsupported smoothing method: {smoothing_method}. Use 'mle', 'laplace', or 'lidstone'.")

        # Cache for freqprob scorers per context
        self._freqprob_cache: dict[str, Any] = {}

        # Initialize the square and non-square contingency table as None
        self._square_ct: dict[tuple[Any, Any], list[list[float]]] | None = None
        self._nonsquare_ct: dict[tuple[Any, Any], list[list[float]]] | None = None

        # Initialize the scorers as None, in a lazy fashion
        self._mle: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._pmi: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._npmi: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._chi2: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._chi2_nonsquare: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._cramersv: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._cramersv_nonsquare: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._fisher: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._theil_u: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._cond_entropy: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._tresoldi: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._mutual_information: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._normalized_mutual_information: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._jaccard_index: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._goodman_kruskal_lambda: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._log_likelihood_ratio: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._log_likelihood_ratio_nonsquare: dict[tuple[Any, Any], tuple[float, float]] | None = None

        # p-value scorers for the statistical tests, lazily computed
        self._chi2_pvalue: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._chi2_pvalue_nonsquare: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._fisher_pvalue: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._log_likelihood_ratio_pvalue: dict[tuple[Any, Any], tuple[float, float]] | None = None
        self._log_likelihood_ratio_pvalue_nonsquare: dict[tuple[Any, Any], tuple[float, float]] | None = None

    def _compute_contingency_table_scorer(
        self, square: bool, computation_func, square_cache_attr: str, nonsquare_cache_attr: str
    ) -> dict[tuple[Any, Any], tuple[float, float]]:
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
            ct_attr = "_square_ct"
        else:
            cache_attr = nonsquare_cache_attr
            ct_attr = "_nonsquare_ct"

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

    def _compute_probabilities(self, pair: tuple[Any, Any]) -> tuple[float, float, float]:
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

    def mle(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return an MLE scorer, computing it if necessary.
        Uses freqprob for robust probability estimation with optional smoothing.
        """

        # Compute the scorer, if necessary
        if not self._mle:
            self._mle = {}

            for pair in product(self.alphabet_x, self.alphabet_y):
                obs = self.obs[pair]

                # Use freqprob for robust probability estimation
                # Create frequency distributions for conditional probabilities

                # P(X|Y) - distribution of X values given Y
                if obs["01"] > 0:  # Y marginal count > 0
                    if self.smoothing_method == "mle":
                        # For MLE, use simple division to avoid log(0) issues
                        xy_score = obs["11"] / obs["01"]
                    else:
                        # For smoothing methods, use freqprob which handles zeros better
                        freq_dist_x_given_y = {pair[0]: obs["11"], f"NOT_{pair[0]}": obs["01"] - obs["11"]}

                        if self._freqprob_scorer_class is None:
                            raise ValueError("Freqprob scorer class not initialized")

                        if self.smoothing_method == "mle" or self.smoothing_method == "laplace":
                            scorer_xy = self._freqprob_scorer_class(freq_dist_x_given_y)  # type: ignore[call-arg]
                        else:  # Lidstone
                            scorer_xy = self._freqprob_scorer_class(freq_dist_x_given_y, gamma=self.smoothing_alpha)  # type: ignore[call-arg]

                        xy_score = math.exp(scorer_xy(pair[0]))
                else:
                    xy_score = 0.0

                # P(Y|X) - distribution of Y values given X
                if obs["10"] > 0:  # X marginal count > 0
                    if self.smoothing_method == "mle":
                        # For MLE, use simple division to avoid log(0) issues
                        yx_score = obs["11"] / obs["10"]
                    else:
                        # For smoothing methods, use freqprob which handles zeros better
                        freq_dist_y_given_x = {pair[1]: obs["11"], f"NOT_{pair[1]}": obs["10"] - obs["11"]}

                        if self._freqprob_scorer_class is None:
                            raise ValueError("Freqprob scorer class not initialized")

                        if self.smoothing_method == "mle" or self.smoothing_method == "laplace":
                            scorer_yx = self._freqprob_scorer_class(freq_dist_y_given_x)  # type: ignore[call-arg]
                        else:  # Lidstone
                            scorer_yx = self._freqprob_scorer_class(freq_dist_y_given_x, gamma=self.smoothing_alpha)  # type: ignore[call-arg]

                        yx_score = math.exp(scorer_yx(pair[1]))
                else:
                    yx_score = 0.0

                self._mle[pair] = (xy_score, yx_score)

        return self._mle

    def get_smoothed_probabilities(self) -> dict[str, dict[tuple[Any, Any], float]]:
        """
        Return smoothed probability estimates using the configured freqprob method.

        Returns
        -------
        Dict[str, Dict[Tuple[Any, Any], float]]
            Dictionary containing 'xy_given_y', 'yx_given_x', 'joint', 'marginal_x', 'marginal_y' probabilities.
        """
        results: dict[str, dict[tuple[Any, Any], float]] = {
            "xy_given_y": {},  # P(X|Y)
            "yx_given_x": {},  # P(Y|X)
            "joint": {},  # P(X,Y)
            "marginal_x": {},  # P(X)
            "marginal_y": {},  # P(Y)
        }

        # Create global frequency distributions

        # Build frequency distributions - cast keys to work with freqprob API
        from typing import cast

        joint_freqdist = cast(dict[Any, int], {pair: obs["11"] for pair, obs in self.obs.items()})
        x_freqdist: dict[Any, int] = {}
        y_freqdist: dict[Any, int] = {}

        for pair in product(self.alphabet_x, self.alphabet_y):
            obs = self.obs[pair]
            if pair[0] not in x_freqdist:
                x_freqdist[pair[0]] = 0
            if pair[1] not in y_freqdist:
                y_freqdist[pair[1]] = 0
            x_freqdist[pair[0]] += obs["10"]
            y_freqdist[pair[1]] += obs["01"]

        # Create freqprob scorers
        if self._freqprob_scorer_class is None:
            raise ValueError("Freqprob scorer class not initialized")

        if self.smoothing_method == "mle" or self.smoothing_method == "laplace":
            joint_scorer = self._freqprob_scorer_class(joint_freqdist)  # type: ignore[call-arg]
            x_scorer = self._freqprob_scorer_class(x_freqdist)  # type: ignore[call-arg]
            y_scorer = self._freqprob_scorer_class(y_freqdist)  # type: ignore[call-arg]
        else:  # Lidstone
            joint_scorer = self._freqprob_scorer_class(joint_freqdist, gamma=self.smoothing_alpha)  # type: ignore[call-arg]
            x_scorer = self._freqprob_scorer_class(x_freqdist, gamma=self.smoothing_alpha)  # type: ignore[call-arg]
            y_scorer = self._freqprob_scorer_class(y_freqdist, gamma=self.smoothing_alpha)  # type: ignore[call-arg]

        for pair in product(self.alphabet_x, self.alphabet_y):
            obs = self.obs[pair]

            # Get smoothed probabilities (note: freqprob returns log probabilities by default)
            joint_prob = math.exp(joint_scorer(pair)) if pair in joint_freqdist else 0.0
            x_prob = math.exp(x_scorer(pair[0])) if pair[0] in x_freqdist else 0.0
            y_prob = math.exp(y_scorer(pair[1])) if pair[1] in y_freqdist else 0.0

            results["joint"][pair] = joint_prob
            results["marginal_x"][pair[0]] = x_prob
            results["marginal_y"][pair[1]] = y_prob

            # Conditional probabilities using Bayes' rule
            if obs["01"] > 0 and y_prob > 0:  # Y marginal count > 0
                results["xy_given_y"][pair] = joint_prob / y_prob
            else:
                results["xy_given_y"][pair] = 0.0

            if obs["10"] > 0 and x_prob > 0:  # X marginal count > 0
                results["yx_given_x"][pair] = joint_prob / x_prob
            else:
                results["yx_given_x"][pair] = 0.0

        return results

    def pmi_smoothed(self, normalized: bool = False) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a PMI scorer using freqprob smoothing for better numerical stability.

        Parameters
        ----------
        normalized : bool
            Whether to return normalized PMI (NPMI) or standard PMI (default: False)

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            PMI scores for each pair (X→Y, Y→X)
        """
        probs = self.get_smoothed_probabilities()
        pmi_scores = {}

        for pair in product(self.alphabet_x, self.alphabet_y):
            p_x = probs["marginal_x"].get(pair[0], 0.0)
            p_y = probs["marginal_y"].get(pair[1], 0.0)
            p_xy = probs["joint"].get(pair, 0.0)

            # PMI(X,Y) = log(P(X,Y) / (P(X) * P(Y)))
            if p_x > 0 and p_y > 0 and p_xy > 0:
                pmi_xy = math.log(p_xy / (p_x * p_y))
                pmi_yx = math.log(p_xy / (p_y * p_x))  # Symmetric for PMI

                if normalized:
                    # NPMI = PMI / -log(P(X,Y))
                    if p_xy > 0:
                        pmi_xy = pmi_xy / (-math.log(p_xy))
                        pmi_yx = pmi_yx / (-math.log(p_xy))
            else:
                pmi_xy = pmi_yx = 0.0

            pmi_scores[pair] = (pmi_xy, pmi_yx)

        return pmi_scores

    def pmi(self, normalized: bool = False) -> dict[tuple[Any, Any], tuple[float, float]]:
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
            assert ret is not None, "PMI should have been computed"  # nosec
        else:
            ret = self._npmi
            assert ret is not None, "NPMI should have been computed"  # nosec

        return ret

    def chi2(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Chi2 scorer.

        Parameters
        ----------
        square_ct : bool
            Whether to return the score over a squared or
            non squared contingency table (default: True)
        """
        return self._compute_contingency_table_scorer(
            square_ct, lambda ct: ss.chi2_contingency(ct)[0], "_chi2", "_chi2_nonsquare"
        )

    def chi2_pvalue(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return the p-values of the chi-square test of independence.

        This is the significance counterpart to :meth:`chi2`: for each pair it
        returns the p-value from ``scipy.stats.chi2_contingency`` (a small value
        indicates a statistically significant association). As with the other
        symmetric statistical tests, both tuple entries hold the same value.

        Parameters
        ----------
        square_ct : bool
            Whether to compute the p-value over a squared (2x2) or non-squared
            (3x2) contingency table (default: True).

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (p-value, p-value) tuples.
        """
        return self._compute_contingency_table_scorer(
            square_ct, lambda ct: ss.chi2_contingency(ct)[1], "_chi2_pvalue", "_chi2_pvalue_nonsquare"
        )

    def cramers_v(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Cramér's V scorer.

        Parameters
        ----------
        square_ct : bool
            Whether to return the score over a squared or
            non squared contingency table (default: True)
        """
        return self._compute_contingency_table_scorer(square_ct, compute_cramers_v, "_cramersv", "_cramersv_nonsquare")

    def fisher(self) -> dict[tuple[Any, Any], tuple[float, float]]:
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
            assert self._square_ct is not None, "Square contingency table should have been computed"  # nosec
            self._fisher = {}
            for pair in self.obs:
                fisher = ss.fisher_exact(self._square_ct[pair])[0]
                self._fisher[pair] = (fisher, fisher)

        return self._fisher

    def fisher_pvalue(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return the p-values of Fisher's exact test.

        This is the significance counterpart to :meth:`fisher`: for each pair it
        returns the (two-sided) p-value from ``scipy.stats.fisher_exact`` over
        the 2x2 contingency table. As with the other symmetric statistical
        tests, both tuple entries hold the same value.

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (p-value, p-value) tuples.
        """

        # Compute the square contingency table, if necessary
        self._compute_contingency_table(True)

        if not self._fisher_pvalue:
            assert self._square_ct is not None, "Square contingency table should have been computed"  # nosec
            self._fisher_pvalue = {}
            for pair in self.obs:
                pvalue = ss.fisher_exact(self._square_ct[pair])[1]
                self._fisher_pvalue[pair] = (pvalue, pvalue)

        return self._fisher_pvalue

    def _joint_counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the global joint-count matrix and its marginals.

        Returns ``(joint, row_sum, col_sum)`` where ``joint[i, j]`` is the
        number of times co-occurrence ``(alphabet_x[i], alphabet_y[j])`` is
        observed, ``row_sum[i]`` is the global count of x symbol ``i`` and
        ``col_sum[j]`` is the global count of y symbol ``j``. Shared by the
        vectorized entropy- and mode-based scorers.
        """
        x_index = {x: i for i, x in enumerate(self.alphabet_x)}
        y_index = {y: j for j, y in enumerate(self.alphabet_y)}

        joint = np.zeros((len(self.alphabet_x), len(self.alphabet_y)))
        for (a, b), count in Counter(self.cooccs).items():
            joint[x_index[a], y_index[b]] += count

        return joint, joint.sum(axis=1), joint.sum(axis=0)

    def _subset_entropy_components(self) -> _SubsetEntropies:
        """
        Vectorized per-pair subset entropies backing :meth:`theil_u`,
        :meth:`cond_entropy`, :meth:`mutual_information` and
        :meth:`normalized_mutual_information`.

        These scorers operate on the subset selected for a pair ``(x, y)`` --
        co-occurrences whose first element is ``x`` or whose second element is
        ``y``. That subset's joint distribution is shaped like a "cross": the
        whole row ``x`` and the whole column ``y`` of the global joint-count
        matrix ``J``, with every other cell empty. This structure lets each of
        the subset entropies be written in closed form from ``J``'s row/column
        sums and a handful of precomputed per-row / per-column vectors, so all
        pairs are evaluated together with array broadcasting.

        This reduces the cost from ``O(|X| * |Y| * n)`` (re-scanning the ``n``
        co-occurrences for every pair) to ``O(|X| * |Y|)``, while reproducing
        the original per-pair results to floating-point precision.
        """

        # Joint-count matrix J[i, j] and its row (R) and column (C) marginals.
        joint, row_sum, col_sum = self._joint_counts()

        # `J log J` (with the standard 0 * log 0 == 0 convention), plus its row
        # and column sums; every logarithm below guards zeros explicitly.
        with np.errstate(divide="ignore", invalid="ignore"):
            joint_log = joint * np.where(joint > 0, np.log(joint), 0.0)
        row_log_sum = joint_log.sum(axis=1)
        col_log_sum = joint_log.sum(axis=0)

        # rowTerm[i] = sum_j J[i,j] log(R[i] / J[i,j]); colTerm[j] analogous.
        # These are the (unnormalized) conditional-entropy contributions of a
        # full row / column and, divided by the subset size, give H(Y|X) and
        # H(X|Y) respectively.
        row_term = row_sum * np.log(row_sum) - row_log_sum
        col_term = col_sum * np.log(col_sum) - col_log_sum

        # Broadcast the marginals over the |X| x |Y| grid. Ntot[i,j] is the size
        # of the subset for pair (i, j): |{first == x}| + |{second == y}| minus
        # the doubly-counted (x, y) cell. It is always positive because every
        # alphabet symbol occurs at least once.
        row_col = row_sum[:, None]
        col_row = col_sum[None, :]
        n_total = row_col + col_row - joint
        log_n_total = np.log(n_total)

        # The subset entropies (see the derivation above). The joint entropy
        # H(X, Y) sums J log J over the "cross" cells (row i plus column j, with
        # the shared cell counted once).
        h_y_given_x = row_term[:, None] / n_total
        h_x_given_y = col_term[None, :] / n_total
        h_x = -(row_col * np.log(row_col) + col_log_sum[None, :] - joint_log) / n_total + log_n_total
        h_y = -(col_row * np.log(col_row) + row_log_sum[:, None] - joint_log) / n_total + log_n_total
        h_xy = -(row_log_sum[:, None] + col_log_sum[None, :] - joint_log) / n_total + log_n_total

        # H(X) is exactly zero iff the subset's x-marginal is a single symbol,
        # i.e. column j is concentrated in row i (C[j] == J[i,j]); likewise
        # H(Y) is zero iff R[i] == J[i,j]. Detecting this from the integer
        # counts reproduces the scalar code's `h == 0.0` special case exactly
        # and avoids dividing by a (floating-point) zero entropy.
        h_y_zero = row_col == joint
        h_x_zero = col_row == joint

        return _SubsetEntropies(
            h_x=h_x,
            h_y=h_y,
            h_xy=h_xy,
            h_x_given_y=h_x_given_y,
            h_y_given_x=h_y_given_x,
            h_x_zero=h_x_zero,
            h_y_zero=h_y_zero,
        )

    def theil_u(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Theil's U uncertainty scorer.

        For every pair ``(x, y)`` this computes Theil's U in both directions
        over the subset of co-occurrences where ``pair[0] == x`` or
        ``pair[1] == y``, matching ``compute_theil_u(Y, X)`` and
        ``compute_theil_u(X, Y)`` on that subset. The underlying entropies are
        computed with a vectorized formulation (see
        :meth:`_subset_entropy_components`) that evaluates all pairs at once
        rather than re-filtering the co-occurrence list for each pair.
        """

        if not self._theil_u:
            comp = self._subset_entropy_components()

            # Theil's U is the fractional reduction in uncertainty, with the
            # scalar code's `h == 0.0 -> 1.0` convention on a degenerate subset.
            with np.errstate(divide="ignore", invalid="ignore"):
                u_y_given_x = np.where(comp.h_y_zero, 1.0, (comp.h_y - comp.h_y_given_x) / comp.h_y)
                u_x_given_y = np.where(comp.h_x_zero, 1.0, (comp.h_x - comp.h_x_given_y) / comp.h_x)

            # Preserve the (compute_theil_u(Y, X), compute_theil_u(X, Y))
            # ordering of the original implementation.
            self._theil_u = {
                (x, y): (float(u_y_given_x[i, j]), float(u_x_given_y[i, j]))
                for i, x in enumerate(self.alphabet_x)
                for j, y in enumerate(self.alphabet_y)
            }

        return self._theil_u

    def cond_entropy(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a corrected conditional entropy scorer.

        For every pair ``(x, y)`` this computes the conditional entropy in both
        directions over the subset of co-occurrences where ``pair[0] == x`` or
        ``pair[1] == y``, matching ``conditional_entropy(Y, X)`` and
        ``conditional_entropy(X, Y)`` on that subset. The entropies come from
        the same vectorized computation as :meth:`theil_u` (see
        :meth:`_subset_entropy_components`).
        """

        if not self._cond_entropy:
            comp = self._subset_entropy_components()

            # Preserve the (conditional_entropy(Y, X), conditional_entropy(X, Y))
            # ordering of the original implementation.
            self._cond_entropy = {
                (x, y): (float(comp.h_y_given_x[i, j]), float(comp.h_x_given_y[i, j]))
                for i, x in enumerate(self.alphabet_x)
                for j, y in enumerate(self.alphabet_y)
            }

        return self._cond_entropy

    def tresoldi(self) -> dict[tuple[Any, Any], tuple[float, float]]:
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
                xy = (
                    -((-pmi[pair][0]) ** (1 - mle[pair][0])) if pmi[pair][0] < 0 else pmi[pair][0] ** (1 - mle[pair][0])
                )

                yx = (
                    -((-pmi[pair][1]) ** (1 - mle[pair][1])) if pmi[pair][1] < 0 else pmi[pair][1] ** (1 - mle[pair][1])
                )

                self._tresoldi[pair] = (xy, yx)

        return self._tresoldi

    def mutual_information(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Mutual Information scorer.

        Computes MI(X;Y) and MI(Y;X) for each symbol pair, measuring the
        overall statistical dependence between the symbols.

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (MI(X;Y), MI(Y;X)) tuples.
        """
        if not self._mutual_information:
            comp = self._subset_entropy_components()

            # Mutual information is symmetric: MI(X;Y) = H(Y) - H(Y|X) = MI(Y;X).
            # Both tuple entries therefore hold the same value, matching the
            # original per-pair implementation to floating-point precision.
            mi = comp.h_y - comp.h_y_given_x

            self._mutual_information = {
                (x, y): (float(mi[i, j]), float(mi[i, j]))
                for i, x in enumerate(self.alphabet_x)
                for j, y in enumerate(self.alphabet_y)
            }

        return self._mutual_information

    def normalized_mutual_information(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Normalized Mutual Information scorer.

        Computes NMI(X;Y) and NMI(Y;X) for each symbol pair, measuring the
        statistical dependence normalized by joint entropy (range [0,1]).

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (NMI(X;Y), NMI(Y;X)) tuples.
        """
        if not self._normalized_mutual_information:
            comp = self._subset_entropy_components()

            # NMI = MI / H(X, Y), also symmetric. When the subset collapses to a
            # single (x, y) cell the joint entropy is zero and MI is zero too;
            # that degenerate case is flagged exactly by the integer-count masks
            # and yields 0.0, matching the scalar implementation.
            mi = comp.h_y - comp.h_y_given_x
            single_cell = comp.h_x_zero & comp.h_y_zero
            with np.errstate(divide="ignore", invalid="ignore"):
                nmi = np.where(single_cell, 0.0, mi / comp.h_xy)

            self._normalized_mutual_information = {
                (x, y): (float(nmi[i, j]), float(nmi[i, j]))
                for i, x in enumerate(self.alphabet_x)
                for j, y in enumerate(self.alphabet_y)
            }

        return self._normalized_mutual_information

    def jaccard_index(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Jaccard Index scorer.

        Computes the Jaccard similarity coefficient between the context sets
        of each symbol pair, measuring overlap in their distributions.

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (J(contexts_x, contexts_y), J(contexts_y, contexts_x)) tuples.
        """
        if not self._jaccard_index:
            self._jaccard_index = {}

            # Pre-compute context sets for each symbol
            x_contexts: dict[Any, list[Any]] = {}  # symbol -> list of contexts where it appears
            y_contexts: dict[Any, list[Any]] = {}  # symbol -> list of contexts where it appears

            for i, (x, y) in enumerate(self.cooccs):
                if x not in x_contexts:
                    x_contexts[x] = []
                if y not in y_contexts:
                    y_contexts[y] = []
                # Use position as context identifier
                x_contexts[x].append(i)
                y_contexts[y].append(i)

            for x in self.alphabet_x:
                for y in self.alphabet_y:
                    # Get contexts for each symbol
                    contexts_x: list[Any] = x_contexts.get(x, [])
                    contexts_y: list[Any] = y_contexts.get(y, [])

                    # Compute Jaccard index for both directions (though it's symmetric)
                    jaccard_xy = compute_jaccard_index(contexts_x, contexts_y)
                    jaccard_yx = compute_jaccard_index(contexts_y, contexts_x)  # Same as xy for Jaccard

                    self._jaccard_index[(x, y)] = (jaccard_xy, jaccard_yx)

        return self._jaccard_index

    def goodman_kruskal_lambda(self) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Goodman-Kruskal Lambda scorer.

        Computes λ(Y|X) and λ(X|Y) for each symbol pair, measuring the
        proportional reduction in error when predicting one variable from another.

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (λ(Y|X), λ(X|Y)) tuples.
        """
        if not self._goodman_kruskal_lambda:
            lambda_y_given_x, lambda_x_given_y = self._goodman_kruskal_lambda_matrices()

            self._goodman_kruskal_lambda = {
                (x, y): (float(lambda_y_given_x[i, j]), float(lambda_x_given_y[i, j]))
                for i, x in enumerate(self.alphabet_x)
                for j, y in enumerate(self.alphabet_y)
            }

        return self._goodman_kruskal_lambda

    def _goodman_kruskal_lambda_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized Goodman-Kruskal lambda backing :meth:`goodman_kruskal_lambda`.

        Lambda is mode-based rather than entropy-based, so it needs the largest
        cell count per row / column of the "cross"-shaped subset (see
        :meth:`_subset_entropy_components` for the subset geometry) rather than
        the entropies. For a pair ``(x=i, y=j)``:

        * predicting ``Y`` from ``X`` -- knowing ``X == i`` narrows ``Y`` to row
          ``i`` (best guess ``max_b J[i, b]``); any other ``X == i'`` only occurs
          with ``Y == j`` (best guess ``J[i', j]``). Summing gives
          ``rowMax[i] + (C[j] - J[i, j])``. The unconditional best guess is the
          most frequent subset ``Y`` value, ``max(C[j], max_{b != j} J[i, b])``.
        * predicting ``X`` from ``Y`` is the transpose.

        With per-row / per-column maxima (and "max excluding this column/row",
        which needs each line's top-two values and its argmax), every pair is
        evaluated together with broadcasting, reproducing the scalar results
        exactly (the computation is over integer counts, with no summation of
        floats).
        """

        joint, row_sum, col_sum = self._joint_counts()
        row_col = row_sum[:, None]
        col_row = col_sum[None, :]
        n_total = row_col + col_row - joint

        col_idx = np.arange(len(self.alphabet_y))[None, :]
        row_idx = np.arange(len(self.alphabet_x))[:, None]

        def line_maxima(axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Per-line max, argmax, and second-largest value along ``axis``."""
            line_max = joint.max(axis=axis)
            line_argmax = joint.argmax(axis=axis)
            size = joint.shape[axis]
            if size >= 2:
                second = np.take(np.partition(joint, size - 2, axis=axis), size - 2, axis=axis)
            else:
                second = np.zeros_like(line_max)
            return line_max, line_argmax, second

        # Predicting Y from X (row-wise): "max excluding column j" drops back to
        # the row's second-largest count exactly at that row's argmax column.
        row_max, row_argmax, row_second = line_maxima(axis=1)
        row_max_excl = np.where(col_idx == row_argmax[:, None], row_second[:, None], row_max[:, None])
        max_marginal_y = np.maximum(col_row, row_max_excl)
        sum_max_conditional_y = row_max[:, None] + (col_row - joint)
        denom_y = n_total - max_marginal_y
        with np.errstate(divide="ignore", invalid="ignore"):
            lambda_y_given_x = np.where(denom_y == 0, 0.0, (sum_max_conditional_y - max_marginal_y) / denom_y)
        lambda_y_given_x = np.maximum(0.0, lambda_y_given_x)

        # Predicting X from Y (column-wise): the transpose of the above.
        col_max, col_argmax, col_second = line_maxima(axis=0)
        col_max_excl = np.where(row_idx == col_argmax[None, :], col_second[None, :], col_max[None, :])
        max_marginal_x = np.maximum(row_col, col_max_excl)
        sum_max_conditional_x = col_max[None, :] + (row_col - joint)
        denom_x = n_total - max_marginal_x
        with np.errstate(divide="ignore", invalid="ignore"):
            lambda_x_given_y = np.where(denom_x == 0, 0.0, (sum_max_conditional_x - max_marginal_x) / denom_x)
        lambda_x_given_y = np.maximum(0.0, lambda_x_given_y)

        return lambda_y_given_x, lambda_x_given_y

    def log_likelihood_ratio(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return a Log-Likelihood Ratio (G²) scorer.

        Computes the G² statistic as an alternative to Chi-square that works
        better with small expected frequencies.

        Parameters
        ----------
        square_ct : bool
            Whether to return the score over a squared or
            non squared contingency table (default: True)

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (G²(X,Y), G²(Y,X)) tuples.
        """
        return self._compute_contingency_table_scorer(
            square_ct, compute_log_likelihood_ratio, "_log_likelihood_ratio", "_log_likelihood_ratio_nonsquare"
        )

    def log_likelihood_ratio_pvalue(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]:
        """
        Return the p-values of the Log-Likelihood Ratio (G²) test.

        This is the significance counterpart to :meth:`log_likelihood_ratio`:
        under the null hypothesis of independence, G² is asymptotically
        chi-square distributed, and this returns the corresponding p-value for
        each pair. As with the other symmetric statistical tests, both tuple
        entries hold the same value.

        Parameters
        ----------
        square_ct : bool
            Whether to compute the p-value over a squared (2x2) or non-squared
            (3x2) contingency table (default: True).

        Returns
        -------
        Dict[Tuple[Any, Any], Tuple[float, float]]
            Dictionary mapping symbol pairs to (p-value, p-value) tuples.
        """
        return self._compute_contingency_table_scorer(
            square_ct,
            compute_log_likelihood_ratio_pvalue,
            "_log_likelihood_ratio_pvalue",
            "_log_likelihood_ratio_pvalue_nonsquare",
        )
