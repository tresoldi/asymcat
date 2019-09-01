# encoding: utf-8

# Import Python standard libraries
from collections import Counter
from itertools import chain, product
import math

# Import 3rd party libraries
import numpy as np
import scipy.stats as ss

# TODO: in all cases, receive the co-occs (which can be properly checked)
# and compute the observations if needed and not provided (could be cached
# and given by the user)

# TODO: add a scorer with theil times npmi? this would probably mean having
# to return 0 instead of none in other cases for consistency


def get_ngrams(seq, order, pad):
    """
    Function for yielding ngrams of a sequence.

    The sequence is padded so that symbols at the extremities will be
    repeated the same number of times that other symbols. The operation also
    guarantees that sequences shorter than the order will be collected.

    :param list seq: The list of elements for ngram collection.
    :param int order: The ngram order.
    :param string pad: The padding symbol.
    :yield: The ngrams of the sequence.
    """

    seq = tuple(chain((pad,) * (order - 1), seq, (pad,) * (order - 1)))

    for ngram in zip(*[seq[i:] for i in range(order)]):
        yield ngram


# TODO: add some smoothing?
def collect_cooccs(seqs, order=None, pad="#"):
    """
    Collects tuples of co-occurring elements in pairs of sequences.

    This function is used for collecting co-occurring elements in order to
    train or improve scorers. Co-occurrences can be collected either
    considering sequences in their entirety, or by aligned ngrams.

    When collecting ngrams, a padding symbol is used internally to guarantee
    that elements at the extremities of the sequences will be collected the
    same number of times that other elements. The padding symbol defaults to
    `"#"`, but given that the tuples where it occurs are automatically
    removed before returning, it is important that such symbol is not a
    part of either of the sequence alphabets. It is the user responsability
    to set an appropriate, non-conflicting pad symbol in case the default
    is used in the data.

    :param list seqs: A list of lists of sequence pairs.
    :param number order: The order of the ngrams to be collected, with `None`
        indicating the collection of the entire sequences (default: None).
    :param string pad: The symbol for internal padding, which must not
        conflict with symbols in the sequences (default: "#").

    :return list: A list of tuples of co-occurring elements.
    """

    # If an ngram order is specified, instead of looking for co-occurrences
    # across the full lenght of the sequences of a pair we need to take
    # overlapping ngrams in those (which must be aligned or, at least,
    # have the same length), creating a new set of sequences to get the
    # cooccurrences which is actually a set of ngram subsequences
    if not order:
        # No ngrams, just copy the sequence pairs, preserving the original
        _seqs = seqs[:]
    else:
        # For ngram collection, we need to guarantee that each in each
        # sequence pair both the sequences have the same length (assuming
        # they are aligned).
        len_check = [len(seq_a) == len(seq_b) for seq_a, seq_b in seqs]
        if not all(len_check):
            raise ValueError("Sequence pair of different lengths.")

        # Collect the ngrams for each sequence in each pair
        ngram_seqs = [
            [get_ngrams(seq_a, order, pad), get_ngrams(seq_b, order, pad)]
            for seq_a, seq_b in seqs
        ]

        # Decompose the list of ngrams in `ngram_seqs`, building new
        # corresponding pairs from where the co-occurrences will be
        # collected. This is done with a non-expansive itertools operation,
        # with no need to cast a list (as the iterable will be consumed
        # into a list below).
        _seqs = chain(*[zip(*ngram_seq) for ngram_seq in ngram_seqs])

    # From the list of sequence pairs, chain a list with all the co-occurring
    # pairs from the product of each pair.
    coocc = chain(*[product(seq_a, seq_b) for seq_a, seq_b in _seqs])

    # Remove `(pad, pad)` entries that will result form ngram collection;
    # the operation is performed even in case of no `order` (i.e., collection
    # over the entire strings), as it allows us to consume `coocc` (at this
    # point, an iterable) into a list.
    coocc = [pair for pair in coocc if pad not in pair]

    return coocc


def collect_observations(cooccs):
    """
    Build a dictionary of observations for all possible correspondences.

    The counts of observations are derived from typical organization in
    a non-squared contingency table, using indexes 0, 1, and 2, where
    0 means that the identity is not considered (so that "00" would refer to
    all correspondence pairs), 1 that there is a match, and 2 that there is
    a mismatch. In more detail, given a pair of symbols `a` and `b`,
    the counts of co-occurrences will be:

    - obs[10]: number of pairs matching `a` (with any `b`)
    - obs[20]: number of pairs not matching `a` (with any `b`)
    - obs[01]: number of pairs matching `b` (with any `a`)
    - obs[02]: number of pairs not matching `b` (with any `a`)
    - obs[11]: number of pairs matching `a` and matching `b`
    - obs[12]: number of pairs matching `a` and not matching `b`
    - obs[21]: number of pairs not matching `a` and matching `b`
    - obs[22]: number of pairs not matching `a` and matching `b`

    Note that obs[22] is not the number of pairs where either `a` or `b`
    are mismatching, but where both necessarily mismatch.

    When doing a->b, a non-squared contingency table will be
    
        |            |  pair[1] | pair[1]==b | pair[1]!=b |
        | pair[0]==a | obs_10   | obs_11     | obs_12     |
        | pair[0]!=a | obs_20   | obs_21     | obs_22     |
    """

    # Obtain the list of symbols of each language (the "alphabets")
    alphabet_a, alphabet_b = [set(alphabet) for alphabet in zip(*cooccs)]

    # Collect observations for all possible pairs
    obs = {}
    for a, b in product(alphabet_a, alphabet_b):
        obs[(a, b)] = {
            "00": len(cooccs),
            "10": len([pair for pair in cooccs if pair[0] == a]),
            "20": len([pair for pair in cooccs if pair[0] != a]),
            "01": len([pair for pair in cooccs if pair[1] == b]),
            "02": len([pair for pair in cooccs if pair[1] != b]),
            "11": cooccs.count((a, b)),
            "22": len(
                [pair for pair in cooccs if not any([pair[0] == a, pair[1] == b])]
            ),
            "12": len([pair for pair in cooccs if pair[0] == a and pair[1] != b]),
            "21": len([pair for pair in cooccs if pair[0] != a and pair[1] == b]),
        }

    return obs


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
