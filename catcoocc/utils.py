# encoding: utf-8

"""
Utility functions for the `catcoocc` library.

The module mostly includes functions for input/output, as well as
pre-computations from raw data.
"""

# TODO: add function for smoothing of co-occurrences after collect_cooccs()
# TODO: add function for reading mushroom and pokemon data

# Import Python standard libraries
import csv
from collections import Counter
from itertools import chain, combinations, product

# Import 3rd party libraries
import numpy as np


def collect_alphabets(cooccs):
    """
    Return the `x` and `y` alphabets from a list of co-occurrences.

    :param list cooccs: The list of co-occurrence tuples.

    :return tuple alphabets: A tuple of two elements, the sorted list of
        symbols in series `x` and the sorted list of symbols in series `y`.
    """

    alphabet_x, alphabet_y = zip(*cooccs)

    return sorted(set(alphabet_x)), sorted(set(alphabet_y))


def collect_ngrams(seq, order, pad):
    """
    Function for yielding the ngrams of a sequence.

    The sequence is padded so that symbols at the extremities will be
    repeated the same number of times that other symbols. This operation also
    guarantees that sequences shorter than the order will be collected.

    :param list seq: The list of elements for ngram collection.
    :param int order: The ngram order.
    :param string pad: The padding symbol.

    :yield: The ngrams of the sequence.
    """

    seq = tuple(chain((pad,) * (order - 1), seq, (pad,) * (order - 1)))

    for ngram in zip(*[seq[i:] for i in range(order)]):
        yield ngram


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
            [collect_ngrams(seq_a, order, pad), collect_ngrams(seq_b, order, pad)]
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
    a mismatch. In more detail, given a pair of symbols `x` and `y`,
    the counts of co-occurrences will be:

    - obs["10"]: number of pairs matching `x` (with any `y`)
    - obs["20"]: number of pairs not matching `x` (with any `y`)
    - obs["01"]: number of pairs matching `y` (with any `x`)
    - obs["02"]: number of pairs not matching `y` (with any `x`)
    - obs["11"]: number of pairs matching `x` and matching `y`
    - obs["12"]: number of pairs matching `x` and not matching `y`
    - obs["21"]: number of pairs not matching `x` and matching `y`
    - obs["22"]: number of pairs not matching `x` and matching `Y`

    Note that obs["22"] is not the number of pairs where either `x` or `y`
    are mismatching, but where both necessarily mismatch.

    When doing x->y, a non-squared contingency table will be

        |            |  pair[1]  | pair[1]==y | pair[1]!=y |
        | pair[0]==x | obs["10"] | obs["11"]  | obs["12"]  |
        | pair[0]!=x | obs["20"] | obs["21"]  | obs["22"]  |

    :param list cooccs: A list of co-occurrence tuples.

    :return dict obs: A dictionary of observations per co-occurrence type.
    """

    # Collect observation counts; while this could be done in linear fashion,
    # especially with Python it takes too long for real data, so we need to
    # properly cache and pre-compute stuff at the expense of memory.
    #
    # "00": len(cooccs)
    # "10": len([p for p in cooccs if p[0] == a])
    # "20": len([p for p in cooccs if p[0] != a])
    # "01": len([p for p in cooccs if p[1] == b])
    # "02": len([p for p in cooccs if p[1] != b])
    # "11": cooccs.count((a, b))
    # "22": len([p for p in cooccs if not any([p[0] == a, p[1] == b])])
    # "12": len([p for p in cooccs if p[0] == a and p[1] != b])
    # "21": len([p for p in cooccs if p[0] != a and p[1] == b])

    # Cache the number of co-occurrences, which is obs_00 for all pairs
    obs_00 = len(cooccs)

    # Obs['11'] is cached with a collections.Counter, so that we rely on
    # the standard library
    obs_11 = Counter(cooccs)

    # Extract the `x` and `y` symbols, so they can be counted as well;
    # the alphabets can be built from their respective sets. We don't
    # use collect_alphabet() as we need the symbols as well, in order to
    # build `x_counter` and `y_counter`
    symbols_x, symbols_y = zip(*cooccs)
    x_counter, y_counter = Counter(symbols_x), Counter(symbols_y)
    alphabet_x = set(symbols_x)
    alphabet_y = set(symbols_y)

    # Build with a dictionary comprehension
    obs = {
        (x, y): {
            "00": obs_00,
            "11": obs_11[(x, y)],
            "10": x_counter[x],
            "20": obs_00 - x_counter[x],
            "01": y_counter[y],
            "02": obs_00 - y_counter[y],
            "12": x_counter[x] - obs_11[(x, y)],
            "21": y_counter[y] - obs_11[(x, y)],
            "22": obs_00 + obs_11[(x, y)] - x_counter[x] - y_counter[y],
        }
        for x, y in product(alphabet_x, alphabet_y)
    }

    return obs


def build_ct(observ, square):
    """
    Build a contingency table from a dictionary of observations.

    The contingency table can be either square (2x2) or not (3x2). Non-squared
    contingency tables include co-occurrences where neither the `x` or `y`
    under investigation occur.

    :param dict observ: A dictionary of observations, as provided by
        utils.collect_observations()
    :param bool square: Whether to return a square (2x2) or non-square (3x2)
            contingency table.
    """

    # Build the contingency tables as np.arrays(), as they will necessary
    # for the functions consuming them and as it allows to use np methods
    # directly (such as .sum())
    if square:
        cont_table = np.array(
            [[observ["11"], observ["12"]], [observ["21"], observ["22"]]]
        )
    else:
        cont_table = np.array(
            [
                [observ["10"], observ["11"], observ["12"]],
                [observ["20"], observ["21"], observ["22"]],
            ]
        )

    return cont_table


def read_sequences(filename, cols=None, col_delim="\t", elem_delim=" "):
    """
    Reads parallel sequences, returning them as a list of lists.

    Datafiles should have one sequence pair per row, with pairs separated
    by `col_delim` and elements delimited by `elem_delim`, as in:

    ```
    Orthography                 Segments
    E X C L A M A T I O N       ɛ k s k l ʌ m eɪ ʃ ʌ n
    C L O S E Q U O T E         k l oʊ z k w oʊ t
    D O U B L E Q U O T E       d ʌ b ʌ l k w oʊ t
    E N D O F Q U O T E         ɛ n d ʌ v k w oʊ t
    E N D Q U O T E             ɛ n d k w oʊ t
    I N Q U O T E S             ɪ n k w oʊ t s
    Q U O T E                   k w oʊ t
    U N Q U O T E               ʌ n k w oʊ t
    H A S H M A R K             h æ m ɑ ɹ k
    ```

    Rows without complete data are not included. The function allows to
    collect data from multiple columns, but it is inteded to pairwise
    comparison; if no column names are provided, as per default,
    the reading will skip the first row (assumed to be header).

    :param str filename: Path to the file to be read.
    :param cols list: List of column names to be collected (default: None)
    :param str col_delim: String used as field delimiter (default: `"\t"`).
    :param str elem_delim: String used as element delimiter (default: `" "`).
    """

    # Column names must always be specified, otherwise we will skip over
    # the header (thus assuming that there are only two columns); the logic
    # is different
    data = []
    with open(filename) as handler:
        if not cols:
            skip_header = True
            for line in handler.readlines():
                if skip_header:
                    skip_header = False
                    continue

                data.append(
                    [
                        column.split(elem_delim)
                        for column in line.strip().split(col_delim)
                    ]
                )
        else:
            reader = csv.DictReader(handler, delimiter=col_delim)
            data = [
                [row[col_name].split(elem_delim) for col_name in cols if row[col_name]]
                for row in reader
            ]

    # Remove incomplete rows
    # NOTE: Checking length is not really effective, but will allow easier
    # expansion in the future
    data = [row for row in data if len(row) == 2]

    return data


# TODO: assuming one taxon per col and locations per row, allow to switch
def read_pa_matrix(filename, delimiter="\t"):
    """
    Reads a presence-absence matrix, returning as equivalent sequences.

    The location must be specified under column name `ID`.

    :param str filename: Path to the file to be read.
    :param str delimiter: String used as field delimiter (default: `"\t"`).
    """

    # We read the matrix and filter each row (location), keeping only
    # the observed taxa
    matrix = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            location = row.pop("ID")
            matrix[location] = sorted(
                [taxon for taxon, observed in row.items() if observed == "1"]
            )

    # Make the corresponding sequence from the combinations
    obs_combinations = list(
        chain.from_iterable(
            [combinations(observed, 2) for location, observed in matrix.items()]
        )
    )

    return obs_combinations
