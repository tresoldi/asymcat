"""
Utility functions for the `asymcat` library.

The module mostly includes functions for input/output, as well as
pre-computations from raw data.
"""

# TODO: add function for smoothing of co-occurrences after collect_cooccs()
# TODO: add function for reading mushroom and pokemon data

import csv
import os

# Import Python standard libraries
from collections import Counter
from collections.abc import Generator
from itertools import chain, combinations, product
from typing import Any

# Import 3rd party libraries
import numpy as np


def collect_alphabets(cooccs: list[tuple]) -> tuple:
    """
    Return the `x` and `y` alphabets from a list of co-occurrences.

    Parameters
    ----------
    cooccs : List[tuple]
        The list of co-occurrence tuples.

    Returns
    -------
    alphabets : tuple
        A tuple of two elements, the sorted list of symbols in series `x` and
        the sorted list of symbols in series `y`.

    Raises
    ------
    ValueError
        If cooccs is empty or contains invalid tuples.
    TypeError
        If cooccs is not a list or contains non-tuple elements.
    """
    if not isinstance(cooccs, list):
        raise TypeError(f"Expected list, got {type(cooccs).__name__}")

    if not cooccs:
        raise ValueError("Empty co-occurrence list provided")

    # Validate that all elements are tuples of length 2
    for i, coocc in enumerate(cooccs):
        if not isinstance(coocc, (tuple, list)) or len(coocc) != 2:
            raise ValueError(f"Invalid co-occurrence at index {i}: expected tuple/list of length 2, got {coocc}")

    try:
        alphabet_x, alphabet_y = zip(*cooccs, strict=False)
    except ValueError as e:
        raise ValueError(f"Failed to unzip co-occurrences: {e}")

    return sorted(set(alphabet_x)), sorted(set(alphabet_y))


# TODO: Use lpngrams?
def collect_ngrams(seq: list[Any] | str, order: int, pad: str) -> Generator[tuple, None, None]:
    """
    Function for yielding the ngrams of a sequence.

    The sequence is padded so that symbols at the extremities will be
    repeated the same number of times that other symbols. This operation also
    guarantees that sequences shorter than the order will be collected.

    Parameters
    ----------
    seq : Union[List[Any], str]
        The list/string of elements for ngram collection.
    order : int
        The ngram order.
    pad : str
        The padding symbol.

    Yields
    ------
    ngram : tuple
        The ngrams of the sequence.

    Raises
    ------
    ValueError
        If order is less than 1 or seq is empty.
    TypeError
        If seq is not a list, tuple, or string, or order is not an integer.
    """
    if not isinstance(seq, (list, tuple, str)):
        raise TypeError(f"Expected list, tuple, or string, got {type(seq).__name__}")

    if not isinstance(order, int):
        raise TypeError(f"Order must be an integer, got {type(order).__name__}")

    if order < 1:
        raise ValueError(f"Order must be at least 1, got {order}")

    if not isinstance(pad, str):
        raise TypeError(f"Pad symbol must be a string, got {type(pad).__name__}")

    tuple_seq = tuple(chain((pad,) * (order - 1), seq, (pad,) * (order - 1)))

    yield from zip(*[tuple_seq[i:] for i in range(order)], strict=False)


# TODO: should yield?
def collect_cooccs(seqs: list[list[list[Any] | str] | tuple], order: int | None = None, pad: str = "#") -> list[tuple]:
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

    Parameters
    ----------
    seqs : List[Union[List[Union[List[Any], str]], tuple]]
        A list of sequence pairs (each sequence can be a list or string).
    order : Optional[int]
        The order of the ngrams to be collected, with `None` indicating the
        collection of the entire sequences (default: None).
    pad : str
        The symbol for internal padding, which must not conflict with symbols
        in the sequences (default: "#").

    Returns
    -------
    coocc : List[tuple]
        A list of tuples of co-occurring elements.

    Raises
    ------
    ValueError
        If seqs is empty, contains invalid sequence pairs, or sequences have mismatched lengths.
    TypeError
        If seqs is not a list or contains invalid data types.
    """
    if not isinstance(seqs, list):
        raise TypeError(f"Expected list, got {type(seqs).__name__}")

    if not seqs:
        raise ValueError("Empty sequence list provided")

    if order is not None and (not isinstance(order, int) or order < 1):
        raise ValueError(f"Order must be a positive integer or None, got {order}")

    if not isinstance(pad, str):
        raise TypeError(f"Pad symbol must be a string, got {type(pad).__name__}")

    # Validate sequence structure
    for i, seq_pair in enumerate(seqs):
        if not isinstance(seq_pair, (list, tuple)) or len(seq_pair) != 2:
            raise ValueError(f"Invalid sequence pair at index {i}: expected list/tuple of length 2, got {seq_pair}")

        seq_a, seq_b = seq_pair
        if not isinstance(seq_a, (list, tuple, str)) or not isinstance(seq_b, (list, tuple, str)):
            raise TypeError(f"Sequences at index {i} must be lists, tuples, or strings")

    # If an ngram order is specified, instead of looking for co-occurrences
    # across the full length of the sequences of a pair we need to take
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
        # Enhanced length validation with specific error information
        for i, (seq_a, seq_b) in enumerate(seqs):
            if len(seq_a) != len(seq_b):
                raise ValueError(f"Sequence pair {i} has mismatched lengths: {len(seq_a)} vs {len(seq_b)}")

        # Collect the ngrams for each sequence in each pair
        ngram_seqs = [[collect_ngrams(seq_a, order, pad), collect_ngrams(seq_b, order, pad)] for seq_a, seq_b in seqs]

        # Decompose the list of ngrams in `ngram_seqs`, building new
        # corresponding pairs from where the co-occurrences will be
        # collected. This is done with a non-expansive itertools operation,
        # with no need to cast a list (as the iterable will be consumed
        # into a list below).
        _seqs = list(chain(*[zip(*ngram_seq, strict=False) for ngram_seq in ngram_seqs]))

    # From the list of sequence pairs, chain a list with all the co-occurring
    # pairs from the product of each pair.
    coocc = list(chain(*[product(seq_a, seq_b) for seq_a, seq_b in _seqs]))

    # Remove `(pad, pad)` entries that will result form ngram collection;
    # the operation is performed even in case of no `order` (i.e., collection
    # over the entire strings), as it allows us to consume `coocc` (at this
    # point, an iterable) into a list.
    coocc = [pair for pair in coocc if pad not in pair]

    return coocc


def collect_observations(cooccs: list[tuple]) -> dict:
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

    Parameters
    ----------
    cooccs : List[tuple]
        A list of tuples of co-occurring elements.

    Returns
    -------
    obs : dict
        A dictionary of observations per co-occurrence type.

    Raises
    ------
    ValueError
        If cooccs is empty or contains invalid tuples.
    TypeError
        If cooccs is not a list.
    """
    if not isinstance(cooccs, list):
        raise TypeError(f"Expected list, got {type(cooccs).__name__}")

    if not cooccs:
        raise ValueError("Empty co-occurrence list provided")

    # Validate co-occurrence structure
    for i, coocc in enumerate(cooccs):
        if not isinstance(coocc, (tuple, list)) or len(coocc) != 2:
            raise ValueError(f"Invalid co-occurrence at index {i}: expected tuple/list of length 2, got {coocc}")

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
    try:
        symbols_x, symbols_y = zip(*cooccs, strict=False)
    except ValueError as e:
        raise ValueError(f"Failed to unzip co-occurrences: {e}")
    x_counter: Counter = Counter(symbols_x)
    y_counter: Counter = Counter(symbols_y)
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


# TODO: does it need to be np.array?
def build_ct(observ, square: bool = True) -> list:
    """
    Build a contingency table from a dictionary of observations.

    The contingency table can be either square (2x2) or not (3x2). Non-squared
    contingency tables include co-occurrences where neither the `x` or `y`
    under investigation occur.

    Parameters
    ----------
    observ : dict
        A dictionary of observations, as provided by
        common.collect_observations().
    square : bool, optional
        Whether to return a square (2x2) or non-square (3x2) contingency table.

    Returns
    -------
    cont_table : np.array
        A contingency table as a numpy array.
    """

    # Build the contingency tables as np.arrays(), as they will necessary
    # for the functions consuming them and as it allows to use np methods
    # directly (such as .sum())
    if square:
        cont_table = np.array([[observ["11"], observ["12"]], [observ["21"], observ["22"]]])
    else:
        cont_table = np.array(
            [
                [observ["10"], observ["11"], observ["12"]],
                [observ["20"], observ["21"], observ["22"]],
            ]
        )

    return cont_table.tolist()


def read_sequences(
    filename: str, cols: list[str] | None = None, col_delim: str = "\t", elem_delim: str = " "
) -> list[list[list[str]]]:
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

    Parameters
    ----------
    filename : str
        Path to the file to be read.
    cols : Optional[List[str]]
        List of column names to be collected (default: None)
    col_delim : str
        String used as field delimiter (default: `"\t"`).
    elem_delim : str
        String used as element delimiter (default: `" "`).

    Returns
    -------
    data : List[List[List[str]]]
        A list of lists, where each list contains the data from a row.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If file is empty or has invalid format.
    PermissionError
        If file cannot be read due to permissions.
    """
    if not isinstance(filename, str):
        raise TypeError(f"Filename must be a string, got {type(filename).__name__}")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    if not os.path.isfile(filename):
        raise ValueError(f"Path is not a file: {filename}")

    if cols is not None and not isinstance(cols, list):
        raise TypeError(f"Columns must be a list or None, got {type(cols).__name__}")

    if not isinstance(col_delim, str) or not isinstance(elem_delim, str):
        raise TypeError("Delimiters must be strings")

    # Column names must always be specified, otherwise we will skip over
    # the header (thus assuming that there are only two columns); the logic
    # is different
    data = []
    try:
        with open(filename, encoding="utf-8") as handler:
            if not cols:
                skip_header = True
                for line in handler.readlines():
                    if skip_header:
                        skip_header = False
                        continue

                    data.append([column.split(elem_delim) for column in line.strip().split(col_delim)])
            else:
                reader = csv.DictReader(handler, delimiter=col_delim)
                try:
                    data = [[row[col_name].split(elem_delim) for col_name in cols if row[col_name]] for row in reader]
                except KeyError as e:
                    raise ValueError(f"Column not found in file: {e}")

    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error: {e}")
    except Exception as e:
        raise OSError(f"Error reading file {filename}: {e}")

    if not data:
        raise ValueError(f"No valid data found in file: {filename}")

    # Remove incomplete rows
    # NOTE: Checking length is not really effective, but will allow easier
    # expansion in the future
    data = [row for row in data if len(row) == 2]

    if not data:
        raise ValueError(f"No complete sequence pairs found in file: {filename}")

    return data


# TODO: assuming one taxon per col and locations per row, allow to switch
def read_pa_matrix(filename: str, delimiter: str = "\t") -> list[tuple]:
    """
    Reads a presence-absence matrix, returning as equivalent sequences.

    The location must be specified under column name `ID`.

    Parameters
    ----------
    filename : str
        Path to the file to be read.
    delimiter : str
        String used as field delimiter (default: `"\t"`).

    Returns
    -------
    obs_combinations : List[tuple]
        A list of tuples representing observed combinations.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If file format is invalid or ID column is missing.
    """
    if not isinstance(filename, str):
        raise TypeError(f"Filename must be a string, got {type(filename).__name__}")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    if not isinstance(delimiter, str):
        raise TypeError(f"Delimiter must be a string, got {type(delimiter).__name__}")

    # We read the matrix and filter each row (location), keeping only
    # the observed taxa
    matrix = {}
    try:
        with open(filename, encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)

            if reader.fieldnames is None or "ID" not in reader.fieldnames:
                raise ValueError("Missing required 'ID' column in presence-absence matrix")

            for row_num, row in enumerate(reader, start=2):  # Start at 2 since header is row 1
                try:
                    location = row.pop("ID")
                except KeyError:
                    raise ValueError(f"Missing ID value in row {row_num}")

                if not location:
                    raise ValueError(f"Empty ID value in row {row_num}")

                # Validate presence-absence values
                for taxon, observed in row.items():
                    if observed not in ("0", "1", ""):
                        raise ValueError(
                            f"Invalid presence-absence value '{observed}' for taxon '{taxon}' in row {row_num}."
                            " Expected '0', '1', or empty."
                        )

                matrix[location] = sorted([taxon for taxon, observed in row.items() if observed == "1"])

    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error: {e}")
    except Exception as e:
        raise OSError(f"Error reading file {filename}: {e}")

    if not matrix:
        raise ValueError(f"No valid data found in presence-absence matrix: {filename}")

    # Make the corresponding sequence from the combinations
    obs_combinations = list(chain.from_iterable([combinations(observed, 2) for location, observed in matrix.items()]))

    return obs_combinations
