# encoding: utf-8

import csv
from itertools import chain, combinations

# TODO: should build co-occurrences (including n-gram ones) already
# from here?

# TODO: allow more than two columns, checking columns
# TODO: skip rows with empty entries
# TODO: allow to skip header
def read_sequences(filename, col_delim="\t", elem_delim=" "):
    """
    Reads parallel sequences, returning them as a list of lists.

    Datafiles should have one sequence pair per row, with pairs separated
    by `col_delim` and elements delimited by `elem_delim`, as in:

    ```
    Orthography             Segments
    E X C L A M A T I O N   ɛ k s k l ʌ m eɪ ʃ ʌ n
    C L O S E Q U O T E     k l oʊ z k w oʊ t
    D O U B L E Q U O T E   d ʌ b ʌ l k w oʊ t
    E N D O F Q U O T E     ɛ n d ʌ v k w oʊ t
    E N D Q U O T E         ɛ n d k w oʊ t
    I N Q U O T E S         ɪ n k w oʊ t s
    Q U O T E               k w oʊ t
    U N Q U O T E           ʌ n k w oʊ t
    H A S H M A R K         h æ m ɑ ɹ k
    ```
    """

    data = []
    with open(filename) as handler:
        for line in handler.readlines():
            data.append(
                [column.split(elem_delim) for column in line.strip().split(col_delim)]
            )

    return data


# TODO: assuming one taxon per col and locations per row, allow to switch
# TODO; assumin column name in position 0
def read_pa_matrix(filename, delimiter="\t"):
    """
    Reads a presence-absence matrix, returning as equivalent sequences.
    """

    # We read the matrix and filter each row (location), keeping only
    # the observed taxa
    matrix = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            location = row.pop("NAME")
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
