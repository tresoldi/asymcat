"""
Test data fixtures for ASymCat tests.

This module provides standardized test datasets used across the test suite,
demonstrating different types of categorical data and expected behaviors.
"""

from pathlib import Path
from typing import List, Tuple

# Test data directory
RESOURCE_DIR = Path(__file__).parent.parent.parent / "resources"


# Small sample derived from CMU pronunciation dictionary
# Demonstrates phoneme-grapheme correspondences
SAMPLE_CMU_DATA = [
    ["ONE", "1 1 1"],  # Perfect correspondence case
    ["A B D O L L A H", "æ b d ɑ l ʌ"],
    ["A M A S S", "ʌ m æ s"],
    ["A N G L O P H I L E", "æ n ɡ l ʌ f aɪ l"],
    ["A N T I C", "æ n t ɪ k"],
    ["A R J O", "ɑ ɹ j oʊ"],
    ["A S T R A D D L E", "ʌ s t ɹ æ d ʌ l"],
    ["B A H L S", "b ɑ l z"],
    ["B L O W E D", "b l oʊ d"],
    ["B O N V I L L A I N", "b ɑ n v ɪ l eɪ n"],
    ["B R A G A", "b ɹ ɑ ɡ ʌ"],
    ["B U R D I", "b ʊ ɹ d i"],
    ["B U R K E R T", "b ɝ k ɝ t"],
    ["B U R R E S S", "b ɝ ʌ s"],
    ["C A E T A N O", "k ʌ t ɑ n oʊ"],
    ["C H E R Y L", "ʃ ɛ ɹ ʌ l"],
    ["C L E M E N C E", "k l ɛ m ʌ n s"],
    ["C O L V I N", "k oʊ l v ɪ n"],
    ["C O N V E N T I O N S", "k ʌ n v ɛ n ʃ ʌ n z"],
    ["C R E A S Y", "k ɹ i s i"],
    ["C R E T I E N", "k ɹ i ʃ j ʌ n"],
    ["C R O C E", "k ɹ oʊ tʃ i"],
]


def get_sample_cmu_processed() -> List[List[List[str]]]:
    """
    Get CMU data processed into the format expected by asymcat.

    Returns:
        List of [orthography_tokens, phonetic_tokens] pairs

    Example:
        >>> data = get_sample_cmu_processed()
        >>> data[0]  # First entry
        [['O', 'N', 'E'], ['1', '1', '1']]
    """
    return [[entry[0].split(), entry[1].split()] for entry in SAMPLE_CMU_DATA]


# Minimal test data for edge cases
MINIMAL_DATA = [
    [["A"], ["B"]],  # Single symbol pair
]

# Perfect correlation data for testing deterministic relationships
PERFECT_CORRELATION_DATA = [
    [["A", "A", "A"], ["B", "B", "B"]],  # Perfect positive correlation
    [["C", "C", "C"], ["D", "D", "D"]],
]

# Independent data for testing lack of association
INDEPENDENT_DATA = [
    [["A", "B"], ["C", "D"]],
    [["A", "B"], ["D", "C"]],
    [["B", "A"], ["C", "D"]],
    [["B", "A"], ["D", "C"]],
]

# Asymmetric relationship data (demonstrates directional dependency)
ASYMMETRIC_DATA = [
    [["A"], ["x"]],  # A always predicts x
    [["A"], ["x"]],
    [["B"], ["y"]],  # B always predicts y
    [["B"], ["y"]],
    [["B"], ["z"]],  # B sometimes predicts z (uncertainty)
]

# N-gram test sequences
NGRAM_TEST_SEQUENCES = [(("abcde", "ABCDE"), ("fgh", "FGH"), ("i", "I"), ("jkl", "JKL"))]


# Expected test results for validation
EXPECTED_CMU_RESULTS = {
    "cooccurrence_count": 879,
    "A_cooccurrences": 92,  # Co-occurrences starting with "A"
    "L_l_cooccurrences": 14,  # "L" to "l" co-occurrences
}

# Expected scoring results for specific pairs
EXPECTED_SCORING_RESULTS = {
    ("ONE", "1"): {
        "mle": (1.0, 1.0),
        "pmi": (5.680172609017068, 5.680172609017068),
        "npmi": (1.0, 1.0),
        "chi2": (609.5807658175393, 609.5807658175393),
        "cramers_v": (0.8325526903114843, 0.8325526903114843),
        "theil_u": (1.0, 1.0),
        "cond_entropy": (0.0, 0.0),
        "tresoldi": (1.0, 1.0),
    },
    ("A", "b"): {
        "mle": (0.11320754716981132, 0.06521739130434782),
        "pmi": (0.07846387631207004, 0.07846387631207004),
        "npmi": (0.015733602612959818, 0.015733602612959818),
        "theil_u": (0.21299752425693524, 0.3356184612000498),
        "cond_entropy": (1.86638224482290279, 0.9999327965500219),
        "tresoldi": (0.0926310345228265, 0.10466500171366895),
    },
}


def get_available_datasets() -> List[Tuple[str, str]]:
    """
    Get list of available test datasets in the resources directory.

    Returns:
        List of (filename, description) pairs
    """
    datasets = []

    # Check for common test files
    test_files = [
        ("toy.tsv", "Small toy dataset for basic testing"),
        ("mushroom-small.tsv", "Mushroom characteristics (small subset)"),
        ("cmudict.sample100.tsv", "CMU pronunciation dictionary (100 samples)"),
        ("cmudict.sample1000.tsv", "CMU pronunciation dictionary (1000 samples)"),
        ("galapagos.tsv", "Galapagos species presence-absence matrix"),
    ]

    for filename, description in test_files:
        filepath = RESOURCE_DIR / filename
        if filepath.exists():
            datasets.append((filename, description))

    return datasets


def validate_test_environment() -> bool:
    """
    Validate that the test environment has required data files.

    Returns:
        True if environment is valid, False otherwise
    """
    required_files = ["toy.tsv", "mushroom-small.tsv"]

    for filename in required_files:
        if not (RESOURCE_DIR / filename).exists():
            return False

    return True
