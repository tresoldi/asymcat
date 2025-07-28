"""
Custom assertions and validation utilities for ASymCat tests.

Provides domain-specific assertions that make tests more readable and maintainable.
"""

from typing import Any

import numpy as np


def assert_valid_cooccurrences(cooccs: list[tuple[Any, Any]]) -> None:
    """
    Assert that co-occurrences are in the expected format.

    Args:
        cooccs: List of co-occurrence tuples to validate

    Example:
        >>> cooccs = [('a', 'b'), ('c', 'd')]
        >>> assert_valid_cooccurrences(cooccs)  # Passes
    """
    assert isinstance(cooccs, list), "Co-occurrences must be a list"
    assert len(cooccs) > 0, "Co-occurrences list cannot be empty"

    for i, coocc in enumerate(cooccs):
        assert isinstance(coocc, (tuple, list)), f"Co-occurrence {i} must be tuple or list"
        assert len(coocc) == 2, f"Co-occurrence {i} must have exactly 2 elements"


def assert_valid_scores(scores: dict[tuple[Any, Any], tuple[float, float]], allow_infinite: bool = False) -> None:
    """
    Assert that scoring results are in the expected format.

    Args:
        scores: Dictionary mapping symbol pairs to (xy, yx) score tuples
        allow_infinite: Whether to allow infinite values (e.g., for Fisher exact test)

    Example:
        >>> scores = {('a', 'b'): (0.5, 0.7)}
        >>> assert_valid_scores(scores)  # Passes
        >>> fisher_scores = {('a', 'b'): (float('inf'), float('inf'))}
        >>> assert_valid_scores(fisher_scores, allow_infinite=True)  # Passes
    """
    assert isinstance(scores, dict), "Scores must be a dictionary"
    assert len(scores) > 0, "Scores dictionary cannot be empty"

    for pair, (xy, yx) in scores.items():
        assert isinstance(pair, (tuple, list)), f"Key {pair} must be tuple or list"
        assert len(pair) == 2, f"Key {pair} must have exactly 2 elements"
        assert isinstance(xy, (int, float)), f"X→Y score for {pair} must be numeric"
        assert isinstance(yx, (int, float)), f"Y→X score for {pair} must be numeric"

        if not allow_infinite:
            assert np.isfinite(xy), f"X→Y score for {pair} must be finite"
            assert np.isfinite(yx), f"Y→X score for {pair} must be finite"
        else:
            # Allow infinite but not NaN values
            assert not np.isnan(xy), f"X→Y score for {pair} cannot be NaN"
            assert not np.isnan(yx), f"Y→X score for {pair} cannot be NaN"


def assert_scores_in_range(
    scores: dict[tuple[Any, Any], tuple[float, float]], min_val: float | None = None, max_val: float | None = None
) -> None:
    """
    Assert that scores fall within expected ranges.

    Args:
        scores: Scoring results to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Example:
        >>> scores = {('a', 'b'): (0.5, 0.7)}
        >>> assert_scores_in_range(scores, min_val=0.0, max_val=1.0)  # Passes
    """
    for pair, (xy, yx) in scores.items():
        if min_val is not None:
            assert xy >= min_val, f"X→Y score {xy} for {pair} below minimum {min_val}"
            assert yx >= min_val, f"Y→X score {yx} for {pair} below minimum {min_val}"

        if max_val is not None:
            assert xy <= max_val, f"X→Y score {xy} for {pair} above maximum {max_val}"
            assert yx <= max_val, f"Y→X score {yx} for {pair} above maximum {max_val}"


def assert_scores_symmetric(scores: dict[tuple[Any, Any], tuple[float, float]], tolerance: float = 1e-10) -> None:
    """
    Assert that scores are symmetric (xy == yx).

    Args:
        scores: Scoring results to validate
        tolerance: Numerical tolerance for equality

    Example:
        >>> scores = {('a', 'b'): (0.5, 0.5)}
        >>> assert_scores_symmetric(scores)  # Passes
    """
    for pair, (xy, yx) in scores.items():
        # Handle infinite values specially
        if np.isinf(xy) and np.isinf(yx):
            # Both infinite with same sign means symmetric
            assert np.sign(xy) == np.sign(yx), f"Infinite scores for {pair} have different signs: {xy} vs {yx}"
        else:
            assert abs(xy - yx) < tolerance, f"Scores for {pair} not symmetric: {xy} vs {yx}"


def assert_scores_asymmetric(scores: dict[tuple[Any, Any], tuple[float, float]], min_difference: float = 1e-6) -> None:
    """
    Assert that at least some scores show asymmetry.

    Args:
        scores: Scoring results to validate
        min_difference: Minimum difference to consider asymmetric

    Example:
        >>> scores = {('a', 'b'): (0.3, 0.7), ('c', 'd'): (0.1, 0.1)}
        >>> assert_scores_asymmetric(scores)  # Passes (first pair is asymmetric)
    """
    asymmetric_count = 0
    for pair, (xy, yx) in scores.items():
        if abs(xy - yx) >= min_difference:
            asymmetric_count += 1

    assert asymmetric_count > 0, "No asymmetric relationships found in scores"


def assert_expected_score_values(
    scores: dict[tuple[Any, Any], tuple[float, float]],
    expected: dict[tuple[Any, Any], tuple[float, float]],
    tolerance: float = 1e-5,
) -> None:
    """
    Assert that scores match expected values within tolerance.

    Args:
        scores: Actual scoring results
        expected: Expected scoring results
        tolerance: Numerical tolerance for comparisons

    Example:
        >>> scores = {('a', 'b'): (0.5, 0.7)}
        >>> expected = {('a', 'b'): (0.5, 0.7)}
        >>> assert_expected_score_values(scores, expected)  # Passes
    """
    for pair, (exp_xy, exp_yx) in expected.items():
        assert pair in scores, f"Expected pair {pair} not found in results"

        act_xy, act_yx = scores[pair]

        assert abs(act_xy - exp_xy) < tolerance, f"X→Y score for {pair}: expected {exp_xy}, got {act_xy}"
        assert abs(act_yx - exp_yx) < tolerance, f"Y→X score for {pair}: expected {exp_yx}, got {act_yx}"


def assert_probabilistic_scores(scores: dict[tuple[Any, Any], tuple[float, float]]) -> None:
    """
    Assert that scores represent valid probabilities [0, 1].

    Args:
        scores: Scoring results to validate as probabilities

    Example:
        >>> scores = {('a', 'b'): (0.3, 0.7)}
        >>> assert_probabilistic_scores(scores)  # Passes
    """
    assert_scores_in_range(scores, min_val=0.0, max_val=1.0)


def assert_information_theoretic_scores(scores: dict[tuple[Any, Any], tuple[float, float]]) -> None:
    """
    Assert that scores represent valid information-theoretic measures.

    Args:
        scores: Information-theoretic scoring results (e.g., MI, PMI)

    Example:
        >>> scores = {('a', 'b'): (0.5, 0.5)}  # Valid MI scores
        >>> assert_information_theoretic_scores(scores)  # Passes
    """
    # Information-theoretic measures should be non-negative
    assert_scores_in_range(scores, min_val=0.0)


def assert_matrices_consistent(
    xy_matrix: np.ndarray, yx_matrix: np.ndarray, x_labels: list[str], y_labels: list[str]
) -> None:
    """
    Assert that score matrices have consistent dimensions and labels.

    Args:
        xy_matrix: X→Y score matrix
        yx_matrix: Y→X score matrix
        x_labels: Labels for X dimension
        y_labels: Labels for Y dimension

    Example:
        >>> xy = np.array([[1, 2], [3, 4]])
        >>> yx = np.array([[1, 3], [2, 4]])
        >>> x_labels = ['a', 'b']
        >>> y_labels = ['c', 'd']
        >>> assert_matrices_consistent(xy, yx, x_labels, y_labels)  # Passes
    """
    # Check matrix dimensions
    assert xy_matrix.shape[0] == len(x_labels), "XY matrix rows don't match X labels"
    assert xy_matrix.shape[1] == len(y_labels), "XY matrix columns don't match Y labels"
    assert yx_matrix.shape[0] == len(y_labels), "YX matrix rows don't match Y labels"
    assert yx_matrix.shape[1] == len(x_labels), "YX matrix columns don't match X labels"

    # Check that matrices are transposes for symmetric measures
    # (This is a property check, not always required)
    assert xy_matrix.shape == yx_matrix.T.shape, "Matrix shapes inconsistent for transpose"


def assert_performance_acceptable(execution_time: float, max_time: float = 30.0) -> None:
    """
    Assert that performance is within acceptable bounds.

    Args:
        execution_time: Actual execution time in seconds
        max_time: Maximum acceptable time in seconds

    Example:
        >>> assert_performance_acceptable(5.0, max_time=10.0)  # Passes
    """
    assert execution_time < max_time, f"Execution time {execution_time:.2f}s exceeds maximum {max_time}s"
    assert execution_time > 0, "Execution time must be positive"
